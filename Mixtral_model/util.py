
"""Alternative implementation using Hugging Face Transformers for HPC environments.

This module provides a drop-in replacement for util.py that uses Hugging Face
transformers instead of API calls. This allows running models directly on HPC
without needing Ollama or API servers.

Usage for HPC:
    1. Install dependencies: pip install -r requirements_hf.txt
    2. Set USE_HUGGINGFACE=1 environment variable, OR
    3. Replace util.py with this file's content, OR
    4. Update imports: from util_hf import llm_map, generate_questions, classify_llm
    
Model Configuration:
    - Set MODEL_CACHE_DIR environment variable to cache models (recommended for HPC)
    - Update MODEL_CONFIGS below with your model paths or HuggingFace model IDs
    - For Mixtral on HPC, you'll need at least 45GB GPU memory (or use quantization)
"""
print("util.py: Starting imports...", flush=True)
import os
print("util.py: os imported", flush=True)
import json
print("util.py: json imported", flush=True)
import random
print("util.py: random imported", flush=True)
import re
print("util.py: re imported", flush=True)
from pathlib import Path
print("util.py: Path imported", flush=True)
from typing import List
print("util.py: typing imported", flush=True)

import numpy as np
print("util.py: numpy imported", flush=True)

print("util.py: About to import langchain_core.prompts...", flush=True)
from langchain_core.prompts import ChatPromptTemplate
print("util.py: ✓ ChatPromptTemplate imported", flush=True)

print("util.py: About to import langchain_core.runnables...", flush=True)
from langchain_core.runnables import RunnableParallel
print("util.py: ✓ RunnableParallel imported", flush=True)

print("util.py: About to import langchain_community.llms...", flush=True)
from langchain_community.llms import HuggingFacePipeline
print("util.py: ✓ HuggingFacePipeline imported", flush=True)

print("util.py: About to import transformers...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
print("util.py: ✓ transformers imported", flush=True)

print("util.py: About to import torch...", flush=True)
import torch
print("util.py: ✓ torch imported", flush=True)

print("util.py: ALL IMPORTS COMPLETE!", flush=True)
from langchain_core.prompts import ChatPromptTemplate


# Model cache directory - set via environment variable or use default
MODEL_CACHE_DIR = os.getenv('HF_MODEL_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))

# Model mapping - UPDATE THESE for your HPC setup
# Note: Llama3 is commented out since it's gated and not needed for this project
# If you need Llama3 later, uncomment and set HF_TOKEN environment variable
MODEL_CONFIGS = {
    # 'llama3': {  # Commented out - gated model, not needed for this project
    #     'model_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
    #     'use_local': False,
    #     'local_path': None,
    #     'use_quantization': False,
    # },
    'mixtral': {
        'model_id': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'use_local': False,
        'local_path': os.getenv('MIXTRAL_MODEL_PATH', None),  # Can set via env var
        'use_quantization': True,  # Recommended for Mixtral (saves ~45GB -> ~23GB VRAM)
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # Note: llm_int8_enable_fp32_cpu_offload doesn't work with 4-bit quantization
            # bitsandbytes 4-bit doesn't support CPU offloading - must fit entirely on GPU
        ) if torch.cuda.is_available() else None,
    }
}

# Global model cache to avoid reloading
_model_cache = {}


class HuggingFaceLLMWrapper:
    """Wrapper to make Hugging Face models compatible with langchain ChatOpenAI interface."""
    
    def __init__(self, model_name: str, temperature: float, model_config: dict):
        self.temperature = temperature
        self.model_name = model_name
        
        # Check cache first
        cache_key = f"{model_name}_{temperature}"
        if cache_key in _model_cache:
            print(f"Using cached model: {model_name}", flush=True)
            cached = _model_cache[cache_key]
            self.tokenizer = cached['tokenizer']
            self.model = cached['model']
            self.pipeline = cached['pipeline']
            self.llm = cached['llm']
            return
        
        # Load model
        model_id = model_config['local_path'] if model_config['use_local'] and model_config['local_path'] else model_config['model_id']
        
        print(f"Loading model: {model_id} (this may take a few minutes on first load...)", flush=True)
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
        
        # Load tokenizer with error handling for gated models
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True,
                token=os.getenv('HF_TOKEN', None)  # Use token if provided
            )
        except Exception as e:
            if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
                raise RuntimeError(
                    f"Cannot access gated model {model_id}. "
                    f"This model requires HuggingFace authentication. "
                    f"Set HF_TOKEN environment variable or log in with: huggingface-cli login"
                ) from e
            raise
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        load_kwargs = {
            'cache_dir': MODEL_CACHE_DIR,
            'trust_remote_code': True,
        }
        
        if torch.cuda.is_available():
            if model_config.get('use_quantization', False) and model_config.get('quantization_config'):
                # Use quantization to save memory (critical for Mixtral)
                print("Using 4-bit quantization to save GPU memory...", flush=True)
                load_kwargs['quantization_config'] = model_config['quantization_config']
                # For 4-bit quantized models, bitsandbytes doesn't support CPU offloading
                # We need to fit everything on GPU. Try loading directly on GPU (device 0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU memory: {gpu_memory:.2f} GB total - Mixtral 4-bit needs ~23GB", flush=True)
                # Load directly on GPU without device_map="auto" to avoid meta tensor issues
                # This will try to fit everything on GPU - if it doesn't fit, we'll get OOM error
                load_kwargs['device_map'] = {"": 0}  # Load all layers on GPU 0
                load_kwargs['dtype'] = torch.float16  # Use dtype instead of torch_dtype (deprecated)
                # Don't use low_cpu_mem_usage with bitsandbytes + device_map to avoid meta tensor errors
                print("Loading model entirely on GPU (bitsandbytes 4-bit doesn't support CPU offloading)...", flush=True)
            else:
                load_kwargs['dtype'] = torch.float16  # Use dtype instead of torch_dtype (deprecated)
                load_kwargs['device_map'] = "auto"
        else:
            load_kwargs['dtype'] = torch.float32  # Use dtype instead of torch_dtype (deprecated)
        
        # Load model with error handling for gated models and OOM
        try:
            print("Loading model weights into GPU memory...", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                token=os.getenv('HF_TOKEN', None),  # Use token if provided
                **load_kwargs
            )
            print("Model weights loaded successfully", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                gpu_info = ""
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    gpu_info = f"Your {gpu_mem:.1f}GB GPU is insufficient for Mixtral even with 4-bit quantization.\n"
                raise RuntimeError(
                    f"GPU out of memory! {gpu_info}"
                    f"Mixtral 4-bit needs ~23GB GPU memory, but loading overhead requires more.\n"
                    f"Solutions:\n"
                    f"  1. Request a GPU with 30GB+ memory (e.g., A100 40GB)\n"
                    f"  2. Use a smaller model\n"
                    f"  3. Use 8-bit quantization instead (supports CPU offloading)"
                ) from e
            raise
        except Exception as e:
            if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
                raise RuntimeError(
                    f"Cannot access gated model {model_id}. "
                    f"This model requires HuggingFace authentication. "
                    f"Set HF_TOKEN environment variable or log in with: huggingface-cli login"
                ) from e
            raise
        
        # Create pipeline

        device = 0 if torch.cuda.is_available() else -1

        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "batch_size": 6, 
            "max_new_tokens": 512,
            "temperature": max(temperature, 0.1) if temperature > 0 else 0.7,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_full_text": False,  # Only return generated text
        }   

        if not hasattr(self.model, 'hf_device_map'):
             pipeline_kwargs["device"] = device

        self.pipeline = pipeline("text-generation", **pipeline_kwargs)
        print("Model loaded and ready!", flush=True)
        
        # Create langchain wrapper
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        # Cache for reuse
        _model_cache[cache_key] = {
            'tokenizer': self.tokenizer,
            'model': self.model,
            'pipeline': self.pipeline,
            'llm': self.llm
        }
    
    def invoke(self, messages):
        """Convert messages to format expected by the model and generate response."""
        # Handle different message formats
        if isinstance(messages, list):
            text = self._format_messages_for_model(messages)
        elif hasattr(messages, 'content'):
            text = messages.content
        elif isinstance(messages, dict) and 'summary' in messages:
            # Handle dict input from chains (e.g., {"summary": "..."})
            text = messages['summary']
        else:
            text = str(messages)
        
        # Generate response
        try:
            result = self.llm.invoke(text)
            if isinstance(result, str):
                content = result
            elif hasattr(result, 'content'):
                content = result.content
            else:
                content = str(result)
        except Exception as e:
            print(f"Error in model invocation: {e}")
            content = ""
        
        # Create a mock message object that has .content attribute
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        return MockMessage(content.strip())
    
    def __or__(self, other):
        """Support pipe operator for LangChain chains."""
        from langchain_core.runnables import RunnableSequence
        return RunnableSequence(self, other)
    
    def __ror__(self, other):
        """Support reverse pipe operator for LangChain chains."""
        from langchain_core.runnables import RunnableSequence
        return RunnableSequence(other, self) 
    def _format_messages_for_model(self, messages):
        """Format messages for Mixtral/Llama chat models using their chat templates."""
        # Convert langchain messages to format expected by chat models
        chat_messages = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
                # Map langchain roles to model roles
                if role == 'system':
                    chat_messages.append({"role": "system", "content": content})
                elif role == 'human' or role == 'user':
                    chat_messages.append({"role": "user", "content": content})
                elif role == 'assistant' or role == 'ai':
                    chat_messages.append({"role": "assistant", "content": content})
            elif hasattr(msg, 'content'):
                role = getattr(msg, 'type', 'user')
                content = msg.content
                if role == 'system':
                    chat_messages.append({"role": "system", "content": content})
                elif role in ['human', 'user']:
                    chat_messages.append({"role": "user", "content": content})
                elif role in ['assistant', 'ai']:
                    chat_messages.append({"role": "assistant", "content": content})
        
        # Use tokenizer's chat template if available (Mixtral/Llama support this)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                print(f"Warning: Could not apply chat template: {e}")
        
        # Fallback: simple formatting
        formatted = []
        for msg in chat_messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted.append(f"System: {content}\n")
            elif role == 'user':
                formatted.append(f"User: {content}\n")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}\n")
        return "".join(formatted)


def DefaultNLELlama3_70bChatOpenAI(temperature: float):
    """Create Llama3 model using Hugging Face.
    
    NOTE: Llama3 is gated and requires HuggingFace authentication.
    This function is commented out since Llama3 is not needed for this project.
    """
    raise NotImplementedError(
        "Llama3 is not configured. It's a gated model requiring HuggingFace authentication. "
        "If you need Llama3, uncomment the config in MODEL_CONFIGS and set HF_TOKEN."
    )
    # config = MODEL_CONFIGS['llama3']
    # wrapper = HuggingFaceLLMWrapper('llama3', temperature, config)
    # return wrapper

def DefaultNLEMixtral_8x7bChatOpenAI(temperature: float):
    """Create Mixtral model using Hugging Face."""
    config = MODEL_CONFIGS['mixtral']
    wrapper = HuggingFaceLLMWrapper('mixtral', temperature, config)
    return wrapper.llm  # Return wrapper, not wrapper.llm
def DefaultNLEMixtral_8x7bChatOpenAI_Direct(temperature: float):
    """Create Mixtral model for direct invoke calls (generate_questions)."""
    config = MODEL_CONFIGS['mixtral']
    wrapper = HuggingFaceLLMWrapper('mixtral', temperature, config)
    return wrapper

# Lazy model loading - only create wrapper functions, don't load models until called
# Llama3 removed since it's gated and not needed - prevents authentication errors
llm_map = {
    # 'llama3': DefaultNLELlama3_70bChatOpenAI,  # Commented out - not needed, requires authentication
    'mixtral': DefaultNLEMixtral_8x7bChatOpenAI
}

llm_map_direct = {
    'mixtral': DefaultNLEMixtral_8x7bChatOpenAI_Direct
    }

# Verify that only requested models will be loaded
# Models are only loaded when llm_map[model_name]() is called


def generate_questions(llm_name: str,
                       question: str,
                       temperature: float,
                       no_questions: int = 10) -> List[str]:
    """
    Generates a number of semantically equivalent questions to the input
    questions using an LLM.
    """
    if llm_name not in llm_map_direct:
        raise ValueError(f"Unknown LLM: {llm_name}. Available models: {list(llm_map.keys())}")
    llm = llm_map_direct[llm_name](temperature=temperature)

    questions = [question]

    if 'mixtral' not in llm_name:
        prompt = [('system', 'You are asked to rephrase a question in a semantically equivalent but syntactically different way.'
                             ' Vary the length of the question as long as you do not alter the meaning of the question.'
                             ' Provide only the rephrased sentence.'
                             ' The original question is the following: {question}.'
                             ' Also, the following list contains some questions that you already generated, do not repeat yourself:\n {alternative_questions}'
                   ),
            ('human', f'Rephrase the original question.')]
    else:
        prompt = [('user',
                   'You are asked to rephrase a question in a semantically equivalent but syntactically different way.'
                   ' Vary the length of the question as long as you do not alter the meaning of the question.'
                   ' Provide only the rephrased sentence with no additional notes.'
                   ' The original question is the following: {question}.'
                   ' Also, the following list contains some questions that you already generated, do not repeat yourself:\n {alternative_questions}.'
                   )]

    i = 0
    while i < no_questions-1:
        chat_prompt = ChatPromptTemplate.from_messages(prompt)

        alternative_questions_str = ''
        for q in questions[-10:]:
            alternative_questions_str += f'- {q}\n'

        alternative_questions_str = re.escape(alternative_questions_str)

        messages = chat_prompt.format_messages(question=question,
                                               alternative_questions=alternative_questions_str)

        new_question = llm.invoke(messages).content

        questions.append(new_question)
        i += 1

    return questions


def classify_llm(llm_name: str,
                 prompts: List,
                 summary: str,
                 temperature: float,
                 no_parallel_calls: int = 1):
    """
    Calls an LLM to perform a classification with a given prompt.
    """
    if llm_name not in llm_map:
        raise ValueError(f"Unknown LLM: {llm_name}. Available models: {list(llm_map.keys())}")
    llm = llm_map[llm_name](temperature=temperature)

    chains = {f"{str(q)}_{str(k)}": (ChatPromptTemplate.from_messages(map(lambda x: tuple(x), prompt)) | llm) for k in range(no_parallel_calls) for q, prompt in enumerate(prompts)}

    map_chain = RunnableParallel(chains)

    results = map_chain.invoke({"summary": summary})
    
    # Wrap string results in objects with .content attribute
    class MessageWrapper:
        def __init__(self, content):
            if isinstance(content, str):
                self.content = content
            elif hasattr(content, 'content'):
                self.content = content.content
            else:
                self.content = str(content)
    
    wrapped_results = {k: MessageWrapper(v) for k, v in results.items()}
    return wrapped_results
