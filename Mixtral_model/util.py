import os
import random
import re
from pathlib import Path
from typing import List
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from typing import List, Callable
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score

MODEL_CACHE_DIR = os.getenv('HF_MODEL_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))

MODEL_CONFIGS = {
    'mixtral': {
        'model_id': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'use_local': False,
        'local_path': os.getenv('MIXTRAL_MODEL_PATH', None),  
        'use_quantization': True,  
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if torch.cuda.is_available() else None,
    }
}
    
_model_cache = {}


class HuggingFaceLLMWrapper:
    """Wrapper to make Hugging Face models compatible with langchain ChatOpenAI interface."""
    
    def __init__(self, model_name: str, temperature: float, model_config: dict):
        self.temperature = temperature
        self.model_name = model_name
        
        cache_key = f"{model_name}_{temperature}"
        if cache_key in _model_cache:
            print(f"Using cached model: {model_name}", flush=True)
            cached = _model_cache[cache_key]
            self.tokenizer = cached['tokenizer']
            self.model = cached['model']
            self.pipeline = cached['pipeline']
            self.llm = cached['llm']
            return  
        
        model_id = model_config['local_path'] if model_config['use_local'] and model_config['local_path'] else model_config['model_id']
        
        print(f"Loading model: {model_id} (this may take a few minutes on first load...)", flush=True)
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True,
                token=os.getenv('HF_TOKEN', None)  
            )
        except Exception as e:
            if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
                raise RuntimeError(
                    f"Cannot access gated model {model_id}. "
                    f"This model requires HuggingFace authentication. "
                    f"Set HF_TOKEN environment variable or log in with: huggingface-cli login"
                ) from e
            raise
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {
            'cache_dir': MODEL_CACHE_DIR,
            'trust_remote_code': True,
        }
        
        if torch.cuda.is_available():
            if model_config.get('use_quantization', False) and model_config.get('quantization_config'):
                print("Using 4-bit quantization to save GPU memory...", flush=True)
                load_kwargs['quantization_config'] = model_config['quantization_config']
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU memory: {gpu_memory:.2f} GB total - Mixtral 4-bit needs ~23GB", flush=True)
                load_kwargs['device_map'] = {"": 0} 
                load_kwargs['dtype'] = torch.float16  
                print("Loading model entirely on GPU (bitsandbytes 4-bit doesn't support CPU offloading)...", flush=True)
            else:
                load_kwargs['dtype'] = torch.float16  
                load_kwargs['device_map'] = "auto"
        else:
            load_kwargs['dtype'] = torch.float32  
        
        try:
            print("Loading model weights into GPU memory...", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                token=os.getenv('HF_TOKEN', None),  
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
        


        device = 0 if torch.cuda.is_available() else -1

        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "batch_size": 1, 
            "max_new_tokens": 512,
            "temperature": max(temperature, 0.1) if temperature > 0 else 0.7,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_full_text": False,  
        }   

        if not hasattr(self.model, 'hf_device_map'):
             pipeline_kwargs["device"] = device

        self.pipeline = pipeline("text-generation", **pipeline_kwargs)
        print("Model loaded and ready!", flush=True)
        
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        _model_cache[cache_key] = {
            'tokenizer': self.tokenizer,
            'model': self.model,
            'pipeline': self.pipeline,
            'llm': self.llm
        }
    
    def invoke(self, messages):
        """Convert messages to format expected by the model and generate response."""
        if isinstance(messages, list):
            text = self._format_messages_for_model(messages)
        elif hasattr(messages, 'content'):
            text = messages.content
        elif isinstance(messages, dict) and 'summary' in messages:
            text = messages['summary']
        else:
            text = str(messages)
        
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
        chat_messages = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
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

def DefaultNLEMixtral_8x7bChatOpenAI(temperature: float):
    """Create Mixtral model using Hugging Face."""
    config = MODEL_CONFIGS['mixtral']
    wrapper = HuggingFaceLLMWrapper('mixtral', temperature, config)
    return wrapper.llm  

def DefaultNLEMixtral_8x7bChatOpenAI_Direct(temperature: float):
    """Create Mixtral model for direct invoke calls (generate_questions)."""
    config = MODEL_CONFIGS['mixtral']
    wrapper = HuggingFaceLLMWrapper('mixtral', temperature, config)
    return wrapper

llm_map = {
    'mixtral': DefaultNLEMixtral_8x7bChatOpenAI
}

llm_map_direct = {
    'mixtral': DefaultNLEMixtral_8x7bChatOpenAI_Direct
    }

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




def parse_json_to_dataframe(input_file: Path) -> pd.DataFrame:
    """
    Parse a JSON file containing a list of dictionaries and convert it into a
     Pandas DataFrame. It assumes a multilabel classification task.

    Each dictionary in the input JSON file should have the following structure:
    {
        "cat_name": str,
        "possible_tags": List[str],
        "prompt": str,
        "question": str,
        "entries": [
            {
                "id": str/int,
                "summary": str,
                "tags": List[str]  # ground truth (multilabel classification)
            }
        ]
    }

    For each entry in the "entries" list of each dictionary, a row will be added to the CSV file.
    The CSV file will contain the following columns:
    - cat_name
    - possible_tags
    - prompt
    - question
    - id
    - summary
    - tags (comma-separated)

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output CSV file.
    :return: pandas DataFrame containing the parsed data.
    """

    # Open the JSON file
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    rows = []
    # Process each dictionary in the JSON data
    for item in data:
        cat_name = item["cat_name"]
        possible_tags = item["possible_tags"]
        prompt = item["prompt"]
        question = item["question"]

        entries = item["entries"]
        for entry in entries:
            entry_id = entry["id"]
            if entry_id == '':
                continue
            summary = entry["summary"]

            difficulty = entry["difficulty"]

            assert len(entry["tags"]) > 0
            tags = ", ".join(entry["tags"])

            # Append row to the list
            rows.append([cat_name, possible_tags, prompt, question,
                         entry_id, summary, tags, difficulty])

    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows, columns=["cat_name", "classes", "prompt",
                                     "question", "id", "summary", "ground_truth", "difficulty"])
    return df


def TVD(distribution1, distributions):
    return 0.5*np.abs(np.expand_dims(distribution1, 0) - distributions).sum(1)


def safe_get_experiment(data_dict: dict, key: str):
    """
    Safely get an experiment from data_dict, returning None if key doesn't exist.
    """
    try:
        return data_dict[key]
    except KeyError:
        return None

def get_available_sample_ids(results_folder: Path,
                             prompt_type: str,
                             llm: str,
                             Q: int,
                             A: int,
                             temp_question: float,
                             temp_answer: float,
                             sample_ids: List[int]) -> List[int]:
    """
    Filter sample_ids to only include those that exist in the results file.
    
    Args:
        results_folder: Path to the results folder
        prompt_type: Prompt type (e.g., 'simple', 'instruct', 'fewshot')
        llm: LLM name
        Q: Number of questions
        A: Number of answers
        temp_question: Temperature for questions
        temp_answer: Temperature for answers
        sample_ids: Original list of sample IDs to filter
        
    Returns:
        List of sample IDs that exist in the results file
    """
    filename = f"results_{prompt_type}.json"
    filepath = Path(results_folder, filename)
    
    if not filepath.exists():
        return []
    
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    
    expected_key_prefix = f"_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"
    available_sample_ids = []
    
    for key in data_dict.keys():
        if key.endswith(expected_key_prefix):
            s_id_str = key.replace(expected_key_prefix, "")
            try:
                s_id = int(s_id_str)
                if s_id in sample_ids:
                    available_sample_ids.append(s_id)
            except ValueError:
                continue
    
    return sorted(available_sample_ids)


def update_results_files_with_distributions(
    results_folder: Path,
    prompt_types: dict,
    class_labels: List[str],
    class_extractor_fun: Callable
):
    """
    Update results JSON files to include 'distribution' and 'info_answers' keys computed from 'raw_answers'.
    
    Args:
        results_folder: Path to the results folder
        prompt_types: Dictionary of prompt types (keys are used for filenames)
        class_labels: List of class labels (including N/A)
        class_extractor_fun: Function to extract class label from answer text
    """
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    n_classes = len(class_labels)
    
    for prompt_type in prompt_types.keys():
        filename = f"results_{prompt_type}.json"
        filepath = Path(results_folder, filename)
        
        if not filepath.exists():
            print(f"Warning: {filename} does not exist, skipping...")
            continue
        
        print(f"Processing {filename}...")
        
        # Load the results file
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
        
        updated_count = 0
        total_experiments = 0
        
        # Process each experiment
        for key, experiment in data_dict.items():
            # Skip non-experiment keys (like question keys)
            if not isinstance(experiment, dict) or 'id' not in experiment:
                continue
            
            total_experiments += 1
            
            needs_update = False
            
            # Skip if raw_answers doesn't exist
            if "raw_answers" not in experiment:
                if "distribution" not in experiment or "info_answers" not in experiment:
                    print(f"Warning: Experiment {key} has neither 'distribution'/'info_answers' nor 'raw_answers'")
                continue
            
            raw_answers = experiment["raw_answers"]
            total_answers = len(raw_answers)
            
            if total_answers == 0:
                print(f"Warning: Experiment {key} has no raw_answers")
                continue
            
            # Compute distribution from raw_answers if missing
            if "distribution" not in experiment:
                label_counts = np.zeros(n_classes)
                
                for answer_key, answer_data in raw_answers.items():
                    answer_content = answer_data.get("content", "")
                    pred_label = class_extractor_fun(answer_content)
                    if pred_label in class_labels_to_id:
                        label_counts[class_labels_to_id[pred_label]] += 1
                
                # Normalize to get distribution
                distribution = (label_counts / total_answers).tolist()
                
                # Add distribution to experiment
                experiment["distribution"] = distribution
                needs_update = True
            
            # Compute info_answers from raw_answers if missing
            if "info_answers" not in experiment:
                info_answers = {}
                
                # Group answers by question ID (raw_answers keys are like "0_0", "0_1", "1_0", etc.)
                for answer_key, answer_data in raw_answers.items():
                    # Parse question_id and answer_id from key (format: "q_a")
                    parts = answer_key.split('_')
                    if len(parts) >= 2:
                        q_id = parts[0]
                        answer_content = answer_data.get("content", "")
                        pred_label = class_extractor_fun(answer_content)
                        
                        # Initialize list for this question if needed
                        if q_id not in info_answers:
                            info_answers[q_id] = []
                        
                        # Append predicted label (as string, matching expected format)
                        info_answers[q_id].append(pred_label)
                
                # Add info_answers to experiment
                experiment["info_answers"] = info_answers
                needs_update = True
            
            if needs_update:
                updated_count += 1
        
        # Save the updated results file
        if updated_count > 0:
            with open(filepath, 'w') as f:
                json.dump(data_dict, f)
            print(f"Updated {updated_count} out of {total_experiments} experiments in {filename}")
        else:
            print(f"No updates needed for {filename} (all experiments already have distributions and info_answers)")


def plot_TVD_info(sample_ids: List[int],
                  prompt_types: dict,
                  llms: List[str],
                  Qs: List[int],
                  temp_questions: List[float],
                  As: List[int],
                  temp_answers: List[float],
                  class_labels: List[str],
                  results_folder: Path):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for prompt_type, prompt in prompt_types.items():
        filename = f"results_{prompt_type}.json"
        print(f'Prompt type: {prompt_type}')

        # Open the JSON file and load its contents into a Python dictionary
        with open(Path(results_folder, filename), 'r') as f:
            data_dict = json.load(f)

        for llm in llms:
            for Q in Qs:
                for A in As:
                    for temp_question in temp_questions:
                        for temp_answer in temp_answers:

                            # compute distribution over class labels predicted over the Q questions
                            samples_distributions = np.zeros((n_samples, n_classes - 1))

                            # compute boolean class assignment matrix
                            boolean_class_matrix = np.zeros((n_samples, n_classes - 1),
                                                            dtype='bool')

                            for idx, s_id in enumerate(sample_ids):
                                key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                experiment = data_dict[key]

                                target = experiment['target']
                                if target == 'not_entailment':
                                    target = 'contradiction/neutral'

                                target_id = class_labels_to_id[target]

                                boolean_class_matrix[
                                    idx, class_labels_to_id[target]] = True

                                samples_distributions[idx] = np.array(experiment["distribution"])[:-1]

                            TVD_matrix = np.zeros((n_samples, n_samples))

                            for idx, s_id in enumerate(sample_ids):
                                TVD_matrix[idx, :] = 1. - TVD(samples_distributions[idx, :-1],
                                                          samples_distributions[:, :-1])

                            fig, axes = plt.subplots(1, n_classes - 1, figsize=(n_classes * 5, 4))  # Create a grid of subplots
                            for c in range(n_classes - 1):
                                class_filter = boolean_class_matrix[:, c]
                                if class_filter.sum() == 0:
                                    continue

                                sns.heatmap(TVD_matrix[class_filter][:, class_filter],
                                            ax=axes[c])
                                axes[c].set_title(f'Pairwise TVD - Class {class_labels[c]}')

                            # plt.tight_layout()
                            plt.savefig(Path(results_folder, f'TVD_matrix_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                            plt.show()

                            fig, axes = plt.subplots(1, n_classes - 1, figsize=(n_classes * 5, 4))  # Create a grid of subplots
                            for c in range(n_classes-1):
                                class_filter = boolean_class_matrix[:, c]
                                if class_filter.sum() == 0:
                                    continue

                                sns.histplot(
                                    np.reshape(TVD_matrix[class_filter][:,
                                               class_filter], -1),
                                    bins=20, stat='probability', kde=False,
                                    ax=axes[c])
                                axes[c].set_xlabel(f'TVD Value')
                                axes[c].set_ylabel(f'Frequency')

                            # plt.tight_layout()
                            plt.savefig(Path(results_folder, f'TVD_histogram_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))



def print_consistency(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path,
                       noise_amount=0.,
                       filter_zero_sensitivity=False):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"
    print(f'Prompt type: {prompt_type}')

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename), 'r') as f:
        data_dict = json.load(f)

    # compute distribution over class labels predicted over the Q questions
    samples_distributions = np.zeros(
        (n_samples, n_classes))

    # compute boolean class assignment matrix
    boolean_class_matrix = np.zeros(
        (n_samples, n_classes - 1),
        dtype='bool')

    for idx, s_id in enumerate(sample_ids):
        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        experiment = data_dict[key]

        target = experiment['target']
        if target == 'not_entailment':
            target = 'contradiction/neutral'

        target_id = class_labels_to_id[target]

        boolean_class_matrix[
            idx, class_labels_to_id[
                target]] = True

        samples_distributions[idx] = np.array(
            experiment["distribution"])[:]

        # Generate a random float between 0 and 1
        random_float = random.random()

        # If the generated float is less than or equal to p, pick a random value from the list
        if random_float < noise_amount:
            sd = np.zeros(len(class_labels))

            for _ in range(Q):
                pred_id = random.choice([i for i in range(len(class_labels))])
                sd[pred_id] += 1

            sd = sd / Q
            print(entropy(sd) / np.log(n_classes))

            samples_distributions[idx] = sd

    consistency = np.zeros(n_classes-1)
    consistency_not_averaged = []
    TVD_matrix_per_class = []
    for c in range(n_classes-1):  # avoid NA

        samples_distributions_c = samples_distributions[boolean_class_matrix[:, c], :]

        if filter_zero_sensitivity: # filter out elements with zero sensitivity, where the prompt rephrasing has no effect
            zero_sensitivity_mask = entropy(samples_distributions_c, axis=1) / np.log(n_classes) == 0.
            samples_distributions_c = samples_distributions_c[~zero_sensitivity_mask]

        n_samples_c = samples_distributions_c.shape[0]

        if n_samples_c > 0:
            TVD_matrix_c = np.zeros((n_samples_c, n_samples_c))

            for idx in range(n_samples_c):
                TVD_matrix_c[idx, :] = 1. - TVD(
                    samples_distributions_c[idx, :],
                    samples_distributions_c[:, :])

        else:
            TVD_matrix_c = np.zeros((1, 1))

        consistency[c] = TVD_matrix_c.mean()
        consistency_not_averaged.append(TVD_matrix_c.reshape(-1))
        TVD_matrix_per_class.append(TVD_matrix_c)

    consistency_not_averaged = np.concatenate(consistency_not_averaged)
    print(f"Avg consistency: {consistency_not_averaged.mean()},"
          f"Std consistency: {consistency_not_averaged.std()}")

    return TVD_matrix_per_class

def print_consistency_over_samples(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path,
                       noise_amount=0.,
                       filter_zero_sensitivity=False):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"
    print(f'Prompt type: {prompt_type}')

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename), 'r') as f:
        data_dict = json.load(f)

    # compute distribution over class labels predicted over the Q questions
    samples_distributions = np.zeros(
        (n_samples, n_classes))

    # compute boolean class assignment matrix
    boolean_class_matrix = np.zeros(
        (n_samples, n_classes - 1),
        dtype='bool')

    for idx, s_id in enumerate(sample_ids):
        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        if key not in data_dict:
            # Skip missing experiments
            continue

        experiment = data_dict[key]

        boolean_class_matrix[
            idx, class_labels_to_id[
                experiment['target']]] = True

        samples_distributions[idx] = np.array(
            experiment["distribution"])[:]

        # Generate a random float between 0 and 1
        random_float = random.random()

        # If the generated float is less than or equal to p, pick a random value from the list
        if random_float < noise_amount:
            sd = np.zeros(len(class_labels))

            for _ in range(Q):
                pred_id = random.choice([i for i in range(len(class_labels))])
                sd[pred_id] += 1

            sd = sd / Q
            print(entropy(sd) / np.log(n_classes))

            samples_distributions[idx] = sd

    consistency = np.zeros(n_classes-1)
    consistency_not_averaged = []
    TVD_matrix_per_class = []
    for c in range(n_classes-1):  # avoid NA

        samples_distributions_c = samples_distributions[boolean_class_matrix[:, c], :]

        if filter_zero_sensitivity: # filter out elements with zero sensitivity, where the prompt rephrasing has no effect
            zero_sensitivity_mask = entropy(samples_distributions_c, axis=1) / np.log(n_classes) == 0.
            samples_distributions_c = samples_distributions_c[~zero_sensitivity_mask]

        n_samples_c = samples_distributions_c.shape[0]

        if n_samples_c > 0:
            TVD_matrix_c = np.zeros((n_samples_c, n_samples_c))

            for idx in range(n_samples_c):
                TVD_matrix_c[idx, :] = 1. - TVD(
                    samples_distributions_c[idx, :],
                    samples_distributions_c[:, :])

        else:
            TVD_matrix_c = np.zeros((1, 1))

        consistency[c] = TVD_matrix_c.mean()
        consistency_not_averaged.append(TVD_matrix_c.reshape(-1))
        TVD_matrix_per_class.append(TVD_matrix_c)


    consistency_not_averaged = np.concatenate(consistency_not_averaged)
    print(f"Avg consistency over classes: {consistency_not_averaged.mean()},"
          f"Std consistency over classes: {consistency_not_averaged.std()}")

    return TVD_matrix_per_class


def plot_questions_vs_predicted_distribution(sample_ids: List[int],
                                             prompt_types: dict,
                                             llms: List[str],
                                             Qs: List[int],
                                             temp_questions: List[float],
                                             As: List[int],
                                             temp_answers: List[float],
                                             class_labels: List[str],
                                             results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for prompt_type, prompt in prompt_types.items():
        filename = f"results_{prompt_type}.json"
        print(f'Prompt type: {prompt_type}')

        # Open the JSON file and load its contents into a Python dictionary
        with open(Path(results_folder, filename), 'r') as f:
            data_dict = json.load(f)

        for llm in llms:
            for Q in Qs:
                for A in As:
                    for temp_question in temp_questions:
                        for temp_answer in temp_answers:

                            fig, axes = plt.subplots(Q, n_classes - 1, figsize=(
                            n_classes * 5, Q * 4))  # Create a grid of subplots

                            for q in range(Q):
                                for c in range(n_classes - 1):
                                    # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                                    q_c_distribution = np.zeros(n_classes)

                                    for idx, s_id in enumerate(sample_ids):
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        experiment = data_dict[key]

                                        target = experiment['target']
                                        if target == 'not_entailment':
                                            target = 'contradiction/neutral'

                                        target_id = class_labels_to_id[target]

                                        pred_id = class_labels_to_id[
                                            experiment["info_answers"][str(q)][0]]

                                        if target_id != c:
                                            continue

                                        q_c_distribution[pred_id] += 1

                                    total = q_c_distribution.sum()
                                    if total == 0:
                                        total = 1.

                                    axes[q, c].bar(np.arange(n_classes),
                                                   q_c_distribution / total)
                                    axes[q, c].set_ylim([0., 1.])
                                    axes[q, c].set_title(f"Class {class_labels[c]}")
                                    axes[q, c].set_xlabel(f"Predicted Class")
                                    axes[q, c].set_xticks(np.arange(n_classes), class_labels,
                                                          rotation='vertical')
                                    axes[q, c].set_ylabel(f"Question ID {q}")

                            plt.tight_layout()
                            plt.savefig(Path(results_folder,
                                             f'question_vs_trueclass_prediction_distributions_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                            plt.show()


def plot_questions_vs_class_sensitivity(sample_ids: List[int],
                                             prompt_types: dict,
                                             llms: List[str],
                                             Qs: List[int],
                                             temp_questions: List[float],
                                             As: List[int],
                                             temp_answers: List[float],
                                             class_labels: List[str],
                                             results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for prompt_type, prompt in prompt_types.items():
        filename = f"results_{prompt_type}.json"
        print(f'Prompt type: {prompt_type}')

        # Open the JSON file and load its contents into a Python dictionary
        with open(Path(results_folder, filename), 'r') as f:
            data_dict = json.load(f)

        for llm in llms:
            for Q in Qs:
                for A in As:
                    for temp_question in temp_questions:
                        for temp_answer in temp_answers:

                            entropy_matrix = np.zeros((Q, n_classes))

                            for q in range(Q):
                                for c in range(n_classes - 1):
                                    # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                                    q_c_distribution = np.zeros(n_classes)

                                    for idx, s_id in enumerate(sample_ids):
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        experiment = data_dict[key]

                                        target = experiment['target']
                                        if target == 'not_entailment':
                                            target = 'contradiction/neutral'

                                        target_id = class_labels_to_id[target]
                                        pred_id = class_labels_to_id[
                                            experiment["info_answers"][str(q)][0]]

                                        if target_id != c:
                                            continue

                                        q_c_distribution[pred_id] += 1

                                    entropy_matrix[q, c] = entropy(q_c_distribution)/np.log(n_classes)

                            ax = sns.heatmap(entropy_matrix, vmax=1.)
                            plt.xlabel('Class ID')
                            plt.ylabel('Question ID')
                            ax.set_xticks(
                                np.arange(len(class_labels)) + 0.5,
                                class_labels, rotation='vertical')

                            plt.tight_layout()
                            plt.savefig(Path(results_folder,
                                             f'question_vs_class_entropy_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                            plt.show()


def print_classification_scores(sample_ids: List[int],
                                prompt_type: str,
                                llm: str,
                                Q: int,
                                temp_question: float,
                                A: int,
                                temp_answer: float,
                                class_labels: List[str],
                                results_folder: Path):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    row_headers = [f"Question {i + 1}" for i in range(Q)]
    col_headers = ["Acc", "Micro F1", "Macro F1"]

    table = [None for _ in range(Q)]

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename), 'r') as f:
        data_dict = json.load(f)

    pred_labels = np.zeros(Q * n_samples)
    true_labels = np.zeros(Q * n_samples)

    for q in range(Q):
        pred_labels_q = np.zeros(n_samples)
        true_labels_q = np.zeros(n_samples)

        for idx, s_id in enumerate(sample_ids):
            key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

            experiment = data_dict[key]

            target = experiment['target']
            if target == 'not_entailment':
                target = 'contradiction/neutral'

            target_id = class_labels_to_id[target]
            true_labels_q[idx] += target_id
            true_labels[q * n_samples + idx] += target_id

            pred_id = class_labels_to_id[
                experiment["info_answers"][str(q)][0]]
            pred_labels_q[idx] += pred_id
            pred_labels[q * n_samples + idx] += pred_id

        acc = accuracy_score(true_labels_q, pred_labels_q)
        microf1 = f1_score(true_labels_q, pred_labels_q, average='micro')
        macrof1 = f1_score(true_labels_q, pred_labels_q, average='macro')

        table[q] = [acc, microf1, macrof1]

        # print(f"Question {q}, accuracy: {acc}, micro f1-score: {microf1}, macro f1-score: {macrof1}")

    # print(Table(row_headers, col_headers, table))

    print(
        f"Global scores, accuracy: {accuracy_score(true_labels_q, pred_labels_q)}, micro f1-score: {f1_score(true_labels_q, pred_labels_q, average='micro')}, macro f1-score: {f1_score(true_labels_q, pred_labels_q, average='macro')}")

    std_over_microf1 = np.std([table[q][1] for q in range(Q)])
    print(f'Standard deviation of microf1 score: {std_over_microf1}')



def plot_avg_sensitivity(sample_ids: List[int],
                 prompt_types: dict,
                 llms: List[str],
                 Qs: List[int],
                 temp_questions: List[float],
                 As: List[int],
                 temp_answers: List[float],
                 class_labels: List[str],
                 results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for llm in llms:
        for Q in Qs:
            for A in As:
                for temp_question in temp_questions:
                    for temp_answer in temp_answers:

                        fig_class = plt.figure()
                        fig_q = plt.figure()

                        for prompt_type, prompt in prompt_types.items():
                            filename = f"results_{prompt_type}.json"

                            # Open the JSON file and load its contents into a Python dictionary
                            with open(Path(results_folder, filename),
                                      'r') as f:
                                data_dict = json.load(f)

                            entropy_matrix = np.zeros((Q, n_classes))

                            for q in range(Q):
                                for c in range(n_classes - 1):
                                    # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                                    q_c_distribution = np.zeros(n_classes)

                                    for idx, s_id in enumerate(sample_ids):
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        experiment = data_dict[key]

                                        target = experiment['target']
                                        if target == 'not_entailment':
                                            target = 'contradiction/neutral'

                                        target_id = class_labels_to_id[target]

                                        pred_id = class_labels_to_id[
                                            experiment["info_answers"][str(q)][0]]

                                        if target_id != c:
                                            continue

                                        q_c_distribution[pred_id] += 1

                                    entropy_matrix[q, c] = entropy(q_c_distribution)/np.log(n_classes)

                            avg_entropy_per_class = entropy_matrix.mean(axis=0)
                            avg_entropy_per_q = entropy_matrix.mean(axis=1)

                            plt.figure(fig_class)

                            #plt.scatter(np.arange(n_classes), avg_entropy_per_class, label=prompt_type)
                            plt.plot(avg_entropy_per_class, label=prompt_type)
                            # Add error bars (standard deviation)
                            # plt.errorbar(np.arange(avg_entropy_per_class.shape[0]), avg_entropy_per_class, yerr=entropy_matrix.std(axis=0), fmt='-o',  solid_capstyle='projecting', capsize=5, label=prompt_type)

                            plt.ylabel('Avg Entropy')
                            plt.xticks(np.arange(len(class_labels)),
                                                class_labels,
                                                rotation='vertical')

                            plt.figure(fig_q)
                            #plt.scatter(np.arange(Q), avg_entropy_per_q, label=prompt_type)
                            # plt.plot(avg_entropy_per_q, label=prompt_type)
                            # Add error bars (standard deviation)
                            plt.errorbar(np.arange(avg_entropy_per_q.shape[0]), avg_entropy_per_q, yerr=entropy_matrix.std(axis=1), fmt='-o',  solid_capstyle='projecting', capsize=5, label=prompt_type)


                            plt.ylabel('Avg Entropy')
                            plt.xticks(np.arange(Q),
                                                [f'Question {i+1}' for i in range(Q)],
                                                rotation='vertical')

                        plt.figure(fig_class)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(Path(results_folder,
                                         f'avg_entropy_per_class_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                        plt.show()

                        plt.figure(fig_q)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(Path(results_folder,
                                         f'avg_entropy_per_question_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                        plt.show()


def print_test_sensitivity_over_samples(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path,
                       noise_amount=0.):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename),
              'r') as f:
        data_dict = json.load(f)

    entropy_matrix = np.zeros(n_samples)

    for idx, s_id in enumerate(sample_ids):
        # compute mean of entropy over samples (over samples) distribution over predicted class labels for a given question and class
        c_distribution = np.zeros(n_classes)

        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        experiment = data_dict[key]

        for q in range(Q):

            pred_id = class_labels_to_id[
                experiment["info_answers"][str(q)][0]]

            # Generate a random float between 0 and 1
            random_float = random.random()

            # If the generated float is less than or equal to p, pick a random value from the list
            if random_float < noise_amount:
                pred_id = random.choice([i for i in range(len(class_labels))])

            c_distribution[pred_id] += 1

        entropy_matrix[idx] = entropy(c_distribution)/np.log(n_classes)

    avg_entropy_over_samples = entropy_matrix.mean()
    std_entropy_over_samples = entropy_matrix.std()

    print(f"Avg Entropy over samples: {avg_entropy_over_samples},"
          f"Std over samples: {std_entropy_over_samples}")

    return entropy_matrix

def sensitivity_per_class(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename),
              'r') as f:
        data_dict = json.load(f)

    sensitivity_matrix = [[] for _ in range(n_classes)]

    s_ids = [[] for _ in range(n_classes)]
    for c in range(n_classes):

        for idx, s_id in enumerate(sample_ids):
            key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

            experiment = data_dict[key]
            target = experiment['target']
            if target == 'not_entailment':
                target = 'contradiction/neutral'

            target_id = class_labels_to_id[target]

            if target_id != c:
                continue

            c_distribution = np.zeros(n_classes)
            for q in range(Q):
                # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                pred_id = class_labels_to_id[
                    experiment["info_answers"][str(q)][0]]
                c_distribution[pred_id] += 1

            s_ids[c].append(s_id)
            sensitivity_matrix[c].append(entropy(c_distribution)/np.log(n_classes))


    # print(class_labels)
    # print(list(zip(sensitivity_matrix[class_labels_to_id['Description']], s_ids[class_labels_to_id['Description']])))

    return sensitivity_matrix

def get_predicted_distribution(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename),
              'r') as f:
        data_dict = json.load(f)

    # Filter to only include sample IDs that exist in the results file
    available_sample_ids = get_available_sample_ids(
        results_folder, prompt_type, llm, Q, A, temp_question, temp_answer, sample_ids
    )
    
    if len(available_sample_ids) == 0:
        print(f"Warning: No samples found in {filename} for {llm}, Q={Q}, A={A}, temp_q={temp_question}, temp_a={temp_answer}")
        return np.zeros((n_samples, n_classes))
    
    samples_distributions = np.zeros((len(available_sample_ids), n_classes))

    for idx, s_id in enumerate(available_sample_ids):
        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        if key not in data_dict:
            print(f"Warning: Key {key} not found in results file")
            continue
            
        experiment = data_dict[key]

        samples_distributions[idx] = np.array(experiment["distribution"])

    return samples_distributions
