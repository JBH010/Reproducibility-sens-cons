"""
Ray Actor for loading and running LLM inference.
This allows loading the model once and processing multiple samples in parallel.
"""

import ray
from estimator import MulticlassEstimator


@ray.remote(num_gpus=1) 
class LLMModelActor:
    """
    Ray Actor that loads the LLM model once and serves inference requests.
    """
    
    def __init__(self, llm, temperature_question, num_questions, llm_rephraser=None):
        """
        Initialize the model actor.
        
        Args:
            llm: LLM name/identifier
            temperature_question: Temperature for question generation
            num_questions: Number of questions to generate
            llm_rephraser: Optional rephraser LLM
        """
        print(f"[Actor] Initializing LLMModelActor with {llm}...", flush=True)
        self.estimator = MulticlassEstimator(
            llm=llm,
            temperature_question=temperature_question,
            num_questions=num_questions,
            llm_rephraser=llm_rephraser,
        )
        print(f"[Actor] Model loaded and ready!", flush=True)
    
    def ready(self):
        """Check if actor is ready (initialization complete)."""
        return True
    
    def generate_questions(self, question):
        """Generate alternative questions."""
        return self.estimator.generate_questions(question=question)
    
    def generate_answers(self, modified_prompts, sample_input, temp_answer, A):
        """
        Generate answers for a sample.
        
        Args:
            modified_prompts: List of modified prompts
            sample_input: Input text to classify
            temp_answer: Temperature for answer generation
            A: Number of calls to the classifier
            
        Returns:
            Dictionary of answers
        """

        try: 
            result =  self.estimator.generate_answers(
            modified_prompts,
            sample_input,
            temp_answer,
            A,
            )

            print(f"[Actor] SUCCESS: Completed processing", flush=True)
            return result

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "OOM" in error_msg or "CUDA out of memory" in error_msg:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    print(f"[Actor] GPU OUT OF MEMORY ERROR! GPU: {gpu_mem:.1f}GB total, {allocated:.1f}GB allocated, {reserved:.1f}GB reserved", flush=True)
                print(f"[Actor] ERROR: {error_msg}", flush=True)
                import sys
                sys.stdout.flush()
                raise RuntimeError(f"GPU out of memory during inference: {error_msg}") from e
            print(f"[Actor] ERROR: {error_msg}", flush=True)
            import sys
            sys.stdout.flush()
            raise
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "OOM" in error_msg:
                print(f"[Actor] MEMORY ERROR: {error_msg}", flush=True)
            else:
                print(f"[Actor] ERROR: {error_msg}", flush=True)
            import sys
            sys.stdout.flush()
            raise
    
    def estimate_quantities(self, answers, labels, class_extractor_fun):
        """Estimate label quantities from answers."""
        return self.estimator.estimate_quantities(
            answers,
            labels=labels,
            class_extractor_fun=class_extractor_fun,
        )
    
    def compute_entropy(self, label_counts, A):
        """Compute entropy from label counts."""
        return self.estimator.compute_entropy(label_counts, A)
    
    def replace_question(self, prompt, question_to_rewrite, alt_question):
        """Replace question in prompt."""
        return self.estimator.replace_question(prompt, question_to_rewrite, alt_question)
