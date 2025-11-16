import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Callable, Any
from datetime import datetime
import threading
import re 


def update_results_files_with_distributions(
    results_folder: Path,
    prompt_types: dict,
    class_labels: List[str],
    class_extractor_fun: Callable
):
    """
    Update results JSON files to include 'distribution' and 'info_answers' keys computed from 'raw_answers'.
    """
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    n_classes = len(class_labels)
    
    for prompt_type in prompt_types.keys():
        filename = f"results_{prompt_type}.json"
        filepath = Path(results_folder, filename)
        
        if not filepath.exists():
            print(f"Warning: {filename} does not exist in {results_folder}, skipping...")
            continue
        
        print(f"Processing {filename}...")
        
        # Load the results file
        try:
            with open(filepath, 'r') as f:
                data_dict = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file {filepath}: {e}. Skipping.")
            continue
        
        updated_count = 0
        total_experiments = 0
        
        # Process each experiment
        for key, experiment in data_dict.items():
            if not isinstance(experiment, dict) or 'id' not in experiment:
                continue
            
            total_experiments += 1
            needs_update = False
            
            if "raw_answers" not in experiment:
                continue
            
            raw_answers = experiment["raw_answers"]
            total_answers = len(raw_answers)
            
            if total_answers == 0:
                continue
            
            # 1. Compute distribution
            if "distribution" not in experiment:
                label_counts = np.zeros(n_classes)
                
                for answer_data in raw_answers.values():
                    answer_content = answer_data.get("content", "")
                    pred_label = class_extractor_fun(answer_content)
                    if pred_label in class_labels_to_id:
                        label_counts[class_labels_to_id[pred_label]] += 1
                
                # Normalize to get distribution
                distribution = (label_counts / total_answers).tolist()
                
                experiment["distribution"] = distribution
                needs_update = True
            
            # 2. Compute info_answers
            if "info_answers" not in experiment:
                info_answers = {}
                
                for answer_key, answer_data in raw_answers.items():
                    # Parse question_id from key (keys are like "😳", "0_1", "1_0", etc.)
                    parts = answer_key.split('_')
                    q_id = parts[0]
                    
                    answer_content = answer_data.get("content", "")
                    pred_label = class_extractor_fun(answer_content)
                    
                    if q_id not in info_answers:
                        info_answers[q_id] = []
                    
                    info_answers[q_id].append(pred_label)
                
                experiment["info_answers"] = info_answers
                needs_update = True
            
            if needs_update:
                updated_count += 1
        
        # Save the updated results file
        if updated_count > 0:
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=4) 
            print(f"Updated {updated_count} out of {total_experiments} experiments in {filename}")
        else:
            print(f"No updates needed for {filename}")

# =================================
# DATASET CONFIGURATION FUNCTIONS
# =================================

def get_cb_config():
    class_labels = ["entailment", "contradiction", "neutral", "N/A"]
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    return {'class_labels': class_labels, 'prompt_types': {"simple": "", "instruct": "", "fewshot": ""}, 'class_extractor_fun': class_extractor_fun}

def get_rte_config():
    class_labels = ["entailment", "contradiction/neutral", "N/A"]
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    return {'class_labels': class_labels, 'prompt_types': {"simple": "", "instruct": "", "fewshot": ""}, 'class_extractor_fun': class_extractor_fun}

def get_trec_config():
    class_labels = ["DESC", "ENTY", "ABBR", "HUM", "LOC", "NUM", "N/A"]
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    return {'class_labels': class_labels, 'prompt_types': {"simple": "", "instruct": "", "fewshot": ""}, 'class_extractor_fun': class_extractor_fun}

def get_dbpedia_config():
    class_labels = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork", "N/A"]
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    return {'class_labels': class_labels, 'prompt_types': {"simple": "", "instruct": "", "fewshot": ""}, 'class_extractor_fun': class_extractor_fun}

def get_wos46985_config():
    class_labels = ["Computer Science", "Electrical Engineering", "Psychology", "Mechanical Engineering", "Civil Engineering", "Medical Science", "Biochemistry", "N/A"]
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    return {'class_labels': class_labels, 'prompt_types': {"simple": "", "instruct": "", "fewshot": ""}, 'class_extractor_fun': class_extractor_fun}

# =========================
# MAIN EXECUTION STRUCTURE
# =========================

# Central dictionary mapping dataset names to their necessary paths and config functions
ALL_DATASET_CONFIGS = {
    'CB': {'results_folder': Path('./data/CB_RESULTS'), 'config_fn': get_cb_config},
    'RTE': {'results_folder': Path('./data/RTE_RESULTS'), 'config_fn': get_rte_config},
    'TREC': {'results_folder': Path('./data/TREC_RESULTS'), 'config_fn': get_trec_config},
    'DBPEDIA': {'results_folder': Path('./data/DBPEDIA_RESULTS'), 'config_fn': get_dbpedia_config},
    'WOS46985': {'results_folder': Path('./data/WOS46985_RESULTS'), 'config_fn': get_wos46985_config},
}

# Define which prompt files to look for (standard in your experiments)
STANDARD_PROMPT_TYPES = {
    "simple": "simple",
    "instruct": "instruct",
    "fewshot": "fewshot"
}


def run_all_datasets(datasets_to_run: List[str] = list(ALL_DATASET_CONFIGS.keys())):
    """Runs the distribution update across all specified datasets."""
    print("--- Starting Multi-Dataset Distribution Update ---")
    
    for dataset_name in datasets_to_run:
        config = ALL_DATASET_CONFIGS[dataset_name]
        
        # 1. Load dataset-specific configuration (labels and extractor function)
        dataset_cfg = config['config_fn']()
        
        # 2. Execute the update function
        update_results_files_with_distributions(
            results_folder=config['results_folder'],
            prompt_types=STANDARD_PROMPT_TYPES,
            class_labels=dataset_cfg['class_labels'],
            class_extractor_fun=dataset_cfg['class_extractor_fun']
        )
        
        print(f"Finished {dataset_name}.")
        print("-" * 50)

    print("\nAll tasks finished. Check your results folders for updated JSON files.")


if __name__ == "__main__":
    # By default, runs all five datasets.
    run_all_datasets()