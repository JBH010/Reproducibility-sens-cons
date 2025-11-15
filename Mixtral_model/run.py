import sys
print("="*80, flush=True)
print("PYTHON SCRIPT STARTED!", flush=True)
print(f"Python version: {sys.version}", flush=True)
print("="*80, flush=True)
sys.stdout.flush()

import os
print("✓ Importing os...", flush=True)
import json
print("✓ Importing json...", flush=True)
from pathlib import Path
print("✓ Importing Path...", flush=True)
from typing import List
print("✓ Importing typing...", flush=True)

print("✓ About to import evaluator...", flush=True)
sys.stdout.flush()
from evaluator import run_grid_search
print("✓ Evaluator imported!", flush=True)
sys.stdout.flush()

# Remove proxy if present
print("✓ Removed proxy", flush=True)
sys.stdout.flush()

# ... rest of your code ...
import os
import json
from pathlib import Path
from typing import List
from evaluator import run_grid_search

# Remove proxy if present
os.environ.pop('http_proxy', None)

# ==============================================================================
# Dataset Parsing Functions
# ==============================================================================

def parse_CB(input_file: Path) -> List[dict]:
    """Parse CB (CommitmentBank) dataset from JSONL file."""
    data_list = []
    
    with open(input_file, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            
            idx = json_obj['idx']
            premise = json_obj['premise']
            hypothesis = json_obj['hypothesis']
            label = json_obj['label']
            
            input_text = f"Premise: {premise}\nHypothesis: {hypothesis}"
            
            new_entry = {
                "id": idx,
                "input": input_text,
                "class": label
            }
            
            data_list.append(new_entry)
    
    return data_list


def parse_RTE(input_file: Path) -> List[dict]:
    """Parse RTE (Recognizing Textual Entailment) dataset from JSONL file."""
    data_list = []
    
    with open(input_file, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            
            idx = json_obj['idx']
            premise = json_obj['premise']
            hypothesis = json_obj['hypothesis']
            label = json_obj['label']
            
            if label == 'not_entailment':
                label = 'contradiction/neutral'
            
            input_text = f"Premise: {premise}\nHypothesis: {hypothesis}"
            
            new_entry = {
                "id": idx,
                "input": input_text,
                "class": label
            }
            
            data_list.append(new_entry)
    
    return data_list


def parse_TREC(input_file: Path) -> List[dict]:
    """Parse TREC dataset from text file."""
    data_list = []
    
    with open(input_file, 'r') as file:
        for idx, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            
            # TREC format: label:text
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            label, text = parts[0], parts[1].strip()
            
            new_entry = {
                "id": idx,
                "input": text,
                "class": label
            }
            
            data_list.append(new_entry)
    
    return data_list


def parse_DBPEDIA(input_file: Path, classes_file: Path) -> List[dict]:
    """Parse DBPEDIA dataset from CSV file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to parse DBPEDIA dataset. Install it with: pip install pandas")
    
    df = pd.read_csv(input_file)
    
    # Load class labels
    with open(classes_file, 'r') as f:
        class_labels = [line.strip() for line in f if line.strip()]
    
    data_list = []
    for idx, row in df.iterrows():
        new_entry = {
            "id": idx + 1,
            "input": row['Text'],  # Assuming column name is 'text'
            "class": class_labels[int(row['Class'])-1]
        }
        data_list.append(new_entry)
    
    return data_list


def parse_WOS46985(input_file: Path, target_file: Path, class_labels: List[str]) -> List[dict]:
    """Parse WOS46985 dataset from text files."""
    data_list = []
    
    # Read input sentences
    with open(input_file, 'r') as f_input:
        inputs = f_input.read().splitlines()
    
    # Read target classes
    with open(target_file, 'r') as f_class:
        classes = f_class.read().splitlines()
    
    # Ensure the number of inputs and classes match
    if len(inputs) != len(classes):
        raise ValueError("Number of input sentences does not match number of classes")
    
    # Create list of dictionaries
    for i, (inp, cls) in enumerate(zip(inputs, classes)):
        sample = {
            "id": i + 1,
            "input": inp,
            "class": class_labels[int(cls)]
        }
        data_list.append(sample)
    
    return data_list


# ==============================================================================
# Dataset Configurations
# ==============================================================================

def get_CB_config():
    """Get configuration for CB dataset."""
    class_labels = ["entailment", "contradiction", "neutral", "N/A"]
    
    question_to_rewrite = "You are given a premise and a hypothesis as input. Determine is there is entailment, contradiction, or a neutral relation between the premise and the hypothesis."
    
    prompt_simple = [
        ["system", "You are a text entailment system."],
        ["user", "You are given a premise and a hypothesis as input. Determine is there is entailment, contradiction, or a neutral relation between the premise and the hypothesis.\nAnswer with the class name only.\nThe possible classes are: entailment, contradiction, and neutral.\nHere is the text: {summary}"]    
    ]
    
    prompt_instruct = [
        ["system", "You are a text entailment system."],
        ["user", "You are given a premise and a hypothesis as input. Determine is there is entailment, contradiction, or a neutral relation between the premise and the hypothesis.\nAnswer with the class name only.\nThe possible classes are: \n- entailment: a logical relationship where the meaning of one text (the hypothesis) is necessarily implied by another text (the premise), \n- contradiction: when two texts express mutually exclusive statements \n- neutral: a state where two texts are independent and their truth values do not affect each other.\nHere is the text: {summary}"]    
    ]
    
    prompt_fewshot = [
        ["system", "You are a text entailment system."],
        ["user",
         "You are given a premise and a hypothesis as input. Determine is there is entailment, contradiction, or a neutral relation between the premise and the hypothesis.\nAnswer with the class name only.\nThe possible classes are: entailment, contradiction, and neutral.\n"
         "Here are a few examples:\n"
         "Example 1:\n"
         "Premise: Your turn. B: Okay. Uh, I don't think they should abolish it. Hypothesis:\n they should abolish it. Label: contradiction\n"
         "Example 2:\n"
         "Premise: And I don't want to have to lie to them. The kidnappers have given us until October the eleventh to deliver the document and I haven't despaired of finding it before then. But if the police learn I 've been to America they 'll ask why. Hypothesis:\n he's been to America. Label: entailment\n"
         "Example 3:\n"
         "Premise: Who knows? The point is, do we go with it or not?'' Do we assume there is a shipment? Hypothesis: there is a shipment. Label: neutral\n"
         "Here is the text: {summary}"]
    ]
    
    prompt_types = {
        "simple": prompt_simple,
        "instruct": prompt_instruct,
        "fewshot": prompt_fewshot
    }
    
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    
    return {
        "class_labels": class_labels,
        "question_to_rewrite": question_to_rewrite,
        "prompt_types": prompt_types,
        "class_extractor_fun": class_extractor_fun
    }


def get_RTE_config():
    """Get configuration for RTE dataset."""
    class_labels = ["entailment", "contradiction/neutral", "N/A"]
    
    question_to_rewrite = "You are given a premise and a hypothesis as input. Determine is there is entailment or a contradiction/neutral statement."
    
    prompt_simple = [
        ["system", "You are a text entailment system."],
        ["user", "You are given a premise and a hypothesis as input. You are given a premise and a hypothesis as input. Determine is there is entailment or a contradiction/neutral statement.\nAnswer with the class name only.\nThe possible classes are: entailment and contradiction/neutral.\nHere is the text: {summary}"]    
    ]
    
    prompt_instruct = [
        ["system", "You are a text entailment system."],
        ["user", "You are given a premise and a hypothesis as input. You are given a premise and a hypothesis as input. Determine is there is entailment or a contradiction/neutral statement.\nAnswer with the class name only.\nThe possible classes are: \n- entailment: a logical relationship where the meaning of one text (the hypothesis) is necessarily implied by another text (the premise), \n- contradiction/neutral: contradictions or neutral statements that are not entailments\n\nHere is the text: {summary}"]    
    ]
    
    prompt_fewshot = [
        ["system", "You are a text entailment system."],
        ["user",
         "You are given a premise and a hypothesis as input. Determine is there is entailment or a contradiction/neutral statement.\nAnswer with the class name only.\nThe possible classes are: entailment and contradiction/neutral.\n"
         "Here are a few examples:\n"
         "Example 1:\n"
         "Premise: Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation. Hypothesis:\n Christopher Reeve had an accident. Label: contradiction/neutral\n"
         "Example 2:\n"
         "Premise: And I don't want to have to lie to them. The kidnappers have given us until October the eleventh to deliver the document and I haven't despaired of finding it before then. But if the police learn I 've been to America they 'll ask why. Hypothesis:\n he's been to America. Label: entailment\n"
         "Here is the text: {summary}"]
    ]
    
    prompt_types = {
        "simple": prompt_simple,
        "instruct": prompt_instruct,
        "fewshot": prompt_fewshot
    }
    
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    
    return {
        "class_labels": class_labels,
        "question_to_rewrite": question_to_rewrite,
        "prompt_types": prompt_types,
        "class_extractor_fun": class_extractor_fun
    }


def get_TREC_config():
    """Get configuration for TREC dataset."""
    class_labels = [
        "DESC", "ENTY", "ABBR", "HUM", "LOC", "NUM", "N/A"
    ]
    
    question_to_rewrite = "Classify the question into one of the following categories: Description, Entity, Abbreviation, Human, Location, or Number."
    
    prompt_simple = [
        ["system", "You are a question classifier."],
        ["user", "Classify the question into one of the following categories: Description, Entity, Abbreviation, Human, Location, or Number.\nAnswer with the class name only.\nThe possible classes are: DESC, ENTY, ABBR, HUM, LOC, NUM.\nHere is the text: {summary}"]    
    ]
    
    prompt_instruct = [
        ["system", "You are a question classifier."],
        ["user", "Classify the question into one of the following categories: Description, Entity, Abbreviation, Human, Location, or Number.\nAnswer with the class name only.\nThe possible classes are: \n- DESC: questions asking for a description\n- ENTY: questions asking about an entity\n- ABBR: questions asking about an abbreviation\n- HUM: questions asking about a person\n- LOC: questions asking about a location\n- NUM: questions asking about a number\nHere is the text: {summary}"]    
    ]
    
    prompt_fewshot = [
        ["system", "You are a question classifier."],
        ["user",
         "Classify the question into one of the following categories: Description, Entity, Abbreviation, Human, Location, or Number.\nAnswer with the class name only.\nThe possible classes are: DESC, ENTY, ABBR, HUM, LOC, NUM.\n"
         "Here are a few examples:\n"
         "Example 1: What is the capital of France? Label: LOC\n"
         "Example 2: Who wrote Romeo and Juliet? Label: HUM\n"
         "Example 3: How tall is Mount Everest? Label: NUM\n"
         "Here is the text: {summary}"]
    ]
    
    prompt_types = {
        "simple": prompt_simple,
        "instruct": prompt_instruct,
        "fewshot": prompt_fewshot
    }
    
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    
    return {
        "class_labels": class_labels,
        "question_to_rewrite": question_to_rewrite,
        "prompt_types": prompt_types,
        "class_extractor_fun": class_extractor_fun
    }


def get_DBPEDIA_config():
    """Get configuration for DBPEDIA dataset."""
    class_labels = [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork", "N/A"
    ]
    
    question_to_rewrite = "Classify the text based on the following categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork."
    
    prompt_simple = [
        ["system", "You are a text classifier."],
        ["user", "Classify the text based on the following categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.\nAnswer with the class name only.\nThe possible classes are: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.\nHere is the text: {summary}"]    
    ]
    
    prompt_instruct = [
        ["system", "You are a text classifier."],
        ["user", "Classify the text based on the following categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.\nAnswer with the class name only.\nThe possible classes are: \n- Company: A business organization\n- EducationalInstitution: Schools, universities, etc.\n- Artist: A person who creates art\n- Athlete: A person who participates in sports\n- OfficeHolder: A person holding a public office\n- MeanOfTransportation: Vehicles, planes, etc.\n- Building: Structures\n- NaturalPlace: Natural locations\n- Village: Small settlements\n- Animal: Living creatures\n- Plant: Flora\n- Album: Music albums\n- Film: Movies\n- WrittenWork: Books and written material\nHere is the text: {summary}"]    
    ]
    
    prompt_fewshot = [
        ["system", "You are a text classifier."],
        ["user",
         "Classify the text based on the following categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.\nAnswer with the class name only.\n"
         "Here are a few examples:\n"
         "Example 1: Microsoft Corporation Label: Company\n"
         "Example 2: Harvard University Label: EducationalInstitution\n"
         "Example 3: The Beatles Label: Artist\n"
         "Here is the text: {summary}"]
    ]
    
    prompt_types = {
        "simple": prompt_simple,
        "instruct": prompt_instruct,
        "fewshot": prompt_fewshot
    }
    
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    
    return {
        "class_labels": class_labels,
        "question_to_rewrite": question_to_rewrite,
        "prompt_types": prompt_types,
        "class_extractor_fun": class_extractor_fun
    }


def get_WOS46985_config():
    """Get configuration for WOS46985 dataset."""
    class_labels = [
        "Computer Science", "Electrical Engineering", "Psychology",
        "Mechanical Engineering", "Civil Engineering", "Medical Science",
        "Biochemistry", "N/A"
    ]
    
    question_to_rewrite = "Classify the text based on whether their field is Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, or Biochemistry."
    
    prompt_simple = [
        ["system", "You are a text classifier."],
        ["user", "Classify the text based on whether their field is Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, or Biochemistry.\nAnswer with the class name only.\nThe possible classes are: Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, Biochemistry.\nHere is the text: {summary}"]    
    ]
    
    prompt_instruct = [
        ["system", "You are a text classifier."],
        ["user", "Classify the text based on whether their field is Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, or Biochemistry.\nAnswer with the class name only.\nThe possible classes are: \n- Computer Science: The study of algorithms, data structures, and software design to solve complex problems using computational systems. \n- Electrical Engineering: Focuses on the design, development, and testing of electrical equipment, from microchips to power station generators. \n- Psychology: The scientific study of the mind and behavior, exploring how people think, feel, and interact.\\n- Mechanical Engineering: Involves the design, analysis, and manufacturing of mechanical systems, ranging from small components to large machinery.\\n- Civil Engineering: Deals with the design, construction, and maintenance of infrastructure projects such as bridges, roads, and buildings.\\n- Medical Science: Encompasses the study and research of the human body and its functions, aiming to improve health and treat diseases.\\n- Biochemistry: The study of chemical processes within and related to living organisms, merging biology and chemistry to understand cellular and molecular mechanisms.\\nHere is the text: {summary}"]    
    ]
    
    prompt_fewshot = [
        ["system", "You are a text classifier."],
        ["user",
         "Classify the text based on whether their field is Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, or Biochemistry.\nAnswer with the class name only.\nThe possible classes are: Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, Biochemistry.\n"
         "Here are a few examples:\n"
         "Example 1: Machine learning algorithms Label: Computer Science\n"
         "Example 2: Circuit design and analysis Label: Electrical Engineering\n"
         "Example 3: Cognitive behavior therapy Label: Psychology\n"
         "Here is the text: {summary}"]
    ]
    
    prompt_types = {
        "simple": prompt_simple,
        "instruct": prompt_instruct,
        "fewshot": prompt_fewshot
    }
    
    def class_extractor_fun(text):
        for l in class_labels:
            if l.lower() in text.lower():
                return l
        return "N/A"
    
    return {
        "class_labels": class_labels,
        "question_to_rewrite": question_to_rewrite,
        "prompt_types": prompt_types,
        "class_extractor_fun": class_extractor_fun
    }


# ==============================================================================
# Main Execution
# ==============================================================================

def run_all_datasets(
    llms: List[str] = None,
    Qs: List[int] = None,
    temp_questions: List[float] = None,
    As: List[int] = None,
    temp_answers: List[float] = None,
    datasets: List[str] = None
):
    """
    Run grid search on all specified datasets.
    
    Args:
        llms: List of LLM names to use (default: ['llama3', 'mixtral'])
        Qs: List of number of question rephrasings (default: [30])
        temp_questions: List of temperatures for question generation (default: [0.])
        As: List of number of answer calls (default: [1])
        temp_answers: List of temperatures for answers (default: [0.])
        datasets: List of dataset names to run (default: all datasets)
    """
    # Default parameters
    if llms is None:
        llms = ['mixtral']
    if Qs is None:
        Qs = [30]
    if temp_questions is None:
        temp_questions = [0.]
    if As is None:
        As = [1]
    if temp_answers is None:
        temp_answers = [0.]
    if datasets is None:
        datasets = ['CB', 'RTE', 'TREC', 'DBPEDIA', 'WOS46985']
    
    # Dataset configurations
    dataset_configs = {
        'CB': {
            'data_file': Path('./data/CB/train.jsonl'),
            'processed_file': Path('./data/CB_RESULTS/data.json'),
            'results_folder': Path('./data/CB_RESULTS'),
            'parser': parse_CB,
            'config_fn': get_CB_config
        },
        'RTE': {
            'data_file': Path('./data/RTE/train.jsonl'),
            'processed_file': Path('./data/RTE_RESULTS/data.json'),
            'results_folder': Path('./data/RTE_RESULTS'),
            'parser': parse_RTE,
            'config_fn': get_RTE_config
        },
        'TREC': {
            'data_file': Path('./data/TREC/train.txt'),
            'processed_file': Path('./data/TREC_RESULTS/data.json'),
            'results_folder': Path('./data/TREC_RESULTS'),
            'parser': parse_TREC,
            'config_fn': get_TREC_config
        },
        'DBPEDIA': {
            'data_file': Path('./data/DBPEDIA/test.csv'),
            'classes_file': Path('./data/DBPEDIA/classes.txt'),
            'processed_file': Path('./data/DBPEDIA_RESULTS/data.json'),
            'results_folder': Path('./data/DBPEDIA_RESULTS'),
            'parser': lambda f: parse_DBPEDIA(f, Path('./data/DBPEDIA/classes.txt')),
            'config_fn': get_DBPEDIA_config
        },
        'WOS46985': {
            'data_file': Path('./data/WOS46985/X.zip'),
            'target_file': Path('./data/WOS46985/YL1.txt'),
            'processed_file': Path('./data/WOS46985_RESULTS/data.json'),
            'results_folder': Path('./data/WOS46985_RESULTS'),
            'parser': None,  # Special handling needed
            'config_fn': get_WOS46985_config
        }
    }
    
    # Run grid search for each dataset
    for dataset_name in datasets:
        if dataset_name not in dataset_configs:
            print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        config = dataset_configs[dataset_name]
        results_folder = config['results_folder']
        processed_file = config['processed_file']
        
        # Create results folder if it doesn't exist
        results_folder.mkdir(parents=True, exist_ok=True)
        
        # Load or parse dat
        
        if processed_file.exists():
            print(f"Loading processed data from {processed_file}")
            with open(processed_file, 'r') as f:
                loaded_data = json.load(f)
                # Handle both formats: plain list or tuple-as-list [data_list, one_shot_examples]
                if isinstance(loaded_data, list) and len(loaded_data) == 2 and isinstance(loaded_data[0], list) and isinstance(loaded_data[1], list):
                    # This is the (data_list, one_shot_examples) format from DBPEDIA/WOS
                    data_list = loaded_data[0]
                    print(f"Loaded data with one-shot examples (format from balanced sampling)")
                else:
                    # Plain list format from CB/RTE/TREC
                    data_list = loaded_data
        else:
            print(f"Parsing data from {config['data_file']}")
            if dataset_name == 'WOS46985':
                # Special handling for WOS46985
                cfg = config['config_fn']()
                # Note: WOS46985 uses X.zip, might need special extraction
                # For now, assume X.txt exists after extraction
                data_file = Path('./data/WOS46985/X.txt')
                if not data_file.exists():
                    print(f"Warning: {data_file} not found. Skipping WOS46985.")
                    continue
                data_list = parse_WOS46985(
                    data_file,
                    config['target_file'],
                    cfg['class_labels']
                )
            else:
                data_list = config['parser'](config['data_file'])
            
            # Save processed data
            json_data = json.dumps(data_list)
            with open(processed_file, 'w') as f:
                f.write(json_data)
        
        print(f"Loaded {len(data_list)} samples")
        
        # Get dataset-specific configuration FIRST (needed for sampling)
        dataset_cfg = config['config_fn']()
        
        # Balance dataset for large datasets (DBPEDIA and WOS46985) - same as original paper
        if dataset_name in ['DBPEDIA', 'WOS46985'] and len(data_list) > 2000:
            import random
            
            print(f"Balancing dataset: sampling 2000 samples from {len(data_list)} total samples...")
            random.seed(42)  # Set seed for reproducibility
            
            # Shuffle the samples
            random.shuffle(data_list)
            
            # Exclude N/A class from balancing
            active_class_labels = [c for c in dataset_cfg['class_labels'] if c != 'N/A']
            n_samples_per_class = 2000 // len(active_class_labels)
            
            balanced_dataset = []
            class_count = {label: 0 for label in active_class_labels}
            remaining_samples = []
            
            for sample in data_list:
                label = sample['class']
                if label in class_count and class_count[label] < n_samples_per_class:
                    balanced_dataset.append(sample)
                    class_count[label] += 1
                else:
                    remaining_samples.append(sample)
            
            # If we still need more samples to reach 2000, randomly sample from remaining samples
            while len(balanced_dataset) < 2000 and remaining_samples:
                random_sample = random.choice(remaining_samples)
                balanced_dataset.append(random_sample)
                remaining_samples.remove(random_sample)
            
            data_list = balanced_dataset[:2000]  # Ensure exactly 2000 samples
            
            print(f"Balanced dataset: {len(data_list)} samples")
            for label in active_class_labels:
                count = sum(1 for s in data_list if s['class'] == label)
                print(f"  Class '{label}': {count} samples")

        # FILTER to only run specific prompt types (skip 'simple')
        all_prompt_types = dataset_cfg['prompt_types']
        dataset_cfg['prompt_types'] = {
            k: v for k, v in all_prompt_types.items()
            if k in ['fewshot', "instruct", "simple"]  # Note: it's 'fewshot' not '1-shot'
        }
        
        print(f"Running prompt types: {list(dataset_cfg['prompt_types'].keys())}")
        
        # Prepare samples
        samples = data_list
        
        # Run grid search
        print(f"Starting grid search for {dataset_name}...", flush=True)
        try:
            run_grid_search(
                samples=samples,
                llms=llms,
                prompt_types=dataset_cfg['prompt_types'],
                Qs=Qs,
                temp_questions=temp_questions,
                question_to_rewrite=dataset_cfg['question_to_rewrite'],
                As=As,
                temp_answers=temp_answers,
                class_labels=dataset_cfg['class_labels'],
                class_extractor_fun=dataset_cfg['class_extractor_fun'],
                results_folder=results_folder,
                num_actors=2,
            )
        except Exception as e:
            print(f"ERROR in grid search for {dataset_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Run only instruct and fewshot prompt types for RTE with Mixtral
    run_all_datasets(
        llms=["mixtral"],
        datasets=["DBPEDIA"],
    )
