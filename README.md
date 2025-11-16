# Reproducibility-sens-cons

Reproduction of sensitivity and consistency experiments for Large Language Models.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Mixtral Model

#### Generate Results

To generate results folders with model predictions, run:

```bash
python Mixtral_model/run.py
```

This will process the datasets and save results to `Mixtral_model/data/*_RESULTS/` folders.

#### Compute Metrics

To compute metrics and generate visualizations for each dataset, run the corresponding Jupyter notebook:

- **CB (CommitmentBank)**: `Mixtral_model/LLM CB.ipynb`
- **DBPEDIA**: `Mixtral_model/LLM DBPEDIA.ipynb`
- **RTE (Recognizing Textual Entailment)**: `Mixtral_model/LLM RTE.ipynb`
- **TREC (Text REtrieval Conference)**: `Mixtral_model/LLM RTE.ipynb`
- **WoS (Web of Science)**: `Mixtral_model/LLM WoS.ipynb`

Each notebook will analyze the results from the corresponding `*_RESULTS` folder and generate metrics, plots, and analysis.

#### Dataset Results

Results are stored in:
- `Mixtral_model/data/CB_RESULTS/`
- `Mixtral_model/data/DBPEDIA_RESULTS/`
- `Mixtral_model/data/RTE_RESULTS/`
- `Mixtral_model/data/TREC_RESULTS/`
- `Mixtral_model/data/WOS46985_RESULTS/`

### Llama-3.2-1B Model

#### Generate Results

To generate results folders with model predictions, run:

```bash
python Llama_model/run.py
```

This will process the datasets and save results to `Llama_model/data/*_RESULTS/` folders.

#### Compute Metrics

To compute metrics and generate visualizations for each dataset, run the corresponding Jupyter notebook:

- **CB (CommitmentBank)**: `Llama_model/LLM CB.ipynb`
- **DBPEDIA**: `Llama_model/LLM DBPEDIA.ipynb`
- **RTE (Recognizing Textual Entailment)**: `Llama_model/LLM RTE.ipynb`
- **TREC (Text REtrieval Conference)**: `Llama_model/LLM RTE.ipynb`
- **WoS (Web of Science)**: `Llama_model/LLM WoS.ipynb`

Each notebook will analyze the results from the corresponding `*_RESULTS` folder and generate metrics, plots, and analysis.

#### Dataset Results

Results are stored in:
- `Llama_model/data/CB_RESULTS/`
- `Llama_model/data/DBPEDIA_RESULTS/`
- `Llama_model/data/RTE_RESULTS/`
- `Llama_model/data/TREC_RESULTS/`
- `Llama_model/data/WOS46985_RESULTS/`
