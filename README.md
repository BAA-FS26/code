# Evaluating Synthetic Data as a Privacy-Enhancing Technology

This repository contains a research prototype for evaluating **synthetic tabular data as a Privacy-Enhancing Technology (PET)** in machine-learning workflows.

The project compares synthetic data against real training data across four evaluation dimensions:

- **Utility**: model performance when classifiers are trained on real or synthetic data and tested on real held-out data
- **Fidelity**: statistical similarity between real and synthetic training data
- **Privacy**: empirical privacy risks measured with Anonymeter and distance-to-closest-record metrics
- **Privacy–Utility trade-off**: combined view of model performance and measured privacy risk

The current implementation uses the **Adult Census Income** dataset as the reference dataset. Results are written as local JSON files and can be explored with the included Streamlit dashboard.

---

## Project structure

```text
src/
  core/                  Shared path, I/O, and data-source helpers
  dataset/               Dataset adapter, dataset configuration, splitting, preprocessing
  modeling/
    classification/      Classifier training, sweeps, and model persistence
    synthesizing/        Non-DP and DP synthetic data generation
  evaluation/            Utility, fidelity, and privacy evaluation scripts
  dashboard/             Streamlit dashboard components
  utility/               Constants, logging, metadata, reproducibility, W&B helpers

assets/                  Static dashboard assets
config/                  Optional model parameter files
data/                    Local input and generated data files
models/                  Local saved classifier and synthesizer models
results/                 Local JSON result files used by the dashboard

dashboard.py             Streamlit dashboard entry point
pyproject.toml           Project metadata and dependencies
```

The main workflow is:

```text
raw data
  -> cleaning and splitting
  -> synthetic data generation
  -> classifier training
  -> utility, fidelity, and privacy evaluation
  -> dashboard visualization
```

---

## Features

- Synthetic data generation with SDV:
  - Gaussian Copula
  - CTGAN
  - TVAE
- Differentially private synthesis with SmartNoise:
  - DP-CTGAN
  - PATE-CTGAN
- Optional CUDA support for neural-network-based synthesizers
- Classifier training on real and synthetic data:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Utility evaluation with Train on Synthetic, Test on Real (TSTR)
- Fidelity evaluation with SDMetrics quality and diagnostic reports
- Privacy evaluation with Anonymeter and SDMetrics DCR metrics
- Local JSON-based result logging
- Optional Weights & Biases integration
- Streamlit dashboard with Utility, Privacy, Trade-off, and Fidelity tabs

---

## Requirements

- Python `>=3.12,<3.13`
- Windows, macOS, or Linux
- Optional: CUDA-capable GPU for faster CTGAN, TVAE, DP-CTGAN, and PATE-CTGAN runs

The pipeline can run on CPU. CUDA is only used when the corresponding `--cuda` flag is passed and the local environment supports it.

---

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/BAA-FS26/code.git
cd code
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project from the repository root:

```bash
python -m pip install --upgrade pip
pip install -e .
```

This makes the local `src` package available for the command-line scripts and dashboard.

Optional Weights & Biases support:

```bash
pip install -e ".[wandb]"
```

If W&B is used, create a local `.env` file and configure the required variables:

```bash
cp .env.example .env
```

Example:

```text
WANDB_ENTITY=your_entity
WANDB_PROJECT=synthetic-data-eval
```

Local JSON files remain the primary output even when W&B logging is enabled.

---

## Data preparation

The implemented dataset adapter downloads and prepares the **Adult Census Income** dataset from the UCI Machine Learning Repository.

Dataset-specific settings are centralized in:

```text
src/dataset/dataset_config.py
```

The default dataset is:

```text
adult_census
```

The cleaned dataset is split deterministically into:

```text
train / validation / test = 60% / 20% / 20%
```

The test split is held out for final utility evaluation.

To prepare the local data files, run the dataset preparation functions from the repository root, for example:

```python
from src.dataset.adult_census import load_cleaned
from src.dataset.data_splitting import split_data, verify_stratification
from src.dataset.dataset_config import get_dataset_config
from src.utility.constants import DATA_DIR, PROCESSED_DATA_DIR

config = get_dataset_config("adult_census")
df = load_cleaned(DATA_DIR)
train, validation, test = split_data(
    df,
    target_col=config.target_col,
    output_dir=PROCESSED_DATA_DIR,
)

print(verify_stratification(train, validation, test, config.target_col))
```

This creates the expected local data structure:

```text
data/
  raw/
    adult_raw.csv
    education_map.json
  cleaned/
    adult_cleaned.csv
  processed/
    train.csv
    validation.csv
    test.csv
```

---

## Evaluation design

| Evaluation | Real data used | Synthetic data used | Main purpose |
| --- | --- | --- | --- |
| Utility | Held-out real test split | Training source for classifiers | Measures downstream model performance |
| Fidelity | Real training split | Synthetic training split | Measures statistical similarity |
| Privacy | Real training split and holdout split | Synthetic training split | Measures re-identification and inference risks |
| Trade-off | Utility JSON results | Privacy JSON results | Combines performance and privacy risk |

Utility follows a TSTR setup for synthetic data:

```text
Train on Synthetic, Test on Real
```

For the real-data baseline, the setup is:

```text
Train on Real, Test on Real
```

Privacy evaluation uses the real training split, the synthetic training split, and a holdout split consisting of validation plus test data.

---

## Running the pipeline

Most commands accept `--dataset adult_census`. This is the default, so the flag can usually be omitted.

### 1. Generate non-DP synthetic data

```bash
python -m src.modeling.synthesizing.synthesize --synthesizer gaussian_copula
python -m src.modeling.synthesizing.synthesize --synthesizer ctgan
python -m src.modeling.synthesizing.synthesize --synthesizer tvae
```

With CUDA support for CTGAN and TVAE:

```bash
python -m src.modeling.synthesizing.synthesize --synthesizer ctgan --cuda
python -m src.modeling.synthesizing.synthesize --synthesizer tvae --cuda
```

Generated files are written to:

```text
data/synthetic/{synthesizer}/default/synthetic_train.csv
```

### 2. Generate DP synthetic data

```bash
python -m src.modeling.synthesizing.synthesize_dp --synthesizer dpctgan --epsilon 0.1
python -m src.modeling.synthesizing.synthesize_dp --synthesizer dpctgan --epsilon 0.5
python -m src.modeling.synthesizing.synthesize_dp --synthesizer dpctgan --epsilon 1.0
python -m src.modeling.synthesizing.synthesize_dp --synthesizer dpctgan --epsilon 5.0
python -m src.modeling.synthesizing.synthesize_dp --synthesizer dpctgan --epsilon 10.0
```

Equivalent commands can be run for `patectgan` where valid runs are available.

With CUDA support for SmartNoise synthesizers:

```bash
python -m src.modeling.synthesizing.synthesize_dp --synthesizer dpctgan --epsilon 1.0 --cuda
python -m src.modeling.synthesizing.synthesize_dp --synthesizer patectgan --epsilon 1.0 --cuda
```

Generated files are written to:

```text
data/synthetic/{synthesizer}/eps_{epsilon}/synthetic_train.csv
```

Note: very small privacy budgets can be difficult for DP synthesizers. In the experiments for this project, PATE-CTGAN did not produce a valid run for `epsilon = 0.1`.

### 3. Train classifiers

Train a real-data baseline:

```bash
python -m src.modeling.classification.classify --mode default --classifier gradient_boosting --data_source real
```

Train on non-DP synthetic data:

```bash
python -m src.modeling.classification.classify --mode default --classifier gradient_boosting --synthesizer ctgan
```

Train on DP synthetic data:

```bash
python -m src.modeling.classification.classify --mode default --classifier gradient_boosting --synthesizer dpctgan --epsilon 0.5
```

The same command structure can be used with:

```text
logistic_regression
random_forest
gradient_boosting
```

If tuned parameter files are available in `config/`, models can also be trained in `best` mode:

```bash
python -m src.modeling.classification.classify --mode best --classifier gradient_boosting --data_source real --params best_gradient_boosting_real.yaml
python -m src.modeling.classification.classify --mode best --classifier gradient_boosting --synthesizer ctgan --params best_gradient_boosting_real.yaml
python -m src.modeling.classification.classify --mode best --classifier gradient_boosting --synthesizer dpctgan --epsilon 0.5 --params best_gradient_boosting_real.yaml
```

### 4. Evaluate utility

Utility is evaluated on the held-out real test split.

```bash
python -m src.evaluation.evaluate_utility --classifier gradient_boosting --data_source real --model_type default
python -m src.evaluation.evaluate_utility --classifier gradient_boosting --synthesizer ctgan --model_type default
python -m src.evaluation.evaluate_utility --classifier gradient_boosting --synthesizer dpctgan --epsilon 0.5 --model_type default
```

For models trained in `best` mode, use:

```bash
python -m src.evaluation.evaluate_utility --classifier gradient_boosting --data_source real --model_type best
python -m src.evaluation.evaluate_utility --classifier gradient_boosting --synthesizer ctgan --model_type best
python -m src.evaluation.evaluate_utility --classifier gradient_boosting --synthesizer dpctgan --epsilon 0.5 --model_type best
```

### 5. Evaluate fidelity

```bash
python -m src.evaluation.evaluate_fidelity --synthesizer ctgan
python -m src.evaluation.evaluate_fidelity --synthesizer dpctgan --epsilon 0.5
```

### 6. Evaluate privacy

```bash
python -m src.evaluation.evaluate_privacy --synthesizer ctgan
python -m src.evaluation.evaluate_privacy --synthesizer dpctgan --epsilon 0.5
```

---

## Result files

Pipeline outputs are written as local JSON files:

```text
results/{category}/YYYY-MM-DD/{run_name}_{HHMMSS}.json
```

Supported result categories:

```text
synthesis
utility
fidelity
privacy
```

Each result file contains run metadata, parameters, summary metrics, history, and artifacts. The dashboard uses these JSON files as its main data source.

---

## Dashboard

Start the dashboard from the repository root:

```bash
streamlit run dashboard.py
```

The dashboard reads local JSON result files from `results/` and provides four tabs:

- **Utility**: classifier performance on the real held-out test set
- **Privacy**: Anonymeter and DCR-based risk metrics
- **Trade-off**: combined privacy-risk and utility view
- **Fidelity**: statistical quality and diagnostic metrics

The sidebar supports filtering by synthesizer, privacy budget epsilon, and run selection.

---

## Reproducibility

The project uses a fixed random seed where supported:

```text
RANDOM_STATE = 42
```

The reproducibility helper sets seeds for Python, NumPy, and PyTorch. Full bitwise determinism is not guaranteed because GPU operations and third-party model internals may remain nondeterministic.

---

## Notes and limitations

- This repository is a research prototype and not a production anonymization system.
- Adult Census is currently the implemented reference dataset.
- Local artifacts in `data/`, `models/`, and `results/` are generated during execution and may not be included in the repository.
- Synthetic data quality and privacy risk depend on the selected synthesizer, privacy budget, dataset, and metric.
- DP synthesizer runs can be computationally expensive.
- Some DP configurations may fail or produce unusable outputs depending on the selected synthesizer and privacy budget.
- Generated synthetic data should be evaluated before being used in downstream workflows.

---

## Thesis context

This repository supports the bachelor thesis:

**Synthetic Data as a Privacy-Enhancing Technology in Machine Learning – Technical Implementation, Evaluation, and Privacy Potential**

The implementation focuses on an empirical evaluation pipeline for synthetic tabular data and its practical use in machine-learning workflows.
