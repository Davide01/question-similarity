# Question similarity classifier

A sentence-transformer-based model for detecting duplicate/similar question pairs. Includes data preprocessing, training with MLflow tracking, evaluation, and a FastAPI serving endpoint.

## Setup

Requires Python 3.14 and Poetry.

```bash
pyenv install 3.14
pyenv local 3.14
poetry env use 3.14
poetry sync
```

## Usage

### Data cleaning

Clean raw question pair data (removes HTML, special characters, and empty rows):

```bash
poetry run python -c "from question_similarity.data.preprocess import clean_data; clean_data('dataset/question_pairs_raw.csv', 'output_dir', force=False)"
```

Note: `train.py` runs cleaning automatically before training.

### Training

Fine-tune a sentence-transformer model:

```bash
poetry run python train.py <source_csv> <run_name> [OPTIONS]
```

Options:
- `--force` — Re-clean data even if a cleaned file already exists
- `--model-name` — Base model (default: `all-MiniLM-L6-v2`)
- `--learning-rate` — Custom learning rate
- `--register-model` — Register the model in the MLflow Model Registry

Example:

```bash
poetry run python train.py --source-file dataset/question_pairs_raw.csv --run-name my_run --model-name all-MiniLM-L6-v2 --learning-rate 2e-5
```

### Evaluation

Evaluate a trained model on the held-out test set:

```bash
poetry run python evaluate.py --model-path <path_to_model> --data <cleaned_csv>
poetry run python evaluate.py --run-id <mlflow_run_id>
```

### API

Serve the model via a FastAPI endpoint. When you train your model you can register it 
as a model in mlflow. After doing so, it will have a model name and model version. 
In order to use them in the API you can pass them as environment variables:

```bash
MODEL_NAME="test_model" MODEL_VERSION="1" poetry run python -m api
```


## Docker

```bash
docker build -t question-sim-api .
docker run --rm -p 8000:8000 question-sim-api
```
