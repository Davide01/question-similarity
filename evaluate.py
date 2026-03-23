import click
import mlflow
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator

from datasets import Dataset
from question_similarity.data.config import TEST_DATA_RATIO, DATASET_SEED


def get_test_dataset(clean_path: Path, test_size: float = TEST_DATA_RATIO):
    """Load and split data, returning only the test portion."""
    df_data = pd.read_csv(clean_path, encoding="utf-8")

    dataset = Dataset.from_dict(
        {
            "sentence1": df_data["question1"],
            "sentence2": df_data["question2"],
            "label": df_data["label"],
        }
    )

    train_test_split = dataset.train_test_split(test_size=test_size, seed=DATASET_SEED)
    valid_test_split = train_test_split["test"].train_test_split(
        test_size=0.5, seed=DATASET_SEED
    )
    test_dataset = valid_test_split["test"]

    return test_dataset


def load_model_from_mlflow(run_id: str, artifact_path: str = "model"):
    """
    Load a model from MLflow using a run ID.

    Args:
        run_id: MLflow run ID
        artifact_path: Path to the model artifact within the run

    Returns:
        Loaded SentenceTransformer model
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    model = SentenceTransformer(local_path)
    return model


def load_model_from_local(model_path: str):
    """
    Load a model from a local path.

    Args:
        model_path: Path to the local model directory

    Returns:
        Loaded SentenceTransformer model
    """
    model = SentenceTransformer(model_path)
    return model


def evaluate_model(model, test_dataset, verbose: bool = True):
    """
    Evaluate a model on the test dataset.

    Args:
        model: SentenceTransformer model
        test_dataset: Test dataset from HuggingFace Dataset
        verbose: Whether to print results

    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = BinaryClassificationEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        labels=test_dataset["label"],
        name="test",
    )

    results = evaluator(model)

    if verbose and results:
        click.echo("\n" + "=" * 50)
        click.echo("TEST EVALUATION RESULTS")
        click.echo("=" * 50)
        for metric_name, metric_value in results.items():
            click.echo(f"  {metric_name}: {metric_value:.4f}")
        click.echo("=" * 50 + "\n")

    return results


@click.command()
@click.option(
    "--run-id", type=str, default=None, help="MLflow run ID to load model from"
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    default=None,
    help="Local path to load model from",
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    default="dataset/question_pairs_clean.csv",
    help="Path to the cleaned question pairs CSV file",
)
@click.option("--log-to-mlflow", is_flag=True, help="Log evaluation metrics to MLflow")
def evaluate(run_id: str, model_path: str, data: str, log_to_mlflow: bool):
    """
    Evaluate a question similarity model on the test dataset.

    Supports loading models from either MLflow (via --run-id) or local path (via --model-path).

    Example usage:
        # Evaluate a model from MLflow
        python evaluate.py --run-id abc123def456

        # Evaluate a local model
        python evaluate.py --model-path ./question_sim/finetuned_question_sim_model

        # Evaluate and log results to MLflow
        python evaluate.py --model-path ./model --log-to-mlflow
    """
    # Validate options
    if not run_id and not model_path:
        click.echo(
            "Error: Please provide either --run-id (for MLflow) or --model-path (for local model)"
        )
        raise click.BadParameter("Must specify either --run-id or --model-path")

    if run_id and model_path:
        click.echo(
            "Error: Please provide only one of --run-id or --model-path, not both"
        )
        raise click.BadParameter("Cannot specify both --run-id and --model-path")

    click.echo(f"Loading test dataset from {data}...")
    test_dataset = get_test_dataset(Path(data))
    click.echo(f"Test dataset size: {len(test_dataset)}")

    if run_id:
        click.echo(f"Loading model from MLflow run: {run_id}")
        model = load_model_from_mlflow(run_id)
    else:
        click.echo(f"Loading model from: {model_path}")
        model = load_model_from_local(model_path)

    click.echo("Running evaluation...")
    results = evaluate_model(model, test_dataset, verbose=True)

    # Optional: Log to MLflow
    if log_to_mlflow:
        click.echo("Logging results to MLflow...")
        # If model came from MLflow run, log to that same run
        if run_id:
            with mlflow.start_run(run_id=run_id, log_system_metrics=False):
                mlflow.log_param("model_source", f"mlflow_run_{run_id}")
                mlflow.log_param("test_dataset_size", len(test_dataset))

                if results:
                    for metric_name, metric_value in results.items():
                        mlflow.log_metric(metric_name, metric_value)

                click.echo("Results logged to MLflow run: " + run_id)
        else:
            # Create new run for local model evaluation
            with mlflow.start_run(log_system_metrics=False):
                mlflow.log_param("model_source", f"local_model_{model_path}")
                mlflow.log_param("test_dataset_size", len(test_dataset))

                if results:
                    for metric_name, metric_value in results.items():
                        mlflow.log_metric(metric_name, metric_value)

                click.echo("Results logged to MLflow")


if __name__ == "__main__":
    evaluate()
