import click
import mlflow
from mlflow.environment_variables import MLFLOW_TRACKING_URI
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from datasets import Dataset
from question_similarity.data.config import TEST_DATA_RATIO, DATASET_SEED
from question_similarity.data.preprocess import clean_data
from question_similarity.training.losses import ContrastiveLoss


def get_dataset_splits(clean_path: Path, test_size: float = TEST_DATA_RATIO):
    df_data = pd.read_csv(clean_path, encoding="utf-8")

    dataset = Dataset.from_dict(
        {
            "sentence1": df_data["question1"],
            "sentence2": df_data["question2"],
            "label": df_data["is_duplicate"],
        }
    )

    train_test_split = dataset.train_test_split(test_size=test_size, seed=DATASET_SEED)
    train_dataset = train_test_split["train"]

    valid_test_split = train_test_split["test"].train_test_split(
        test_size=0.5, seed=DATASET_SEED
    )
    valid_dataset = valid_test_split["train"]
    test_dataset = valid_test_split["test"]

    return train_dataset, valid_dataset, test_dataset


def setup_evaluators(valid_dataset, test_dataset):
    dev_evaluator = BinaryClassificationEvaluator(
        sentences1=valid_dataset["sentence1"],
        sentences2=valid_dataset["sentence2"],
        labels=valid_dataset["label"],
        name="valid",
    )

    test_evaluator = BinaryClassificationEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        labels=test_dataset["label"],
        name="test",
    )

    return dev_evaluator, test_evaluator


def setup_training(
    model,
    train_dataset,
    validator,
    run_name: str,
    workspace: Path,
    learning_rate: float = None,
):
    loss = ContrastiveLoss(model=model)

    output_dir = f"{workspace}/checkpoints"

    training_args = {
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "warmup_ratio": 0.1,
        "fp16": True,
        "run_name": run_name,
    }

    if learning_rate is not None:
        training_args["learning_rate"] = learning_rate

    args = SentenceTransformerTrainingArguments(**training_args)

    trainer = SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=validator,
    )

    return trainer


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("run_name", type=str)
@click.option(
    "--force", is_flag=True, help="Force re-cleaning even if output file already exists"
)
@click.option(
    "--model-name",
    type=str,
    default="all-MiniLM-L6-v2",
    help="Model name to use for training",
)
@click.option(
    "--learning-rate", type=float, default=None, help="Learning rate for the optimizer"
)
@click.option(
    "--register-model",
    is_flag=True,
    default=False,
    help="Whether to register the model in MLflow Model Registry",
)
def train(
    source: str,
    run_name: str,
    force: bool,
    model_name: str,
    learning_rate: float,
    register_model: bool,
) -> None:
    """
    Train a question similarity model using cleaned data from a source CSV file.

    Args:
        source: Path to the source CSV file
        run_name: Folder path for storing experiment artifacts
        force: If set, remove existing output file and re-clean the data
        model_name: Model name to use for training (default: all-MiniLM-L6-v2)
        learning_rate: Learning rate for the optimizer (optional)
        register_model: Whether to register the model in MLflow Model Registry
    """
    # Set up MLflow experiment and run
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.get())
    mlflow.set_experiment(model_name)

    workspace = Path(".workspace") / run_name
    workspace.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        clean_path = clean_data(source, workspace, force)

        train_dataset, valid_dataset, test_dataset = get_dataset_splits(clean_path)

        # Log training data size
        mlflow.log_param("training_data_size", len(train_dataset))

        # Log learning rate if provided
        if learning_rate is not None:
            mlflow.log_param("learning_rate", learning_rate)

        model = SentenceTransformer(model_name)

        dev_evaluator, test_evaluator = setup_evaluators(valid_dataset, test_dataset)
        initial_results = test_evaluator(model)

        trainer = setup_training(
            model, train_dataset, dev_evaluator, run_name, workspace, learning_rate
        )
        train_result = trainer.train()

        final_results = test_evaluator(model)

        # Save the model
        model_path = Path(workspace) / "model"
        model.save(str(model_path))
        model_info = mlflow.sentence_transformers.log_model(
            model,
            registered_model_name=model_name + "_tuned" if register_model else None,
        )

        click.echo("Logging metrics to MLflow...")
        if initial_results:
            for metric_name, metric_value in initial_results.items():
                mlflow.log_metric(
                    f"initial_{metric_name}", metric_value, model_id=model_info.model_id
                )

        if final_results:
            for metric_name, metric_value in final_results.items():
                mlflow.log_metric(
                    f"final_{metric_name}", metric_value, model_id=model_info.model_id
                )

        click.echo(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
