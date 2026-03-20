from pathlib import Path

import click
from question_similarity.data.preprocess import clean_data


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("experiment_folder", type=click.Path())
@click.option(
    "--force", is_flag=True, help="Force re-cleaning even if output file already exists"
)
def clean(source: str, experiment_folder: str, force: bool) -> None:
    """
    Clean question data from a source CSV file and save with "_cleaned" suffix.

    Args:
        source: Path to the source CSV file
        experiment_folder: Folder path for storing experiment artifacts
        force: If set, remove existing output file and re-clean the data
    """

    workspace = Path(".workspace") / experiment_folder
    workspace.mkdir(parents=True, exist_ok=True)

    _ = clean_data(source, experiment_folder, force)


if __name__ == "__main__":
    clean()
