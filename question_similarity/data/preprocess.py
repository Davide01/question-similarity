from pathlib import Path
import re
import pandas as pd


def preprocess(csv_path: str):
    df_data = pd.read_csv(csv_path, encoding="utf-8")

    df_data = df_data.dropna()

    df_data["question1"] = df_data["question1"].apply(clean_text)
    df_data["question2"] = df_data["question2"].apply(clean_text)

    # Drop rows where cleaning produced empty strings
    df_data = df_data[df_data["question1"].str.len() > 0]
    df_data = df_data[df_data["question2"].str.len() > 0]

    df_data[["question1", "question2"]] = df_data[["question1", "question2"]].astype(
        str
    )

    return df_data


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    return text


def clean_data(source: str, run_name: str, force: bool) -> Path:
    """
    Clean question data from a source CSV file and save with "_cleaned" suffix.

    Args:
        source: Path to the source CSV file
        experiment_folder: Folder path for storing experiment artifacts
        force: If set, remove existing output file and re-clean the data
    """
    source_path = Path(source)
    exp_path = Path(run_name)
    exp_path.mkdir(parents=True, exist_ok=True)

    # Generate target filename with "_cleaned" suffix
    target_filename = source_path.stem + "_cleaned" + source_path.suffix
    target_path = exp_path / target_filename

    # Check if target already exists
    if target_path.exists():
        if not force:
            print(f"Output file {target_path} already exists. Skipping cleaning.")
            print("Use --force to re-clean and overwrite the existing file.")
            return target_path
        else:
            print(f"Removing existing file {target_path}...")
            target_path.unlink()

    # Process the data
    print(f"Processing data from {source}...")
    df_cleaned = preprocess(source)

    # Save to target file in experiment folder
    df_cleaned.to_csv(target_path, index=False)
    print(f"Cleaned data saved to {target_path}")

    # Print summary
    print(f"\nData cleaning summary:")
    print(f"  Rows processed: {len(df_cleaned)}")
    print(f"  Columns: {', '.join(df_cleaned.columns)}")

    return target_path
