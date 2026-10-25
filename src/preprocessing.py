import pandas as pd
import re
import emoji
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "final.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed_dataset.csv"


def clean_text(text):
    """Clean text while preserving useful information."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Convert emojis into readable text descriptions
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    # Load the balanced dataset
    df = pd.read_csv(DATA_PATH)

    # Verify required columns
    required_columns = {"text", "emotion", "sentiment"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Remove rows with missing text or targets
    df = df.dropna(subset=["text", "emotion", "sentiment"]).copy()

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty text
    df = df[df["clean_text"].str.len() > 0].copy()

    # Save the processed dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing completed successfully!")
    print(f"Original dataset: {DATA_PATH}")
    print(f"Processed dataset: {OUTPUT_PATH}")
    print(f"Rows retained: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()