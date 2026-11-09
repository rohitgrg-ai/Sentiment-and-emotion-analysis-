"""Clean final.csv and create a stratified train/test split.

Reads  data/final.csv  (leak-free, balanced; see balance_dataset.py)
Writes data/train.csv and data/test.csv (with a clean_text column).
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import emoji
except ImportError:  # emoji is optional
    emoji = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def clean_text(text: str) -> str:
    text = str(text).lower()
    if emoji is not None:
        text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    df = pd.read_csv(DATA / "final.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    # Stratify on emotion x sentiment so both targets stay balanced in each split.
    strata = df["label"].astype(str) + "_" + df["sentiment"]
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=strata
    )
    train_df.to_csv(DATA / "train.csv", index=False)
    test_df.to_csv(DATA / "test.csv", index=False)

    print(f"Total {len(df):,} | train {len(train_df):,} | test {len(test_df):,}")
    print("\nTrain emotion:\n", train_df["emotion"].value_counts().sort_index().to_string())
    print("\nTrain sentiment:\n", train_df["sentiment"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
