import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = {"clean_text", "sentiment", "emotion"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.dropna(subset=["clean_text", "sentiment", "emotion"]).copy()
df = df[df["clean_text"].str.strip().ne("")].copy()

# Stratify by the joint emotion-sentiment combination.
# This preserves the relationship between both targets
# while using one shared train/test split.
joint_strata = df["emotion"].astype(str) + "|" + df["sentiment"].astype(str)

# Each joint category needs enough examples for both splits.
if joint_strata.value_counts().min() < 2:
    raise ValueError(
        "At least one emotion-sentiment combination has fewer than "
        "two examples. A joint stratified split is not possible."
    )

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=joint_strata
)

print(f"\nTotal samples: {len(df):,}")
print(f"Training samples: {len(train_df):,}")
print(f"Testing samples: {len(test_df):,}")


# --------------------------------------------------
# TF-IDF feature extraction
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])

print(f"Training feature matrix: {X_train.shape}")
print(f"Testing feature matrix: {X_test.shape}")


# --------------------------------------------------
# Save vectorizer
# --------------------------------------------------

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

print("TF-IDF vectorizer saved.")


# --------------------------------------------------
# Train and evaluate both classifiers
# --------------------------------------------------

tasks = {
    "sentiment": {
        "target": "sentiment",
        "filename": "sentiment_model.joblib"
    },
    "emotion": {
        "target": "emotion",
        "filename": "emotion_model.joblib"
    }
}

for task_name, config in tasks.items():

    print(f"\n{'=' * 50}")
    print(f"TRAINING {task_name.upper()} MODEL")
    print(f"{'=' * 50}")

    y_train = train_df[config["target"]]
    y_test = test_df[config["target"]]

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{task_name.title()} accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    )

    model_path = MODEL_DIR / config["filename"]
    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")


print("\nAll models trained and saved successfully!")