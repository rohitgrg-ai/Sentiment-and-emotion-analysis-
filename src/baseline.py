"""TF-IDF + Linear SVM baseline, trained on data/final.csv (made by balance_dataset.py).

    python src/baseline.py

Scores the baseline on its own test split and on human-labelled GoEmotions text.
The app shows its answer for comparison only. The transformer makes every final decision.
"""
import argparse
import json
import re
from pathlib import Path

import emoji
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TASKS = ("emotion", "sentiment")


def clean_text(text):
    text = emoji.demojize(str(text).lower(), delimiters=(" ", " "))
    return re.sub(r"\s+", " ", re.sub(r"http\S+|www\S+", "", text)).strip()


def load_keyword_model(models_dir=MODELS):
    """Return a function giving the baseline's emotion, or None if the baseline is not trained."""
    paths = models_dir / "tfidf_vectorizer.joblib", models_dir / "emotion_linear_svm.joblib"
    if not all(path.exists() for path in paths):
        return None
    vectorizer, classifier = (joblib.load(path) for path in paths)
    return lambda text: str(classifier.predict(vectorizer.transform([clean_text(text)]))[0])


def scores(gold, predicted):
    return {"accuracy": round(accuracy_score(gold, predicted), 4), "macro_f1": round(f1_score(gold, predicted, average="macro"), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=MODELS)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "baseline_metrics.json")
    args = parser.parse_args()

    rows = pd.read_csv(args.data / "final.csv")
    rows["clean_text"] = rows["text"].map(clean_text)
    train, test = train_test_split(rows, test_size=0.20, random_state=42, stratify=rows["label"].astype(str) + "_" + rows["sentiment"])
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train, x_test = vectorizer.fit_transform(train["clean_text"]), vectorizer.transform(test["clean_text"])
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.out / "tfidf_vectorizer.joblib")

    metrics, models = {}, {}
    for task in TASKS:
        models[task] = LinearSVC(C=0.5, random_state=42).fit(x_train, train[task])
        joblib.dump(models[task], args.out / f"{task}_linear_svm.joblib")
        metrics[f"{task}_own_test"] = scores(test[task], models[task].predict(x_test))

    natural = pd.read_csv(args.data / "natural_emotion_test.csv")
    natural = natural[natural["emotion"] != "neutral"]  # the baseline has no neutral class
    predicted = models["emotion"].predict(vectorizer.transform(natural["text"].map(clean_text)))
    metrics["emotion_natural_text"] = scores(natural["emotion"], predicted)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
