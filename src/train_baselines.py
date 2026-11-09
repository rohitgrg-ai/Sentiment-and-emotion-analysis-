"""Baseline models (TF-IDF + linear classifiers) for emotion and sentiment.

Run after preprocessing.py. Writes metrics, confusion matrices, misclassified
examples and the best model per task to results/ and models/.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, MODELS = ROOT / "data", ROOT / "results", ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

# Vectorizer is fit on training text only.
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vectorizer.fit_transform(train["clean_text"].fillna(""))
X_test = vectorizer.transform(test["clean_text"].fillna(""))
joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.joblib")

TASKS = {"emotion": "emotion", "sentiment": "sentiment"}  # task -> target column
MODELS_TO_RUN = {
    "majority_baseline": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, C=4.0),
    "linear_svm": lambda: LinearSVC(C=0.5),
    "naive_bayes": lambda: MultinomialNB(alpha=0.3),
}

summary = {}
for task, col in TASKS.items():
    y_train, y_test = train[col], test[col]
    summary[task] = {}
    best_name, best_f1, best_model, best_pred = None, -1, None, None
    for name, make in MODELS_TO_RUN.items():
        model = make().fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        summary[task][name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"{task:9s} {name:20s} acc={acc:.4f} macro_f1={f1:.4f}")
        if name != "majority_baseline" and f1 > best_f1:
            best_name, best_f1, best_model, best_pred = name, f1, model, pred

    summary[task]["best_model"] = best_name
    print(f"\n[{task}] best: {best_name}\n")
    report = classification_report(y_test, best_pred, digits=3)
    print(report)
    (RESULTS / f"{task}_classification_report.txt").write_text(f"Best model: {best_name}\n\n{report}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, ax=ax, cmap="Blues", normalize="true",
        values_format=".2f", xticks_rotation=45)
    ax.set_title(f"{task.title()} - {best_name} (row-normalised)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{task}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    wrong = test.assign(predicted=best_pred)
    wrong = wrong[wrong[col] != wrong["predicted"]][["text", col, "predicted"]]
    wrong.sample(n=min(200, len(wrong)), random_state=42).to_csv(
        RESULTS / f"{task}_misclassified_sample.csv", index=False)
    joblib.dump(best_model, MODELS / f"{task}_{best_name}.joblib")

(RESULTS / "baseline_results.json").write_text(json.dumps(summary, indent=2))
