"""Experimental emoji-aware models (Experiment 2). The original models are untouched.

Model B  demojized TF-IDF:   emojis -> names inline ("happy 😭" -> "happy loudly crying face"),
                              existing clean_text, TF-IDF (1-2 grams), Linear SVM.
Model C  hybrid features:     text TF-IDF (emojis removed)  +  emoji-name TF-IDF  +  numeric emoji
                              features (count, frequency, repetition), joined with scipy.sparse.hstack,
                              Linear SVM.

Each is trained twice on the SAME 80% training split as the existing models:
  *_orig  original training text (it contains no emojis at all)
  *_aug   a synthetic emoji-augmented copy of the training text (see augment_with_emojis):
          labels are never changed; emojis are added with deliberate noise so the model
          cannot learn "emoji = label". This injects an assumed emoji prior, so *_aug results
          show what a model learns from that prior, not from real emoji usage.

All SVMs use the baseline settings (C=0.5) and random_state=42. Probabilities come from
sigmoid calibration (CalibratedClassifierCV, 3-fold, on the training split only); labels come
from the raw SVM. The ambiguous test set is never used here.

Train:  python src/experimental_models.py     (writes models/experimental/, results/emoji_comparison/)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emoji_features import (demojize, emoji_description, extract_emojis,  # noqa: E402
                            numeric_emoji_features, strip_emojis)
from preprocessing import clean_text  # noqa: E402

DATA, EXP_MODELS = ROOT / "data", ROOT / "models" / "experimental"
RESULTS = ROOT / "results" / "emoji_comparison"
SEED = 42
VARIANTS = ["B_orig", "C_orig", "B_aug", "C_aug"]
TFIDF = dict(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

# ---------------------------------------------------------------- synthetic augmentation
# Emojis commonly used with each emotion (assumed prior, used ONLY for training augmentation).
AUG_POOLS = {
    "joy": "😂😄😁😊🥳🎉😆🤣😃👍", "sadness": "😢😭😞😔💔😩🥺😿",
    "anger": "😡😠🤬😤🙄😒💢", "fear": "😨😰😱😬😟🙀",
    "love": "❤😍🥰😘💕💖🤗", "surprise": "😲😮🤯😳😯",
}
AMBIGUOUS_POOL = "🙂🤔😅👀💀🙏😐🔥💯🙃"
AUG_SETTINGS = {"share_of_rows": 0.5, "p_consistent": 0.60, "p_ambiguous": 0.25, "p_contradicting": 0.15,
                "n_emojis": {1: 0.7, 2: 0.2, 3: 0.1}, "p_at_start": 0.2}


def augment_with_emojis(df, seed=SEED):
    """Copy of df where about half the texts get 1-3 emojis. Labels are unchanged.
    60% emojis match the row's emotion, 25% are ambiguous, 15% contradict it (noise)."""
    rng = np.random.default_rng(seed)
    s = AUG_SETTINGS
    out = df.copy()
    emotions = list(AUG_POOLS)
    texts = []
    for text, emo in zip(out["text"], out["emotion"]):
        if rng.random() >= s["share_of_rows"]:
            texts.append(text)
            continue
        kind = rng.choice(["consistent", "ambiguous", "contradicting"],
                          p=[s["p_consistent"], s["p_ambiguous"], s["p_contradicting"]])
        if kind == "consistent":
            pool = AUG_POOLS[emo]
        elif kind == "ambiguous":
            pool = AMBIGUOUS_POOL
        else:
            pool = AUG_POOLS[rng.choice([e for e in emotions if e != emo])]
        n = int(rng.choice(list(s["n_emojis"]), p=list(s["n_emojis"].values())))
        emoji = str(rng.choice(list(pool)))
        chunk = emoji * n if rng.random() < 0.5 else "".join(str(e) for e in rng.choice(list(pool), size=n))
        texts.append(f"{chunk} {text}" if rng.random() < s["p_at_start"] else f"{text} {chunk}")
    out["text"] = texts
    return out


# ---------------------------------------------------------------- models
def _svm():
    return LinearSVC(C=0.5, random_state=SEED, max_iter=5000)


class _Base:
    def _fit_classifier(self, X, y):
        self.calibrated = CalibratedClassifierCV(_svm(), cv=3, method="sigmoid", ensemble=False).fit(X, y)
        self.svm = self.calibrated.calibrated_classifiers_[0].estimator  # SVM refit on all training rows
        self.classes_ = self.svm.classes_
        return self

    def predict(self, texts):
        return self.svm.predict(self.transform(texts))

    def decision_function(self, texts):
        return self.svm.decision_function(self.transform(texts))

    def predict_proba(self, texts):
        """Sigmoid-calibrated probabilities (calibrated on the training split)."""
        return self.calibrated.predict_proba(self.transform(texts))


class ModelB(_Base):
    """Demojized text -> TF-IDF -> Linear SVM."""
    def fit(self, texts, y):
        self.vec = TfidfVectorizer(**TFIDF)
        return self._fit_classifier(self.vec.fit_transform([clean_text(demojize(t)) for t in texts]), y)

    def transform(self, texts):
        return self.vec.transform([clean_text(demojize(t)) for t in texts])

    def contributions(self, text):
        """Ablation: decision scores with emojis vs with emojis removed (per class)."""
        with_e = self.decision_function([text])[0]
        without = self.decision_function([strip_emojis(text)])[0]
        return {"text_only_scores": dict(zip(self.classes_, without)),
                "emoji_effect": dict(zip(self.classes_, with_e - without))}


class ModelC(_Base):
    """[text TF-IDF | emoji-name TF-IDF | numeric emoji features] -> Linear SVM."""
    def fit(self, texts, y):
        self.text_vec = TfidfVectorizer(**TFIDF)
        X_text = self.text_vec.fit_transform([clean_text(strip_emojis(t)) for t in texts])
        descriptions = [emoji_description(t) for t in texts]
        if any(descriptions):
            self.emoji_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            X_emoji = self.emoji_vec.fit_transform(descriptions)
        else:  # original training data has no emojis: the emoji channel has no columns to learn
            self.emoji_vec = None
            X_emoji = csr_matrix((len(texts), 0))
        self.n_text, self.n_emoji = X_text.shape[1], X_emoji.shape[1]
        X = hstack([X_text, X_emoji, numeric_emoji_features(texts)]).tocsr()
        return self._fit_classifier(X, y)

    def transform(self, texts):
        X_text = self.text_vec.transform([clean_text(strip_emojis(t)) for t in texts])
        X_emoji = (self.emoji_vec.transform([emoji_description(t) for t in texts]) if self.emoji_vec
                   else csr_matrix((len(texts), 0)))
        return hstack([X_text, X_emoji, numeric_emoji_features(texts)]).tocsr()

    def contributions(self, text):
        """Exact split of each class's decision score into text / emoji / intercept parts."""
        x = self.transform([text])
        w, b = self.svm.coef_, self.svm.intercept_
        t_end, e_end = self.n_text, self.n_text + self.n_emoji
        text_part = x[:, :t_end] @ w[:, :t_end].T
        emoji_part = x[:, t_end:] @ w[:, t_end:].T  # emoji TF-IDF + numeric features
        return {"text": dict(zip(self.classes_, np.ravel(text_part))),
                "emoji": dict(zip(self.classes_, np.ravel(emoji_part))),
                "intercept": dict(zip(self.classes_, b))}


# ---------------------------------------------------------------- loading / prediction
_CACHE = {}


def load_variant(name):
    if name not in _CACHE:
        path = EXP_MODELS / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run: python src/experimental_models.py")
        _CACHE[name] = joblib.load(path)
    return _CACHE[name]


def predict_experimental(texts, index=None, variants=VARIANTS):
    """Predictions + calibrated confidence for each experimental variant (one row per text)."""
    out = pd.DataFrame(index=index if index is not None else range(len(texts)))
    for v in variants:
        bundle = load_variant(v)
        for task in ["emotion", "sentiment"]:
            m = bundle[task]
            out[f"{v}_{task}"] = m.predict(texts)
            out[f"{v}_{task}_conf"] = np.round(m.predict_proba(texts).max(axis=1), 3)
    return out


# ---------------------------------------------------------------- training
def _metrics(y, p):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": round(accuracy_score(y, p), 4), "macro_precision": round(pr, 4),
            "macro_recall": round(rc, 4), "macro_f1": round(f, 4),
            "weighted_f1": round(f1_score(y, p, average="weighted"), 4)}


def train_all():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train_aug = augment_with_emojis(train, seed=SEED)
    test_aug = augment_with_emojis(test, seed=SEED + 1)  # synthetic check only, never used for training
    EXP_MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    n_aug = int((train_aug["text"] != train["text"]).sum())
    print(f"Training rows: {len(train):,}; synthetic emoji rows in augmented copy: {n_aug:,}")

    results = {"augmentation": {**AUG_SETTINGS, "rows_with_emojis": n_aug, "seed": SEED}, "models": {}}
    for name in VARIANTS:
        cls = ModelB if name.startswith("B") else ModelC
        data = train_aug if name.endswith("aug") else train
        bundle = {}
        for task in ["emotion", "sentiment"]:
            model = cls().fit(data["text"].tolist(), data[task])
            bundle[task] = model
            res = results["models"].setdefault(name, {})
            p_test = model.predict(test["text"].tolist())
            res[f"{task}_original_test"] = _metrics(test[task], p_test)
            res[f"{task}_synthetic_emoji_test"] = _metrics(test_aug[task], model.predict(test_aug["text"].tolist()))
            prob_label = model.classes_[model.predict_proba(test["text"].tolist()).argmax(axis=1)]
            res[f"{task}_calibrated_vs_svm_label_disagreement"] = round(float((prob_label != p_test).mean()), 4)
            print(f"{name} {task:9s} original test macro-F1 {res[f'{task}_original_test']['macro_f1']:.4f} | "
                  f"synthetic-emoji test macro-F1 {res[f'{task}_synthetic_emoji_test']['macro_f1']:.4f}")
        joblib.dump(bundle, EXP_MODELS / f"{name}.joblib")

    # Existing model A on the same two test sets, for reference.
    from predictor import Predictor
    pa = Predictor()
    results["models"]["A_existing"] = {}
    for label, frame in [("original_test", test), ("synthetic_emoji_test", test_aug)]:
        X = pa.vectorizer.transform([clean_text(t) for t in frame["text"]])
        for task, m in [("emotion", pa.emotion_model), ("sentiment", pa.sentiment_model)]:
            results["models"]["A_existing"][f"{task}_{label}"] = _metrics(frame[task], m.predict(X))
    (RESULTS / "training_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved models to {EXP_MODELS} and results to {RESULTS / 'training_results.json'}")


if __name__ == "__main__":
    import experimental_models  # so pickled classes are stored as experimental_models.ModelB/C
    experimental_models.train_all()
