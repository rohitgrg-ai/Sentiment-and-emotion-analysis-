"""Zero-shot comparison: pretrained transformers vs the existing TF-IDF systems.

Run from the project root (needs `pip install transformers torch`; a GPU is optional):
    python src/transformer_zero_shot.py
    python src/transformer_zero_shot.py --limit 40      # quick smoke test

The pretrained models are used AS-IS: no training, no tuning, no threshold changes on the
ambiguous set. This is evaluation only, so the 500 rows stay a clean test set.

Default models (swap with --emotion-model / --sentiment-model):
    sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest   -> negative / neutral / positive
    emotion:   j-hartmann/emotion-english-distilroberta-base      -> anger, disgust, fear, joy,
                                                                    neutral, sadness, surprise
Label mapping (fixed BEFORE looking at results, and printed in the report):
    disgust -> anger.  'neutral' has no matching class in the project's six emotions, and the
    pretrained emotion model has no 'love', so a 'neutral' prediction counts as wrong and
    'love' rows cannot be matched. Both limits are reported, not hidden.

Existing systems A (text SVM) and D (SVM + emoji voting) are read from
results/ambiguity/ai_suggested_all/predictions_detailed.csv (run evaluate_ambiguity.py first).
Outputs: results/transformers/zero_shot/{predictions.csv, metrics.csv, report.txt}
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ambiguous_social_media_test.csv"
EXISTING = ROOT / "results" / "ambiguity" / "ai_suggested_all" / "predictions_detailed.csv"
OUT = ROOT / "results" / "transformers" / "zero_shot"

EMOTION_MAP = {"anger": "anger", "disgust": "anger", "fear": "fear", "joy": "joy",
               "sadness": "sadness", "surprise": "surprise", "love": "love", "neutral": "neutral"}


def top_label(result):
    """pipeline output is a dict or a list of dicts, depending on version."""
    while isinstance(result, list):
        result = result[0]
    return result["label"].lower(), float(result["score"])


def run_model(model_name, texts, batch_size):
    from transformers import pipeline  # imported here so the file compiles without it
    clf = pipeline("text-classification", model=model_name, truncation=True, max_length=128,
                   batch_size=batch_size, device=-1 if not _cuda() else 0)
    out = [top_label(r) for r in clf(list(texts))]
    return [o[0] for o in out], [o[1] for o in out]


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def metrics(gold, pred, labels):
    return {"accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0)}


def build_table(df):
    """df has gold_*, secondary_emotion and <system>_emotion/<system>_sentiment columns."""
    systems = [c[:-8] for c in df.columns if c.endswith("_emotion") and not c.startswith(("gold", "secondary", "emoji"))]
    rows = []
    subsets = {"all rows": df.index == df.index,
               "emoji rows": df["emoji_flag"], "no-emoji rows": ~df["emoji_flag"]}
    for cat, sub in df.groupby("category"):
        subsets[f"category: {cat}"] = df.index.isin(sub.index)
    for name, mask in subsets.items():
        d = df[mask]
        if len(d) == 0:
            continue
        for s in systems:
            emo_ok = (d[f"{s}_emotion"] == d["gold_emotion"])
            lenient = emo_ok | (d[f"{s}_emotion"] == d["secondary_emotion"])
            rows.append({"subset": name, "n": len(d), "system": s,
                         "sentiment_acc": accuracy_score(d["gold_sentiment"], d[f"{s}_sentiment"]),
                         "emotion_acc_strict": emo_ok.mean(), "emotion_acc_lenient": lenient.mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment-model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--emotion-model", default="j-hartmann/emotion-english-distilroberta-base")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    amb = pd.read_csv(DATA)
    if a.limit:
        amb = amb.head(a.limit)
    old = pd.read_csv(EXISTING).set_index("id")
    amb = amb.set_index("id").join(old[["A_emotion", "A_sentiment", "D_emotion", "D_sentiment"]])
    df = pd.DataFrame({"text": amb["text"], "category": amb["category"],
                       "gold_emotion": amb["expected_emotion"], "gold_sentiment": amb["expected_sentiment"],
                       "secondary_emotion": amb["secondary_emotion"].fillna(""),
                       "emoji_flag": amb["emoji_present"].astype(str).str.lower().isin(["true", "1", "yes"]),
                       "A_emotion": amb["A_emotion"], "A_sentiment": amb["A_sentiment"],
                       "D_emotion": amb["D_emotion"], "D_sentiment": amb["D_sentiment"]})

    sent, sent_p = run_model(a.sentiment_model, df["text"], a.batch_size)
    emo, emo_p = run_model(a.emotion_model, df["text"], a.batch_size)
    df["T_sentiment"] = sent
    df["T_sentiment_conf"] = sent_p
    df["T_emotion"] = [EMOTION_MAP.get(e, e) for e in emo]
    df["T_emotion_raw"] = emo
    df["T_emotion_conf"] = emo_p

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "predictions.csv")
    table = build_table(df)
    table.to_csv(OUT / "metrics.csv", index=False)

    overall = table[table["subset"] == "all rows"].set_index("system")
    lines = ["Zero-shot transformer vs TF-IDF systems on the ambiguous set (labels: AI-suggested, not human-verified)",
             f"Sentiment model: {a.sentiment_model}", f"Emotion model:   {a.emotion_model}",
             "Mapping: disgust->anger. 'neutral' predictions are counted wrong (no neutral class in the gold labels).",
             "System T = pretrained transformer used as-is; A = text SVM; D = SVM + emoji voting.", "",
             overall.round(3).to_string(), "",
             "Share of rows the transformer called 'neutral' emotion: "
             f"{(df['T_emotion'] == 'neutral').mean():.1%}",
             "Gold 'love' rows (transformer has no love class): "
             f"{(df['gold_emotion'] == 'love').sum()}", "",
             "Caveats: one test set, AI-suggested labels, no confidence intervals here. Differences of a few "
             "points on 500 rows may be noise; run the McNemar test in evaluate_ambiguity.py before claiming a win."]
    (OUT / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
