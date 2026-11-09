"""Evaluate prediction systems on the ambiguous social-media test set.

Usage (from the project root):
    python src/evaluate_ambiguity.py                  # all 500 rows, AI-suggested labels
    python src/evaluate_ambiguity.py --build-verified # only human_verified rows, reviewed labels win

Systems evaluated (decision logic is never changed here):
    A  existing text-only Linear SVM            (words only; emojis ignored)
    D  existing predictor.py                    (A + rule-based emoji voting)
    B*, C*  experimental emoji-aware SVMs       (added automatically if models/experimental/ exists)

The ambiguous set is evaluation-only: nothing here trains or tunes on it.
Outputs go to results/ambiguity/<label_set>/.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictor import EMOTION_TO_SENTIMENT, Predictor  # noqa: E402

DATA = ROOT / "data"
OUT_ROOT = ROOT / "results" / "ambiguity"
EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]


# ----------------------------------------------------------------------------- data
def load_dataset(verified=False):
    df = pd.read_csv(DATA / "ambiguous_social_media_test.csv", keep_default_na=False, dtype=str)
    df["emoji_present"] = df["emoji_present"].str.lower().eq("true")
    if not verified:
        df["gold_sentiment"] = df["expected_sentiment"]
        df["gold_emotion"] = df["expected_emotion"]
        return df, "ai_suggested_all"
    df = df[df["annotation_status"] == "human_verified"].copy()
    df["gold_sentiment"] = np.where(df["reviewed_sentiment"] != "", df["reviewed_sentiment"], df["expected_sentiment"])
    df["gold_emotion"] = np.where(df["reviewed_emotion"] != "", df["reviewed_emotion"], df["expected_emotion"])
    df["gold_emotion"] = df["gold_emotion"].replace({"none": "", "-": ""})
    out = df.drop(columns=["gold_sentiment", "gold_emotion"]).assign(
        final_sentiment=df["gold_sentiment"], final_emotion=df["gold_emotion"])
    out.to_csv(DATA / "ambiguous_human_verified.csv", index=False)
    return df, "human_verified"


# ----------------------------------------------------------------------------- systems
def run_existing(df):
    """Systems A and D from the unchanged predictor.py, plus its internals for the CSV."""
    predictor = Predictor()
    rows = []
    for text in df["text"]:
        r = predictor.predict(text)
        emoji_emotions = [emo for _, emo in r["emojis_used"]]
        if emoji_emotions:
            top = Counter(emoji_emotions).most_common(1)[0][0]
            emoji_emotion, emoji_sentiment = top, EMOTION_TO_SENTIMENT[top]
        else:
            emoji_emotion = emoji_sentiment = ""
        rows.append({
            "A_emotion": r["emotion_from_text"], "A_sentiment": r["sentiment_from_text"],
            "A_emotion_conf": round(r["emotion_text_confidence"], 3),
            "A_sentiment_conf": round(r["sentiment_text_confidence"], 3),
            "emoji_vote_emotion": emoji_emotion, "emoji_vote_sentiment": emoji_sentiment,
            "emojis_recognised": " ".join(e for e, _ in r["emojis_used"]),
            "known_words": r["known_words"],
            "D_emotion": r["emotion"], "D_sentiment": r["sentiment"],
            "D_emotion_conf": round(max(r["emotion_scores"].values()), 3),
            "D_sentiment_conf": round(max(r["sentiment_scores"].values()), 3),
        })
    return pd.DataFrame(rows, index=df.index)


def available_systems(pred):
    return [s for s in ["A", "B_orig", "C_orig", "B_aug", "C_aug", "D"] if f"{s}_emotion" in pred.columns]


SYSTEM_NAMES = {
    "A": "A  text-only SVM (existing)",
    "B_orig": "B  demojized TF-IDF SVM, original training data",
    "C_orig": "C  hybrid text+emoji SVM, original training data",
    "B_aug": "B' demojized TF-IDF SVM, synthetic emoji-augmented training",
    "C_aug": "C' hybrid text+emoji SVM, synthetic emoji-augmented training",
    "D": "D  existing predictor.py (text SVM + emoji voting)",
}


# ----------------------------------------------------------------------------- metrics
def scores(y_true, y_pred, labels):
    if len(y_true) == 0:
        return {"n": 0}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"n": int(len(y_true)), "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_precision": round(p, 3), "macro_recall": round(r, 3), "macro_f1": round(f, 3),
            "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 3)}


def subset_metrics(df, pred, system, mask):
    d, p = df[mask], pred[mask]
    sent = scores(d["gold_sentiment"], p[f"{system}_sentiment"], SENTIMENTS)
    has_emo = d["gold_emotion"] != ""
    emo = scores(d.loc[has_emo, "gold_emotion"], p.loc[has_emo, f"{system}_emotion"], EMOTIONS)
    if emo.get("n"):
        lenient = (p.loc[has_emo, f"{system}_emotion"] == d.loc[has_emo, "gold_emotion"]) | \
                  (p.loc[has_emo, f"{system}_emotion"] == d.loc[has_emo, "secondary_emotion"])
        emo["lenient_accuracy"] = round(lenient.mean(), 3)
    return sent, emo


def subsets(df):
    paired = df["paired_group_id"] != ""
    out = {
        "all": pd.Series(True, index=df.index),
        "emoji_present": df["emoji_present"],
        "emoji_free": ~df["emoji_present"],
        "paired_examples": paired,
        "sarcasm_and_irony": df["category"].isin(["sarcasm", "irony"]),
    }
    for cat in sorted(df["category"].unique()):
        out[f"category: {cat}"] = df["category"] == cat
    for lvl in ["Easy", "Medium", "Hard"]:
        out[f"level: {lvl}"] = df["ambiguity_level"] == lvl
    return out


def group_consistency(df, pred, system):
    """Share of paired groups where EVERY member is predicted correctly."""
    res = {}
    for task, gold in [("sentiment", "gold_sentiment"), ("emotion", "gold_emotion")]:
        d = df[(df["paired_group_id"] != "") & (df[gold] != "")]
        ok = (pred.loc[d.index, f"{system}_{task}"] == d[gold]).groupby(d["paired_group_id"]).all()
        res[task] = f"{ok.mean():.1%} of {len(ok)} groups"
    return res


def confusion(df, pred, system, task, out_dir):
    gold = df[f"gold_{task}"]
    mask = gold != ""
    labels = SENTIMENTS if task == "sentiment" else EMOTIONS
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(gold[mask], pred.loc[mask, f"{system}_{task}"], labels=labels,
                                            ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{task.title()} — {SYSTEM_NAMES[system].split('(')[0].strip()}\n(ambiguous set, counts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{task}_{system}.png", dpi=130)
    plt.close(fig)


def bootstrap_ci(correct, n_boot=2000, seed=42):
    """95% bootstrap confidence interval for an accuracy (correct = boolean Series)."""
    values = np.asarray(correct, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def mcnemar_p(correct_a, correct_b):
    """Exact McNemar test (two-sided) on paired correct/incorrect outcomes."""
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    if only_a + only_b == 0:
        return 1.0, only_a, only_b
    return binomtest(only_a, only_a + only_b, 0.5).pvalue, only_a, only_b


def significance_lines(df, detail, systems):
    """Is each system different from the A baseline? (McNemar, emoji-containing rows and all rows)."""
    lines = ["SIGNIFICANCE vs A (exact McNemar; 'wins' = rows only this system got right, 'losses' = only A got right)"]
    for subset_name, mask in [("all rows", pd.Series(True, index=df.index)), ("emoji rows", df["emoji_present"])]:
        for task in ["sentiment", "emotion"]:
            m = mask & (df[f"gold_{task}"] != "")
            for s in systems:
                if s == "A":
                    continue
                if task == "sentiment":
                    ca, cb = detail.loc[m, "A_sentiment_correct"], detail.loc[m, f"{s}_sentiment_correct"]
                else:
                    ca, cb = detail.loc[m, "A_emotion_correct"] == "True", detail.loc[m, f"{s}_emotion_correct"] == "True"
                p, la, lb = mcnemar_p(cb, ca)
                lines.append(f"  {subset_name:10s} {task:9s} {s:7s} wins {la:3d} / losses {lb:3d}  p = {p:.4f}"
                             + ("  (significant at 0.05)" if p < 0.05 else ""))
    lines.append("")
    return lines


# ----------------------------------------------------------------------------- main
def evaluate(df, pred, label_set):
    out_dir = OUT_ROOT / label_set
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = available_systems(pred)

    detail = df[["id", "text", "category", "ambiguity_level", "emoji_present", "paired_group_id",
                 "gold_sentiment", "gold_emotion", "secondary_emotion", "review_flags"]].join(pred)
    for s in systems:
        detail[f"{s}_sentiment_correct"] = detail[f"{s}_sentiment"] == detail["gold_sentiment"]
        detail[f"{s}_emotion_correct"] = np.where(detail["gold_emotion"] == "", "",
                                                  (detail[f"{s}_emotion"] == detail["gold_emotion"]).astype(str))
    detail.to_csv(out_dir / "predictions_detailed.csv", index=False)

    table, summary = [], {"label_set": label_set, "rows": int(len(df)), "systems": {}}
    for s in systems:
        summary["systems"][s] = {"name": SYSTEM_NAMES[s], "paired_group_all_correct": group_consistency(df, pred, s)}
        for name, mask in subsets(df).items():
            sent, emo = subset_metrics(df, pred, s, mask)
            summary["systems"][s][name] = {"sentiment": sent, "emotion": emo}
            table.append({"system": s, "subset": name, "n_sent": sent.get("n", 0),
                          "sent_acc": sent.get("accuracy"), "sent_macro_f1": sent.get("macro_f1"),
                          "n_emo": emo.get("n", 0), "emo_acc": emo.get("accuracy"),
                          "emo_macro_f1": emo.get("macro_f1"), "emo_lenient_acc": emo.get("lenient_accuracy")})
        for task in ["sentiment", "emotion"]:
            confusion(df, pred, s, task, out_dir)
    table = pd.DataFrame(table)
    table.to_csv(out_dir / "metrics_by_subset.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"AMBIGUOUS SET EVALUATION — labels: {label_set} ({len(df)} rows)",
             "Labels are AI-suggested and NOT human-verified." if label_set == "ai_suggested_all"
             else "Labels: human-verified rows only (reviewed labels override suggestions).",
             "Confidence: A and D = softmax of SVM decision scores (uncalibrated); B/C variants = "
             "sigmoid-calibrated probabilities (calibrated on the training split).",
             "Accuracy 95% CI = bootstrap over rows (2000 resamples, seed 42).", ""]
    for s in systems:
        lines.append(SYSTEM_NAMES[s])
        sub = table[table.system == s].set_index("subset")
        for name in ["all", "emoji_present", "emoji_free", "paired_examples", "sarcasm_and_irony"]:
            r = sub.loc[name]
            mask = subsets(df)[name]
            ci_s = bootstrap_ci(detail.loc[mask, f"{s}_sentiment_correct"].astype(bool))
            em = mask & (df["gold_emotion"] != "")
            ci_e = bootstrap_ci(detail.loc[em, f"{s}_emotion_correct"] == "True")
            lines.append(f"  {name:18s} sentiment acc {r.sent_acc:.3f} [{ci_s[0]:.2f}-{ci_s[1]:.2f}] F1 {r.sent_macro_f1:.3f} (n={r.n_sent}) | "
                         f"emotion acc {r.emo_acc:.3f} [{ci_e[0]:.2f}-{ci_e[1]:.2f}] F1 {r.emo_macro_f1:.3f} lenient {r.emo_lenient_acc:.3f} (n={r.n_emo})")
        lines.append(f"  paired groups fully correct: {summary['systems'][s]['paired_group_all_correct']}")
        lines.append("")
    lines += significance_lines(df, detail, systems)
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return detail, table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-verified", action="store_true", help="use only human_verified rows")
    args = parser.parse_args()
    df, label_set = load_dataset(verified=args.build_verified)
    if df.empty:
        print("No human_verified rows yet - review data/ambiguous_social_media_test.csv first "
              "(see data/ambiguous_annotation_guide.md).")
        return
    pred = run_existing(df)
    try:
        from experimental_models import predict_experimental  # added in Phase 4
        pred = pred.join(predict_experimental(df["text"].tolist(), index=df.index))
    except FileNotFoundError:
        print("(experimental models not trained yet - evaluating A and D only)\n")
    except ImportError:
        pass
    evaluate(df, pred, label_set)


if __name__ == "__main__":
    main()
