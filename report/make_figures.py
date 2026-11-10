"""Draw the report's exploration and results graphs with matplotlib.

    python report/make_figures.py --data data

Saves PNGs to report/figures/ and every number used in the report to results/report_numbers.json.
Needs the datasets, the trained model, and the trained baseline.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import emoji
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline import clean_text  # noqa: E402
from model import (EMOJI_EMOTION, EMOJI_WEIGHT, EMOTION_ID, EMOTION_SENTIMENT, EMOTIONS, EXCLUDED_WHEN_SARCASTIC, MODEL_DIR,  # noqa: E402
                   POSITIVE, SARCASM_THRESHOLD, load_trained, probabilities)

FIGURES = ROOT / "report" / "figures"
BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#b8bcc2", "#1f2a37", "#5b6675", "#e6e8ec"
WIDTH = 6.7
BASELINE_EMOTIONS = [e for e in EMOTIONS if e != "neutral"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def bars(ax, labels, values, color=BLUE, fmt="{:,.0f}", limit=None):
    """Horizontal bars, largest-first order as given, each with its value at the end."""
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions, values, color=color, height=0.68)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, limit or max(values) * 1.18)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for y, value in zip(positions, values):
        ax.text(value + ax.get_xlim()[1] * 0.012, y, fmt.format(value), va="center", color=INK)


def heatmap(ax, matrix, xlabels, ylabels, fmt="{:.0f}", vmax=None):
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax or matrix.max(), aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(axis="x", colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cutoff = (vmax or matrix.max()) * 0.55
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, fmt.format(value), ha="center", va="center", fontsize=8, color="white" if value > cutoff else INK)


def read_all(data):
    frames = {"nat": [], "tweet": [], "sarcasm": []}
    for key, stem in (("nat", "natural_emotion"), ("tweet", "tweet_emotion"), ("sarcasm", "sarcasm")):
        frames[key] = pd.concat([pd.read_csv(data / f"{stem}_{split}.csv").assign(split=split) for split in ("train", "val", "test")], ignore_index=True)
    frames["sentiment_test"] = pd.read_csv(data / "natural_sentiment_test.csv")
    frames["final"] = pd.read_csv(data / "final.csv")
    frames["raw_labels"] = pd.read_csv(data / "text2.csv", usecols=["label"])["label"].map(dict(enumerate(["sadness", "joy", "love", "anger", "fear", "surprise"])))
    frames["balance"] = json.loads((data / "balancing_summary.json").read_text())
    return frames


def exploration(frames, numbers):
    final, nat, tweet, sarcasm = frames["final"], frames["nat"], frames["tweet"], frames["sarcasm"]
    named = {"Baseline data": final, "GoEmotions comments": nat, "Tweets (SemEval)": tweet, "Sarcasm and irony tweets": sarcasm}

    sizes = {"Baseline data, raw": frames["balance"]["input_rows"], "Baseline data, balanced": len(final),
             "GoEmotions comments": len(nat), "TweetEval sentiment (test only)": len(frames["sentiment_test"]), "Sarcasm and irony tweets": len(sarcasm), "Tweets (SemEval emotion)": len(tweet)}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    bars(ax, list(sizes), list(sizes.values()))
    save(fig, "eda_dataset_sizes")

    b = frames["balance"]
    steps = {"Raw rows": b["input_rows"], "After removing conflicting labels": b["input_rows"] - b["rows_removed_for_conflicting_labels"],
             "After removing duplicates": b["clean_unique_text_rows_before_balance"], "After balancing": b["rows_after_balance"]}
    fig, ax = plt.subplots(figsize=(WIDTH, 1.6))
    bars(ax, list(steps), list(steps.values()))
    save(fig, "eda_cleaning_steps")

    shares = {"Baseline, before balancing": frames["raw_labels"].value_counts(normalize=True), "Baseline, after balancing": final["emotion"].value_counts(normalize=True),
              "GoEmotions comments": nat["emotion"].value_counts(normalize=True), "Tweets (SemEval)": tweet["emotion"].value_counts(normalize=True)}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6))
    for ax, (title, share) in zip(axes.flat, shares.items()):
        values = [100 * share.get(e, 0) for e in EMOTIONS]
        bars(ax, EMOTIONS, values, fmt="{:.1f}%", limit=62)
        ax.set_title(title)
    fig.tight_layout(h_pad=1.5, w_pad=2)
    save(fig, "eda_emotion_balance")
    numbers["emotion_share_percent"] = {k: {e: round(100 * float(v.get(e, 0)), 1) for e in EMOTIONS} for k, v in shares.items()}

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.8), sharex=True)
    numbers["median_words"] = {}
    for ax, (title, frame) in zip(axes.flat, named.items()):
        words = frame["text"].astype(str).str.split().str.len()
        numbers["median_words"][title] = float(words.median())
        ax.hist(words.clip(upper=70), bins=np.arange(0, 72, 2), color=BLUE, edgecolor="white", linewidth=0.5, weights=np.full(len(words), 100 / len(words)))
        ax.axvline(words.median(), color=ORANGE, linewidth=1.6)
        ax.text(0.98, 0.86, f"median {words.median():.0f} words", color=INK, ha="right", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel("% of rows")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Words in the post (70 or more shown at 70)")
    fig.tight_layout(h_pad=1.2, w_pad=2)
    save(fig, "eda_text_length")

    has_feel = {k: 100 * float(v["text"].astype(str).str.contains(r"\bfeel(?:ing|s)?\b", case=False, regex=True).mean()) for k, v in named.items()}
    has_emoji = {k: 100 * float(v["text"].astype(str).map(lambda t: emoji.emoji_count(t) > 0).mean()) for k, v in named.items()}
    numbers["percent_with_word_feel"], numbers["percent_with_emoji"] = has_feel, has_emoji
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.9))
    bars(axes[0], list(has_feel), list(has_feel.values()), fmt="{:.1f}%", limit=105)
    axes[0].set_title('Rows using "feel", "feels" or "feeling"')
    bars(axes[1], list(has_emoji), list(has_emoji.values()), color=ORANGE, fmt="{:.1f}%", limit=max(has_emoji.values()) * 1.3)
    axes[1].set_title("Rows containing an emoji")
    axes[1].set_yticklabels([])
    fig.tight_layout(w_pad=1.5)
    save(fig, "eda_writing_style")

    counts = sarcasm.groupby("source")["sarcastic"].agg(["size", "sum"])
    numbers["sarcasm_rows"] = {source: {"rows": int(row["size"]), "sarcastic": int(row["sum"])} for source, row in counts.iterrows()}
    labels = {"isarcasm": "iSarcasm", "isarcasm_rephrase": "iSarcasm plain rephrases", "semeval2018_irony": "SemEval irony"}
    names, sarcastic = [labels[source] for source in counts.index], counts["sum"].values
    plain = counts["size"].values - sarcastic
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    positions = np.arange(len(names))[::-1]
    ax.barh(positions, plain, color=GREY, height=0.62, label="Not sarcastic", edgecolor="white", linewidth=2)
    ax.barh(positions, sarcastic, left=plain, color=ORANGE, height=0.62, label="Sarcastic or ironic", edgecolor="white", linewidth=2)
    ax.set_yticks(positions, names)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for y, a, b_ in zip(positions, plain, sarcastic):
        ax.text(a / 2, y, f"{a:,}", ha="center", va="center")
        if b_:
            ax.text(a + b_ / 2, y, f"{b_:,}", ha="center", va="center")
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
    save(fig, "eda_sarcasm_balance")

    table = pd.crosstab(final["emotion"], final["sentiment"]).reindex(index=BASELINE_EMOTIONS, columns=["negative", "neutral", "positive"]).fillna(0)
    numbers["baseline_emotion_by_sentiment"] = table.astype(int).to_dict("index")
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    heatmap(ax, table.values, table.columns, table.index, fmt="{:,.0f}")
    ax.set_xlabel("Word-list sentiment label")
    ax.tick_params(axis="x", labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_ha("center")
    save(fig, "eda_emotion_vs_sentiment")


def box(ax, x, y, w, h, text, fill="#eaf2fb", edge=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08", facecolor=fill, edgecolor=edge, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.8)


def arrow(ax, start, end, color=MUTED):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1, "shrinkA": 0, "shrinkB": 0})


def pipeline():
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    box(ax, 0.05, 3.3, 1.6, 0.9, "Social media\npost")
    box(ax, 2.2, 3.3, 2.1, 0.9, "RoBERTa\n(shared encoder)")
    box(ax, 4.9, 3.85, 1.9, 0.72, "Emotion scores\n(7 feelings)")
    box(ax, 4.9, 2.85, 1.9, 0.72, "Sarcasm\nprobability")
    box(ax, 7.4, 3.85, 2.5, 0.72, "Emoji vote added\n(weight 0.25)")
    box(ax, 7.4, 2.05, 2.5, 1.12, "Sarcasm rule: at 90% or\nmore, joy or love becomes\nthe next likely feeling")
    box(ax, 4.9, 1.05, 1.9, 0.72, "Final emotion", "#fdeee7", ORANGE)
    box(ax, 2.2, 1.05, 2.1, 0.72, "Sentiment\n(from the emotion)", "#fdeee7", ORANGE)
    box(ax, 0.05, 0.1, 6.75, 0.62, "TF-IDF + Linear SVM: its answer is shown for comparison only", "#f1f2f4", GREY)
    arrow(ax, (1.7, 3.75), (2.17, 3.75))
    arrow(ax, (4.33, 3.9), (4.87, 4.18))
    arrow(ax, (4.33, 3.6), (4.87, 3.25))
    arrow(ax, (6.83, 4.2), (7.37, 4.2))
    arrow(ax, (8.65, 3.82), (8.65, 3.2))
    arrow(ax, (6.83, 3.05), (7.37, 2.85))
    arrow(ax, (7.37, 2.1), (6.83, 1.5))
    arrow(ax, (4.87, 1.41), (4.33, 1.41))
    arrow(ax, (0.85, 3.27), (0.85, 0.75), GREY)
    save(fig, "method_pipeline")


def predictions(name, texts, model, tokenizer, cache):
    path = cache / f"{name}.npz" if cache else None
    if path and path.exists():
        stored = np.load(path)
        return stored["emotion"], stored["sarcasm"]
    emotion, sarcasm = probabilities(model(), tokenizer(), texts)
    if path:
        np.savez(path, emotion=emotion, sarcasm=sarcasm)
    return emotion, sarcasm


def blended(texts, emotion):
    scores = emotion.copy()
    for row, text in enumerate(texts):
        found = [EMOJI_EMOTION[c] for c in text if c in EMOJI_EMOTION]
        if found:
            vote = np.array([found.count(name) for name in EMOTIONS], dtype=float) / len(found)
            weight = EMOJI_WEIGHT if re.search(r"[A-Za-z0-9]", "".join(" " if c in EMOJI_EMOTION else c for c in text)) else 1.0
            scores[row] = (1 - weight) * scores[row] + weight * vote
    return scores


def decide(scores, sarcasm, threshold):
    adjusted, flip = scores.copy(), (sarcasm >= threshold) & np.isin(scores.argmax(1), POSITIVE)
    adjusted[np.ix_(flip, EXCLUDED_WHEN_SARCASTIC)] = -1.0
    return adjusted.argmax(1)


def results(frames, data, cache, numbers):
    state = {}
    def model():
        state.setdefault("pair", load_trained(MODEL_DIR))
        return state["pair"][0]
    def tokenizer():
        state.setdefault("pair", load_trained(MODEL_DIR))
        return state["pair"][1]

    def run(name, frame):
        texts = frame["text"].astype(str).tolist()
        emotion, sarcasm = predictions(name, texts, model, tokenizer, cache)
        return decide(blended(texts, emotion), sarcasm, SARCASM_THRESHOLD), emotion, sarcasm, texts

    per_class, matrices = {}, {}
    for key, title in (("nat", "GoEmotions comments"), ("tweet", "Tweets (SemEval)")):
        frame = frames[key][frames[key]["split"] == "test"].reset_index(drop=True)
        gold = frame["emotion"].map(EMOTION_ID).values
        predicted = run(f"{key}_test", frame)[0]
        classes = f1_score(gold, predicted, average=None, labels=range(len(EMOTIONS)), zero_division=0)
        per_class[title] = classes
        counts = confusion_matrix(gold, predicted, labels=range(len(EMOTIONS)))
        matrices[title] = 100 * counts / np.maximum(counts.sum(1, keepdims=True), 1)
        numbers[f"{key}_test"] = {"rows": len(frame), "accuracy": float(accuracy_score(gold, predicted)), "macro_f1": float(classes.mean()),
                                  "per_class_f1": dict(zip(EMOTIONS, np.round(classes, 3).tolist())), "support": dict(zip(EMOTIONS, np.bincount(gold, minlength=len(EMOTIONS)).tolist()))}

    natural = frames["nat"][frames["nat"]["split"] == "test"].reset_index(drop=True)
    keep = natural["emotion"] != "neutral"
    predicted = run("nat_test", natural)[0]
    names = np.array(EMOTIONS)[predicted][keep.values]
    vectorizer, svm = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib"), joblib.load(ROOT / "models" / "emotion_linear_svm.joblib")
    baseline = svm.predict(vectorizer.transform(natural.loc[keep, "text"].map(clean_text)))
    gold = natural.loc[keep, "emotion"]
    same_rows = {"rows": int(keep.sum()),
                 "tfidf": {"accuracy": float(accuracy_score(gold, baseline)), "macro_f1": float(f1_score(gold, baseline, average="macro", labels=BASELINE_EMOTIONS, zero_division=0))},
                 "roberta": {"accuracy": float(accuracy_score(gold, names)), "macro_f1": float(f1_score(gold, names, average="macro", labels=BASELINE_EMOTIONS, zero_division=0))}}
    numbers["same_rows_comparison"] = same_rows

    positions, height = np.arange(len(EMOTIONS))[::-1], 0.36
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    for offset, color, (title, values) in zip((height / 2 + 0.02, -height / 2 - 0.02), (BLUE, ORANGE), per_class.items()):
        ax.barh(positions + offset, values, height=height, color=color, label=title)
        for y, value in zip(positions + offset, values):
            ax.text(value + 0.012, y, f"{value:.2f}", va="center", fontsize=8)
    ax.set_yticks(positions, EMOTIONS)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 score (1.0 is perfect)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
    save(fig, "results_per_emotion_f1")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), sharey=True)
    for ax, (title, matrix) in zip(axes, matrices.items()):
        heatmap(ax, matrix, EMOTIONS, EMOTIONS, vmax=100)
        ax.set_title(title)
        ax.set_xlabel("What the model said")
    axes[0].set_ylabel("True emotion")
    fig.tight_layout(w_pad=1.5)
    save(fig, "results_confusion_matrices")

    test = frames["sarcasm"][frames["sarcasm"]["split"] == "test"].reset_index(drop=True)
    _, _, sarcasm_probs, _ = run("sarcasm_test", test)
    for source, rows in test.groupby("source"):
        p, r, f, _ = precision_recall_fscore_support(rows["sarcastic"], sarcasm_probs[rows.index.values] >= SARCASM_THRESHOLD, average="binary", zero_division=0)
        numbers[f"sarcasm_{source}"] = {"rows": len(rows), "precision": float(p), "recall": float(r), "f1": float(f)}

    sentiment = frames["sentiment_test"]
    predicted = run("sentiment_test", sentiment)[0]
    words = [EMOTION_SENTIMENT[EMOTIONS[i]] for i in predicted]
    numbers["sentiment_test"] = {"rows": len(sentiment), "accuracy": float(accuracy_score(sentiment["sentiment"], words)), "macro_f1": float(f1_score(sentiment["sentiment"], words, average="macro"))}

    i, s = numbers["sarcasm_isarcasm"], numbers["sarcasm_semeval2018_irony"]
    overview = {"Emotion, GoEmotions (macro-F1)": numbers["nat_test"]["macro_f1"], "Emotion, tweets (macro-F1)": numbers["tweet_test"]["macro_f1"],
                "Sentiment, TweetEval (macro-F1)": numbers["sentiment_test"]["macro_f1"], "Irony, SemEval (F1)": s["f1"], "Sarcasm, iSarcasm (F1)": i["f1"]}
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    bars(ax, list(overview), list(overview.values()), fmt="{:.3f}", limit=1.0)
    save(fig, "results_overview")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.9), sharey=True)
    bars(axes[0], ["TF-IDF + SVM", "RoBERTa"], [same_rows["tfidf"]["macro_f1"], same_rows["roberta"]["macro_f1"]], fmt="{:.3f}", limit=1.0)
    axes[0].set_title("Macro-F1, same comment rows")
    bars(axes[1], ["TF-IDF + SVM", "RoBERTa"], [same_rows["tfidf"]["accuracy"], same_rows["roberta"]["accuracy"]], color=ORANGE, fmt="{:.3f}", limit=1.0)
    axes[1].set_title("Accuracy, same comment rows")
    fig.tight_layout(w_pad=2)
    save(fig, "results_baseline_vs_roberta")

    words_per_emotion = {}
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.9))
    vocabulary = vectorizer.get_feature_names_out()
    for ax, (row, name) in zip(axes.flat, enumerate(svm.classes_)):
        top = np.argsort(svm.coef_[row])[-6:][::-1]
        words_per_emotion[name] = [str(vocabulary[j]) for j in top]
        bars(ax, [vocabulary[j] for j in top], svm.coef_[row][top], fmt="{:.1f}", limit=svm.coef_.max() * 1.3)
        ax.set_title(name)
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, "results_baseline_keywords")
    numbers["baseline_top_words"] = words_per_emotion

    val_emotion = pd.concat([frames[k][frames[k]["split"] == "val"] for k in ("nat", "tweet")], ignore_index=True)
    texts = val_emotion["text"].astype(str).tolist()
    emotion, sarcasm = predictions("emotion_val", texts, model, tokenizer, cache)
    scores, gold = blended(texts, emotion), val_emotion["emotion"].map(EMOTION_ID).values
    macro = lambda pred: float(f1_score(gold, pred, average="macro", labels=range(len(EMOTIONS)), zero_division=0))
    no_rule = macro(scores.argmax(1))
    emotion_curve = [macro(decide(scores, sarcasm, t)) for t in THRESHOLDS]
    val_sarcasm = frames["sarcasm"][frames["sarcasm"]["split"] == "val"].reset_index(drop=True)
    _, probs = predictions("sarcasm_val", val_sarcasm["text"].astype(str).tolist(), model, tokenizer, cache)
    pr = [precision_recall_fscore_support(val_sarcasm["sarcastic"], probs >= t, average="binary", zero_division=0)[:2] for t in THRESHOLDS]
    numbers["threshold_curve"] = {"no_rule_macro_f1": no_rule, "thresholds": THRESHOLDS, "emotion_macro_f1": emotion_curve, "precision": [float(a) for a, _ in pr], "recall": [float(b) for _, b in pr]}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    x = np.arange(len(THRESHOLDS))
    axes[0].plot(x, [a for a, _ in pr], color=BLUE, linewidth=2, marker="o", markersize=4)
    axes[0].plot(x, [b for _, b in pr], color=ORANGE, linewidth=2, marker="o", markersize=4)
    axes[0].text(x[-1], pr[-1][0] + 0.03, "precision", ha="right")
    axes[0].text(x[-1], pr[-1][1] - 0.075, "recall", ha="right")
    axes[0].set_title("Sarcasm flags")
    axes[0].set_ylim(0, 1)
    axes[1].plot(x, emotion_curve, color=BLUE, linewidth=2, marker="o", markersize=4)
    axes[1].axhline(no_rule - 0.005, color=GREY, linestyle="--", linewidth=1)
    axes[1].text(0, no_rule - 0.0045, "largest loss we allowed", color=MUTED, va="bottom")
    axes[1].set_title("Emotion macro-F1")
    for ax in axes:
        ax.set_xticks(x, [str(t) for t in THRESHOLDS])
        ax.set_xlabel("Sarcasm threshold")
        ax.axvline(THRESHOLDS.index(SARCASM_THRESHOLD), color=GREY, linewidth=1)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    save(fig, "results_threshold_choice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, help="optional folder that stores model outputs between runs")
    args = parser.parse_args()
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    frames, numbers = read_all(args.data), {}
    exploration(frames, numbers)
    pipeline()
    results(frames, args.data, args.cache, numbers)
    path = ROOT / "results" / "report_numbers.json"
    path.write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
