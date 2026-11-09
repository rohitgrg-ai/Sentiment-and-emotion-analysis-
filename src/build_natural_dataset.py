"""Build NEW human-labelled, natural-text datasets for the transformer phase.

Run once from the project root on a machine with internet access:
    pip install datasets            # only needed for this step
    python src/build_natural_dataset.py

Sources (human-labelled; check each dataset's licence/terms before redistributing):
    emotion   : GoEmotions "simplified" (Reddit comments, 27 emotions + neutral)  google-research-datasets/go_emotions
    sentiment : TweetEval "sentiment" (tweets, negative / neutral / positive)       cardiffnlp/tweet_eval

Rules enforced here
    * data/final.csv and every existing file are NEVER touched; only new data/natural_* files are written.
    * The 500-row ambiguous test set is never used for training: any text that matches it after
      normalisation is removed from every split (see the report for the count).
    * Official train/validation/test splits are kept; a text that appears in more than one split is
      removed from the earlier (train, then validation) split, so test stays clean.
    * Emotion: only comments with exactly ONE original label are kept (multi-label rows are noisy for a
      single-label classifier). Labels with no clean home in the six emotions are dropped, not forced.
    * Emojis, slang and hashtags are left in the text on purpose.
    * Nothing is learned from labels here; the mapping below is fixed and written to data/natural_label_mapping.md.

Outputs (data/):
    natural_emotion_{train,val,test}.csv    columns: text, emotion, source_label
    natural_sentiment_{train,val,test}.csv  columns: text, sentiment
    natural_label_mapping.md, natural_dataset_report.md
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AMBIGUOUS = DATA / "ambiguous_social_media_test.csv"
SEED = 42

# GoEmotions label -> project emotion. Fixed BEFORE looking at any result. 'neutral' is kept as its own class.
EMOTION_MAP = {
    "joy": "joy", "amusement": "joy", "excitement": "joy", "relief": "joy", "pride": "joy",
    "love": "love", "caring": "love", "gratitude": "love", "admiration": "love",
    "anger": "anger", "annoyance": "anger", "disgust": "anger", "disapproval": "anger",
    "fear": "fear", "nervousness": "fear",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness", "remorse": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}
DROPPED = ["approval", "confusion", "curiosity", "desire", "embarrassment", "optimism", "realization"]
SENTIMENT_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def norm(text):
    """Key used only for duplicate / leakage checks; the saved text keeps its original casing and emojis."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tidy(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def remove_overlaps(frames, blocked):
    """frames: {'train','val','test'} -> DataFrame with 'text'. Returns cleaned frames and a count table."""
    counts = {}
    keys = {k: f["text"].map(norm) for k, f in frames.items()}
    out = {}
    later = {"train": ["val", "test"], "val": ["test"], "test": []}
    for k, f in frames.items():
        key = keys[k]
        bad_block = key.isin(blocked)
        bad_dup = key.duplicated(keep="first")
        bad_cross = pd.Series(False, index=f.index)
        for other in later[k]:
            bad_cross |= key.isin(set(keys[other]))
        counts[k] = {"rows_before": len(f), "removed_ambiguous_overlap": int(bad_block.sum()),
                     "removed_duplicates": int((bad_dup & ~bad_block).sum()),
                     "removed_in_later_split": int((bad_cross & ~bad_block & ~bad_dup).sum())}
        out[k] = f[~(bad_block | bad_dup | bad_cross)].reset_index(drop=True)
        counts[k]["rows_after"] = len(out[k])
    return out, counts


def build_emotion(raw, blocked):
    """raw: {'train','val','test'} -> DataFrame(text, label_names=list of str)."""
    frames, stats = {}, {}
    for k, df in raw.items():
        single = df[df["label_names"].map(len) == 1].copy()
        single["source_label"] = single["label_names"].map(lambda x: x[0])
        single = single[single["source_label"].isin(EMOTION_MAP)]
        single["emotion"] = single["source_label"].map(EMOTION_MAP)
        single["text"] = single["text"].map(tidy)
        single = single[single["text"] != ""]
        stats[k] = {"raw_rows": len(df), "single_label_mapped": len(single),
                    "multi_label_dropped": int((df["label_names"].map(len) != 1).sum()),
                    "unmapped_label_dropped": int(df["label_names"].map(len).eq(1).sum()) - len(single)}
        frames[k] = single[["text", "emotion", "source_label"]].sample(frac=1, random_state=SEED).reset_index(drop=True)
    cleaned, counts = remove_overlaps(frames, blocked)
    return cleaned, stats, counts


def build_sentiment(raw, blocked):
    frames = {}
    for k, df in raw.items():
        d = pd.DataFrame({"text": df["text"].map(tidy), "sentiment": df["label"].map(SENTIMENT_NAMES)})
        d = d[(d["text"] != "") & d["sentiment"].notna()]
        frames[k] = d.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return remove_overlaps(frames, blocked)


def load_raw():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Please run: pip install datasets")
    go = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = go["train"].features["labels"].feature.names
    emo_raw = {}
    for hf, k in SPLITS.items():
        d = go[hf].to_pandas()
        d["label_names"] = d["labels"].map(lambda ids: [names[i] for i in ids])
        emo_raw[k] = d[["text", "label_names"]]
    te = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    sent_raw = {k: te[hf].to_pandas()[["text", "label"]] for hf, k in SPLITS.items()}
    return emo_raw, sent_raw


def counts_md(frames, col):
    lines = ["| split | rows | " + " | ".join(sorted(frames["train"][col].unique())) + " |",
             "|---|---|" + "---|" * frames["train"][col].nunique()]
    labels = sorted(frames["train"][col].unique())
    for k, f in frames.items():
        vc = f[col].value_counts()
        lines.append(f"| {k} | {len(f)} | " + " | ".join(str(int(vc.get(l, 0))) for l in labels) + " |")
    return "\n".join(lines)


def write_outputs(emo, sent, emo_stats, emo_counts, sent_counts):
    for k in ("train", "val", "test"):
        emo[k].to_csv(DATA / f"natural_emotion_{k}.csv", index=False)
        sent[k].to_csv(DATA / f"natural_sentiment_{k}.csv", index=False)
    mapping = ["# Label mapping (fixed before any model was trained on this data)", "",
               "GoEmotions label -> project emotion. Only comments with exactly one original label are kept.", "",
               "| project emotion | GoEmotions labels |", "|---|---|"]
    for e in ["joy", "love", "anger", "fear", "sadness", "surprise", "neutral"]:
        mapping.append(f"| {e} | {', '.join(k for k, v in EMOTION_MAP.items() if v == e)} |")
    mapping += ["", f"Dropped (no clean match, never forced into an emotion): {', '.join(DROPPED)}.", "",
                "Sentiment: TweetEval labels used as given (0 negative, 1 neutral, 2 positive).", "",
                "'neutral' is an extra emotion class here; the project's earlier data had only six emotions. "
                "Decide and document how it is evaluated against the ambiguous set (whose emotion labels have no neutral)."]
    (DATA / "natural_label_mapping.md").write_text("\n".join(mapping))
    rep = ["# Natural dataset build report", "", "Sources: GoEmotions (emotion), TweetEval sentiment. Human-labelled; "
           "check licences before redistributing. final.csv untouched.", "",
           "## Emotion rows kept per split", "", counts_md(emo, "emotion"), "",
           "## Emotion filtering", "", pd.DataFrame(emo_stats).T.to_markdown() if _has_tabulate() else str(emo_stats), "",
           "## Emotion overlap / duplicate removal", "", pd.DataFrame(emo_counts).T.to_markdown() if _has_tabulate() else str(emo_counts), "",
           "## Sentiment rows kept per split", "", counts_md(sent, "sentiment"), "",
           "## Sentiment overlap / duplicate removal", "", pd.DataFrame(sent_counts).T.to_markdown() if _has_tabulate() else str(sent_counts), "",
           "Notes: emojis and slang are kept in the text. 'removed_ambiguous_overlap' counts rows matching the 500-row "
           "ambiguous test set; it must never be trained on."]
    (DATA / "natural_dataset_report.md").write_text("\n".join(rep))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    amb = pd.read_csv(AMBIGUOUS)
    blocked = set(amb["text"].map(norm))
    emo_raw, sent_raw = load_raw()
    emo, emo_stats, emo_counts = build_emotion(emo_raw, blocked)
    sent, sent_counts = build_sentiment(sent_raw, blocked)
    write_outputs(emo, sent, emo_stats, emo_counts, sent_counts)
    print((DATA / "natural_dataset_report.md").read_text())


if __name__ == "__main__":
    main()
