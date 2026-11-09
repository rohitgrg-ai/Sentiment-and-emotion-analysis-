"""Build tweet datasets for sarcasm detection and emotion in emoji-rich text.

Run from the project root once the raw files are in one folder:
    python src/build_sarcasm_emoji_dataset.py --raw path/to/raw --ambiguous data/ambiguous_social_media_test.csv

Raw files expected in --raw (check each source's terms before redistributing):
    irony/{train,validation,test}.parquet   SemEval-2018 Task 3 through cardiffnlp/tweet_eval, config "irony"
                                            https://huggingface.co/datasets/cardiffnlp/tweet_eval
    isarcasm/train.En.csv                   iSarcasmEval (SemEval-2022 Task 6), labelled by the tweets' authors
    isarcasm/task_A_En_test.csv             https://github.com/iabufarha/iSarcasmEval
    ait2018/all-data.zip                    SemEval-2018 Task 1 (Affect in Tweets), crowd-annotated
                                            http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip

Rules enforced here
    * The 500-row ambiguous test set is never used for training: matching text is removed from every split.
    * Official test splits are kept. A text found in two splits is removed from the earlier one.
    * Texts that appear with conflicting labels in one split are removed entirely.
    * Hashtags that announce the label (#not, #irony, #sarcasm...) are removed from the sarcasm text.
    * Author rephrases of sarcastic tweets become non-sarcastic TRAINING rows only.
    * Emojis and slang are kept. Handles and links are normalised to @user and http.
    * Classes are not balanced here. Use class weights when training.

Outputs (--out):
    sarcasm_{train,val,test}.csv        columns: text, sarcastic, source, has_emoji
    tweet_emotion_{train,val,test}.csv  columns: text, emotion, source_labels, has_emoji
    sarcasm_emoji_dataset_report.md
"""
import argparse
import csv
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
VAL_SHARE = 0.10
SPLITS = ("train", "val", "test")

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
LABEL_HASHTAGS = re.compile(r"#(?:not|irony|ironic|ironically|sarcasm|sarcastic|sarcastically)\b", re.IGNORECASE)

# Fixed before any model was trained on this data, following build_natural_dataset.py.
AFFECT_MAP = {"anger": "anger", "disgust": "anger", "fear": "fear", "joy": "joy",
              "love": "love", "sadness": "sadness", "surprise": "surprise"}
AFFECT_NO_HOME = ["anticipation", "optimism", "pessimism", "trust"]
AFFECT_LABELS = list(AFFECT_MAP) + AFFECT_NO_HOME
AFFECT_FILES = {"train": "train", "val": "dev", "test": "test-gold"}


def clean(text):
    text = re.sub(r"https?://\S+|www\.\S+", "http", str(text))
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_label_hashtags(text):
    return re.sub(r"\s+", " ", LABEL_HASHTAGS.sub(" ", text)).strip()


def overlap_key(text):
    """Duplicate check only: letters, digits and emojis. Emojis can change the meaning."""
    return re.sub("[^a-z0-9\U0001F300-\U0001FAFF☀-➿]", "", re.sub(r"@user|http", "", str(text).lower()))


def remove_overlaps(frames, label, blocked):
    keys = {split: frame["text"].map(overlap_key) for split, frame in frames.items()}
    later = {"train": ["val", "test"], "val": ["test"], "test": []}
    cleaned, counts = {}, {}
    for split, frame in frames.items():
        key = keys[split]
        empty = key.eq("")
        ambiguous = key.isin(blocked)
        conflict = frame.groupby(key)[label].transform("nunique").gt(1)
        duplicate = key.duplicated(keep="first")
        in_later = key.isin(set().union(*[set(keys[other]) for other in later[split]]))
        drop = empty | ambiguous | conflict | duplicate | in_later
        cleaned[split] = frame[~drop].sample(frac=1, random_state=SEED).reset_index(drop=True)
        counts[split] = {"rows_before": len(frame), "empty": int(empty.sum()),
                         "ambiguous_overlap": int(ambiguous.sum()), "conflicting_labels": int(conflict.sum()),
                         "duplicates": int(duplicate.sum()), "in_later_split": int(in_later.sum()),
                         "rows_after": len(cleaned[split])}
    return cleaned, counts


def load_irony(raw):
    files = {"train": "train", "val": "validation", "test": "test"}
    return {split: pd.DataFrame({"text": (d := pd.read_parquet(raw / f"irony/{name}.parquet"))["text"],
                                 "sarcastic": d["label"].astype(int), "source": "semeval2018_irony"})
            for split, name in files.items()}


def load_isarcasm(raw):
    full = pd.read_csv(raw / "isarcasm/train.En.csv").dropna(subset=["tweet"])
    train, val = train_test_split(full, test_size=VAL_SHARE, random_state=SEED, stratify=full["sarcastic"])
    rephrases = train[(train["sarcastic"] == 1) & train["rephrase"].notna()]
    rephrases = rephrases[rephrases["rephrase"].map(lambda t: overlap_key(clean(t))) !=
                          rephrases["tweet"].map(lambda t: overlap_key(clean(t)))]
    frame = lambda text, label, source: pd.DataFrame(
        {"text": text.values, "sarcastic": label.astype(int).values, "source": source})
    test = pd.read_csv(raw / "isarcasm/task_A_En_test.csv").dropna(subset=["text"])
    return {
        "train": pd.concat([frame(train["tweet"], train["sarcastic"], "isarcasm"),
                            frame(rephrases["rephrase"], rephrases["sarcastic"] * 0, "isarcasm_rephrase")]),
        "val": frame(val["tweet"], val["sarcastic"], "isarcasm"),
        "test": frame(test["text"], test["sarcastic"], "isarcasm"),
    }


def build_sarcasm(raw, blocked):
    irony, isarcasm = load_irony(raw), load_isarcasm(raw)
    frames, marker_rows = {}, {}
    for split in SPLITS:
        frame = pd.concat([isarcasm[split], irony[split]], ignore_index=True)
        frame["text"] = frame["text"].map(clean)
        marker_rows[split] = int(frame["text"].str.contains(LABEL_HASHTAGS).sum())
        frame["text"] = frame["text"].map(strip_label_hashtags)
        frames[split] = frame
    cleaned, counts = remove_overlaps(frames, "sarcastic", blocked)
    return cleaned, counts, marker_rows


def affect_row_emotion(row):
    """One project emotion, 'neutral' for a tweet with no labels, or None when the row is dropped."""
    mapped = {AFFECT_MAP[name] for name in AFFECT_MAP if row[name] == 1}
    if len(mapped) == 1:
        return mapped.pop()
    if not mapped and row[AFFECT_LABELS].sum() == 0:
        return "neutral"
    return None


def build_tweet_emotion(raw, blocked):
    frames, stats = {}, {}
    with zipfile.ZipFile(raw / "ait2018/all-data.zip") as archive:
        for split, name in AFFECT_FILES.items():
            path = f"SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-{name}.txt"
            with archive.open(path) as handle:
                data = pd.read_csv(handle, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8")
            data["emotion"] = data.apply(affect_row_emotion, axis=1)
            data["source_labels"] = data.apply(lambda r: "+".join(n for n in AFFECT_LABELS if r[n] == 1), axis=1)
            stats[split] = {"raw_rows": len(data), "kept": int(data["emotion"].notna().sum()),
                            "dropped_mixed_or_no_home": int(data["emotion"].isna().sum())}
            kept = data[data["emotion"].notna()]
            frames[split] = pd.DataFrame({"text": kept["Tweet"].map(clean), "emotion": kept["emotion"],
                                          "source_labels": kept["source_labels"]})
    cleaned, counts = remove_overlaps(frames, "emotion", blocked)
    return cleaned, counts, stats


def add_emoji_flag(frames):
    for frame in frames.values():
        frame["has_emoji"] = frame["text"].map(lambda text: bool(EMOJI.search(text)))


def counts_table(frames, column):
    table = pd.DataFrame({split: frame[column].value_counts() for split, frame in frames.items()}).fillna(0).astype(int)
    return table.to_string()


def write_report(out, sarcasm, tweet_emotion, sarcasm_counts, emotion_counts, marker_rows, emotion_stats):
    source_rates = "\n".join(
        f"{split}: " + frame.groupby("source")["sarcastic"].agg(["size", "mean"]).round(3).to_dict("index").__str__()
        for split, frame in sarcasm.items())
    emoji_share = lambda frames: {split: round(100 * float(frame["has_emoji"].mean()), 1) for split, frame in frames.items()}
    sections = [
        "# Sarcasm and tweet-emotion dataset report", "",
        "Labels: iSarcasm by the tweets' authors; SemEval-2018 irony and Affect in Tweets by annotators.",
        "`sarcastic` uses irony as a stand-in for SemEval rows. Use `source` to train or score them separately.", "",
        "## Sarcasm rows by label", "```", counts_table(sarcasm, "sarcastic"), "```",
        "## Sarcasm rows and sarcastic rate by source", "```", source_rates, "```",
        f"Rows containing a label hashtag before removal: {marker_rows}", "",
        f"Rows with an emoji (%): {emoji_share(sarcasm)}", "",
        "## Sarcasm overlap removal", "```", pd.DataFrame(sarcasm_counts).T.to_string(), "```",
        "## Tweet emotion rows", "```", counts_table(tweet_emotion, "emotion"), "```",
        f"Rows with an emoji (%): {emoji_share(tweet_emotion)}", "",
        "## Tweet emotion filtering", "```", pd.DataFrame(emotion_stats).T.to_string(), "```",
        "## Tweet emotion overlap removal", "```", pd.DataFrame(emotion_counts).T.to_string(), "```",
        "Mapping: anger and disgust become anger; fear, joy, love, sadness and surprise keep their names.",
        f"{', '.join(AFFECT_NO_HOME)} are ignored beside one mapped emotion. Rows with only those labels are dropped.",
        "Rows with two or more project emotions are dropped. Rows with no labels become neutral.",
    ]
    (out / "sarcasm_emoji_dataset_report.md").write_text("\n".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--ambiguous", type=Path, default=ROOT / "data/ambiguous_social_media_test.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    blocked = set(pd.read_csv(args.ambiguous)["text"].map(lambda text: overlap_key(clean(text))))
    sarcasm, sarcasm_counts, marker_rows = build_sarcasm(args.raw, blocked)
    tweet_emotion, emotion_counts, emotion_stats = build_tweet_emotion(args.raw, blocked)
    add_emoji_flag(sarcasm)
    add_emoji_flag(tweet_emotion)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        sarcasm[split].to_csv(args.out / f"sarcasm_{split}.csv", index=False)
        tweet_emotion[split].to_csv(args.out / f"tweet_emotion_{split}.csv", index=False)
    write_report(args.out, sarcasm, tweet_emotion, sarcasm_counts, emotion_counts, marker_rows, emotion_stats)
    print((args.out / "sarcasm_emoji_dataset_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
