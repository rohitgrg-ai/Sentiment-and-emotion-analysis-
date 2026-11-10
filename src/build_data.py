"""Build the emotion and sarcasm datasets.

    python src/build_data.py --raw path/to/raw

--raw must hold (check each source's terms before sharing the data):
    irony/{train,validation,test}.parquet   SemEval-2018 Task 3, from cardiffnlp/tweet_eval (config "irony")
    isarcasm/train.En.csv, task_A_En_test.csv   https://github.com/iabufarha/iSarcasmEval
    ait2018/all-data.zip                    SemEval-2018 Task 1 (Affect in Tweets)
GoEmotions is downloaded automatically and needs the `datasets` package.

Writes data/{natural_emotion,tweet_emotion,sarcasm}_{train,val,test}.csv
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
SPLITS = ("train", "val", "test")
EMOJI = "\U0001F300-\U0001FAFF☀-➿"
LABEL_HASHTAGS = re.compile(r"#(?:not|irony|ironic|ironically|sarcasm|sarcastic|sarcastically)\b", re.IGNORECASE)
GO_EMOTIONS = {
    "joy": "joy", "amusement": "joy", "excitement": "joy", "relief": "joy", "pride": "joy",
    "love": "love", "caring": "love", "gratitude": "love", "admiration": "love",
    "anger": "anger", "annoyance": "anger", "disgust": "anger", "disapproval": "anger",
    "fear": "fear", "nervousness": "fear", "surprise": "surprise", "neutral": "neutral",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness", "remorse": "sadness",
}
AFFECT = {"anger": "anger", "disgust": "anger", "fear": "fear", "joy": "joy", "love": "love", "sadness": "sadness", "surprise": "surprise"}
AFFECT_OTHER = ["anticipation", "optimism", "pessimism", "trust"]


def clean(text):
    text = re.sub(r"https?://\S+|www\.\S+", "http", str(text))
    return re.sub(r"\s+", " ", re.sub(r"@\w+", "@user", text)).strip()


def strip_label_hashtags(text):
    """These hashtags announce the label, so a model could score well just by spotting them."""
    return re.sub(r"\s+", " ", LABEL_HASHTAGS.sub(" ", text)).strip()


def overlap_key(text):
    return re.sub(f"[^a-z0-9{EMOJI}]", "", re.sub(r"@user|http", "", str(text).lower()))


def remove_overlaps(frames, label):
    """Drop empty, duplicated and conflicting rows. A text in two splits stays in the later one."""
    keys = {split: frame["text"].map(overlap_key) for split, frame in frames.items()}
    later = {"train": ["val", "test"], "val": ["test"], "test": []}
    cleaned = {}
    for split, frame in frames.items():
        key = keys[split]
        conflict = frame.groupby(key)[label].transform("nunique").gt(1)
        in_later = key.isin(set().union(*[set(keys[other]) for other in later[split]]))
        cleaned[split] = frame[~(key.eq("") | conflict | key.duplicated() | in_later)].sample(frac=1, random_state=SEED).reset_index(drop=True)
    return cleaned


def go_emotions():
    from datasets import load_dataset
    data = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = data["train"].features["labels"].feature.names
    frames = {}
    for source, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        rows = data[source].to_pandas()
        rows = rows[rows["labels"].map(len) == 1]
        labels = rows["labels"].map(lambda ids: names[ids[0]])
        kept = labels.isin(GO_EMOTIONS)
        frames[split] = pd.DataFrame({"text": rows.loc[kept, "text"].map(clean), "emotion": labels[kept].map(GO_EMOTIONS)})
    return remove_overlaps(frames, "emotion")


def tweet_emotions(raw):
    def emotion(row):
        found = {AFFECT[name] for name in AFFECT if row[name] == 1}
        if len(found) == 1:
            return found.pop()
        return "neutral" if not found and row[list(AFFECT) + AFFECT_OTHER].sum() == 0 else None

    frames = {}
    with zipfile.ZipFile(raw / "ait2018/all-data.zip") as archive:
        for split, name in (("train", "train"), ("val", "dev"), ("test", "test-gold")):
            with archive.open(f"SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-{name}.txt") as handle:
                rows = pd.read_csv(handle, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8")
            rows["emotion"] = rows.apply(emotion, axis=1)
            rows = rows[rows["emotion"].notna()]
            frames[split] = pd.DataFrame({"text": rows["Tweet"].map(clean), "emotion": rows["emotion"]})
    return remove_overlaps(frames, "emotion")


def sarcasm(raw):
    def frame(text, label, source):
        return pd.DataFrame({"text": text.values, "sarcastic": label.astype(int).values, "source": source})

    full = pd.read_csv(raw / "isarcasm/train.En.csv").dropna(subset=["tweet"])
    train, val = train_test_split(full, test_size=0.10, random_state=SEED, stratify=full["sarcastic"])
    rephrases = train[(train["sarcastic"] == 1) & train["rephrase"].notna()]
    rephrases = rephrases[rephrases["rephrase"].map(lambda t: overlap_key(clean(t))) != rephrases["tweet"].map(lambda t: overlap_key(clean(t)))]
    test = pd.read_csv(raw / "isarcasm/task_A_En_test.csv").dropna(subset=["text"])
    parts = {"train": [frame(train["tweet"], train["sarcastic"], "isarcasm"), frame(rephrases["rephrase"], rephrases["sarcastic"] * 0, "isarcasm_rephrase")],
             "val": [frame(val["tweet"], val["sarcastic"], "isarcasm")], "test": [frame(test["text"], test["sarcastic"], "isarcasm")]}
    for split, name in (("train", "train"), ("val", "validation"), ("test", "test")):
        irony = pd.read_parquet(raw / f"irony/{name}.parquet")
        parts[split].append(frame(irony["text"], irony["label"], "semeval2018_irony"))
    frames = {}
    for split in SPLITS:
        joined = pd.concat(parts[split], ignore_index=True)
        joined["text"] = joined["text"].map(lambda t: strip_label_hashtags(clean(t)))
        frames[split] = joined
    return remove_overlaps(frames, "sarcastic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frames in (("natural_emotion", go_emotions()), ("tweet_emotion", tweet_emotions(args.raw)), ("sarcasm", sarcasm(args.raw))):
        for split in SPLITS:
            frames[split].to_csv(args.out / f"{name}_{split}.csv", index=False)
            print(name, split, len(frames[split]))


if __name__ == "__main__":
    main()
