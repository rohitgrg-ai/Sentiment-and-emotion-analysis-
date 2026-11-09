"""Label sarcastic training tweets with the emotion of their authors' plain rephrases.

    python src/label_sarcasm_emotions.py --raw path/to/isarcasm/train.En.csv

Run after a first training pass. Neutral rephrases and rephrases below 60% confidence are skipped.
"""
import argparse
from pathlib import Path

import pandas as pd

from build_data import clean, overlap_key, strip_label_hashtags
from model import EMOTIONS, MODEL_DIR, ROOT, load_trained, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "data/sarcasm_emotion_pseudo.csv")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw).dropna(subset=["tweet", "rephrase"])
    raw = raw[raw["sarcastic"] == 1]
    raw = raw.assign(key=raw["tweet"].map(lambda t: overlap_key(strip_label_hashtags(clean(t)))))
    train = pd.read_csv(ROOT / "data/sarcasm_train.csv")
    train = train[(train["source"] == "isarcasm") & (train["sarcastic"] == 1)]
    pairs = train.assign(key=train["text"].map(overlap_key))[["text", "key"]].merge(raw[["key", "rephrase"]].drop_duplicates("key"), on="key")

    model, tokenizer = load_trained(args.model)
    probs, _ = probabilities(model, tokenizer, pairs["rephrase"].astype(str).tolist())
    pairs["emotion"] = [EMOTIONS[i] for i in probs.argmax(1)]
    pairs["confidence"] = probs.max(1)
    kept = pairs[(pairs["emotion"] != "neutral") & (pairs["confidence"] >= args.min_confidence)]
    kept[["text", "emotion", "confidence"]].to_csv(args.out, index=False)
    print(f"kept {len(kept)} of {len(pairs)}")


if __name__ == "__main__":
    main()
