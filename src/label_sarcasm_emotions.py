"""Give sarcastic iSarcasm training tweets an emotion taken from the author's plain rephrase.

Run from the project root after build_sarcasm_emoji_dataset.py and a --sincere-negatives training run:
    python src/label_sarcasm_emotions.py --raw path/to/isarcasm/train.En.csv --model models/sarcasm_emotion_v2

Sarcastic text is read literally by an emotion model, but the rephrase says what the author meant.
So the rephrase is scored and its emotion is attached to the sarcastic tweet.

Rules, fixed before any result was seen:
    * Training rows only. Validation and test sets are never labelled this way.
    * Neutral rephrases and rephrases below --min-confidence (0.6) are skipped.
    * These are machine-made labels, so they carry the labelling model's mistakes.

Output: data/sarcasm_emotion_pseudo.csv with columns text, emotion, rephrase, confidence
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_sarcasm_emoji_dataset import clean, overlap_key, strip_label_hashtags  # noqa: E402
from train_sarcasm_emotion import EMOTIONS, EmotionSarcasmModel, predict  # noqa: E402


def load_trained_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir), add_pooling_layer=False)
    model = EmotionSarcasmModel(encoder)
    load_model(model, str(model_dir / "model.safetensors"))
    return model.eval(), tokenizer


def sarcastic_training_pairs(raw_path, train_path):
    """Sarcastic tweets as saved in the training file, joined to their authors' rephrases."""
    raw = pd.read_csv(raw_path).dropna(subset=["tweet", "rephrase"])
    raw = raw[raw["sarcastic"] == 1]
    raw = raw.assign(key=raw["tweet"].map(lambda t: overlap_key(strip_label_hashtags(clean(t)))))
    train = pd.read_csv(train_path)
    train = train[(train["source"] == "isarcasm") & (train["sarcastic"] == 1)]
    train = train.assign(key=train["text"].map(overlap_key))
    return train[["text", "key"]].merge(raw[["key", "rephrase"]].drop_duplicates("key"), on="key")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="iSarcasm train.En.csv")
    parser.add_argument("--model", type=Path, required=True, help="trained model folder used for labelling")
    parser.add_argument("--train", type=Path, default=ROOT / "data/sarcasm_train.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/sarcasm_emotion_pseudo.csv")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args()

    model, tokenizer = load_trained_model(args.model)
    pairs = sarcastic_training_pairs(args.raw, args.train)
    settings = SimpleNamespace(batch_size=32, max_len=64)
    probs, _ = predict(model, tokenizer, pairs["rephrase"].astype(str).tolist(), torch.device("cpu"), settings)
    pairs["emotion"] = [EMOTIONS[i] for i in probs.argmax(1)]
    pairs["confidence"] = probs.max(1)

    kept = pairs[(pairs["emotion"] != "neutral") & (pairs["confidence"] >= args.min_confidence)]
    kept[["text", "emotion", "rephrase", "confidence"]].to_csv(args.out, index=False)
    print(f"pairs {len(pairs)}, kept {len(kept)}")
    print("all rephrases:", pairs["emotion"].value_counts().to_dict())
    print("kept:", kept["emotion"].value_counts().to_dict())


if __name__ == "__main__":
    main()
