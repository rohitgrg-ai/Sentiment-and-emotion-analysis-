"""Train the emotion and sarcasm model, then score it on the test sets.

    python src/train.py --base cardiffnlp/twitter-roberta-base-2022-154m
    python src/train.py --base cardiffnlp/twitter-roberta-base-2022-154m --extra-emotion data/sarcasm_emotion_pseudo.csv

The best epoch is chosen on validation data. Test sets are only used for the final scores.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from model import (EMOTION_ID, EMOTIONS, MAX_TOKENS, MODEL_DIR, ROOT, SARCASM_THRESHOLD, EmotionSarcasmModel,
                   apply_sarcasm_rule, blend_emojis, load_trained, normalise, probabilities)

IGNORE = -100


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def emotion_rows(args, split):
    rows = [pd.read_csv(args.data / f"natural_emotion_{split}.csv")[["text", "emotion"]].assign(origin="natural"),
            pd.read_csv(args.data / f"tweet_emotion_{split}.csv")[["text", "emotion"]].assign(origin="tweet")]
    if split == "train" and args.extra_emotion:
        rows += [pd.read_csv(args.extra_emotion)[["text", "emotion"]].assign(origin="pseudo")] * args.extra_repeat
    rows = pd.concat(rows, ignore_index=True)
    return rows.sample(min(args.limit, len(rows)), random_state=args.seed) if args.limit else rows


def sarcasm_rows(args, split):
    rows = pd.read_csv(args.data / f"sarcasm_{split}.csv")
    return rows.sample(min(args.limit, len(rows)), random_state=args.seed) if args.limit else rows


def class_weights(labels, classes, target):
    counts = np.bincount(labels, minlength=classes).astype(float)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float, device=target)


def task_loss(loss_fn, logits, labels):
    return loss_fn(logits, labels) if (labels != IGNORE).any() else logits.sum() * 0.0


def validate(model, tokenizer, args, target):
    emotions, sarcasm = emotion_rows(args, "val"), sarcasm_rows(args, "val")
    emotion_probs, _ = probabilities(model, tokenizer, emotions["text"].tolist(), target)
    _, sarcasm_probs = probabilities(model, tokenizer, sarcasm["text"].tolist(), target)
    gold = emotions["emotion"].map(EMOTION_ID).values
    emotion_f1 = f1_score(gold, emotion_probs.argmax(1), average="macro", labels=range(len(EMOTIONS)), zero_division=0)
    return (emotion_f1 + f1_score(sarcasm["sarcastic"], sarcasm_probs >= 0.5, zero_division=0)) / 2


def train(args, target):
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    emotions, sarcasm = emotion_rows(args, "train"), sarcasm_rows(args, "train")
    texts = [normalise(text) for text in pd.concat([emotions["text"], sarcasm["text"]])]
    y_emotion = np.concatenate([emotions["emotion"].map(EMOTION_ID).values, np.full(len(sarcasm), IGNORE)])
    # GoEmotions comments also teach the sarcasm output what ordinary text looks like.
    sincere = np.where(emotions["origin"].eq("natural"), 0, IGNORE)
    y_sarcasm = np.concatenate([sincere, sarcasm["sarcastic"].astype(int).values])

    model = EmotionSarcasmModel(AutoModel.from_pretrained(args.base, add_pooling_layer=False)).to(target)
    emotion_loss = nn.CrossEntropyLoss(class_weights(y_emotion[y_emotion != IGNORE], len(EMOTIONS), target), ignore_index=IGNORE)
    sarcasm_loss = nn.CrossEntropyLoss(class_weights(y_sarcasm[y_sarcasm != IGNORE], 2, target), ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(texts) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)

    args.out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.RandomState(args.seed + epoch).permutation(len(texts))
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            encoded = tokenizer([texts[i] for i in batch], truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt").to(target)
            emotion_logits, sarcasm_logits = model(**encoded)
            loss = (task_loss(emotion_loss, emotion_logits, torch.tensor(y_emotion[batch], device=target)) +
                    task_loss(sarcasm_loss, sarcasm_logits, torch.tensor(y_sarcasm[batch], device=target)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        score = validate(model, tokenizer, args, target)
        print(f"epoch {epoch}: validation score {score:.4f}", flush=True)
        if score > best:
            best = score
            save_model(model, str(args.out / "model.safetensors"))
    model.encoder.config.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)


def evaluate(args, target):
    model, tokenizer = load_trained(args.out)
    model.to(target)
    metrics = {}
    for name, frame in [("natural_emotion", pd.read_csv(args.data / "natural_emotion_test.csv")),
                        ("tweet_emotion", pd.read_csv(args.data / "tweet_emotion_test.csv"))]:
        frame = frame.head(args.limit) if args.limit else frame
        texts = frame["text"].astype(str).tolist()
        emotion_probs, sarcasm_probs = probabilities(model, tokenizer, texts, target)
        scores = np.array([apply_sarcasm_rule(blend_emojis(t, e), float(s)) for t, e, s in zip(texts, emotion_probs, sarcasm_probs)])
        gold = frame["emotion"].map(EMOTION_ID).values
        metrics[name] = {"rows": len(frame), "accuracy": float(accuracy_score(gold, scores.argmax(1))),
                         "macro_f1": float(f1_score(gold, scores.argmax(1), average="macro", labels=range(len(EMOTIONS)), zero_division=0))}
    sarcasm = pd.read_csv(args.data / "sarcasm_test.csv")
    sarcasm = sarcasm.head(args.limit) if args.limit else sarcasm
    _, sarcasm_probs = probabilities(model, tokenizer, sarcasm["text"].astype(str).tolist(), target)
    for source, rows in sarcasm.groupby("source"):
        p, r, f, _ = precision_recall_fscore_support(rows["sarcastic"], sarcasm_probs[rows.index.values] >= SARCASM_THRESHOLD, average="binary", zero_division=0)
        metrics[f"sarcasm_{source}"] = {"rows": len(rows), "precision": float(p), "recall": float(r), "f1": float(f)}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="pretrained model name or folder")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "metrics.json")
    parser.add_argument("--extra-emotion", type=Path, help="CSV from label_sarcasm_emotions.py")
    parser.add_argument("--extra-repeat", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="use only this many rows, for a quick check")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    target = device()
    if not args.eval_only:
        train(args, target)
    evaluate(args, target)


if __name__ == "__main__":
    main()
