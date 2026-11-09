"""Train one Twitter-based transformer to predict emotion and sarcasm together.

Run from the project root (Apple GPU or CUDA strongly recommended):
    python src/train_sarcasm_emotion.py --base cardiffnlp/twitter-roberta-base-2022-154m --limit 400 --epochs 1
    python src/train_sarcasm_emotion.py --base cardiffnlp/twitter-roberta-base-2022-154m

Training data: natural_emotion_train + tweet_emotion_train (emotion head), sarcasm_train (sarcasm head).
Each row trains only the head it has a label for. With --sincere-negatives, GoEmotions rows
also train the sarcasm head as not sarcastic, so it learns what ordinary comments look like.
A few of those comments are really sarcastic, so this adds a small amount of label noise.
With --extra-emotion, sarcastic tweets labelled by label_sarcasm_emotions.py also train the
emotion head, each counted --extra-repeat times.

Fair-test rules
    * Settings match finetune_emotion.py: lr 2e-5, 3 epochs, batch 32, 64 tokens, seed 42, sqrt class weights.
    * The best epoch is chosen on validation only: mean of emotion macro-F1 and sarcastic-class F1.
    * Test sets and the ambiguous set are scored after training. Nothing is tuned on them.
    * Fixed before training:
        emoji fusion w = EMOJI_WEIGHT["emotion"], the value predictor.py already uses
        sarcasm rule: if P(sarcastic) >= 0.5 and the top emotion is joy or love, take the best
        emotion other than joy or love

Outputs: models/sarcasm_emotion/ and results/sarcasm_emotion/{history.csv, metrics.json, report.txt,
predictions_ambiguous.csv}
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_model, save_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictor import EMOJI_EMOTION, EMOJI_WEIGHT  # noqa: E402

EMOTIONS = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
EMOTION_ID = {name: i for i, name in enumerate(EMOTIONS)}
NEUTRAL = EMOTION_ID["neutral"]
POSITIVE = [EMOTION_ID["joy"], EMOTION_ID["love"]]
IGNORE = -100
SARCASM_THRESHOLD = 0.5


class EmotionSarcasmModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.emotion_head = nn.Linear(hidden, len(EMOTIONS))
        self.sarcasm_head = nn.Linear(hidden, 2)

    def forward(self, **encoded):
        cls = self.dropout(self.encoder(**encoded).last_hidden_state[:, 0])
        return self.emotion_head(cls), self.sarcasm_head(cls)


def normalise(text):
    text = re.sub(r"https?://\S+|www\.\S+", "http", str(text))
    return re.sub(r"\s+", " ", re.sub(r"@\w+", "@user", text)).strip()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def class_weights(labels, classes, power=0.5):
    counts = np.bincount(labels, minlength=classes).astype(float)
    weights = (counts.sum() / np.maximum(counts, 1)) ** power
    return weights / weights.mean()


def task_loss(loss_fn, logits, labels):
    if (labels != IGNORE).any():
        return loss_fn(logits, labels)
    return logits.sum() * 0.0


def load_splits(args):
    def emotion(split):
        frames = [pd.read_csv(args.data / f"natural_emotion_{split}.csv")[["text", "emotion"]].assign(origin="natural"),
                  pd.read_csv(args.new_data / f"tweet_emotion_{split}.csv")[["text", "emotion"]].assign(origin="tweet")]
        if split == "train" and args.extra_emotion:
            extra = pd.read_csv(args.extra_emotion)[["text", "emotion"]].assign(origin="sarcasm_pseudo")
            frames += [extra] * args.extra_repeat
        return pd.concat(frames, ignore_index=True)

    sarcasm = {s: pd.read_csv(args.new_data / f"sarcasm_{s}.csv") for s in ("train", "val")}
    data = {"emotion_train": emotion("train"), "emotion_val": emotion("val"),
            "sarcasm_train": sarcasm["train"], "sarcasm_val": sarcasm["val"]}
    if args.limit:
        data = {k: v.sample(min(args.limit, len(v)), random_state=args.seed) for k, v in data.items()}
    return data


@torch.no_grad()
def predict(model, tokenizer, texts, device, args):
    model.eval()
    emotion, sarcasm = [], []
    batch = args.batch_size * 2
    for start in range(0, len(texts), batch):
        encoded = tokenizer([normalise(t) for t in texts[start:start + batch]], truncation=True,
                            max_length=args.max_len, padding=True, return_tensors="pt").to(device)
        e, s = model(**encoded)
        emotion.append(torch.softmax(e, -1).float().cpu().numpy())
        sarcasm.append(torch.softmax(s, -1)[:, 1].float().cpu().numpy())
    return np.concatenate(emotion), np.concatenate(sarcasm)


def validate(model, tokenizer, data, device, args):
    emotion_probs, _ = predict(model, tokenizer, data["emotion_val"]["text"].tolist(), device, args)
    _, sarcasm_probs = predict(model, tokenizer, data["sarcasm_val"]["text"].tolist(), device, args)
    gold = data["emotion_val"]["emotion"].map(EMOTION_ID).values
    emotion_f1 = f1_score(gold, emotion_probs.argmax(1), average="macro", labels=range(len(EMOTIONS)), zero_division=0)
    sarcasm_f1 = f1_score(data["sarcasm_val"]["sarcastic"], sarcasm_probs >= SARCASM_THRESHOLD, zero_division=0)
    return float(emotion_f1), float(sarcasm_f1)


def train(args, tokenizer, device):
    data = load_splits(args)
    emotion_y = data["emotion_train"]["emotion"].map(EMOTION_ID).values
    sarcasm_y = data["sarcasm_train"]["sarcastic"].astype(int).values
    texts = [normalise(t) for t in pd.concat([data["emotion_train"]["text"], data["sarcasm_train"]["text"]])]
    y_emotion = np.concatenate([emotion_y, np.full(len(sarcasm_y), IGNORE)])
    is_sincere = data["emotion_train"]["origin"].eq("natural").values & args.sincere_negatives
    y_sarcasm = np.concatenate([np.where(is_sincere, 0, IGNORE), sarcasm_y])

    encoder = AutoModel.from_pretrained(args.base, add_pooling_layer=False)
    model = EmotionSarcasmModel(encoder).to(device)
    weight = lambda labels, n: torch.tensor(class_weights(labels, n), dtype=torch.float, device=device)
    emotion_loss = nn.CrossEntropyLoss(weight=weight(emotion_y, len(EMOTIONS)), ignore_index=IGNORE)
    sarcasm_loss = nn.CrossEntropyLoss(weight=weight(y_sarcasm[y_sarcasm != IGNORE], 2), ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(texts) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    print(f"device={device} rows={len(texts)} steps={steps} sincere_negatives={int(is_sincere.sum())}", flush=True)

    history, best = [], -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.RandomState(args.seed + epoch).permutation(len(texts))
        total, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            encoded = tokenizer([texts[i] for i in batch], truncation=True, max_length=args.max_len,
                                padding=True, return_tensors="pt").to(device)
            emotion_logits, sarcasm_logits = model(**encoded)
            loss = (task_loss(emotion_loss, emotion_logits, torch.tensor(y_emotion[batch], device=device)) +
                    task_loss(sarcasm_loss, sarcasm_logits, torch.tensor(y_sarcasm[batch], device=device)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total += loss.item(); batches += 1
            if batches % 100 == 0:
                print(f"epoch {epoch} step {batches} loss {total / batches:.4f}", flush=True)
        emotion_f1, sarcasm_f1 = validate(model, tokenizer, data, device, args)
        score = (emotion_f1 + sarcasm_f1) / 2
        history.append({"epoch": epoch, "train_loss": total / batches, "val_emotion_macro_f1": emotion_f1,
                        "val_sarcasm_f1": sarcasm_f1, "val_score": score})
        print(history[-1], flush=True)
        if score > best:
            best = score
            save_model(model, str(args.out / "model.safetensors"))
    encoder.config.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    (args.out / "labels.json").write_text(json.dumps({"emotions": EMOTIONS, "sarcasm_threshold": SARCASM_THRESHOLD}))
    args.results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(args.results / "history.csv", index=False)
    return model


def with_emojis(texts, probs):
    weight = EMOJI_WEIGHT["emotion"]
    out = probs.copy()
    for i, text in enumerate(texts):
        found = [EMOJI_EMOTION[ch] for ch in str(text) if ch in EMOJI_EMOTION]
        if not found:
            continue
        vote = np.array([found.count(name) for name in EMOTIONS], dtype=float)
        vote /= vote.sum()
        words = "".join(" " if ch in EMOJI_EMOTION else ch for ch in str(text))
        share = weight if re.search(r"[A-Za-z0-9]", words) else 1.0
        out[i] = (1 - share) * probs[i] + share * vote
    return out


def six_class(probs):
    masked = probs.copy()
    masked[:, NEUTRAL] = -1.0
    return masked


def with_sarcasm_rule(probs, sarcasm_probs):
    masked = six_class(probs)
    flipped = (sarcasm_probs >= SARCASM_THRESHOLD) & np.isin(masked.argmax(1), POSITIVE)
    masked[np.ix_(flipped, POSITIVE)] = -1.0
    return masked


def evaluate(args, model, tokenizer, device):
    run = lambda frame: predict(model, tokenizer, frame["text"].astype(str).tolist(), device, args)
    trim = lambda frame: frame.head(args.limit) if args.limit else frame
    metrics, lines = {}, []

    natural = trim(pd.read_csv(args.data / "natural_emotion_test.csv"))
    probs, _ = run(natural)
    gold = natural["emotion"].map(EMOTION_ID).values
    non_neutral = gold != NEUTRAL
    metrics["natural_test"] = {
        "accuracy_7class": float(accuracy_score(gold, probs.argmax(1))),
        "macro_f1_7class": float(f1_score(gold, probs.argmax(1), average="macro", labels=range(7), zero_division=0)),
        "accuracy_six_emotions": float(accuracy_score(gold[non_neutral], six_class(probs[non_neutral]).argmax(1))),
    }

    tweets = trim(pd.read_csv(args.new_data / "tweet_emotion_test.csv"))
    probs, _ = run(tweets)
    gold = tweets["emotion"].map(EMOTION_ID).values
    metrics["tweet_emotion_test"] = {
        "accuracy": float(accuracy_score(gold, probs.argmax(1))),
        "macro_f1": float(f1_score(gold, probs.argmax(1), average="macro", labels=range(7), zero_division=0)),
    }

    sarcasm = trim(pd.read_csv(args.new_data / "sarcasm_test.csv"))
    _, sarcasm_probs = run(sarcasm)
    metrics["sarcasm_test"] = {}
    for source, rows in sarcasm.groupby("source"):
        predicted = sarcasm_probs[rows.index.values] >= SARCASM_THRESHOLD
        p, r, f, _ = precision_recall_fscore_support(rows["sarcastic"], predicted, average="binary", zero_division=0)
        metrics["sarcasm_test"][source] = {"n": len(rows), "precision": float(p), "recall": float(r), "f1": float(f)}

    ambiguous = pd.read_csv(args.data / "ambiguous_social_media_test.csv")
    ambiguous = trim(ambiguous[ambiguous["expected_emotion"].notna()].reset_index(drop=True))
    texts = ambiguous["text"].astype(str).tolist()
    probs, sarcasm_probs = run(ambiguous)
    fused = with_emojis(texts, probs)
    systems = {"text_only": six_class(probs), "with_emojis": six_class(fused),
               "with_emojis_and_sarcasm_rule": with_sarcasm_rule(fused, sarcasm_probs)}
    labelled = ambiguous[["id", "text", "category", "expected_emotion"]].copy()
    labelled["p_sarcastic"] = sarcasm_probs
    gold = ambiguous["expected_emotion"].values
    has_emoji = ambiguous["emoji_present"].astype(str).str.lower().isin(["true", "1", "yes"]).values
    metrics["ambiguous"] = {"n": len(ambiguous), "labels": "AI-suggested, not human-verified"}
    for name, scores in systems.items():
        predicted = np.array(EMOTIONS)[scores.argmax(1)]
        labelled[name] = predicted
        correct = pd.Series(predicted == gold)
        metrics["ambiguous"][name] = {
            "strict": float(correct.mean()), "emoji_rows": float(correct[has_emoji].mean()),
            "no_emoji_rows": float(correct[~has_emoji].mean()),
            "by_category": {k: float(v) for k, v in correct.groupby(ambiguous["category"]).mean().round(3).items()},
        }
    flagged = pd.Series(sarcasm_probs >= SARCASM_THRESHOLD)
    metrics["ambiguous"]["sarcasm_flag_rate_by_category"] = {
        k: float(v) for k, v in flagged.groupby(ambiguous["category"]).mean().round(3).items()}
    labelled.to_csv(args.results / "predictions_ambiguous.csv", index=False)

    lines.append(json.dumps(metrics, indent=1))
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (args.results / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base model name or local folder")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="folder with natural_* and ambiguous files")
    parser.add_argument("--new-data", type=Path, default=ROOT / "data", help="folder with sarcasm_* and tweet_emotion_*")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "sarcasm_emotion")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "sarcasm_emotion")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="smoke test: use only this many rows per file")
    parser.add_argument("--eval-only", action="store_true", help="score the saved model without training")
    parser.add_argument("--sincere-negatives", action="store_true", help="use GoEmotions rows as not-sarcastic examples")
    parser.add_argument("--extra-emotion", type=Path, default=None, help="CSV of extra emotion training rows")
    parser.add_argument("--extra-repeat", type=int, default=3, help="how many times each extra row counts")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device()
    if args.eval_only:
        tokenizer = AutoTokenizer.from_pretrained(args.out, local_files_only=True)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(args.out), add_pooling_layer=False)
        model = EmotionSarcasmModel(encoder)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=Path(args.base).exists())
        model = train(args, tokenizer, device).cpu()
    load_model(model, str(args.out / "model.safetensors"))
    args.results.mkdir(parents=True, exist_ok=True)
    evaluate(args, model.to(device), tokenizer, device)


if __name__ == "__main__":
    main()
