"""Fine-tune a pretrained transformer for 7-class emotion (six emotions + neutral) on the natural dataset.

Run from the project root (GPU strongly recommended: free Colab T4 or a Mac with Apple GPU):
    python src/finetune_emotion.py --epochs 1 --limit 2000     # quick smoke test first
    python src/finetune_emotion.py                              # full run

Needs: pip install transformers torch   (no extra packages; plain PyTorch loop, no Trainer/accelerate)
Inputs : data/natural_emotion_{train,val,test}.csv  (from build_natural_dataset.py)
Fair-test rules
    * Trains on natural_emotion_train only. The best epoch is chosen on natural_emotion_val (macro-F1).
    * natural_emotion_test and the 500-row ambiguous set are each evaluated ONCE, after training.
      Nothing is tuned on either of them.
    * Hyperparameters are fixed up front (seed 42). Any change you try is a new, reported experiment.
Outputs: models/transformer_emotion/ (model + tokenizer), results/transformers/finetune_emotion/
    history.csv, report.txt, metrics.json, predictions_natural_test.csv, predictions_ambiguous.csv
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "transformers" / "finetune_emotion"
MODEL_DIR = ROOT / "models" / "transformer_emotion"
LABELS = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
SIX = [l for l in LABELS if l != "neutral"]
NEUTRAL_ID = LABELS.index("neutral")
L2I = {l: i for i, l in enumerate(LABELS)}


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(y, power=0.5):
    """Square-root inverse frequency: helps rare classes (fear, surprise) without over-boosting them."""
    counts = np.bincount(y, minlength=len(LABELS)).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return w / w.mean()


def restricted_argmax(logits):
    """Best of the six emotions, ignoring the neutral class (for comparison with six-class systems)."""
    masked = logits.copy()
    masked[:, NEUTRAL_ID] = -1e9
    return masked.argmax(1)


def predict_logits(model, tok, texts, device, batch_size, max_len):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def train(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    set_seed(args.seed)
    device = pick_device()
    tr = pd.read_csv(DATA / "natural_emotion_train.csv")
    va = pd.read_csv(DATA / "natural_emotion_val.csv")
    if args.limit:
        tr, va = tr.sample(min(args.limit, len(tr)), random_state=args.seed), va.head(args.limit // 4 or 1)
    ytr = tr["emotion"].map(L2I).values
    yva = va["emotion"].map(L2I).values
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(ytr), dtype=torch.float, device=device))
    texts = tr["text"].tolist()
    history, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = np.random.RandomState(args.seed + ep).permutation(len(texts))
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            labels = torch.tensor(ytr[idx], device=device)
            loss = loss_fn(model(**enc).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item() * len(idx)
        val_pred = predict_logits(model, tok, va["text"].tolist(), device, args.batch_size * 2, args.max_len).argmax(1)
        f1 = f1_score(yva, val_pred, average="macro", labels=range(len(LABELS)), zero_division=0)
        history.append({"epoch": ep, "train_loss": total / len(texts), "val_macro_f1": f1,
                        "val_accuracy": accuracy_score(yva, val_pred)})
        print(history[-1], flush=True)
        if f1 > best:  # model selection on VALIDATION only
            best = f1
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)
    return history, device


def svm_text_only(texts):
    """Existing text-only SVM (system A). Imported lazily so this file works without the old code."""
    sys.path.insert(0, str(ROOT / "src"))
    from predictor import Predictor
    p = Predictor()
    return [p.predict(t)["emotion_from_text"] for t in texts]


def evaluate(args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    lines, metrics = [], {}

    # ---- natural test (evaluated once) ----
    te = pd.read_csv(DATA / "natural_emotion_test.csv")
    if args.limit:
        te = te.head(args.limit)
    logits = predict_logits(model, tok, te["text"].tolist(), device, args.batch_size * 2, args.max_len)
    y = te["emotion"].map(L2I).values
    pred = logits.argmax(1)
    metrics["natural_test_7class"] = {"accuracy": accuracy_score(y, pred),
                                      "macro_f1": f1_score(y, pred, average="macro", labels=range(7), zero_division=0)}
    lines += ["NATURAL TEST (7 classes, evaluated once)", f"n={len(te)}",
              classification_report(y, pred, labels=range(7), target_names=LABELS, zero_division=0, digits=3),
              "Confusion matrix (rows gold, cols predicted; order: " + ", ".join(LABELS) + ")",
              str(confusion_matrix(y, pred, labels=range(7))), ""]
    nn = te["emotion"] != "neutral"
    a_pred = svm_text_only(te.loc[nn, "text"].tolist())
    t_pred = [LABELS[i] for i in restricted_argmax(logits[nn.values])]
    gold6 = te.loc[nn, "emotion"].tolist()
    metrics["natural_test_six_emotions_only"] = {
        "n": int(nn.sum()),
        "svm_A_macro_f1": f1_score(gold6, a_pred, labels=SIX, average="macro", zero_division=0),
        "svm_A_accuracy": accuracy_score(gold6, a_pred),
        "transformer_macro_f1": f1_score(gold6, t_pred, labels=SIX, average="macro", zero_division=0),
        "transformer_accuracy": accuracy_score(gold6, t_pred)}
    lines += ["NATURAL TEST, non-neutral rows only (same rows, six emotions; transformer's neutral class masked)",
              json.dumps(metrics["natural_test_six_emotions_only"], indent=1),
              "Note: the old SVM was trained on a different (keyword-style) dataset, so this shows how well it transfers, "
              "not a like-for-like training comparison.", ""]
    pd.DataFrame({"text": te["text"], "gold": te["emotion"], "pred": [LABELS[i] for i in pred]}) \
        .to_csv(OUT / "predictions_natural_test.csv", index=False)

    # ---- ambiguous set (evaluated once, never trained or tuned on) ----
    amb = pd.read_csv(DATA / "ambiguous_social_media_test.csv")
    amb = amb[amb["expected_emotion"].notna()].reset_index(drop=True)
    if args.limit:
        amb = amb.head(args.limit)
    al = predict_logits(model, tok, amb["text"].tolist(), device, args.batch_size * 2, args.max_len)
    p7 = [LABELS[i] for i in al.argmax(1)]
    p6 = [LABELS[i] for i in restricted_argmax(al)]
    sec = amb["secondary_emotion"].fillna("")
    gold = amb["expected_emotion"]
    metrics["ambiguous"] = {
        "n_with_gold_emotion": len(amb),
        "strict_acc_7class_neutral_counts_wrong": float((np.array(p7) == gold).mean()),
        "strict_acc_neutral_masked": float((np.array(p6) == gold).mean()),
        "lenient_acc_neutral_masked": float(((np.array(p6) == gold) | (np.array(p6) == sec)).mean()),
        "share_predicted_neutral_7class": float((np.array(p7) == "neutral").mean())}
    lines += ["AMBIGUOUS SET (labels are AI-suggested, NOT human-verified; evaluated once, never tuned on)",
              json.dumps(metrics["ambiguous"], indent=1),
              "Compare with earlier systems on the same 436 rows: A strict 0.275, D strict 0.401 "
              "(results/ambiguity/ai_suggested_all/report.txt).", ""]
    emoji = amb["emoji_present"].astype(str).str.lower().isin(["true", "1", "yes"])
    for name, key, m in (("emoji rows", "emoji_rows", emoji), ("no-emoji rows", "no_emoji_rows", ~emoji)):
        acc = float((np.array(p6)[m.values] == gold[m].values).mean())
        metrics["ambiguous"][f"strict_acc_neutral_masked_{key}"] = acc
        lines.append(f"  {name}: n={int(m.sum())} strict (neutral masked) = {acc:.3f}")
    pd.DataFrame({"id": amb["id"], "text": amb["text"], "gold": gold, "pred_7class": p7, "pred_neutral_masked": p6}) \
        .to_csv(OUT / "predictions_ambiguous.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (OUT / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use only this many rows")
    args = ap.parse_args()
    history, device = train(args)
    evaluate(args, device)


if __name__ == "__main__":
    main()
