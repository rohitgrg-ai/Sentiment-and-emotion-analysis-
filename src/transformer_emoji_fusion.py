"""Fine-tuned emotion transformer + emoji handling, evaluated fairly.

Run from the project root after src/finetune_emotion.py has saved models/transformer_emotion/:
    python src/transformer_emoji_fusion.py --limit 300     # quick smoke test
    python src/transformer_emoji_fusion.py                  # full run

Systems compared (emotion only)
    T          fine-tuned transformer, text as typed (emojis go to the tokenizer untouched)
    T_demoji   same model, emojis replaced by their names first ("upside down face")
    T+E(w)     late fusion, like predictor.py:  p = (1 - w) * p_transformer + w * p_emoji_vote
               (emoji vote = predictor.EMOJI_EMOTION table; emoji-only posts use the emoji vote alone)

How the fusion weight is chosen (no tuning on the 500 ambiguous rows)
    PRIMARY   w = 0.25, fixed in advance: the same emotion weight predictor.py (system D) already uses.
    SECONDARY w picked on a SYNTHETIC dev set: natural_emotion_val with emojis added by the same
              augmentation as experimental_models.py. Caveat (also printed): the augmentation pools and the
              emoji vote table were written by the same author and agree on 42 of 43 emojis, so this choice
              mostly reflects the assumed 60/25/15 match/ambiguous/contradict mix. Treat it as exploratory.

The 500-row ambiguous set is evaluated ONCE per system listed above; nothing is chosen on it.
Outputs: results/transformers/emoji_fusion/{dev_grid.csv, report.txt, metrics.json, predictions_ambiguous.csv}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emoji_features import demojize  # noqa: E402
from experimental_models import AMBIGUOUS_POOL, augment_with_emojis  # noqa: E402
from predictor import Predictor  # noqa: E402  (only the static split_emojis is used; no SVM is loaded)

DATA = ROOT / "data"
MODEL_DIR = ROOT / "models" / "transformer_emotion"
OUT = ROOT / "results" / "transformers" / "emoji_fusion"
EXISTING = ROOT / "results" / "ambiguity" / "ai_suggested_all" / "predictions_detailed.csv"
LABELS = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
NEUTRAL_ID = LABELS.index("neutral")
FIXED_W = 0.25
GRID = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
SEED = 42


# ------------------------------------------------------------------ pieces that need no model
def emoji_vote(text):
    """Vector over LABELS from the hand-made emoji table, or None if no known emoji is present."""
    _, found = Predictor.split_emojis(text)
    if not found:
        return None
    v = np.zeros(len(LABELS))
    for _, emo in found:
        v[LABELS.index(emo)] += 1
    return v / v.sum()


def has_words(text):
    rest, _ = Predictor.split_emojis(text)
    return bool(re.search(r"[A-Za-z0-9]", rest))


def fuse(p_text, texts, w):
    """Same rule as predictor.py: no emoji -> text; emoji but no words -> emoji vote; else weighted mix."""
    out = p_text.copy()
    for i, t in enumerate(texts):
        v = emoji_vote(t)
        if v is None:
            continue
        ww = 1.0 if not has_words(t) else w
        out[i] = (1 - ww) * p_text[i] + ww * v
    return out


def masked_argmax(p):
    q = p.copy()
    q[:, NEUTRAL_ID] = -1.0
    return q.argmax(1)


def synthetic_dev(val):
    """Emoji-augmented copy of the natural validation split. Neutral rows get ambiguous emojis at the
    same 50% rate, so 'has an emoji' alone does not reveal 'not neutral'."""
    rng = np.random.default_rng(SEED)
    emo = val[val["emotion"] != "neutral"]
    neu = val[val["emotion"] == "neutral"].copy()
    emo_aug = augment_with_emojis(emo[["text", "emotion"]], seed=SEED)
    texts = []
    for t in neu["text"]:
        if rng.random() < 0.5:
            n = int(rng.choice([1, 2], p=[0.8, 0.2]))
            texts.append(f"{t} {''.join(rng.choice(list(AMBIGUOUS_POOL), size=n))}")
        else:
            texts.append(t)
    neu["text"] = texts
    return pd.concat([emo_aug, neu[["text", "emotion"]]]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def mcnemar(a_ok, b_ok):
    w = int((a_ok & ~b_ok).sum()); l = int((~a_ok & b_ok).sum())
    return w, l, (binomtest(w, w + l, 0.5).pvalue if w + l else 1.0)


def paired_groups_correct(df, ok):
    g = df.assign(ok=ok)[df["paired_group_id"].notna()].groupby("paired_group_id")["ok"].all()
    return f"{int(g.sum())} of {len(g)}"


# ------------------------------------------------------------------ model
def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not MODEL_DIR.exists():
        sys.exit("models/transformer_emotion/ not found. Run src/finetune_emotion.py first.")
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if labels != LABELS:
        sys.exit(f"Unexpected label order in saved model: {labels}")

    def probs(texts, batch_size=64, max_len=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to(dev)
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))
    return probs


# ------------------------------------------------------------------ main
def run(probs, limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    lines, metrics = [], {}

    # 1) choose w on the synthetic dev set (secondary result)
    val = pd.read_csv(DATA / "natural_emotion_val.csv")
    if limit:
        val = val.head(limit)
    dev = synthetic_dev(val)
    y = dev["emotion"].map(LABELS.index).values
    emoji_rows = dev["text"].map(lambda t: emoji_vote(t) is not None).values
    base = {"T": probs(dev["text"].tolist()), "T_demoji": probs(dev["text"].map(demojize).tolist())}
    grid = []
    for name, p in base.items():
        for w in GRID:
            pred = fuse(p, dev["text"].tolist(), w).argmax(1)
            grid.append({"input": name, "w": w, "dev_macro_f1": f1_score(y, pred, average="macro", labels=range(7), zero_division=0),
                         "dev_accuracy": accuracy_score(y, pred),
                         "dev_emoji_rows_accuracy": accuracy_score(y[emoji_rows], pred[emoji_rows]) if emoji_rows.any() else np.nan})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "dev_grid.csv", index=False)
    best = grid.sort_values(["dev_macro_f1", "w"], ascending=[False, True]).iloc[0]
    metrics["dev_choice"] = {"input": best["input"], "w": float(best["w"]), "dev_macro_f1": float(best["dev_macro_f1"])}
    lines += ["SYNTHETIC DEV SET (natural validation + added emojis; used only to pick the secondary w)",
              f"rows={len(dev)}, rows with a known emoji={int(emoji_rows.sum())}",
              grid.round(3).to_string(index=False),
              f"Chosen on dev: input={best['input']}, w={best['w']}",
              "Caveat: pools and emoji table share an author and agree on 42 of 43 emojis, so this choice mostly reflects "
              "the assumed 60/25/15 mix. Exploratory only.", ""]

    # 2) systems fixed BEFORE looking at the ambiguous set
    systems = {"T (no emoji handling)": ("T", 0.0),
               f"T+E fixed w={FIXED_W} (PRIMARY)": ("T", FIXED_W),
               "T_demoji (emoji names as words)": ("T_demoji", 0.0),
               f"best on dev: {best['input']} w={best['w']} (SECONDARY)": (best["input"], float(best["w"]))}

    amb = pd.read_csv(DATA / "ambiguous_social_media_test.csv")
    amb = amb[amb["expected_emotion"].notna()].reset_index(drop=True)
    if limit:
        amb = amb.head(limit)
    old = pd.read_csv(EXISTING)[["id", "A_emotion", "D_emotion"]]
    amb = amb.merge(old, on="id", how="left")
    gold = amb["expected_emotion"].values
    sec = amb["secondary_emotion"].fillna("").values
    emoji_mask = amb["emoji_present"].astype(str).str.lower().isin(["true", "1", "yes"]).values
    texts = amb["text"].tolist()
    amb_base = {"T": probs(texts), "T_demoji": probs([demojize(t) for t in texts])}
    a_ok = amb["A_emotion"].values == gold
    d_ok = amb["D_emotion"].values == gold
    lines += [f"AMBIGUOUS SET ({len(amb)} rows with a gold emotion; AI-suggested labels, NOT human-verified; evaluated once)",
              "Main metric: strict accuracy with the neutral class masked (gold labels never use neutral).",
              f"Reference on the same rows: A strict {a_ok.mean():.3f}, D strict {d_ok.mean():.3f}", ""]
    preds_out = amb[["id", "text", "category", "expected_emotion"]].copy()
    metrics["ambiguous"] = {"A_strict": float(a_ok.mean()), "D_strict": float(d_ok.mean())}
    for label, (inp, w) in systems.items():
        p = fuse(amb_base[inp], texts, w)
        pred = np.array([LABELS[i] for i in masked_argmax(p)])
        p7 = np.array([LABELS[i] for i in p.argmax(1)])
        ok = pred == gold
        m = {"strict": float(ok.mean()), "lenient": float((ok | (pred == sec)).mean()),
             "strict_neutral_counts_wrong": float((p7 == gold).mean()),
             "share_neutral_7class": float((p7 == "neutral").mean()),
             "emoji_rows": float(ok[emoji_mask].mean()), "no_emoji_rows": float(ok[~emoji_mask].mean()),
             "paired_groups_fully_correct": paired_groups_correct(amb, ok)}
        for ref, ref_ok in (("A", a_ok), ("D", d_ok)):
            w_, l_, pv = mcnemar(ok, ref_ok)
            m[f"vs_{ref}"] = f"wins {w_} / losses {l_}, p={pv:.4f}"
        metrics["ambiguous"][label] = m
        preds_out[label] = pred
        lines += [label] + [f"  {k}: {v if isinstance(v, str) else round(v, 3)}" for k, v in m.items()] + [""]
    preds_out.to_csv(OUT / "predictions_ambiguous.csv", index=False)

    # 3) sanity check: emoji handling must not hurt plain natural text
    te = pd.read_csv(DATA / "natural_emotion_test.csv")
    if limit:
        te = te.head(limit)
    yt = te["emotion"].map(LABELS.index).values
    te_base = {"T": probs(te["text"].tolist()), "T_demoji": probs(te["text"].map(demojize).tolist())}
    sanity = {}
    for label, (inp, w) in systems.items():
        p = te_base[inp]
        pred = fuse(p, te["text"].tolist(), w).argmax(1)
        sanity[label] = float(f1_score(yt, pred, average="macro", labels=range(7), zero_division=0))
    metrics["natural_test_macro_f1_7class"] = sanity
    lines += ["SANITY: natural test (mostly emoji-free), 7-class macro-F1; should barely change"] + \
             [f"  {k}: {v:.3f}" for k, v in sanity.items()]
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (OUT / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(load_model(), a.limit)


if __name__ == "__main__":
    main()
