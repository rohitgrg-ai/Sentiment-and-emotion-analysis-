"""Charts + comparison report for the emoji/ambiguity experiments.

Run after experimental_models.py, evaluate_ambiguity.py and evaluate_paired_emoji.py:
    python src/report_experiments.py
Writes results/emoji_comparison/accuracy_by_subset.png, paired_sensitivity_vs_stability.png
and comparison_report.md (all numbers are read from the saved result files).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AMB = ROOT / "results" / "ambiguity" / "ai_suggested_all" / "metrics_by_subset.csv"
PAIRED = ROOT / "results" / "emoji_comparison" / "paired_robustness_metrics.csv"
OUT = ROOT / "results" / "emoji_comparison"

ORDER = ["A", "B_orig", "C_orig", "B_aug", "C_aug", "D"]
LABELS = {"A": "A text-only", "B_orig": "B demojized", "C_orig": "C hybrid", "B_aug": "B′ demojized+aug",
          "C_aug": "C′ hybrid+aug", "D": "D emoji voting"}
BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#1f1f1e", "#6b6a64", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def accuracy_panels():
    t = pd.read_csv(AMB)
    subsets = [("all", "All 500 rows"), ("emoji_present", "Emoji rows"), ("emoji_free", "Emoji-free rows"),
               ("sarcasm_and_irony", "Sarcasm + irony")]
    tasks = [("sent_acc", "Sentiment accuracy", 1 / 3), ("emo_acc", "Emotion accuracy", 1 / 6)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True, sharey=True)
    for r, (col, task_label, chance) in enumerate(tasks):
        for c, (subset, title) in enumerate(subsets):
            ax = axes[r, c]
            d = t[t.subset == subset].set_index("system").reindex(ORDER)
            y = range(len(ORDER))
            ax.barh(list(y), d[col], color=BLUE, height=0.62)
            for yi, v in zip(y, d[col]):
                ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=8, color=INK,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
            ax.axvline(chance, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
            ax.set_xlim(0, 1)
            ax.set_yticks(list(y), [LABELS[s] for s in ORDER])
            ax.invert_yaxis()
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(task_label, fontsize=10, color=INK)
    fig.suptitle("Ambiguous social-media set: accuracy by system (AI-suggested labels, not yet human-verified)",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.005, "Dashed line = random guessing (1/3 sentiment, 1/6 emotion). "
             "Original test-set accuracy for every system: ~0.95 sentiment, ~0.96 emotion.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "accuracy_by_subset.png", dpi=150)
    plt.close(fig)


def tradeoff_panels():
    t = pd.read_csv(PAIRED)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, task in zip(axes, ["sentiment", "emotion"]):
        d = t[t.task == task].set_index("system").reindex(ORDER)
        ax.scatter(d["stability"], d["sensitivity"], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        placed = {}
        for s, r in d.iterrows():
            key = (round(r.stability, 2), round(r.sensitivity, 2))
            label = LABELS[s] if key not in placed else None
            if label is None:
                placed[key] += " = " + LABELS[s].split()[0]
                continue
            placed[key] = label
        keys = list(placed)
        for (x, y), label in placed.items():
            # put the label on the left if another point is close by on the right
            crowded = any(0 < kx - x < 0.25 and abs(ky - y) < 0.08 for kx, ky in keys)
            offset, ha = ((-6, 4), "right") if crowded else ((6, 4), "left")
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8, color=INK, ha=ha)
        ax.set_xlim(-0.03, 1.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(task.title(), fontsize=10, color=INK)
        ax.set_xlabel("Stability: keeps its answer when the emoji\nshould NOT change it")
        ax.grid(color=GRID, lw=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Sensitivity: reacts when the emoji\nSHOULD change the answer")
    fig.suptitle("Same text, different emoji: no system is both sensitive and stable (ideal = top right)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "paired_sensitivity_vs_stability.png", dpi=150)
    plt.close(fig)


def _row(name, m):
    return (f"| {name} | {m['accuracy']:.3f} | {m['macro_precision']:.3f} | {m['macro_recall']:.3f} | "
            f"{m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |")


HEADER = "| System | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |\n|---|---|---|---|---|---|"


def write_report():
    train = json.loads((OUT / "training_results.json").read_text())["models"]
    amb_dir = AMB.parent
    amb = json.loads((amb_dir / "metrics_summary.json").read_text())["systems"]
    sig = (amb_dir / "report.txt").read_text().split("SIGNIFICANCE vs A", 1)[1]
    paired = pd.read_csv(PAIRED)
    verified = ROOT / "results" / "ambiguity" / "human_verified" / "metrics_summary.json"
    key_train = {"A": "A_existing", "B_orig": "B_orig", "C_orig": "C_orig", "B_aug": "B_aug", "C_aug": "C_aug"}
    L = ["# Emoji & Ambiguity Experiments — Comparison Report", "",
         "Research question: *To what extent can emoji-aware feature extraction improve sentiment and emotion "
         "classification on ambiguous social media text compared with conventional text-only machine learning "
         "and rule-based emoji integration?*", "",
         "Systems: **A** existing text-only Linear SVM (primary benchmark) · **B** demojized TF-IDF SVM · **C** hybrid "
         "text + emoji-feature SVM · **B′/C′** the same trained on a synthetic emoji-augmented copy of the training "
         "split · **D** existing `predictor.py` (A + rule-based emoji voting).", "",
         "> **Three different evaluations — do not compare their numbers directly.** (1) the original test split "
         "(11,796 rows, same distribution as training, no emojis); (2) the ambiguous set (500 hand-written social-media "
         "sentences, very different style) with AI-suggested labels; (3) the same set restricted to human-verified "
         "labels. A drop from (1) to (2) measures how badly the model transfers, not a single \"true\" accuracy.", ""]
    for task in ["sentiment", "emotion"]:
        L += [f"## 1. Original test split — {task} (existing benchmark and experimental models)", "", HEADER]
        for s in ["A", "B_orig", "C_orig", "B_aug", "C_aug"]:
            L.append(_row(LABELS[s], train[key_train[s]][f"{task}_original_test"]))
        L.append("| D emoji voting | identical to A on this split (it contains no emojis) | | | | |")
        L.append("")
    L += ["## 2. Ambiguous set — AI-suggested labels (NOT human-verified)", "",
          "Emotion metrics exclude the 64 rows whose emotion fits none of the six classes.", ""]
    for subset, title in [("all", "All 500 rows"), ("emoji_present", "Emoji-containing rows (158)"),
                          ("emoji_free", "Emoji-free rows (342)"), ("paired_examples", "Paired examples (161)"),
                          ("sarcasm_and_irony", "Sarcasm and irony (80)")]:
        for task in ["sentiment", "emotion"]:
            L += [f"### {title} — {task}", "", HEADER]
            for s in ORDER:
                L.append(_row(LABELS[s], amb[s][subset][task]))
            L.append("")
    L += ["Confusion matrices for every system: `results/ambiguity/ai_suggested_all/confusion_*.png`.", "",
          "### Significance vs A (exact McNemar test on the same rows)", "", "```", sig.strip(), "```", "",
          "![Accuracy by subset](accuracy_by_subset.png)", "",
          "## 3. Paired emoji robustness (same text, different emoji)", "",
          "| System | Task | Variant accuracy | Sensitivity | Stability | Text overrides emoji | Harmful emoji flip | "
          "Emoji dominance |", "|---|---|---|---|---|---|---|---|"]
    for _, r in paired.iterrows():
        L.append(f"| {LABELS[r.system]} | {r.task} | {r.variant_accuracy:.2f} | {r.sensitivity:.2f} | "
                 f"{r.stability:.2f} | {r.text_overrides_emoji:.2f} | {r.harmful_emoji_flip:.2f} | "
                 f"{r.emoji_dominance:.2f} |")
    L += ["", "![Sensitivity vs stability](paired_sensitivity_vs_stability.png)", "",
          "## 4. Human-verified evaluation", ""]
    if verified.exists():
        v = json.loads(verified.read_text())
        L += [f"{v['rows']} human-verified rows.", "", HEADER]
        for task in ["sentiment", "emotion"]:
            L.append(f"| **{task}** | | | | | |")
            for s in ORDER:
                if s in v["systems"]:
                    L.append(_row(LABELS[s], v["systems"][s]["all"][task]))
    else:
        L += ["Not available yet: no rows are marked `human_verified`. After review, run "
              "`python src/evaluate_ambiguity.py --build-verified` and then this script again."]
    (OUT / "comparison_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    accuracy_panels()
    tradeoff_panels()
    write_report()
    print("Saved charts and comparison_report.md in", OUT)
