"""Experiment 3: controlled emoji robustness on the paired groups of the ambiguous set.

Two kinds of groups in data/ambiguous_social_media_test.csv:
  P01-P25  same text, different emoji   ("I'm fine", "I'm fine 😊", "I'm fine 🙂🔪", ...)
  S01-S08  same emoji, different text   ("My dog passed away 😭", "This song is too good 😭", ...)

For P groups each emoji variant is compared with the group's no-emoji sentence:
  sensitivity      : expected label changes  -> does the prediction change?   (emoji used when it should be)
  stability        : expected label the same -> does the prediction stay?     (emoji not over-trusted)
  correct_when_changed : expected label changes -> is the new prediction right?
  text_overrides_emoji : expected label changes but prediction stays = 1 - sensitivity
  harmful_emoji_flip   : expected label the same, no-emoji prediction was right, emoji made it wrong
For S groups (emoji fixed): do predictions differ when the expected labels differ (text_sensitivity),
and how often does one emoji force the same answer on a whole group (emoji_dominance)?

Labels are the AI-suggested ones unless --verified is given (human_verified rows only).
Run: python src/evaluate_paired_emoji.py
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_ambiguity import SYSTEM_NAMES, available_systems, load_dataset, run_existing  # noqa: E402

OUT = ROOT / "results" / "emoji_comparison"


def rate(numer, denom):
    return (round(numer / denom, 3), denom) if denom else (float("nan"), 0)


def p_group_metrics(df, pred, system, task):
    gold_col, pred_col = f"gold_{task}", f"{system}_{task}"
    c = dict(sens_n=0, sens_d=0, stab_n=0, stab_d=0, right_n=0, harm_n=0, harm_d=0, correct_n=0, correct_d=0)
    for gid, g in df[df["pair_type"] == "same_text_different_emoji"].groupby("paired_group_id"):
        base = g[~g["emoji_present"]]
        if base.empty:
            continue
        b = base.index[0]
        g0, p0 = df.at[b, gold_col], pred.at[b, pred_col]
        for i in g.index.drop(b):
            gv, pv = df.at[i, gold_col], pred.at[i, pred_col]
            if gv == "" or g0 == "":
                continue
            c["correct_d"] += 1
            c["correct_n"] += pv == gv
            if gv != g0:
                c["sens_d"] += 1
                c["sens_n"] += pv != p0
                c["right_n"] += pv == gv
            else:
                c["stab_d"] += 1
                c["stab_n"] += pv == p0
                if p0 == g0:
                    c["harm_d"] += 1
                    c["harm_n"] += pv != gv
    sens = rate(c["sens_n"], c["sens_d"])
    return {
        "variant_accuracy": rate(c["correct_n"], c["correct_d"]),
        "sensitivity": sens,
        "stability": rate(c["stab_n"], c["stab_d"]),
        "correct_when_changed": rate(c["right_n"], c["sens_d"]),
        "text_overrides_emoji": (round(1 - sens[0], 3), sens[1]) if sens[1] else sens,
        "harmful_emoji_flip": rate(c["harm_n"], c["harm_d"]),
    }


def s_group_metrics(df, pred, system, task):
    gold_col, pred_col = f"gold_{task}", f"{system}_{task}"
    diff_n = diff_d = dominated = groups = 0
    for gid, g in df[df["pair_type"] == "same_emoji_different_text"].groupby("paired_group_id"):
        g = g[g[gold_col] != ""]
        if len(g) < 2:
            continue
        for i, j in itertools.combinations(g.index, 2):
            if df.at[i, gold_col] != df.at[j, gold_col]:
                diff_d += 1
                diff_n += pred.at[i, pred_col] != pred.at[j, pred_col]
        if g[gold_col].nunique() > 1:
            groups += 1
            dominated += pred.loc[g.index, pred_col].nunique() == 1
    return {"text_sensitivity": rate(diff_n, diff_d), "emoji_dominance": rate(dominated, groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verified", action="store_true")
    args = parser.parse_args()
    df, label_set = load_dataset(verified=args.verified)
    df = df[df["paired_group_id"] != ""].copy()
    if df.empty:
        print("No paired rows available for this label set.")
        return
    pred = run_existing(df)
    try:
        from experimental_models import predict_experimental
        pred = pred.join(predict_experimental(df["text"].tolist(), index=df.index))
    except FileNotFoundError:
        pass
    systems = available_systems(pred)

    rows, lines = [], [f"PAIRED EMOJI ROBUSTNESS — labels: {label_set}; "
                       f"{df['paired_group_id'].nunique()} groups, {len(df)} sentences",
                       "Values: rate (number of comparisons). Higher is better except text_overrides_emoji "
                       "and harmful_emoji_flip (lower is better) and emoji_dominance (lower is better).", ""]
    for s in systems:
        lines.append(SYSTEM_NAMES[s])
        for task in ["sentiment", "emotion"]:
            m = {**p_group_metrics(df, pred, s, task), **s_group_metrics(df, pred, s, task)}
            rows.append({"system": s, "task": task, **{k: v[0] for k, v in m.items()},
                         **{f"{k}_n": v[1] for k, v in m.items()}})
            lines.append(f"  {task:9s} " + " | ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in m.items()))
        lines.append("")

    # Example table: one group, every system
    show = df[df["paired_group_id"].isin(["P01", "P03", "P05", "S01"])]
    lines.append("EXAMPLES (expected emotion/sentiment -> each system's emotion/sentiment)")
    for i, r in show.iterrows():
        exp = f"{r.gold_emotion or '-'}/{r.gold_sentiment}"
        preds = "  ".join(f"{s}:{pred.at[i, s + '_emotion']}/{pred.at[i, s + '_sentiment'][:3]}" for s in systems)
        lines.append(f"  [{r.paired_group_id}] {r.text:34s} exp {exp:17s} {preds}")

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if label_set == "ai_suggested_all" else "_verified"
    pd.DataFrame(rows).to_csv(OUT / f"paired_robustness_metrics{suffix}.csv", index=False)
    cols = ["id", "paired_group_id", "pair_type", "text", "gold_sentiment", "gold_emotion"]
    df[cols].join(pred[[c for c in pred.columns if c.endswith(("_emotion", "_sentiment"))]]).to_csv(
        OUT / f"paired_variant_predictions{suffix}.csv", index=False)
    (OUT / f"paired_robustness_report{suffix}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
