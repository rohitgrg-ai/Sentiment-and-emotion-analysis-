"""Prepare a clean dataset balanced on both emotion and sentiment labels.

The input sentiment labels in this project were generated heuristically. This
script preserves them but does not treat them as human-verified annotations.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "text2.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "final.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "balancing_summary.json"
EMOTION_NAMES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
SENTIMENTS = ("negative", "neutral", "positive")
# These suffixes were synthetically appended in the project data to encode
# targets. Remove them before model training to prevent target leakage.
SYNTHETIC_SUFFIX = re.compile(
    r"\s+(?:😢|😄|❤️|😡|😨|😲)\s+(?:🙂|😐|🙁)\s*$"
)


class Dinic:
    """Integer max-flow implementation for feasible joint-label quotas."""

    def __init__(self, node_count: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int) -> None:
        forward = [target, capacity, len(self.graph[target])]
        reverse = [source, 0, len(self.graph[source])]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        node_count = len(self.graph)
        while True:
            level = [-1] * node_count
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for target, capacity, _ in self.graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total

            cursor = [0] * node_count

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, capacity, reverse_index = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        pushed = send(target, min(amount, capacity))
                        if pushed:
                            edge[1] -= pushed
                            self.graph[target][reverse_index][1] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 10**18)
                if not pushed:
                    break
                total += pushed


def build_joint_quotas(
    joint_counts: pd.DataFrame,
    emotion_target: int,
    emotion_labels: list[int],
    sentiment_labels: list[str],
) -> dict[tuple[int, str], int]:
    """Find integer cell quotas with equal emotion and sentiment marginals."""
    source = 0
    emotion_start = 1
    sentiment_start = emotion_start + len(emotion_labels)
    sink = sentiment_start + len(sentiment_labels)
    flow = Dinic(sink + 1)

    for i, emotion in enumerate(emotion_labels):
        flow.add_edge(source, emotion_start + i, emotion_target)
    for j, sentiment in enumerate(sentiment_labels):
        flow.add_edge(sentiment_start + j, sink, emotion_target * len(emotion_labels) // len(sentiment_labels))

    pair_edges: dict[tuple[int, str], tuple[int, int]] = {}
    for i, emotion in enumerate(emotion_labels):
        for j, sentiment in enumerate(sentiment_labels):
            capacity = int(joint_counts.loc[sentiment, emotion]) if sentiment in joint_counts.index and emotion in joint_counts.columns else 0
            node = emotion_start + i
            edge_index = len(flow.graph[node])
            flow.add_edge(node, sentiment_start + j, capacity)
            pair_edges[(emotion, sentiment)] = (node, edge_index)

    requested = emotion_target * len(emotion_labels)
    achieved = flow.max_flow(source, sink)
    if achieved != requested:
        raise ValueError(
            "Cannot balance both targets using undersampling alone for this "
            f"target size (needed {requested} rows, feasible {achieved})."
        )

    quotas: dict[tuple[int, str], int] = {}
    for key, (node, edge_index) in pair_edges.items():
        target, remaining_capacity, _ = flow.graph[node][edge_index]
        initial_capacity = int(joint_counts.loc[key[1], key[0]])
        quotas[key] = initial_capacity - remaining_capacity
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"text", "label", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    source_rows = len(df)
    df = df.dropna(subset=["text", "label", "sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    if not set(df["label"].unique()).issubset(EMOTION_NAMES):
        raise ValueError("Unexpected emotion IDs; expected 0 through 5.")
    if not set(df["sentiment"].unique()).issubset(SENTIMENTS):
        raise ValueError(f"Unexpected sentiment values; expected {SENTIMENTS}.")

    synthetic_removed = 0
    if "emoji_augmented" in df.columns:
        augmented = df["emoji_augmented"].fillna(False).astype(bool)
        before = df.loc[augmented, "text"].copy()
        df.loc[augmented, "text"] = before.map(lambda value: SYNTHETIC_SUFFIX.sub("", value).strip())
        synthetic_removed = int((before != df.loc[augmented, "text"]).sum())

    # Preserve a stable pointer to the input record for auditing.
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "source_index"})
    elif "source_index" not in df.columns:
        df.insert(0, "source_index", df.index)

    df = df[df["text"].ne("")].copy()
    before_duplicate_filter = len(df)
    text_targets = df.groupby("text").agg(
        emotion_classes=("label", "nunique"),
        sentiment_classes=("sentiment", "nunique"),
        rows=("source_index", "size"),
    )
    conflicting_texts = text_targets.index[
        (text_targets["emotion_classes"] > 1) | (text_targets["sentiment_classes"] > 1)
    ]
    conflict_group_count = len(conflicting_texts)
    conflict_row_count = int(text_targets.loc[conflicting_texts, "rows"].sum()) if conflict_group_count else 0
    df = df[~df["text"].isin(conflicting_texts)].copy()

    # Collapse repeated copies when the exact same text has the same two labels.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text", "label", "sentiment"], keep="first").copy()
    duplicate_rows_removed = before_dedup - len(df)

    emotion_labels = sorted(EMOTION_NAMES)
    sentiment_labels = list(SENTIMENTS)
    emotion_counts = df["label"].value_counts()
    missing_emotions = set(emotion_labels) - set(emotion_counts.index)
    missing_sentiments = set(sentiment_labels) - set(df["sentiment"].unique())
    if missing_emotions or missing_sentiments:
        raise ValueError(f"Missing emotion classes {missing_emotions} or sentiments {missing_sentiments} after cleanup.")

    # Pure undersampling keeps one real row per selected example. A max-flow
    # allocation balances both marginals simultaneously while respecting each
    # observed emotion × sentiment cell's available row count.
    emotion_target = int(emotion_counts.min())
    joint_counts = pd.crosstab(df["sentiment"], df["label"])
    quotas = build_joint_quotas(joint_counts, emotion_target, emotion_labels, sentiment_labels)

    selected_parts = []
    for (emotion, sentiment), quota in quotas.items():
        cell = df[(df["label"] == emotion) & (df["sentiment"] == sentiment)]
        if quota:
            selected_parts.append(cell.sample(n=quota, replace=False, random_state=args.seed))
    final = pd.concat(selected_parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    final["emotion"] = final["label"].map(EMOTION_NAMES)
    final = final[["source_index", "text", "label", "emotion", "sentiment"]]

    # Refuse to write if class balancing invariants did not hold.
    emotion_final = final["label"].value_counts().sort_index()
    sentiment_final = final["sentiment"].value_counts().reindex(sentiment_labels)
    if emotion_final.nunique() != 1 or sentiment_final.nunique() != 1:
        raise AssertionError("Final output is not exactly balanced on both targets.")
    if final["text"].duplicated().any():
        raise AssertionError("Final output still contains duplicate text rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    summary = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "random_seed": args.seed,
        "input_rows": source_rows,
        "rows_after_missing_text_or_label_removal": before_duplicate_filter,
        "synthetic_emoji_rows_cleaned": synthetic_removed,
        "conflicting_text_groups_removed": conflict_group_count,
        "rows_removed_for_conflicting_labels": conflict_row_count,
        "same_label_duplicate_rows_removed": duplicate_rows_removed,
        "clean_unique_text_rows_before_balance": len(df),
        "rows_after_balance": len(final),
        "emotion_samples_each": int(emotion_final.iloc[0]),
        "sentiment_samples_each": int(sentiment_final.iloc[0]),
        "emotion_counts": {EMOTION_NAMES[int(k)]: int(v) for k, v in emotion_final.items()},
        "sentiment_counts": {str(k): int(v) for k, v in sentiment_final.items()},
        "joint_cell_quotas": {
            f"{EMOTION_NAMES[e]}|{s}": int(n) for (e, s), n in sorted(quotas.items())
        },
        "sentiment_label_note": "Sentiment labels in the input were generated by a simple lexicon heuristic and are not manually verified.",
        "balancing_method": "Remove label-coded synthetic emoji suffixes; drop texts with conflicting targets; deduplicate exact texts; use seeded undersampling with integer max-flow quotas to balance both marginal targets without replacement.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {len(final):,} rows to {args.output}")
    print("Emotion counts:")
    print(final["emotion"].value_counts().sort_index().to_string())
    print("\nSentiment counts:")
    print(final["sentiment"].value_counts().sort_index().to_string())
    print(f"\nDropped {source_rows - len(final):,} rows total; summary: {args.summary}")


if __name__ == "__main__":
    main()
