"""Check whether predictions follow the emoji in a text (same sentence, different emojis).

Uses exactly the app's prediction logic (src/predictor.py). The sentences here
are different from the ones used to choose the emoji weights, so this is a fair
check. "Expected" values are what a human would answer.

Run from the project root:  python src/test_emoji_robustness.py
Writes results/emoji_robustness_predictions.csv and results/emoji_robustness_report.txt
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictor import EMOTION_TO_SENTIMENT, Predictor  # noqa: E402

RESULTS = ROOT / "results"
predictor = Predictor()

EMOJIS = {  # 3 emojis per emotion
    "joy": ["😁", "🤣", "🎉"],
    "love": ["😍", "🥰", "💕"],
    "anger": ["😠", "🤬", "😤"],
    "fear": ["😱", "😰", "😧"],
    "sadness": ["😭", "😞", "💔"],
    "surprise": ["😮", "🤯", "😳"],
}
# Sentences with no clear emotion of their own: the emoji should decide.
NEUTRAL_TEXTS = [
    "my brother came over today",
    "the bus was late again",
    "we are going to the beach on sunday",
    "she replied to my message",
    "got my phone back from repair",
    "the teacher announced the results",
    "dinner is ready",
    "that is what he said",
    "first day at the new office",
    "watched the match last night",
]
# Sentences with a clear emotion: when an emoji disagrees, the text should win.
EMOTIONAL_TEXTS = {
    "i am so delighted and grateful for everything": "joy",
    "i feel heartbroken and alone tonight": "sadness",
    "i am so angry and annoyed with my roommate": "anger",
    "i feel scared and nervous about the interview": "fear",
    "i adore my little sister she is so sweet and caring": "love",
    "i was stunned and surprised by the plot twist": "surprise",
}
UNMAPPED_EMOJIS = ["🥲", "🙃", "😅", "🤔", "😐", "💀"]  # ambiguous: deliberately not mapped

rows = []
def add(test, text, base, emoji, exp_emotion, exp_sentiment):
    r = predictor.predict(text)
    rows.append({
        "test": test, "text": text, "base_text": base, "emoji": emoji,
        "expected_emotion": exp_emotion, "predicted_emotion": r["emotion"],
        "expected_sentiment": exp_sentiment, "predicted_sentiment": r["sentiment"],
        "emotion_from_text_only": r["emotion_from_text"],
        "emotion_ok": (r["emotion"] == exp_emotion) if exp_emotion else None,
        "sentiment_ok": (r["sentiment"] == exp_sentiment) if exp_sentiment else None,
    })

for base in NEUTRAL_TEXTS:
    add("0_no_emoji", base, base, "", None, None)
    for emo, emojis in EMOJIS.items():
        sent = EMOTION_TO_SENTIMENT[emo]
        for e in emojis:
            add("1_emoji_at_end", f"{base} {e}", base, e, emo, sent)
            add("2_emoji_at_start", f"{e} {base}", base, e, emo, sent)
        add("3_repeated_emoji", f"{base} {emojis[0] * 3}", base, emojis[0] * 3, emo, sent)
        add("4_two_same_emotion_emojis", f"{base} {emojis[0]}{emojis[1]}", base, emojis[0] + emojis[1], emo, sent)

for emo, emojis in EMOJIS.items():
    for e in emojis:
        add("5_emoji_only", e, "", e, emo, EMOTION_TO_SENTIMENT[emo])

for base, text_emo in EMOTIONAL_TEXTS.items():
    add("6_emotional_text_no_emoji", base, base, "", text_emo, EMOTION_TO_SENTIMENT[text_emo])
    for emo, emojis in EMOJIS.items():
        same = emo == text_emo
        add("7_text_and_emoji_agree" if same else "8_text_and_emoji_conflict",
            f"{base} {emojis[0]}", base, emojis[0], text_emo, EMOTION_TO_SENTIMENT[text_emo])

for base in NEUTRAL_TEXTS[:5]:
    for e in UNMAPPED_EMOJIS:
        add("9_unmapped_emoji", f"{base} {e}", base, e, None, None)

df = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "emoji_robustness_predictions.csv", index=False)

lines = []
def out(s=""):
    lines.append(s)
    print(s)

def pct(series):
    series = series.dropna().astype(bool)
    return f"{series.mean() * 100:5.1f}% ({series.sum()}/{len(series)})" if len(series) else "  n/a"

out("EMOJI ROBUSTNESS TEST (expected = human judgement)")
out("=" * 78)
out(f"{'test':30s} {'emotion correct':>20s} {'sentiment correct':>20s}")
for test, g in df.groupby("test"):
    if test in ("0_no_emoji", "9_unmapped_emoji"):
        continue
    out(f"{test:30s} {pct(g.emotion_ok):>20s} {pct(g.sentiment_ok):>20s}")

single = df[df.test.isin(["1_emoji_at_end", "2_emoji_at_start"])]
out("\nSame sentence + one emoji, by emoji emotion:")
for emo, g in single.groupby("expected_emotion"):
    wrong = g[~g.sentiment_ok.astype(bool)].predicted_sentiment.value_counts().to_dict()
    out(f"  {emo:9s} emotion {pct(g.emotion_ok)}  sentiment {pct(g.sentiment_ok)}  wrong sentiment -> {wrong}")

out("\nWithout an emoji, neutral sentences get an arbitrary emotion (no 'neutral' emotion class):")
for _, r in df[df.test == "0_no_emoji"].iterrows():
    out(f"  {r.text:38s} -> {r.predicted_emotion} / {r.predicted_sentiment}")

out("\nFailures (text + emoji conflicts and others):")
fails = df[(df.emotion_ok == False) | (df.sentiment_ok == False)]  # noqa: E712
for _, r in fails.iterrows():
    out(f"  [{r.test}] {r.text:50s} -> {r.predicted_emotion}/{r.predicted_sentiment}"
        f"  (expected {r.expected_emotion}/{r.expected_sentiment})")

out("\nEmojis not in the table (ambiguous ones like 🥲 🙃 😅 🤔 😐 💀) are ignored:")
un = df[df.test == "9_unmapped_emoji"]
ref = df[df.test == "0_no_emoji"].set_index("base_text").predicted_emotion
same = (un.predicted_emotion.values == un.base_text.map(ref).values).mean()
out(f"  prediction identical to the sentence without emoji: {same * 100:.0f}% of {len(un)} cases")

(RESULTS / "emoji_robustness_report.txt").write_text("\n".join(lines) + "\n")
