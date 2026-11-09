# Annotation Guide — Ambiguous Social Media Test Set

File: `data/ambiguous_social_media_test.csv` (500 rows). Purpose: an **independent, evaluation-only** set for testing how the system handles sarcasm, irony, slang, mixed feelings and emojis. It must never be used for training or tuning.

## Where the labels came from (be honest about this)
- All 500 sentences and their `expected_*` labels were **written and suggested by an AI assistant (Claude)**. `label_source = ai_suggested`, `annotation_status = pending_review` on every row.
- The model was **not** used to label anything, and labels were **not** assigned by the emoji lookup table.
- Caveat: the same assistant wrote the emoji table in `predictor.py`. Its label suggestions may share the same assumptions about emojis, which could favour the emoji-voting system. That is why independent human review matters.
- No sentence was copied from the training data (0 exact overlaps with `final.csv`).

## Label definitions (same classes as the project)
**Sentiment** — the writer's *intended* overall attitude, not the surface words.
- `positive` — the writer is pleased, approving, affectionate, amused.
- `negative` — the writer is unhappy, annoyed, afraid, hurt, disapproving.
- `neutral` — no clear attitude, or genuinely balanced.
- Sarcasm/irony: label the **intended meaning** ("Great, my laptop crashed" → negative).

**Emotion** — the writer's dominant felt emotion; one of: `joy, sadness, anger, fear, love, surprise`.
- joy: happiness, amusement, excitement, pride, relief. sadness: loss, disappointment, loneliness, weariness, hurt.
- anger: annoyance, frustration, irritation, outrage. fear: worry, anxiety, dread, nervousness.
- love: affection, warmth toward someone/something. surprise: shock, amazement, disbelief.
- **No forced labels.** If none of the six fits (embarrassment, guilt, suspicion, plain factual text), leave `expected_emotion` **empty** — the row is flagged `no_valid_emotion_class` and excluded from emotion metrics. We do not add a new class.
- `secondary_emotion`: a second emotion that is clearly present (mixed feelings). Metrics report a *strict* score (primary only) and a *lenient* score (primary or secondary).

## Columns
| Column | Meaning |
|---|---|
| id | AMB0001 … AMB0500 |
| text | Sentence including emojis |
| expected_sentiment / expected_emotion | Suggested labels (empty emotion = no valid class) |
| secondary_emotion | Optional second emotion |
| category | One of 10 ambiguity types (see below) |
| explanation | Why this interpretation was suggested |
| ambiguity_level | Easy / Medium / Hard |
| emoji_present | True if the text contains an emoji |
| paired_group_id | P01–P25: identical text, different emoji; S01–S08: same emoji, different text; empty = not paired |
| pair_type | same_text_different_emoji / same_emoji_different_text |
| annotation_status | pending_review → human_verified (or disputed) |
| label_source | ai_suggested |
| review_flags | no_valid_emotion_class; mixed_emotions; secondary_emotion; subjective_interpretation |
| reviewed_sentiment / reviewed_emotion | Fill in **only if you disagree** with the suggestion |
| reviewer / review_notes | Your name, and any comment |

Categories: sarcasm, irony, positive_words_negative_meaning, negative_words_positive_meaning, emoji_dependent_sentiment, context_dependent_emotion, mixed_emotions, internet_slang, contradictory_text_emoji, neutral_ambiguous.

## How to review (for Rohit / classmates)
1. Open the CSV in Excel / Google Sheets. **Cover the `expected_*` and `explanation` columns** first and decide your own sentiment and emotion, then compare. (This avoids just agreeing with the suggestion.)
2. If you agree: set `annotation_status = human_verified`, put your name in `reviewer`.
3. If you disagree but are confident: write your labels in `reviewed_sentiment` / `reviewed_emotion`, set `human_verified`, and explain in `review_notes`. Your labels then override the suggestion.
4. If the sentence is genuinely unclear even to you: set `annotation_status = disputed`. Disputed rows are left out of the verified evaluation.
5. Ideally two people review independently; rows where you disagree should become `disputed` or be discussed.
6. Then run: `python src/evaluate_ambiguity.py --build-verified` — this writes `data/ambiguous_human_verified.csv` with only verified rows (reviewed labels taking priority) and re-runs the evaluation on it.

## Known properties of the set
- 500 rows; 158 contain emojis; 161 rows in 33 paired groups (130 same-text/different-emoji, 31 same-emoji/different-text).
- Sentiment: 264 negative, 154 positive, 82 neutral. Emotion: joy 121, sadness 111, anger 108, fear 44, surprise 26, love 26, no valid class 64. Love and surprise are small, so their per-class scores are noisy.
- 256 rows carry at least one review flag; 106 are rated Hard.
- The style is short, social-media-like and mostly not "i feel …" — very different from the training data (96.7% of training texts start with "i"/"im"). Results on this set are therefore **not directly comparable** to the original test-set scores.
