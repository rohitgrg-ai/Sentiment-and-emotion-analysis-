# Plan: pretrained and fine-tuned transformers (Phase 3)

Goal: test whether a pretrained transformer handles ambiguous social-media text (emojis, sarcasm, mixed feelings) better than the TF-IDF Linear SVM, which stays as the baseline.

## Rules that carry over
- The 500-row ambiguous set is **test only**. Never train, tune thresholds or pick models on it.
- Keep the existing TF-IDF models, `predictor.py` and all results unchanged.
- Labels stay "AI-suggested" until a human reviews them; say so in every report.
- Report any comparison on the same rows, with the McNemar test, before claiming a win.

## Step 1: Zero-shot (script ready: `src/transformer_zero_shot.py`)
Pretrained models used as-is, no training. Compares T (transformer) with A and D on the 500 rows.
Known limits to report: the emotion model has no `love` class and predicts `neutral`, which the gold labels never use.
Run: `pip install transformers torch` then `python src/transformer_zero_shot.py` (try `--limit 40` first).

## Step 2: Fine-tune (only if Step 1 shows promise or you want the full pipeline)
1. **Training data:** a separate human-labelled, natural-text dataset (e.g. GoEmotions or TweetEval emotion).
   Map its labels to the project's six emotions; decide explicitly what to do with neutral (keep a 7th class
   in training, and map or drop it at evaluation, and say which).
2. **No overlap:** drop any training text that also appears in the 500-row set (exact match after lowercasing).
3. **Model:** a small pretrained model (distilroberta-base, or a twitter-roberta/BERTweet checkpoint).
4. **Setup:** max_length 128, 3 epochs, lr 2e-5, batch 16-32, fixed seed 42; hold out a validation split of the
   *training* data for early stopping. Run on free Colab GPU.
5. **Evaluate once** on the 500 rows. Report strict and lenient emotion accuracy, sentiment accuracy, macro-F1,
   and the paired-emoji sensitivity/stability test (reuse `evaluate_paired_emoji.py` logic).

## Step 3: Optional fallback
Use TF-IDF when its confidence is high, otherwise the transformer. Pick the confidence threshold on a
validation split of the training data, never on the 500 rows. Treat as an experiment: TF-IDF is confidently
wrong on sarcasm, so expect limited gain.

## What to write in the viva
- Why TF-IDF cannot see context, and what the transformer adds (pretrained on large text, uses word order).
- The fair-test rules above.
- Honest limits: AI-suggested labels, 500 rows, emotion label mismatch with pretrained models.
