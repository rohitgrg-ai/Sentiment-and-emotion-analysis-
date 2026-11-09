"""Emotion and sarcasm from the Twitter-based transformer, sentiment from the existing SVM.

The emotion score combines three steps, all defined in train_sarcasm_emotion.py:
    1. Transformer emotion probabilities.
    2. Emoji fusion: (1 - w) * p_transformer + w * p_emoji_vote. Emoji with no words uses the vote alone.
    3. Sarcasm rule: if P(sarcastic) >= APP_SARCASM_THRESHOLD and the top emotion is joy or love,
       take the best emotion other than joy, love or neutral.

Neutral stays available here for text that is not flagged. The ambiguous-set evaluation hid
neutral on every row because its labels never use it.
There is no sentiment transformer yet, so sentiment still comes from Predictor and ignores sarcasm.
"""
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

from predictor import EMOJI_WEIGHT, MODELS, Predictor
from train_sarcasm_emotion import (EMOTION_ID, EMOTIONS, POSITIVE, EmotionSarcasmModel,
                                   predict as model_probabilities, with_emojis)

TRANSFORMER_DIR = "sarcasm_emotion"
# Lowest threshold costing at most 0.005 validation emotion macro-F1. The sarcasm head only saw
# tweets in training, so at 0.5 it flags about a quarter of ordinary comments and hurts emotion.
APP_SARCASM_THRESHOLD = 0.998
DEVICE = torch.device("cpu")
INFERENCE = SimpleNamespace(batch_size=32, max_len=64)  # max_len matches training
EXCLUDED_WHEN_SARCASTIC = POSITIVE + [EMOTION_ID["neutral"]]


class HybridPredictor(Predictor):
    def __init__(self, models_dir=MODELS, emoji_weight=EMOJI_WEIGHT):
        super().__init__(models_dir, emoji_weight)
        model_dir = models_dir / TRANSFORMER_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir), add_pooling_layer=False)
        self.transformer = EmotionSarcasmModel(encoder)
        load_model(self.transformer, str(model_dir / "model.safetensors"))
        self.transformer.to(DEVICE).eval()

    def probabilities(self, texts):
        """Emotion probabilities and P(sarcastic) for each text, before emojis or the sarcasm rule."""
        return model_probabilities(self.transformer, self.tokenizer, list(texts), DEVICE, INFERENCE)

    @staticmethod
    def apply_sarcasm_rule(scores, p_sarcastic):
        """Return the scores to report: unchanged, or renormalised without the excluded emotions."""
        if p_sarcastic < APP_SARCASM_THRESHOLD or int(np.argmax(scores)) not in POSITIVE:
            return scores
        adjusted = scores.copy()
        adjusted[EXCLUDED_WHEN_SARCASTIC] = 0.0
        return adjusted / adjusted.sum()

    def predict(self, text):
        result = super().predict(text)
        p_text, p_sarcastic = self.probabilities([str(text)])
        p_emoji = with_emojis([str(text)], p_text)[0]
        p_final = self.apply_sarcasm_rule(p_emoji, float(p_sarcastic[0]))
        result.update(
            emotion=EMOTIONS[int(np.argmax(p_final))],
            emotion_from_text=EMOTIONS[int(np.argmax(p_text[0]))],
            emotion_before_sarcasm=EMOTIONS[int(np.argmax(p_emoji))],
            emotion_scores=dict(zip(EMOTIONS, p_final)),
            emotion_text_confidence=float(p_text[0].max()),
            sarcasm_probability=float(p_sarcastic[0]),
            sarcasm_flagged=bool(p_sarcastic[0] >= APP_SARCASM_THRESHOLD),
        )
        return result
