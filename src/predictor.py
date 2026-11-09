"""Prediction logic shared by the Streamlit app and the tests.

How a prediction is made ("late fusion" of text and emoji):
1. Emojis listed in EMOJI_EMOTION are taken out of the text and counted.
2. The remaining text goes through clean_text -> TF-IDF -> Linear SVM. The SVM
   scores are turned into relative confidences with a softmax (p_text).
3. The emojis give their own vote (p_emoji): e.g. one 😄 = 100% joy / positive.
4. Final = (1 - w) * p_text + w * p_emoji, with w = EMOJI_WEIGHT.
   - Text with no clear emotion (low confidence) -> the emoji decides.
   - Strongly emotional text -> the text wins, the emoji cannot overrule it.
   - Text with no known words (e.g. emoji only) -> the emoji alone decides.
Why: the training data contains no emojis (the label-coded emoji suffixes were
removed to prevent leakage), so the SVM cannot read emojis by itself.
"""
from pathlib import Path

import joblib
import numpy as np

from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
# Share of the final vote given to emojis. The sentiment SVM is more confident on
# its own than the emotion SVM, so it needs a slightly larger emoji share.
EMOJI_WEIGHT = {"emotion": 0.25, "sentiment": 0.30}

_EMOJI_GROUPS = {
    "joy": "😄😀😁😃😊😂🤣😆🥳🎉😎😌🙂☺😸😹🤩😇😋👍🔥",
    "love": "❤😍🥰😘💕💖💗💓💞😻🤗💛💙💚💜🧡💘😚😙",
    "anger": "😡😠🤬👿💢😤😾🙄😒🖕",
    "fear": "😨😰😱😧😬🙀😖😣",
    "sadness": "😢😭😞😔😟😿🥺☹🙁💔😩😫😥😓😪🥀",
    "surprise": "😲😮😯🤯😳😵🫢😦",
}
EMOJI_EMOTION = {ch: emo for emo, chars in _EMOJI_GROUPS.items() for ch in chars}
EMOTION_TO_SENTIMENT = {"joy": "positive", "love": "positive", "anger": "negative",
                        "fear": "negative", "sadness": "negative", "surprise": "neutral"}


def _softmax(scores):
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


class Predictor:
    def __init__(self, models_dir=MODELS, emoji_weight=EMOJI_WEIGHT):
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
        self.emotion_model = joblib.load(models_dir / "emotion_linear_svm.joblib")
        self.sentiment_model = joblib.load(models_dir / "sentiment_linear_svm.joblib")
        self.emoji_weight = emoji_weight

    @staticmethod
    def split_emojis(text):
        """Return (text without mapped emojis, list of (emoji, emotion))."""
        found, kept = [], []
        for ch in str(text):
            if ch in EMOJI_EMOTION:
                found.append((ch, EMOJI_EMOTION[ch]))
                kept.append(" ")
            else:
                kept.append(ch)
        return "".join(kept).replace("️", ""), found

    def _fuse(self, task, model, features, emoji_labels):
        classes = list(model.classes_)
        p_text = _softmax(model.decision_function(features)[0])
        if not emoji_labels:
            return classes, p_text, p_text, None
        p_emoji = np.array([emoji_labels.count(c) for c in classes], dtype=float)
        p_emoji /= p_emoji.sum()
        w = 1.0 if features.nnz == 0 else self.emoji_weight[task]
        return classes, p_text, (1 - w) * p_text + w * p_emoji, p_emoji

    def predict(self, text):
        plain, emojis = self.split_emojis(text)
        features = self.vectorizer.transform([clean_text(plain)])
        emotions = [emo for _, emo in emojis]
        sentiments = [EMOTION_TO_SENTIMENT[emo] for emo in emotions]
        result = {"emojis_used": emojis, "known_words": int(features.nnz)}
        for task, model, labels in [("emotion", self.emotion_model, emotions),
                                    ("sentiment", self.sentiment_model, sentiments)]:
            classes, p_text, p_final, p_emoji = self._fuse(task, model, features, labels)
            result[task] = classes[int(np.argmax(p_final))]
            result[f"{task}_from_text"] = classes[int(np.argmax(p_text))]
            result[f"{task}_scores"] = dict(zip(classes, p_final))
            result[f"{task}_text_confidence"] = float(p_text.max())
        return result
