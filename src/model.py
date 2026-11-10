"""The emotion and sarcasm model, and the prediction logic used by the app."""
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "emotion_sarcasm"

EMOTIONS = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
EMOTION_ID = {name: i for i, name in enumerate(EMOTIONS)}
EMOTION_SENTIMENT = {"anger": "negative", "fear": "negative", "sadness": "negative", "joy": "positive",
                     "love": "positive", "surprise": "neutral", "neutral": "neutral"}
POSITIVE = [EMOTION_ID["joy"], EMOTION_ID["love"]]
EXCLUDED_WHEN_SARCASTIC = POSITIVE + [EMOTION_ID["neutral"]]
MAX_TOKENS = 64
SARCASM_THRESHOLD = 0.90  # lowest threshold costing at most 0.005 validation emotion macro-F1
EMOJI_WEIGHT = 0.25
EMOJI_EMOTION = {char: emotion for emotion, chars in {
    "joy": "😄😀😁😃😊😂🤣😆🥳🎉😎😌🙂☺😸😹🤩😇😋👍🔥",
    "love": "❤😍🥰😘💕💖💗💓💞😻🤗💛💙💚💜🧡💘😚😙",
    "anger": "😡😠🤬👿💢😤😾🙄😒🖕",
    "fear": "😨😰😱😧😬🙀😖😣",
    "sadness": "😢😭😞😔😟😿🥺☹🙁💔😩😫😥😓😪🥀",
    "surprise": "😲😮😯🤯😳😵🫢😦",
}.items() for char in chars}


class EmotionSarcasmModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.emotion_head = nn.Linear(encoder.config.hidden_size, len(EMOTIONS))
        self.sarcasm_head = nn.Linear(encoder.config.hidden_size, 2)

    def forward(self, **encoded):
        cls = self.dropout(self.encoder(**encoded).last_hidden_state[:, 0])
        return self.emotion_head(cls), self.sarcasm_head(cls)


def normalise(text):
    text = re.sub(r"https?://\S+|www\.\S+", "http", str(text))
    return re.sub(r"\s+", " ", re.sub(r"@\w+", "@user", text)).strip()


def load_trained(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir), add_pooling_layer=False)
    model = EmotionSarcasmModel(encoder)
    load_model(model, str(model_dir / "model.safetensors"))
    return model.eval(), tokenizer


@torch.no_grad()
def probabilities(model, tokenizer, texts, device=torch.device("cpu"), batch_size=64):
    """Emotion probabilities and P(sarcastic) for each text."""
    model.eval()
    emotion, sarcasm = [], []
    for start in range(0, len(texts), batch_size):
        batch = [normalise(text) for text in texts[start:start + batch_size]]
        encoded = tokenizer(batch, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt").to(device)
        emotion_logits, sarcasm_logits = model(**encoded)
        emotion.append(torch.softmax(emotion_logits, -1).float().cpu().numpy())
        sarcasm.append(torch.softmax(sarcasm_logits, -1)[:, 1].float().cpu().numpy())
    return np.concatenate(emotion), np.concatenate(sarcasm)


def blend_emojis(text, scores):
    """Mix the model's scores with a vote from known emojis. Emojis decide alone when there are no words."""
    found = [EMOJI_EMOTION[char] for char in str(text) if char in EMOJI_EMOTION]
    if not found:
        return scores
    vote = np.array([found.count(name) for name in EMOTIONS], dtype=float) / len(found)
    words = "".join(" " if char in EMOJI_EMOTION else char for char in str(text))
    weight = EMOJI_WEIGHT if re.search(r"[A-Za-z0-9]", words) else 1.0
    return (1 - weight) * scores + weight * vote


def apply_sarcasm_rule(scores, p_sarcastic):
    """Sarcastic joy or love rarely means it, so take the best emotion other than joy, love or neutral."""
    if p_sarcastic < SARCASM_THRESHOLD or int(np.argmax(scores)) not in POSITIVE:
        return scores
    adjusted = scores.copy()
    adjusted[EXCLUDED_WHEN_SARCASTIC] = 0.0
    return adjusted / adjusted.sum()


class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model, self.tokenizer = load_trained(model_dir)

    def predict(self, text):
        text = str(text)
        emotion_probs, sarcasm_probs = probabilities(self.model, self.tokenizer, [text])
        from_text, p_sarcastic = emotion_probs[0], float(sarcasm_probs[0])
        with_emojis = blend_emojis(text, from_text)
        final = apply_sarcasm_rule(with_emojis, p_sarcastic)
        emotion = EMOTIONS[int(np.argmax(final))]
        sentiment_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for name, score in zip(EMOTIONS, final):
            sentiment_scores[EMOTION_SENTIMENT[name]] += float(score)
        return {
            "emotion": emotion,
            "emotion_from_text": EMOTIONS[int(np.argmax(from_text))],
            "emotion_before_sarcasm": EMOTIONS[int(np.argmax(with_emojis))],
            "emotion_scores": dict(zip(EMOTIONS, final)),
            "text_confidence": float(from_text.max()),
            "sentiment": EMOTION_SENTIMENT[emotion],
            "sentiment_scores": sentiment_scores,
            "sarcasm_probability": p_sarcastic,
            "sarcasm_flagged": p_sarcastic >= SARCASM_THRESHOLD,
            "emojis_used": [(char, EMOJI_EMOTION[char]) for char in text if char in EMOJI_EMOTION],
        }
