"""Streamlit demo: predict emotion and sentiment of a social-media text.

Run from the project root:  streamlit run src/app.py
Needs the models created by train_baselines.py and train_sarcasm_emotion.py (models/).
Emotion and sarcasm come from the transformer (src/transformer_predictor.py); sentiment
and the emoji table come from the SVM predictor (src/predictor.py).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transformer_predictor import HybridPredictor  # noqa: E402

EMOTION_ICONS = {"anger": "😡", "fear": "😨", "joy": "😄", "love": "❤️", "neutral": "😐", "sadness": "😢", "surprise": "😲"}
SENTIMENT_ICONS = {"negative": "🙁", "neutral": "😐", "positive": "🙂"}
EXAMPLES = [
    "i feel so lonely and miserable today",
    "i am so happy and excited about my new job",
    "i cant believe how rude he was i feel furious",
    "just finished my exams 🎉",
]
LOW_CONFIDENCE = 0.50  # display hint only: below this the transformer's top emotion is a weak guess


@st.cache_resource(show_spinner="Loading models...")
def load_predictor():
    return HybridPredictor()


st.set_page_config(page_title="Emotion & Sentiment Analyzer", page_icon="💬")
st.title("💬 Emotion & Sentiment Analyzer")
st.caption("Type a social-media style post (emojis welcome) and the model predicts its emotion and sentiment.")

try:
    predictor = load_predictor()
except Exception as err:  # missing files or scikit-learn version mismatch
    st.error(
        "Could not load the trained models. Run `python src/preprocessing.py`, "
        "`python src/train_baselines.py` and `python src/train_sarcasm_emotion.py --base "
        "cardiffnlp/twitter-roberta-base-2022-154m` from the project root, then restart the app."
    )
    st.exception(err)
    st.stop()

if "text" not in st.session_state:
    st.session_state.text = ""

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for i, example in enumerate(EXAMPLES):
    if cols[i].button(f"Example {i + 1}", use_container_width=True):
        st.session_state.text = example

text = st.text_area("Your text", key="text", height=140, placeholder="e.g. finally got the results 😁")

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        r = predictor.predict(text)

        left, right = st.columns(2)
        left.metric("Emotion", f"{EMOTION_ICONS[r['emotion']]} {r['emotion'].title()}")
        right.metric("Sentiment", f"{SENTIMENT_ICONS[r['sentiment']]} {r['sentiment'].title()}")
        if r["sarcasm_flagged"]:
            note = f"Possible sarcasm ({r['sarcasm_probability']:.0%} likely). "
            if r["emotion_before_sarcasm"] != r["emotion"]:
                note += (
                    f"Emotion changed from {r['emotion_before_sarcasm'].title()} to "
                    f"{r['emotion'].title()} because sarcastic praise rarely means it. "
                )
            st.info(note + "Sentiment comes from a separate model and does not account for sarcasm.")

        for col, task in [(left, "emotion"), (right, "sentiment")]:
            scores = pd.Series(r[f"{task}_scores"]).mul(100).round(1).sort_values(ascending=False)
            col.bar_chart(scores, horizontal=True)

        with st.expander("How was this decided?", expanded=True):
            if r["emojis_used"]:
                emoji_list = ", ".join(f"{e} = {emo}" for e, emo in r["emojis_used"])
                st.write(f"**Emojis recognised:** {emoji_list}")
            else:
                st.write("**Emojis recognised:** none")
            st.write(
                f"**Words only:** emotion = {r['emotion_from_text']} "
                f"(confidence {r['emotion_text_confidence']:.0%}), "
                f"sentiment = {r['sentiment_from_text']} "
                f"(confidence {r['sentiment_text_confidence']:.0%})"
            )
            if r["known_words"] == 0 and r["emojis_used"]:
                st.write("No known words, so the emojis decided on their own.")
            elif r["emojis_used"]:
                st.write(
                    "Final result = words + emojis combined. The emoji decides when the words "
                    "show no clear emotion; strongly emotional words can overrule it."
                )

        if r["known_words"] == 0 and not r["emojis_used"]:
            st.info("None of these words or emojis were recognised, so the prediction is unreliable.")
        elif not r["emojis_used"] and r["emotion_text_confidence"] < LOW_CONFIDENCE:
            st.info("The model is unsure about the emotion in these words, so treat it as a weak guess.")

st.divider()
st.caption(
    "Models: emotion and sarcasm use one Twitter-based transformer trained on human-labelled "
    "comments and tweets; sentiment uses TF-IDF + Linear SVM. Only very confident sarcasm is "
    "flagged, so many sarcastic posts are missed. Emojis are handled by a rule table combined with the "
    "model scores. Sentiment labels in the SVM's training data came from a word-list heuristic, "
    "so treat sentiment as indicative only. Scores are relative, not exact probabilities."
)
