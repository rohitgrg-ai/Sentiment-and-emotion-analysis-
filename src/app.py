"""Streamlit demo: predict the emotion, sentiment and possible sarcasm of a social-media post.

Run from the project root:  python -m streamlit run src/app.py
Needs the models created by train_baselines.py and train_sarcasm_emotion.py (models/).
Emotion and sarcasm come from the transformer (src/transformer_predictor.py); sentiment
and the emoji table come from the SVM predictor (src/predictor.py).
Page styling lives in src/app_style.css.
"""
import sys
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transformer_predictor import HybridPredictor  # noqa: E402

INK = "#15213b"
WHITE = "#ffffff"
# background colour, text colour on that background (each pair meets WCAG AA contrast)
EMOTION_COLOURS = {
    "anger": ("#d62828", WHITE),
    "fear": ("#6a2cb8", WHITE),
    "joy": ("#ffc233", INK),
    "love": ("#ff7aa8", INK),
    "neutral": ("#c3c9d8", INK),
    "sadness": ("#2563d9", WHITE),
    "surprise": ("#17c7b2", INK),
}
# highlight, main and shade for each result page; the text colour passes AA against all three
EMOTION_GRADIENTS = {
    "anger": ("#c81e5b", "#d62828", "#8a1238"),
    "fear": ("#8b3fd1", "#6a2cb8", "#2f1a78"),
    "joy": ("#ffe066", "#ffc233", "#ff9a3c"),
    "love": ("#ffb3a7", "#ff7aa8", "#f25c9c"),
    "neutral": ("#eef1f7", "#c3c9d8", "#a9b3c9"),
    "sadness": ("#1f6fd6", "#2563d9", "#12307f"),
    "surprise": ("#7cf0a6", "#17c7b2", "#12a8c9"),
}
SENTIMENT_COLOURS = {"negative": INK, "neutral": "#8d97b0", "positive": "#c3cce0"}
# Each post was checked against the model: it returns the emotion it is listed under.
EXAMPLES = {
    "anger": "I just love it when my phone dies at 2%",
    "joy": "I got the job!!!",
    "sadness": "i miss my grandma every single day",
    "love": "you are the sweetest person i know ❤️",
    "surprise": "wait, WHAT? i did not see that coming",
    "fear": "my hands won't stop shaking before this interview",
    "neutral": "The meeting is at 3pm on Tuesday.",
}
LOW_CONFIDENCE = 0.50  # display hint only: below this the transformer's top emotion is a weak guess
FINE_PRINT = (
    "Sarcasm flags can be wrong, and many sarcastic posts are missed. Sentiment comes from an "
    "older word-based model, so treat it as a rough guide. Percentages are relative scores."
)


@st.cache_resource(show_spinner="Loading models...")
def load_predictor():
    return HybridPredictor()


def read_post(text):
    st.session_state.reading = (text, load_predictor().predict(text))


def read_typed_post():
    text = st.session_state.text.strip()
    st.session_state.missing_text = not text
    if text:
        read_post(text)


def read_example(example):
    st.session_state.text = example
    st.session_state.missing_text = False
    read_post(example)


def start_over():
    st.session_state.text = ""
    st.session_state.reading = None


def apply_styles(reading):
    css = (ROOT / "src" / "app_style.css").read_text(encoding="utf-8")
    page = ""
    if reading:
        emotion = reading[1]["emotion"]
        background, foreground = EMOTION_COLOURS[emotion]
        highlight, main, shade = EMOTION_GRADIENTS[emotion]
        wash = f"radial-gradient(130% 120% at 88% 0%, {highlight} 0%, {main} 48%, {shade} 100%)"
        page = f".stApp {{ --page: {background}; --on: {foreground}; --wash: {wash}; }}"
    st.markdown(f"<style>{' '.join(css.split())} {page}</style>", unsafe_allow_html=True)


def sarcasm_html(result):
    if not result["sarcasm_flagged"]:
        return ""
    likely = f"This looks sarcastic ({result['sarcasm_probability']:.0%} likely)."
    if result["emotion_before_sarcasm"] != result["emotion"]:
        likely = (
            f"This reads like {result['emotion_before_sarcasm']}, but it looks sarcastic "
            f"({result['sarcasm_probability']:.0%} likely), so the emotion moved to {result['emotion']}."
        )
    return f'<p class="sarcasm">{likely}<small>Sentiment does not account for sarcasm.</small></p>'


def reliability_html(result):
    note = ""
    if result["known_words"] == 0 and not result["emojis_used"]:
        note = "These words and emojis were not recognised, so this result is unreliable."
    elif not result["emojis_used"] and result["emotion_text_confidence"] < LOW_CONFIDENCE:
        note = "The model is unsure about the emotion in these words, so treat it as a weak guess."
    return f'<p class="note">{note}</p>' if note else ""


def result_band(text, result):
    return (
        '<section class="band">'
        f'<blockquote class="post">{escape(text)}</blockquote>'
        f'<h2 class="verdict">{result["emotion"].title()}</h2>'
        f'<p class="sentiment">Sentiment: {result["sentiment"]}</p>'
        f"{sarcasm_html(result)}{reliability_html(result)}</section>"
    )


def mix_html(title, scores, colours):
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    segments = "".join(
        f'<span style="width:{value * 100:.2f}%;background:{colours[name]}"></span>'
        for name, value in ranked if value >= 0.005
    )
    legend = "".join(
        f'<li><i style="background:{colours[name]}"></i>{name.title()}<b>{value:.0%}</b></li>'
        for name, value in ranked
    )
    return (
        f'<section class="mix"><h3>{title}</h3><div class="bar" aria-hidden="true">{segments}</div>'
        f'<ul class="legend">{legend}</ul></section>'
    )


def show_explanation(result):
    with st.expander("How this was decided"):
        if result["emojis_used"]:
            found = ", ".join(f"{emoji} = {emotion}" for emoji, emotion in result["emojis_used"])
            st.write(f"**Emojis recognised:** {found}")
        else:
            st.write("**Emojis recognised:** none")
        st.write(
            f"**Words only:** emotion = {result['emotion_from_text']} "
            f"(confidence {result['emotion_text_confidence']:.0%}), "
            f"sentiment = {result['sentiment_from_text']} "
            f"(confidence {result['sentiment_text_confidence']:.0%})"
        )
        if result["known_words"] == 0 and result["emojis_used"]:
            st.write("With no known words, the emojis decided on their own.")
        elif result["emojis_used"]:
            st.write("The final result combines the words and the emojis. Strongly emotional words can overrule an emoji.")


def show_reading(text, result):
    st.html(result_band(text, result))
    st.button("Try another post", key="again", on_click=start_over)
    with st.container(key="sheet"):
        emotion_colours = {name: pair[0] for name, pair in EMOTION_COLOURS.items()}
        st.html(mix_html("Emotion mix", result["emotion_scores"], emotion_colours))
        st.html(mix_html("Sentiment mix", result["sentiment_scores"], SENTIMENT_COLOURS))
        show_explanation(result)


def show_wall():
    with st.container(key="wall"):
        for emotion, example in EXAMPLES.items():
            st.button(example, key=f"ex_{emotion}", on_click=read_example, args=(example,))


st.set_page_config(page_title="What is this post really saying?", page_icon="💬", layout="wide")
st.session_state.setdefault("text", "")
st.session_state.setdefault("reading", None)
apply_styles(st.session_state.reading)

try:
    load_predictor()
except Exception as err:  # missing files or scikit-learn version mismatch
    st.error(
        "The trained models could not be loaded. Run `python src/preprocessing.py`, "
        "`python src/train_baselines.py` and `python src/train_sarcasm_emotion.py --base "
        "cardiffnlp/twitter-roberta-base-2022-154m` from the project root, then restart the app."
    )
    st.exception(err)
    st.stop()

left, right = st.columns([5, 7], gap="large")
with left:
    st.markdown(
        '<header class="masthead"><h1>What is this post really saying?</h1>'
        "<p>Paste a post, or pick one from the wall. Emojis count.</p></header>",
        unsafe_allow_html=True,
    )
    st.text_area("Your post", key="text", height=120, placeholder="finally got the results 😁")
    st.button("Read this post", key="read", on_click=read_typed_post)
    if st.session_state.pop("missing_text", False):
        st.warning("Write or paste a post first.")
    st.markdown(f'<p class="fineprint">{FINE_PRINT}</p>', unsafe_allow_html=True)

with right:
    if st.session_state.reading:
        show_reading(*st.session_state.reading)
    else:
        show_wall()
