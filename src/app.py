"""Streamlit app. Run from the project root:  python -m streamlit run src/app.py"""
from html import escape
from pathlib import Path

import streamlit as st

from model import Predictor

INK, WHITE = "#15213b", "#ffffff"
# Each emotion: main colour, text colour, gradient highlight, gradient shade. All text pairs meet WCAG AA.
PALETTE = {
    "anger": ("#d62828", WHITE, "#c81e5b", "#8a1238"),
    "fear": ("#6a2cb8", WHITE, "#8b3fd1", "#2f1a78"),
    "joy": ("#ffc233", INK, "#ffe066", "#ff9a3c"),
    "love": ("#ff7aa8", INK, "#ffb3a7", "#f25c9c"),
    "neutral": ("#c3c9d8", INK, "#eef1f7", "#a9b3c9"),
    "sadness": ("#2563d9", WHITE, "#1f6fd6", "#12307f"),
    "surprise": ("#17c7b2", INK, "#7cf0a6", "#12a8c9"),
}
SENTIMENT_COLOURS = {"negative": INK, "neutral": "#8d97b0", "positive": "#c3cce0"}
EXAMPLES = {
    "anger": "I just love it when my phone dies at 2%",
    "joy": "I got the job!!!",
    "sadness": "i miss my grandma every single day",
    "love": "you are the sweetest person i know ❤️",
    "surprise": "wait, WHAT? i did not see that coming",
    "fear": "my hands won't stop shaking before this interview",
    "neutral": "The meeting is at 3pm on Tuesday.",
}
FINE_PRINT = "Sarcasm flags can be wrong, and many sarcastic posts are missed. Sentiment follows from the emotion. Percentages are relative scores."


@st.cache_resource(show_spinner="Loading the model...")
def load_predictor():
    return Predictor()


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
    css = (Path(__file__).parent / "app_style.css").read_text(encoding="utf-8")
    page = ""
    if reading:
        main, text, highlight, shade = PALETTE[reading[1]["emotion"]]
        wash = f"radial-gradient(130% 120% at 88% 0%, {highlight} 0%, {main} 48%, {shade} 100%)"
        page = f".stApp {{ --page: {main}; --on: {text}; --wash: {wash}; }}"
    st.markdown(f"<style>{' '.join(css.split())} {page}</style>", unsafe_allow_html=True)


def result_html(text, result):
    extra = ""
    if result["sarcasm_flagged"]:
        chance = f"{result['sarcasm_probability']:.0%}"
        extra += f'<p class="sarcasm">This looks sarcastic ({chance} likely).</p>'
        if result["emotion_before_sarcasm"] != result["emotion"]:
            extra = f'<p class="sarcasm">This reads like {result["emotion_before_sarcasm"]}, but it looks sarcastic ({chance} likely), so the emotion moved to {result["emotion"]}.</p>'
    if not result["emojis_used"] and result["text_confidence"] < 0.5:
        extra += '<p class="note">The model is unsure about the emotion in these words, so treat it as a weak guess.</p>'
    return (f'<section class="band"><blockquote class="post">{escape(text)}</blockquote>'
            f'<h2 class="verdict">{result["emotion"].title()}</h2>'
            f'<p class="sentiment">Sentiment: {result["sentiment"]}</p>{extra}</section>')


def mix_html(title, scores, colours):
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    segments = "".join(f'<span style="width:{value * 100:.2f}%;background:{colours[name]}"></span>' for name, value in ranked if value >= 0.005)
    legend = "".join(f'<li><i style="background:{colours[name]}"></i>{name.title()}<b>{value:.0%}</b></li>' for name, value in ranked)
    return f'<section class="mix"><h3>{title}</h3><div class="bar" aria-hidden="true">{segments}</div><ul class="legend">{legend}</ul></section>'


def show_reading(text, result):
    st.html(result_html(text, result))
    st.button("Try another post", key="again", on_click=start_over)
    with st.container(key="sheet"):
        st.html(mix_html("Emotion mix", result["emotion_scores"], {name: colours[0] for name, colours in PALETTE.items()}))
        st.html(mix_html("Sentiment mix", result["sentiment_scores"], SENTIMENT_COLOURS))
        with st.expander("How this was decided"):
            found = ", ".join(f"{emoji} = {emotion}" for emoji, emotion in result["emojis_used"]) or "none"
            st.write(f"**Emojis recognised:** {found}")
            st.write(f"**Words only:** {result['emotion_from_text']} ({result['text_confidence']:.0%} confidence)")
            st.write(f"**Sarcasm:** {result['sarcasm_probability']:.0%} likely")


st.set_page_config(page_title="What is this post really saying?", page_icon="💬", layout="wide")
st.session_state.setdefault("text", "")
st.session_state.setdefault("reading", None)
apply_styles(st.session_state.reading)

try:
    load_predictor()
except OSError as err:
    st.error("The trained model is missing. Train it with `python src/train.py`, as described in the README, then restart the app.")
    st.exception(err)
    st.stop()

left, right = st.columns([5, 7], gap="large")
with left:
    st.markdown('<header class="masthead"><h1>What is this post really saying?</h1>'
                "<p>Paste a post, or pick one from the wall. Emojis count.</p></header>", unsafe_allow_html=True)
    st.text_area("Your post", key="text", height=120, placeholder="finally got the results 😁")
    st.button("Read this post", key="read", on_click=read_typed_post)
    if st.session_state.pop("missing_text", False):
        st.warning("Write or paste a post first.")
    st.markdown(f'<p class="fineprint">{FINE_PRINT}</p>', unsafe_allow_html=True)

with right:
    if st.session_state.reading:
        show_reading(*st.session_state.reading)
    else:
        with st.container(key="wall"):
            for emotion, example in EXAMPLES.items():
                st.button(example, key=f"ex_{emotion}", on_click=read_example, args=(example,))
