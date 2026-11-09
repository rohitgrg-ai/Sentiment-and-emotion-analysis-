"""'Advanced Emoji & Ambiguity Testing' page for the Streamlit app (src/app.py).

Shows, side by side, what every system predicts for one text:
  A  existing text-only SVM            D  existing predictor.py (text + emoji voting)
  B / C  experimental emoji-aware SVMs (original data)   B' / C'  (synthetic emoji-augmented data)
Confidence columns are labelled honestly: A and D give a softmax of SVM decision scores
(uncalibrated); B/C give sigmoid-calibrated probabilities.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emoji_features import emoji_name, extract_emojis  # noqa: E402
from predictor import EMOJI_EMOTION, Predictor  # noqa: E402

EXP_NAMES = {
    "B_orig": "B · demojized TF-IDF (original data)",
    "C_orig": "C · hybrid text+emoji (original data)",
    "B_aug": "B′ · demojized TF-IDF (synthetic emoji data)",
    "C_aug": "C′ · hybrid text+emoji (synthetic emoji data)",
}
QUICK_EMOJIS = ["😊", "🙂", "😂", "😭", "💀", "🙄", "🙃", "😡", "😱", "❤️", "👍", "🔥", "🔪"]


@st.cache_resource
def _predictor():
    return Predictor()


@st.cache_resource
def _experimental():
    from experimental_models import VARIANTS, load_variant
    return {v: load_variant(v) for v in VARIANTS}


@st.cache_data
def _dataset():
    path = ROOT / "data" / "ambiguous_social_media_test.csv"
    return pd.read_csv(path, keep_default_na=False, dtype=str) if path.exists() else None


def _append_emoji(e):
    st.session_state.adv_text = (st.session_state.get("adv_text", "") + " " + e).strip()


def _load_example():
    row = st.session_state.get("adv_example_row")
    if row is not None:
        st.session_state.adv_text = row["text"]


def predict_all(text):
    """One row per system: prediction, confidence and how that confidence was computed."""
    r = _predictor().predict(text)
    rows = [
        {"system": "A · text-only SVM (existing)", "emotion": r["emotion_from_text"],
         "emotion score": r["emotion_text_confidence"], "sentiment": r["sentiment_from_text"],
         "sentiment score": r["sentiment_text_confidence"], "score type": "softmax of SVM scores"},
        {"system": "D · predictor.py, text + emoji voting (existing)", "emotion": r["emotion"],
         "emotion score": max(r["emotion_scores"].values()), "sentiment": r["sentiment"],
         "sentiment score": max(r["sentiment_scores"].values()), "score type": "softmax + emoji vote"},
    ]
    explanations = {"D": r}
    try:
        models = _experimental()
    except FileNotFoundError:
        return pd.DataFrame(rows), explanations, False
    for key, bundle in models.items():
        row = {"system": EXP_NAMES[key], "score type": "calibrated probability"}
        for task in ["emotion", "sentiment"]:
            m = bundle[task]
            proba = m.predict_proba([text])[0]
            row[task] = m.predict([text])[0]
            row[f"{task} score"] = float(proba.max())
            explanations[f"{key}_{task}"] = (m, dict(zip(m.classes_, proba)))
        rows.append(row)
    return pd.DataFrame(rows), explanations, True


def _contribution_text(key, model, text, label):
    c = model.contributions(text)
    if key.startswith("C"):
        t, e = c["text"][label], c["emoji"][label]
        return f"score for **{label}**: text part {t:+.2f}, emoji part {e:+.2f} (exact split of a linear model)"
    effect = c["emoji_effect"][label]
    return (f"the emojis change the score for **{label}** by {effect:+.2f} "
            "(prediction with emojis minus prediction with emojis removed)")


def render():
    st.header("🧪 Advanced Emoji & Ambiguity Testing")
    st.caption("Compare the existing system with the experimental emoji-aware models on tricky text: "
               "sarcasm, slang, mixed feelings and emojis that change the meaning.")

    data = _dataset()
    if data is not None:
        with st.expander("Load an example from the ambiguous test set"):
            cats = ["(all)"] + sorted(data["category"].unique())
            cat = st.selectbox("Category", cats)
            sub = data if cat == "(all)" else data[data["category"] == cat]
            choice = st.selectbox("Example", sub.index, format_func=lambda i: f"{sub.at[i, 'id']}  {sub.at[i, 'text']}")
            st.session_state.adv_example_row = sub.loc[choice].to_dict()
            st.button("Use this example", on_click=_load_example)

    st.write("Quick-insert an emoji (or type/paste any emoji yourself):")
    cols = st.columns(len(QUICK_EMOJIS))
    for col, e in zip(cols, QUICK_EMOJIS):
        col.button(e, key=f"adv_emoji_{e}", on_click=_append_emoji, args=(e,), use_container_width=True)
    text = st.text_area("Text to analyse", key="adv_text", height=110, placeholder="e.g. Great job 👍🙄")

    if not st.button("Compare all models", type="primary"):
        return
    if not text.strip():
        st.warning("Please enter some text first.")
        return

    table, expl, has_exp = predict_all(text)
    if not has_exp:
        st.info("Experimental models not found. Run `python src/experimental_models.py` to train them; "
                "showing the existing systems only.")

    example = st.session_state.get("adv_example_row")
    if example and example.get("text") == text:
        status = example.get("annotation_status", "")
        exp_emo = example.get("reviewed_emotion") or example.get("expected_emotion") or "(no valid class)"
        exp_sent = example.get("reviewed_sentiment") or example.get("expected_sentiment")
        st.write(f"**Expected** ({status.replace('_', ' ')}): emotion **{exp_emo}**, sentiment **{exp_sent}**  \n"
                 f"_{example.get('category', '').replace('_', ' ')} — {example.get('explanation', '')}_")
        table["emotion ✓"] = (table["emotion"] == exp_emo).map({True: "✅", False: "❌"})
        table["sentiment ✓"] = (table["sentiment"] == exp_sent).map({True: "✅", False: "❌"})

    shown = table.copy()
    for c in ["emotion score", "sentiment score"]:
        shown[c] = shown[c].map(lambda v: f"{v:.0%}")
    st.dataframe(shown, hide_index=True, use_container_width=True)

    n_emo, n_sent = table["emotion"].nunique(), table["sentiment"].nunique()
    if n_emo == 1 and n_sent == 1:
        st.success("All models agree.")
    else:
        parts = [f"{n} different {task}s" for n, task in [(n_emo, "emotion"), (n_sent, "sentiment")] if n > 1]
        st.warning("Models disagree: " + " and ".join(parts) + ".")

    with st.expander("Why do the models differ?", expanded=True):
        found = extract_emojis(text)
        r = expl["D"]
        if found:
            names = ", ".join(f"{e} = “{emoji_name(e)}”" for e in found)
            st.write(f"**Emojis found:** {names}")
            unknown = [e for e in found if not any(ch in EMOJI_EMOTION for ch in e)]
            if unknown:
                st.write(f"**Not in D's emoji table (D ignores them):** {' '.join(unknown)}")
        else:
            st.write("**No emojis** — every model relies on the words only, so differences come from training data.")
        st.write(f"**D:** words alone → {r['emotion_from_text']} / {r['sentiment_from_text']}; "
                 f"emoji table votes: {', '.join(f'{e}={emo}' for e, emo in r['emojis_used']) or 'none'}; "
                 f"final → {r['emotion']} / {r['sentiment']}.")
        if has_exp:
            for key in EXP_NAMES:
                m, _ = expl[f"{key}_emotion"]
                label = m.predict([text])[0]
                st.write(f"**{EXP_NAMES[key].split(' · ')[0]}** emotion: " + _contribution_text(key, m, text, label))
        st.caption("A and C (original data) were trained on text with no emojis, so they cannot react to emojis. "
                   "B reads emoji *names* as words. B′/C′ learned emojis from synthetic examples, and D uses a "
                   "hand-made emoji table — both encode assumptions about what emojis mean, so they can over-trust "
                   "an emoji (e.g. 😭 used for happy tears). None of these models understands sarcasm.")
