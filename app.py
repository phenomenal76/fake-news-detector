"""Streamlit web app.  Run:  streamlit run app.py"""
import streamlit as st

from src.predict import BASELINE_PATH, BaselineDetector, BertDetector, bert_available
from src.train_baseline import train

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News & Misinformation Detector")
st.caption("Paste a headline or article. The model looks at *writing patterns* and tells you how "
           "likely it is to be misleading. It is a helper, not a fact-checker.")


@st.cache_resource
def get_baseline():
    if not BASELINE_PATH.exists():
        with st.spinner("No trained model found - training a quick demo model on the sample data..."):
            train(quiet=True)
    return BaselineDetector()


@st.cache_resource
def get_bert():
    return BertDetector()


options = ["TF-IDF + Logistic Regression"]
if bert_available():
    options.append("BERT (fine-tuned)")
choice = st.sidebar.radio("Model", options)
st.sidebar.info("Train on the full dataset for real accuracy - see the README.")

examples = {
    "- pick an example -": "",
    "Sensational": "SHOCKING!! Doctors HATE this one secret miracle cure the government is hiding from you!!!",
    "Normal report": "The central bank kept interest rates unchanged on Wednesday, saying inflation was easing, according to a statement from the bank.",
}
pick = st.selectbox("Try an example", list(examples))
text = st.text_area("Headline or article text", value=examples[pick], height=180)

if st.button("Analyse", type="primary"):
    if len(text.split()) < 3:
        st.warning("Please enter at least a short headline.")
        st.stop()

    detector = get_bert() if choice.startswith("BERT") else get_baseline()
    res = detector.predict(text)

    if res["fake_prob"] >= 0.5:
        st.error(f"⚠️ {res['verdict']}")
    else:
        st.success(f"✅ {res['verdict']}")

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
    c2.metric("Chance it is misleading", f"{res['fake_prob'] * 100:.1f}%")
    st.progress(res["fake_prob"])

    st.subheader("Supporting evidence")
    if "words_pointing_to_fake" in res:
        a, b = st.columns(2)
        a.markdown("**Words pushing towards FAKE**")
        for w, v in res["words_pointing_to_fake"] or []:
            a.write(f"🔴 `{w}`  ({v:+.3f})")
        if not res["words_pointing_to_fake"]:
            a.write("None found")
        b.markdown("**Words pushing towards REAL**")
        for w, v in res["words_pointing_to_real"] or []:
            b.write(f"🟢 `{w}`  ({v:+.3f})")
        if not res["words_pointing_to_real"]:
            b.write("None found")

    st.markdown("**Red flags in the writing style**")
    if res["red_flags"]:
        for f in res["red_flags"]:
            st.write(f"🚩 {f}")
    else:
        st.write("No obvious red flags.")
    st.caption(f"Model used: {res['model']}")
