# Author: Lindsey Yang
# Modified: 2026-11-23
"""Group Theta sentiment-analysis app prototype; run with streamlit run app.py."""

from pathlib import Path

import streamlit as st

from sentiment import EXAMPLES, MODEL_FILES, get_demo_models, load_team_model, predict_review

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
st.set_page_config(page_title="Food Review Sentiment | Group Theta", page_icon="🍽️", layout="wide")


@st.cache_resource
def demo_models():
    return get_demo_models()


@st.cache_resource
def team_model(name, artifact_path, modified_ns, size):
    # File metadata is part of the cache key so replacing an artifact reloads it.
    return load_team_model(name, Path(artifact_path).parent)


def use_example():
    choice = st.session_state["example"]
    if choice in EXAMPLES:
        st.session_state["summary"] = ""
        st.session_state["review"] = EXAMPLES[choice]


st.title("Food Review Sentiment")
st.caption("CSE 3521 · Group Theta · Amazon Fine Food Reviews")

with st.sidebar:
    st.header("Models")
    mode = st.radio("Model source", ["Demo", "Team models"], key="mode")
    if mode == "Demo":
        available_models = list(demo_models())
    else:
        available_models = [
            name for name, filename in MODEL_FILES.items()
            if (MODELS_DIR / filename).is_file()
        ]
    selected = st.multiselect("Compare models", available_models, default=available_models,
                              key=f"selected_models_{mode}")
    st.divider()
    st.markdown("**Sentiment labels**")
    st.caption("Negative: 1–2 stars\n\nNeutral: 3 stars\n\nPositive: 4–5 stars")

if mode == "Demo":
    st.warning(
        "DEMO ONLY — Both models use 18 invented food reviews, not the Amazon dataset. "
        "Predictions demonstrate the interface and must not be reported as project results."
    )
elif available_models:
    st.info("Team models are fitted on Amazon Fine Food Reviews.")
else:
    st.warning("No trained team models are available. Add the fitted pipelines to models/.")

st.subheader("Try a review")
st.selectbox("Try an example", ["Write my own", *EXAMPLES], key="example", on_change=use_example)
summary = st.text_input("Summary (optional)", placeholder="Enter the review title…",
                        max_chars=10000, key="summary")
review = st.text_area("Review text", placeholder="Paste an English food review here…", height=150,
                      max_chars=10000, key="review")
st.caption("Summary and review text are joined with one space, up to 10,000 characters in total.")

if st.button("Analyze review", type="primary"):
    if not review.strip():
        st.warning("Enter a review before running a prediction.")
    elif not selected:
        st.warning("Select at least one model to compare.")
    else:
        review_input = f"{summary} {review}"
        st.subheader("Model predictions")
        for column, name in zip(st.columns(len(selected)), selected):
            with column:
                st.markdown(f"### {name}")
                try:
                    if mode == "Demo":
                        model = demo_models()[name]
                    else:
                        artifact = MODELS_DIR / MODEL_FILES[name]
                        if not artifact.is_file():
                            raise ValueError(f"Missing team artifact: models/{artifact.name}.")
                        metadata = artifact.stat()
                        model = team_model(name, str(artifact), metadata.st_mtime_ns, metadata.st_size)
                    result = predict_review(model, review_input)
                except Exception as error:
                    st.error(str(error) if isinstance(error, ValueError)
                             else "Prediction failed. Ask the model author to verify the exported pipeline.")
                    continue
                st.metric("Prediction", result["label"].title())
                if result["probabilities"] is None:
                    st.caption("Class probabilities are unavailable for this model.")
                else:
                    st.dataframe([
                        {"Sentiment": label.title(), "Model probability": f"{value:.1%}"}
                        for label, value in result["probabilities"].items()
                    ], hide_index=True, width="stretch")
                    st.caption("Uncalibrated model probabilities; not a guarantee of correctness.")

with st.expander("About the models"):
    st.write(
        "Naive Bayes uses Bag-of-Words counts. Logistic Regression and the "
        "feedforward neural network use TF-IDF. The team predictions use fitted "
        "pipelines, and the demo predictions use sample data."
    )
