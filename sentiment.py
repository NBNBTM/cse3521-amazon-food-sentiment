# Author: Lindsey Yang
# Modified: 2026-11-03
# AI assistance: Codex assisted with this prototype implementation.
"""Small sentiment-model adapter; demo data are synthetic, not Amazon reviews."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

LABELS = ("negative", "neutral", "positive")
MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}
EXAMPLES = {
    "Positive": "Delicious fresh food with great flavor. I love it and will buy again.",
    "Neutral": "The flavor is average. It is okay, but nothing special.",
    "Negative": "This tastes terrible and stale. I am disappointed and will not buy again.",
}

# shortcut: these 18 invented examples only exercise the app; replace with team-trained artifacts for project use.
DEMO_REVIEWS = (
    ("The food tastes terrible and stale. I will not buy again.", "negative"),
    ("Awful bitter flavor and bad smell. Very disappointed.", "negative"),
    ("This arrived spoiled and tasted disgusting. A waste of money.", "negative"),
    ("I hate this snack. It is dry, stale, and unpleasant.", "negative"),
    ("Poor quality food with a terrible taste. Not recommended.", "negative"),
    ("The package was damaged and the food was bad. Disappointed.", "negative"),
    ("The flavor is average. It is okay, but nothing special.", "neutral"),
    ("An ordinary snack. Neither good nor bad for the price.", "neutral"),
    ("This is acceptable food with an average taste.", "neutral"),
    ("It is okay. I have no strong opinion about this product.", "neutral"),
    ("The taste is mild and ordinary. Nothing special to report.", "neutral"),
    ("A standard product. It meets expectations without standing out.", "neutral"),
    ("Delicious fresh food with great flavor. I love it and will buy again.", "positive"),
    ("Excellent taste and wonderful quality. Highly recommended.", "positive"),
    ("I love this tasty snack. It is fresh and delicious.", "positive"),
    ("Amazing flavor and great value. I am very happy.", "positive"),
    ("The food is wonderful and fresh. My favorite product.", "positive"),
    ("Perfect snack with delicious flavor. I will order again.", "positive"),
)


def get_demo_models():
    """Fit two real classifiers on invented examples; no evaluation is claimed."""
    texts, labels = zip(*DEMO_REVIEWS)
    models = {
        "Naive Bayes": Pipeline([
            ("features", CountVectorizer(ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("features", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(random_state=13, max_iter=1000)),
        ]),
    }
    return {name: model.fit(texts, labels) for name, model in models.items()}


def validate_model(model):
    """Require the raw-text pipeline and the team's agreed string label names."""
    if not isinstance(model, Pipeline):
        raise ValueError("Export a complete scikit-learn Pipeline, including text preprocessing.")
    classes = list(getattr(model, "classes_", []))
    if len(classes) != 3 or set(classes) != set(LABELS):
        raise ValueError("Model classes must be exactly negative, neutral, and positive.")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("The pipeline must provide predict().")
    return model


def load_team_model(name, models_dir):
    """Load only a fixed, trusted local team artifact; never accept uploads."""
    if name not in MODEL_FILES:
        raise ValueError("Unknown model selection.")
    path = Path(models_dir) / MODEL_FILES[name]
    if not path.is_file():
        raise ValueError(f"Missing team artifact: models/{path.name}.")
    try:
        model = joblib.load(path)
    except Exception as error:
        raise ValueError(f"Could not load models/{path.name}; ask its author to re-export it.") from error
    return validate_model(model)


def predict_review(model, text):
    """Use training-time transformations unchanged and validate model output."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a review before running a prediction.")
    if len(text) > 10000:
        raise ValueError("Keep the review within 10,000 characters.")
    validate_model(model)
    prediction = np.asarray(model.predict([text]))
    if prediction.shape != (1,) or prediction[0] not in LABELS:
        raise ValueError("The model returned an invalid sentiment label.")
    result = {"label": str(prediction[0]), "probabilities": None}
    if callable(getattr(model, "predict_proba", None)):
        values = np.asarray(model.predict_proba([text]), dtype=float)
        if (values.shape != (1, 3) or not np.all(np.isfinite(values))
                or np.any(values < 0) or np.any(values > 1)
                or not np.isclose(values.sum(), 1.0, atol=1e-6)):
            raise ValueError("The model returned invalid class probabilities.")
        by_class = dict(zip(model.classes_, values[0].tolist()))
        result["probabilities"] = {label: by_class[label] for label in LABELS}
    return result
