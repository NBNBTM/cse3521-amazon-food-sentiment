# Author: Lindsey Yang
# Modified: 2026-11-03
# AI assistance: Codex assisted with these prototype checks.
"""Run with python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

from sentiment import EXAMPLES, LABELS, MODEL_FILES, get_demo_models, load_team_model, predict_review

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TMP = ROOT / "tests" / "tmp"


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = get_demo_models()
        LOCAL_TMP.mkdir(exist_ok=True)

    def test_demo_predictions_have_valid_labels_and_probabilities(self):
        for name, model in self.models.items():
            for expected, review in EXAMPLES.items():
                with self.subTest(model=name, example=expected):
                    result = predict_review(model, review)
                    self.assertEqual(result["label"], expected.lower())
                    self.assertEqual(tuple(result["probabilities"]), LABELS)
                    self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

    def test_empty_and_oversized_input_are_rejected(self):
        for review in ("", "  \n", "a" * 10001):
            with self.assertRaises(ValueError):
                predict_review(self.models["Naive Bayes"], review)

    def test_team_artifact_handoff_and_bad_artifacts(self):
        with tempfile.TemporaryDirectory(dir=LOCAL_TMP) as directory:
            for name, filename in MODEL_FILES.items():
                path = Path(directory) / filename
                with self.assertRaisesRegex(ValueError, "Missing team artifact"):
                    load_team_model(name, directory)
                path.write_bytes(b"not a joblib artifact")
                with self.assertRaisesRegex(ValueError, "Could not load"):
                    load_team_model(name, directory)
                joblib.dump(self.models[name], path)
                loaded = load_team_model(name, directory)
                self.assertEqual(predict_review(loaded, EXAMPLES["Positive"])["label"], "positive")
            joblib.dump({"fake": "model"}, Path(directory) / MODEL_FILES["Naive Bayes"])
            with self.assertRaisesRegex(ValueError, "complete scikit-learn Pipeline"):
                load_team_model("Naive Bayes", directory)

    def test_invalid_classes_and_probabilities_are_rejected(self):
        model = self.models["Naive Bayes"]
        classifier = model.named_steps["classifier"]
        with patch.object(classifier, "classes_", np.array([0, 1, 2])):
            with self.assertRaisesRegex(ValueError, "Model classes"):
                predict_review(model, "A review")
        for probabilities in (np.array([[np.nan, 0.5, 0.5]]), np.array([[0.5, 0.5]]),
                              np.array([[0.2, 0.2, 0.2]])):
            with patch.object(model, "predict_proba", return_value=probabilities):
                with self.assertRaisesRegex(ValueError, "invalid class probabilities"):
                    predict_review(model, "A review")

    def test_app_empty_input_and_two_model_predictions(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("DEMO ONLY" in item.value for item in app.warning))
        app.button[0].click().run()
        self.assertTrue(any("Enter a review" in item.value for item in app.warning))
        for expected in EXAMPLES:
            app.selectbox[0].select(expected).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual([item.value for item in app.metric], [expected, expected])
            self.assertEqual(len(app.dataframe), 2)

    def test_app_team_mode_has_no_silent_demo_fallback(self):
        with tempfile.TemporaryDirectory(dir=LOCAL_TMP) as directory:
            app = self.app_with_team_directory(directory)
            app.radio[0].set_value("Team models").run()
            app.text_area[0].input(EXAMPLES["Positive"])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 2)
            self.assertFalse(app.metric)

    def test_app_runs_exported_team_pipelines(self):
        with tempfile.TemporaryDirectory(dir=LOCAL_TMP) as directory:
            models_dir = Path(directory) / "models"
            models_dir.mkdir()
            for name, filename in MODEL_FILES.items():
                joblib.dump(self.models[name], models_dir / filename)
            app = self.app_with_team_directory(directory)
            app.radio[0].set_value("Team models").run()
            app.text_area[0].input(EXAMPLES["Negative"])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual([item.value for item in app.metric], ["Negative", "Negative"])

    def app_with_team_directory(self, directory):
        source = (ROOT / "app.py").read_text()
        source = source.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({directory!r})")
        test_app = Path(directory) / "app_test.py"
        test_app.write_text(source)
        return AppTest.from_file(str(test_app)).run(timeout=30)


if __name__ == "__main__":
    unittest.main()
