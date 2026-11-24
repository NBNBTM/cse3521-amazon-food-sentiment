# Team model handoff

Author: Lindsey Yang · Modified: 2026-11-23

The [training scripts](../training/README.md) export the fitted pipelines. Demo mode fits two small classifiers on the 18 invented examples in `sentiment.py` for interface checks.

For **Team models** mode, export trusted, fitted scikit-learn `Pipeline` objects to:

- `models/naive_bayes.joblib`
- `models/logistic_regression.joblib`
- `models/feedforward_nn.joblib`

Each pipeline must accept a list of raw strings in `predict()` and contain all training-time text transformations. Its `classes_` must contain exactly the strings `negative`, `neutral`, and `positive`. Optional `predict_proba()` must return one three-class probability row per input, in `classes_` order. Numeric class IDs require an explicit label mapping before export; the app will not guess.

Use `Summary + " " + Text` for each input, with an empty string when the summary is absent. The app uses the same composition. Exclude review `Score` from model inputs. Record dependency versions with the exported artifact and reproduce those versions for inference.

```python
import joblib

# pipeline is already fitted using the training split only.
joblib.dump(pipeline, "models/logistic_regression.joblib")
```

Team mode displays models with a local fitted pipeline. The feedforward neural network uses the same scikit-learn Pipeline interface as Naive Bayes and Logistic Regression. Load artifacts from trusted sources because joblib files can execute Python code. Replacing an artifact refreshes the cache when its modification timestamp or size changes. Restart the app if both values stay the same.
