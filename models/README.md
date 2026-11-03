# Team model handoff

Author: Lindsey Yang · Modified: 2026-11-03

No Amazon-trained models are included yet. Demo mode fits two small classifiers on the 18 invented examples in `sentiment.py`. Those predictions are interface checks, not project evaluation results.

For **Team models** mode, export trusted, fitted scikit-learn `Pipeline` objects to:

- `models/naive_bayes.joblib`
- `models/logistic_regression.joblib`

Each pipeline must accept a list of raw strings in `predict()` and contain all training-time text transformations. Its `classes_` must contain exactly the strings `negative`, `neutral`, and `positive`. Optional `predict_proba()` must return one three-class probability row per input, in `classes_` order. Numeric class IDs require an explicit label mapping before export; the app will not guess.

Agree on input composition with the feature owner before training (the charter proposes `Summary + Text`). The app passes the entered string unchanged to the pipeline. Never use review `Score` as an input feature. Record dependency versions with the exported artifact and reproduce those versions for inference.

```python
import joblib

# pipeline is already fitted using the training split only.
joblib.dump(pipeline, "models/logistic_regression.joblib")
```

Only load artifacts produced and reviewed by trusted team members: joblib/pickle files can execute code. The app deliberately has no model-upload interface. Missing or incompatible artifacts produce an error rather than silently using demo models. Replacing an artifact refreshes the cache when its modification timestamp or size changes; restart the app if you preserve both metadata values.

The neural network needs a separate reviewed adapter when the team delivers it. It is not represented by a fake model slot here.
