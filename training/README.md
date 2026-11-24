# Training and evaluation (Milestone 3)

Author: Ziwei Li · Group Theta · Seed 49

Run from the repository root after downloading the original dataset described in [data/README.md](../data/README.md) and placing the full `Reviews.csv` at `data/Reviews.csv`. The scripts draw the seed-49 sample.

## Environment

Use the pinned environment in `requirements-lock.txt`, which includes pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, and matplotlib 3.11.2:

```sh
python -m pip install -r requirements-lock.txt
```

Use the comparison table and `results_final.json` from the same run when checking report metrics. The JSON records the training environment, and elapsed training time varies by machine.

## Commands

```sh
# 1. Validation tuning (fit split -> validation split): LR C, GMM components, NN epoch
python training/run_tuning.py data/Reviews.csv

# 2. Refit all models on the full training split, evaluate once on the test split, export pipelines
python training/run_final.py data/Reviews.csv models
```

Outputs:

- `output/comparison_table.csv`: the M3 table (accuracy, macro precision/recall/F1, training time).
- `output/results_final.json`: per-class metrics and confusion matrices for every model, plus the tuning grid.
- `output/nn_loss_curves.csv` and `output/fig_nn_loss.png`: per-epoch training and validation loss for the feedforward NN.
- `models/naive_bayes.joblib`, `models/logistic_regression.joblib`, `models/feedforward_nn.joblib`: fitted `Pipeline` objects that accept raw strings and output `negative`, `neutral`, `positive` (validated with `sentiment.validate_model`).

## Protocol

- Sample: `reviews.sample(n=50000, random_state=49)`; drop missing `Text`; `Summary` filled with empty string; drop duplicate `Text` (keep first) → 46,391 reviews.
- Input text: `Summary + " " + Text`. `Score` is used only to build the label.
- Split: stratified 80/20 (`random_state=49`) → training 37,112 / test 9,279. Tuning uses a stratified 80/20 split of the training set → fit 29,689 / validation 7,423.
- The test split is used once, after all choices are fixed.
- The validation run selects LR's regularization strength, the GMM's component count, and the NN epoch with the lowest validation loss. The NN uses one hidden layer of 64 ReLU units, Adam (lr 1e-3), L2 alpha 1e-4, batch size 256, and balanced sample weights.
- The final run reads the selected settings from `output/results_partial.json` and fits all models on the full training split, including retraining the NN for the selected epoch count.

## Known limits

- The NN is a sklearn `MLPClassifier`, not a PyTorch model; no GPU is used.
- The GMM is a generative baseline on dense SVD features. It is included to compare a classical probabilistic model, not as a candidate for the app.
- Only one NN architecture was tuned (hidden size and alpha fixed). A larger search could change the deep-model result.
- `feedforward_nn.joblib` is about 55 MB. Retrain it with the script rather than committing it if the repository size is a concern.
