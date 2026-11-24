# Training and evaluation (Milestone 3)

Author: Ziwei Li · Group Theta · Seed 49

Reproduces every number in the M3 model comparison from the seed-49 sample. Run from the repository root after placing the sample described in [data/README.md](../data/README.md) at `data/Reviews.csv` (or edit the path in the commands).

## Environment

Pinned in `requirements-lock.txt` (pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1). Use that file for an identical environment:

```sh
python -m pip install -r requirements-lock.txt
```

Note: the numbers in the M3 report come from a sandbox run with pandas 3.0.5 and numpy 2.5.3. Small differences (about 0.1 points) occur across pandas versions, so rerun the commands locally to confirm before citing them.

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
- Chosen settings: LR `C=2` (balanced class weights); GMM with full covariance on 100 SVD components, 1 component per class; NN with one hidden layer of 64 ReLU units, Adam (lr 1e-3), L2 alpha 1e-4, batch size 256, balanced sample weights, trained for 4 epochs (the epoch with the lowest validation loss).
- The NN's 4-epoch count was tuned on the fit/validation split; the final model is retrained on the full training split for the same 4 epochs.

## Known limits

- The NN is a sklearn `MLPClassifier`, not a PyTorch model; no GPU is used.
- The GMM is a generative baseline on dense SVD features. It is included to compare a classical probabilistic model, not as a candidate for the app.
- Only one NN architecture was tuned (hidden size and alpha fixed). A larger search could change the deep-model result.
- `feedforward_nn.joblib` is about 55 MB. Retrain it with the script rather than committing it if the repository size is a concern.
