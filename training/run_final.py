import json, sys, time, os
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from prep import load_splits, LABELS, SEED

CSV = sys.argv[1]; OUT = "output"; EXPORT = sys.argv[2] if len(sys.argv) > 2 else "models"
R = json.load(open(f"{OUT}/results_partial.json"))
bestC, bestK, best_ep = R["best_C"], R["best_k"], R["nn"]["best_epoch"]
log = lambda *a: print(*a, flush=True)
df, train, test, fit, val = load_splits(CSV)
X_tr, y_tr = train["doc"].tolist(), train["label"].tolist()
X_te, y_te = test["doc"].tolist(), test["label"].tolist()
ytr = np.array(y_tr)

def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average="macro", zero_division=0)
    pc, rc, fc, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average=None, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_precision": p, "macro_recall": r, "macro_f1": f,
            "per_class_precision": dict(zip(LABELS, pc.tolist())), "per_class_recall": dict(zip(LABELS, rc.tolist())),
            "per_class_f1": dict(zip(LABELS, fc.tolist())), "confusion": confusion_matrix(y, pred, labels=LABELS).tolist()}

results, preds = {}, {}
# 1 Majority baseline
t = time.perf_counter(); maj = Pipeline([("bow", CountVectorizer(max_features=20000)), ("clf", DummyClassifier(strategy="most_frequent"))]).fit(X_tr, y_tr)
tt = time.perf_counter() - t; preds["Majority baseline"] = maj.predict(X_te); results["Majority baseline"] = dict(metrics(y_te, preds["Majority baseline"]), train_seconds=tt)
log("maj", round(results["Majority baseline"]["macro_f1"],4), round(tt,2))

# 2 Naive Bayes (BoW, unsmoothed alpha=1e-9, as in M2)
t = time.perf_counter()
nb = Pipeline([("bow", CountVectorizer(max_features=20000, lowercase=True)), ("nb", MultinomialNB(alpha=1e-9))]).fit(X_tr, y_tr)
tt = time.perf_counter() - t; preds["Naive Bayes"] = nb.predict(X_te); results["Naive Bayes"] = dict(metrics(y_te, preds["Naive Bayes"]), train_seconds=tt)
log("nb", round(results["Naive Bayes"]["accuracy"],4), round(results["Naive Bayes"]["macro_f1"],4), round(tt,2))

# 3 Logistic Regression (TF-IDF 1-2 grams)
t = time.perf_counter()
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, lowercase=True)
lr = Pipeline([("tfidf", tfidf), ("lr", LogisticRegression(C=bestC, class_weight="balanced", max_iter=1000, random_state=SEED))]).fit(X_tr, y_tr)
tt = time.perf_counter() - t; preds["Logistic Regression"] = lr.predict(X_te); results["Logistic Regression"] = dict(metrics(y_te, preds["Logistic Regression"]), train_seconds=tt, C=bestC)
log("lr", round(results["Logistic Regression"]["accuracy"],4), round(results["Logistic Regression"]["macro_f1"],4), round(tt,2))

# 4 GMM (TF-IDF -> SVD100 -> per-class diagonal GMM + Bayes rule)
t = time.perf_counter()
tf_g = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, lowercase=True)
Xg = tf_g.fit_transform(X_tr); svd = TruncatedSVD(n_components=100, random_state=SEED); Zg = svd.fit_transform(Xg)
gm, lp = {}, {}
for c in LABELS:
    Zc = Zg[ytr == c]
    gm[c] = GaussianMixture(n_components=bestK, covariance_type="full", reg_covar=1e-3, random_state=SEED, max_iter=200).fit(Zc)
    lp[c] = np.log(len(Zc) / len(ytr))
tt = time.perf_counter() - t
Zte = svd.transform(tf_g.transform(X_te))
S = np.column_stack([gm[c].score_samples(Zte) + lp[c] for c in LABELS])
preds["GMM (per-class)"] = np.array(LABELS)[S.argmax(1)]
results["GMM (per-class)"] = dict(metrics(y_te, preds["GMM (per-class)"]), train_seconds=tt, n_components=bestK)
log("gmm", round(results["GMM (per-class)"]["accuracy"],4), round(results["GMM (per-class)"]["macro_f1"],4), round(tt,2))

# 5 Feedforward NN (TF-IDF -> MLP), trained for best_ep epochs on the full training split
t = time.perf_counter()
tf_n = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, lowercase=True)
Xn = tf_n.fit_transform(X_tr)
mlp = MLPClassifier(hidden_layer_sizes=(R["nn"]["hidden"],), activation="relu", solver="adam", alpha=R["nn"]["alpha"],
                    learning_rate_init=1e-3, batch_size=256, random_state=SEED)
sw = compute_sample_weight("balanced", ytr); rng = np.random.RandomState(SEED); classes = np.array(LABELS)
for ep in range(best_ep):
    idx = rng.permutation(Xn.shape[0]); mlp.partial_fit(Xn[idx], ytr[idx], classes=classes, sample_weight=sw[idx])
tt = time.perf_counter() - t
nn = Pipeline([("tfidf", tf_n), ("mlp", mlp)])
preds["Feedforward NN"] = nn.predict(X_te); results["Feedforward NN"] = dict(metrics(y_te, preds["Feedforward NN"]), train_seconds=tt, epochs=best_ep, hidden=R["nn"]["hidden"], n_params=int(sum(w.size for w in mlp.coefs_)+sum(b.size for b in mlp.intercepts_)))
log("nn", round(results["Feedforward NN"]["accuracy"],4), round(results["Feedforward NN"]["macro_f1"],4), round(tt,2))

# Drop Adam optimizer state (≈3x weight size) before export; prediction does not need it
# Drop training-only state: Adam moments and the early-stopping copy of the weights (each ~ one full weight matrix)
for attr in ('_optimizer', '_best_coefs', '_best_intercepts'):
    if hasattr(mlp, attr): setattr(mlp, attr, None)
# Save results, table, exports, figures
json.dump({"env": R["env"], "n": R["n"], "tuning": {k: R[k] for k in ["tune_lr","tune_gmm","best_C","best_k"]}, "nn_curves": R["nn"]["curves"], "test": results}, open(f"{OUT}/results_final.json","w"), indent=2, default=float)
rows = []
for name, r in results.items():
    rows.append({"Model": name, "Accuracy": f"{r['accuracy']*100:.2f}%", "Macro precision": f"{r['macro_precision']*100:.2f}%",
                 "Macro recall": f"{r['macro_recall']*100:.2f}%", "Macro F1": f"{r['macro_f1']*100:.2f}%",
                 "Train time (s)": f"{r['train_seconds']:.1f}"})
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/comparison_table.csv", index=False); print(tab.to_string(index=False))
if EXPORT:
    os.makedirs(EXPORT, exist_ok=True)
    joblib.dump(nb, f"{EXPORT}/naive_bayes.joblib"); joblib.dump(lr, f"{EXPORT}/logistic_regression.joblib"); joblib.dump(nn, f"{EXPORT}/feedforward_nn.joblib")
    log("exported", os.listdir(EXPORT))

# Loss curves figure
cur = pd.DataFrame(R["nn"]["curves"])
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
ax.plot(cur.epoch, cur.train_loss, marker="o", ms=3, label="Training loss", color="#2563eb")
ax.plot(cur.epoch, cur.val_loss, marker="o", ms=3, label="Validation loss", color="#dc2626")
ax.axvline(best_ep, ls="--", color="gray", lw=1, label=f"Chosen epoch = {best_ep}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.grid(alpha=0.3); ax.legend(frameon=False)
ax.set_title("Feedforward NN loss curves (tuning run: fit -> validation)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_nn_loss.png"); plt.close(fig)
