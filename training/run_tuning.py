import json, sys, time, platform
import numpy as np, pandas as pd, joblib, sklearn, pandas as pd_, numpy as np_
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from prep import load_splits, LABELS, SEED

CSV = sys.argv[1]
OUT = "output"
import os; os.makedirs(OUT, exist_ok=True)
log = lambda *a: (print(*a, flush=True))

df, train, test, fit, val = load_splits(CSV)
X_tr, y_tr = train["doc"].tolist(), train["label"].tolist()
X_fit, y_fit = fit["doc"].tolist(), fit["label"].tolist()
X_val, y_val = val["doc"].tolist(), val["label"].tolist()
X_te, y_te = test["doc"].tolist(), test["label"].tolist()

def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average="macro", zero_division=0)
    pc, rc, fc, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average=None, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_precision": p, "macro_recall": r, "macro_f1": f,
            "per_class_f1": dict(zip(LABELS, fc.tolist())), "per_class_recall": dict(zip(LABELS, rc.tolist())),
            "confusion": confusion_matrix(y, pred, labels=LABELS).tolist()}

R = {"env": {"python": platform.python_version(), "pandas": pd_.__version__, "numpy": np_.__version__,
             "sklearn": sklearn.__version__, "joblib": joblib.__version__},
     "n": {"clean": len(df), "train": len(train), "test": len(test), "fit": len(fit), "val": len(val)}}

# ---- Tuning on validation split (fit -> val) ----
t0 = time.perf_counter()
tfidf_fit = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, lowercase=True)
Xf = tfidf_fit.fit_transform(X_fit); Xv = tfidf_fit.transform(X_val)
log("tfidf tuning matrix", Xf.shape, round(time.perf_counter()-t0,1))

# LR: tune C
R["tune_lr"] = {}
for C in [0.5, 1, 2, 4, 8, 16]:
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED)
    m.fit(Xf, y_fit); pv = m.predict(Xv)
    R["tune_lr"][C] = metrics(y_val, pv)["macro_f1"]; log("LR C", C, round(R["tune_lr"][C],4))
bestC = max(R["tune_lr"], key=R["tune_lr"].get); R["best_C"] = bestC

# GMM: SVD on fit split, per-class GMM, Bayes rule
svd_fit = TruncatedSVD(n_components=100, random_state=SEED)
Zf = svd_fit.fit_transform(Xf); Zv = svd_fit.transform(Xv)
log("svd done, explained var", round(float(svd_fit.explained_variance_ratio_.sum()),3))
def gmm_fit(Z, y, k):
    gm, logpri = {}, {}
    for c in LABELS:
        Zc = Z[np.array(y) == c]
        gm[c] = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-3, random_state=SEED, max_iter=200).fit(Zc)
        logpri[c] = np.log(len(Zc) / len(y))
    return gm, logpri
def gmm_predict(gm, logpri, Z):
    S = np.column_stack([gm[c].score_samples(Z) + logpri[c] for c in LABELS])
    return np.array(LABELS)[S.argmax(1)]
R["tune_gmm"] = {}
for k in [1, 2]:
    gm, lp = gmm_fit(Zf, y_fit, k)
    R["tune_gmm"][k] = metrics(y_val, gmm_predict(gm, lp, Zv))["macro_f1"]; log("GMM k", k, round(R["tune_gmm"][k],4))
bestK = max(R["tune_gmm"], key=R["tune_gmm"].get); R["best_k"] = bestK; R["gmm_cov"] = "full"

# NN: MLP with per-epoch train/val log-loss on fit->val
NN_H, NN_ALPHA, MAX_EP, PATIENCE = 64, 1e-4, 30, 3
classes = np.array(LABELS)
ytf = np.array(y_fit); yvf = np.array(y_val)
sw = compute_sample_weight("balanced", ytf)
mlp = MLPClassifier(hidden_layer_sizes=(NN_H,), activation="relu", solver="adam", alpha=NN_ALPHA,
                    learning_rate_init=1e-3, batch_size=256, random_state=SEED)
curves, best, bad = [], (np.inf, 0), 0
rng = np.random.RandomState(SEED)
for ep in range(1, MAX_EP+1):
    idx = rng.permutation(Xf.shape[0])
    mlp.partial_fit(Xf[idx], ytf[idx], classes=classes, sample_weight=sw[idx])
    ptr = mlp.predict_proba(Xf); pva = mlp.predict_proba(Xv)
    tl = log_loss(ytf, ptr, labels=classes); vl = log_loss(yvf, pva, labels=classes)
    va = accuracy_score(yvf, classes[pva.argmax(1)])
    curves.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_acc": va})
    log(f"NN ep {ep} train {tl:.4f} val {vl:.4f} valacc {va:.4f}")
    if vl < best[0] - 1e-4: best, bad = (vl, ep), 0
    else:
        bad += 1
        if bad >= PATIENCE: break
best_ep = best[1]; R["nn"] = {"hidden": NN_H, "alpha": NN_ALPHA, "best_epoch": best_ep, "curves": curves}
pd.DataFrame(curves).to_csv(f"{OUT}/nn_loss_curves.csv", index=False)
json.dump(R, open(f"{OUT}/results_partial.json", "w"), indent=2, default=float)
log("tuning done. bestC", bestC, "bestK", bestK, "best_epoch", best_ep, round(time.perf_counter()-t0,1), "s")
