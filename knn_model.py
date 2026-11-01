"""
K-Nearest Neighbours implemented from scratch (NumPy) on the WDBC dataset.

Split      : 70% train / 20% validation / 10% test (stratified, seed 42)
Scaling    : StandardScaler fitted on the TRAIN split only (KNN is distance based)
Tuning     : k in 1..30 x {euclidean, manhattan} x {uniform, distance} weighting,
             scored on the VALIDATION split (F2, so malignant recall is favoured)
Final model: fit on train only with best params, evaluated once on TEST
Sanity     : predictions are cross-checked against sklearn's KNeighborsClassifier

Outputs (CSV, in --out directory) for plotting:
  knn_split_summary.csv    sizes and class balance per split
  knn_k_sweep.csv          train/val metrics for every (k, metric, weights)
  knn_metrics.csv          final metrics on train / val / test
  knn_confusion_matrix.csv long-format confusion matrices
  knn_roc_curve.csv        fpr, tpr, threshold per split
  knn_pr_curve.csv         precision, recall, threshold per split
  knn_predictions.csv      per-sample true label, prediction, P(malignant)
  knn_pca_projection.csv   2-D PCA coordinates + labels/predictions for scatter plots
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42


class KNN:
    """Binary/multi-class KNN with euclidean or manhattan distance.

    weights="uniform"  : each neighbour votes equally
    weights="distance" : neighbour votes weighted by 1 / (distance + eps)
    Ties in the vote (possible for even k) go to the positive class (label 1,
    malignant), which favours recall.
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        assert metric in ("euclidean", "manhattan")
        assert weights in ("uniform", "distance")
        self.k, self.metric, self.weights = k, metric, weights

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def _distances(self, Q):
        if self.metric == "euclidean":
            d2 = ((Q ** 2).sum(1)[:, None] - 2 * Q @ self.X_.T
                  + (self.X_ ** 2).sum(1)[None, :])
            return np.sqrt(np.maximum(d2, 0.0))
        return np.abs(Q[:, None, :] - self.X_[None, :, :]).sum(axis=2)

    def predict_proba(self, Q):
        Q = np.asarray(Q, dtype=float)
        D = self._distances(Q)
        k = min(self.k, len(self.X_))
        nn = np.argpartition(D, k - 1, axis=1)[:, :k]            # k nearest idx
        nn_d = np.take_along_axis(D, nn, axis=1)
        nn_y = self.y_[nn]
        w = np.ones_like(nn_d) if self.weights == "uniform" else 1.0 / (nn_d + 1e-9)
        proba = np.zeros((len(Q), len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            proba[:, ci] = (w * (nn_y == c)).sum(1)
        return proba / proba.sum(1, keepdims=True)

    def predict(self, Q):
        p = self.predict_proba(Q)
        # argmax over reversed columns so ties pick the LAST class (malignant = 1)
        return self.classes_[p.shape[1] - 1 - np.argmax(p[:, ::-1], axis=1)]


def load_and_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns="diagnosis")
    y = df["diagnosis"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1 / 3, stratify=y_tmp, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    return ((scaler.transform(X_tr), y_tr),
            (scaler.transform(X_val), y_val),
            (scaler.transform(X_te), y_te))


def evaluate(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "avg_precision": average_precision_score(y_true, score),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cleaned_data.csv")
    ap.add_argument("--out", default="knn_outputs")
    ap.add_argument("--max-k", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_and_split(args.data)
    splits = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}

    pd.DataFrame([{"split": n, "n_samples": len(y), "n_malignant": int(y.sum()),
                   "pct_malignant": 100 * y.mean()}
                  for n, (_, y) in splits.items()]
                 ).to_csv(f"{args.out}/knn_split_summary.csv", index=False)

    # ---- sweep k / metric / weighting on the validation set ------------
    rows = []
    for metric in ("euclidean", "manhattan"):
        for weights in ("uniform", "distance"):
            for k in range(1, args.max_k + 1):
                m = KNN(k, metric, weights).fit(Xtr, ytr)
                p_tr, p_va = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xva)[:, 1]
                tr_m = evaluate(ytr, (p_tr >= 0.5).astype(int), p_tr)
                va_m = evaluate(yva, (p_va >= 0.5).astype(int), p_va)
                rows.append({"k": k, "metric": metric, "weights": weights,
                             "train_accuracy": tr_m["accuracy"],
                             "train_error": 1 - tr_m["accuracy"],
                             "val_error": 1 - va_m["accuracy"],
                             **{f"val_{a}": b for a, b in va_m.items()}})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(f"{args.out}/knn_k_sweep.csv", index=False)

    best = sweep.sort_values(["val_f2", "val_accuracy", "k"],
                             ascending=[False, False, False]).iloc[0]
    k, metric, weights = int(best["k"]), best["metric"], best["weights"]
    print(f"Best (chosen on validation): k={k}, metric={metric}, weights={weights}")

    # ---- final model ---------------------------------------------------
    model = KNN(k, metric, weights).fit(Xtr, ytr)

    metric_rows, cm_rows, roc_rows, pr_rows, pred_rows = [], [], [], [], []
    for name, (X, y) in splits.items():
        score = model.predict_proba(X)[:, 1]          # P(malignant)
        pred = model.predict(X)
        metric_rows.append({"split": name, **evaluate(y, pred, score)})

        cm = confusion_matrix(y, pred, labels=[0, 1])
        for i, t in enumerate(["benign", "malignant"]):
            for j, p in enumerate(["benign", "malignant"]):
                cm_rows.append({"split": name, "true": t, "predicted": p,
                                "count": int(cm[i, j])})

        fpr, tpr, thr = roc_curve(y, score)
        roc_rows += [{"split": name, "fpr": a, "tpr": b, "threshold": c}
                     for a, b, c in zip(fpr, tpr, thr)]
        prec, rec, thr = precision_recall_curve(y, score)
        thr = np.append(thr, np.nan)
        pr_rows += [{"split": name, "precision": a, "recall": b, "threshold": c}
                     for a, b, c in zip(prec, rec, thr)]
        pred_rows += [{"split": name, "sample_idx": int(i), "y_true": int(t),
                       "y_pred": int(p), "p_malignant": float(s),
                       "correct": int(t == p)}
                      for i, t, p, s in zip(y.index, y, pred, score)]

    pd.DataFrame(metric_rows).to_csv(f"{args.out}/knn_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(f"{args.out}/knn_confusion_matrix.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(f"{args.out}/knn_roc_curve.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(f"{args.out}/knn_pr_curve.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{args.out}/knn_predictions.csv", index=False)

    # ---- 2-D PCA projection for scatter plots --------------------------
    pca = PCA(n_components=2, random_state=SEED).fit(Xtr)
    pca_rows = []
    for name, (X, y) in splits.items():
        Z, pred = pca.transform(X), model.predict(X)
        pca_rows += [{"split": name, "pc1": z[0], "pc2": z[1],
                      "y_true": int(t), "y_pred": int(p)}
                     for z, t, p in zip(Z, y, pred)]
    pd.DataFrame(pca_rows).to_csv(f"{args.out}/knn_pca_projection.csv", index=False)

    # ---- sanity check against sklearn ----------------------------------
    sk = KNeighborsClassifier(
        n_neighbors=k, weights=weights,
        p=2 if metric == "euclidean" else 1).fit(Xtr, ytr)
    agree = (sk.predict(Xte) == model.predict(Xte)).mean()
    print(f"Agreement with sklearn KNeighborsClassifier on test: {agree:.3f} "
          "(tiny differences possible on exact vote ties)")

    print(pd.DataFrame(metric_rows)[["split", "accuracy", "precision", "recall",
                                     "specificity", "f1", "roc_auc"]]
          .round(4).to_string(index=False))
    print(f"CSV files written to ./{args.out}/")


if __name__ == "__main__":
    main()
