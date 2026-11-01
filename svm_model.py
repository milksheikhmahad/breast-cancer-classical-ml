"""
SVM on the WDBC breast cancer dataset.

Split      : 70% train / 20% validation / 10% test (stratified, seed 42)
Scaling    : StandardScaler fitted on the TRAIN split only
Tuning     : grid search over kernel / C / gamma, scored on the VALIDATION split
             (F2 score, so recall on malignant cases is weighted over precision)
Final model: refit on train only with the best params, evaluated once on TEST

Outputs (CSV, in --out directory) for plotting:
  svm_split_summary.csv    sizes and class balance per split
  svm_grid_results.csv     every hyperparameter combo with train/val metrics
  svm_metrics.csv          final metrics on train / val / test
  svm_confusion_matrix.csv long-format confusion matrices (val, test)
  svm_roc_curve.csv        fpr, tpr, threshold per split
  svm_pr_curve.csv         precision, recall, threshold per split
  svm_learning_curve.csv   train/val accuracy vs training-set size
  svm_predictions.csv      per-sample true label, prediction, decision score
"""
import argparse
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def load_and_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns="diagnosis")
    y = df["diagnosis"]

    # 70 / 30 first, then split the 30 into 20 (val) and 10 (test) -> test_size = 1/3
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1 / 3, stratify=y_tmp, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)  # train statistics only: no leakage
    tr = scaler.transform(X_tr)
    va = scaler.transform(X_val)
    te = scaler.transform(X_te)
    return (tr, y_tr), (va, y_val), (te, y_te)


def evaluate(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),  # sensitivity (malignant = 1)
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
    ap.add_argument("--out", default="svm_outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_and_split(args.data)
    splits = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}

    # ---- split summary -------------------------------------------------
    pd.DataFrame([{"split": n, "n_samples": len(y), "n_malignant": int(y.sum()),
                   "pct_malignant": 100 * y.mean()}
                  for n, (_, y) in splits.items()]
                 ).to_csv(f"{args.out}/svm_split_summary.csv", index=False)

    # ---- grid search on the validation set -----------------------------
    Cs = [0.01, 0.1, 1, 10, 100]
    gammas = ["scale", 0.001, 0.01, 0.1, 1]
    grid = [("linear", C, "n/a", "n/a") for C in Cs]
    grid += [("rbf", C, g, "n/a") for C, g in product(Cs, gammas)]
    grid += [("poly", C, g, 3) for C, g in product(Cs, gammas)]

    rows = []
    for kernel, C, gamma, degree in grid:
        kw = {"kernel": kernel, "C": C, "random_state": SEED}
        if kernel != "linear":
            kw["gamma"] = gamma
        if kernel == "poly":
            kw["degree"] = degree
        m = SVC(**kw).fit(Xtr, ytr)
        tr_m = evaluate(ytr, m.predict(Xtr), m.decision_function(Xtr))
        va_m = evaluate(yva, m.predict(Xva), m.decision_function(Xva))
        rows.append({"kernel": kernel, "C": C, "gamma": gamma, "degree": degree,
                     "train_accuracy": tr_m["accuracy"],
                     **{f"val_{k}": v for k, v in va_m.items()}})
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(f"{args.out}/svm_grid_results.csv", index=False)

    # best = highest val F2, ties broken by val accuracy, then smaller C
    best = grid_df.sort_values(["val_f2", "val_accuracy", "C"],
                               ascending=[False, False, True]).iloc[0]
    kw = {"kernel": best["kernel"], "C": float(best["C"]),
          "random_state": SEED}
    if best["kernel"] != "linear":
        g = best["gamma"]
        kw["gamma"] = g if g == "scale" else float(g)
    if best["kernel"] == "poly":
        kw["degree"] = int(best["degree"])
    print("Best params (chosen on validation):", kw)

    # ---- final model: train only, test touched once --------------------
    model = SVC(**kw).fit(Xtr, ytr)

    metric_rows, cm_rows, roc_rows, pr_rows, pred_rows = [], [], [], [], []
    for name, (X, y) in splits.items():
        score = model.decision_function(X)
        pred = model.predict(X)
        m = evaluate(y, pred, score)
        metric_rows.append({"split": name, **m})

        cm = confusion_matrix(y, pred, labels=[0, 1])
        for i, t in enumerate(["benign", "malignant"]):
            for j, p in enumerate(["benign", "malignant"]):
                cm_rows.append({"split": name, "true": t, "predicted": p,
                                "count": int(cm[i, j])})

        fpr, tpr, thr = roc_curve(y, score)
        roc_rows += [{"split": name, "fpr": a, "tpr": b, "threshold": c}
                     for a, b, c in zip(fpr, tpr, thr)]
        prec, rec, thr = precision_recall_curve(y, score)
        thr = np.append(thr, np.nan)  # pr curve has one more point than thresholds
        pr_rows += [{"split": name, "precision": a, "recall": b, "threshold": c}
                     for a, b, c in zip(prec, rec, thr)]

        pred_rows += [{"split": name, "sample_idx": int(i), "y_true": int(t),
                       "y_pred": int(p), "decision_score": float(s),
                       "correct": int(t == p)}
                      for i, t, p, s in zip(y.index, y, pred, score)]

    pd.DataFrame(metric_rows).to_csv(f"{args.out}/svm_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(f"{args.out}/svm_confusion_matrix.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(f"{args.out}/svm_roc_curve.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(f"{args.out}/svm_pr_curve.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{args.out}/svm_predictions.csv", index=False)

    # ---- learning curve (train on growing fractions of the train split) -
    rng = np.random.RandomState(SEED)
    lc = []
    for frac in np.linspace(0.1, 1.0, 10):
        n = int(frac * len(ytr))
        idx = rng.permutation(len(ytr))[:n]
        if ytr.iloc[idx].nunique() < 2:
            continue
        m = SVC(**kw).fit(Xtr[idx], ytr.iloc[idx])
        lc.append({"train_fraction": frac, "n_train": n,
                   "train_accuracy": accuracy_score(ytr.iloc[idx], m.predict(Xtr[idx])),
                   "val_accuracy": accuracy_score(yva, m.predict(Xva)),
                   "val_recall": recall_score(yva, m.predict(Xva))})
    pd.DataFrame(lc).to_csv(f"{args.out}/svm_learning_curve.csv", index=False)

    print(pd.DataFrame(metric_rows)[["split", "accuracy", "precision", "recall",
                                     "specificity", "f1", "roc_auc"]]
          .round(4).to_string(index=False))
    print(f"CSV files written to ./{args.out}/")


if __name__ == "__main__":
    main()
