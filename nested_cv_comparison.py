"""
Fair comparison of all 8 models under IDENTICAL conditions (repeated nested CV).

Why: a single 80/20 split has only 42 malignant test cases, so one extra miss
moves recall by 2.4 points and the ranking can change with the split.
Here every model is evaluated on the SAME 50 outer test folds:

  Outer loop : RepeatedStratifiedKFold, 5 folds x 10 repeats, seed 42, on all 569 samples
               -> every sample is tested 10 times, by every model
  Inner loop : GridSearchCV, stratified 5-fold, same grids as compare_all_models.py,
               model selected by F2 (recall-weighted), tuned on the outer-train part only
  Scaling    : StandardScaler inside the Pipeline for LR / KNN / SVM (no leakage)
  Threshold  : default decision rule (0.5 probability / SVM sign)

Metrics per outer fold: accuracy, balanced accuracy, precision, recall (sensitivity),
specificity, F1, F2, MCC, ROC-AUC, PR-AUC, TN/FP/FN/TP, chosen hyperparameters.

Needs : cleaned_data.csv, compare_all_models.py, tree_models.py
Run   : python nested_cv_comparison.py          (about 40 min on 2 CPU cores)
          - results are saved after every fold, so an interrupted run resumes
Then  : python analyse_nested_cv.py
Output: nested_results/fold_metrics.csv, nested_results/oof_predictions.csv
"""
import os
import json
import time
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, fbeta_score, matthews_corrcoef,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             make_scorer)

from compare_all_models import build_models, score_of

SEED = 42
N_SPLITS = 5
N_REPEATS = 10
OUT = "nested_results"
os.makedirs(OUT, exist_ok=True)
FOLD_FILE = os.path.join(OUT, "fold_metrics.csv")
PRED_FILE = os.path.join(OUT, "oof_predictions.csv")


def fold_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "f2": fbeta_score(y, pred, beta=2),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    df = pd.read_csv("cleaned_data.csv")
    X = df.drop(columns="diagnosis").values
    y = df["diagnosis"].astype(int).values
    print(f"Data {X.shape}, malignant {y.sum()}  |  outer {N_SPLITS}x{N_REPEATS} = "
          f"{N_SPLITS * N_REPEATS} folds")

    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f2 = make_scorer(fbeta_score, beta=2)

    # resume support
    done = set()
    if os.path.exists(FOLD_FILE):
        old = pd.read_csv(FOLD_FILE)
        for _, r in old.iterrows():
            done.add((r["model"], int(r["fold"])))

    folds = list(outer.split(X, y))
    models = build_models()
    t_start = time.time()

    for fold_id in range(len(folds)):
        train_idx, test_idx = folds[fold_id]
        repeat = fold_id // N_SPLITS
        for name in models:
            if (name, fold_id) in done:
                continue
            est, grid = models[name]
            # one core per model; the grid search uses both cores
            if "n_jobs" in est.get_params():
                est.set_params(n_jobs=1)

            t0 = time.time()
            gs = GridSearchCV(est, grid, cv=inner, scoring=f2, refit=True, n_jobs=-1)
            gs.fit(X[train_idx], y[train_idx])
            best = gs.best_estimator_
            pred = best.predict(X[test_idx])
            score = score_of(best, X[test_idx])

            params = {}
            for k, v in gs.best_params_.items():
                if isinstance(v, (np.integer, np.floating)):
                    v = v.item()
                params[k.replace("clf__", "")] = v

            row = {"model": name, "fold": fold_id, "repeat": repeat,
                   "n_test": len(test_idx), **fold_metrics(y[test_idx], pred, score),
                   "inner_cv_f2": gs.best_score_, "fit_seconds": time.time() - t0,
                   "best_params": json.dumps(params, default=str, sort_keys=True)}
            pd.DataFrame([row]).to_csv(FOLD_FILE, mode="a", index=False,
                                       header=not os.path.exists(FOLD_FILE))

            preds = pd.DataFrame({"model": name, "fold": fold_id, "repeat": repeat,
                                  "sample": test_idx, "y_true": y[test_idx],
                                  "y_pred": pred, "score": score})
            preds.to_csv(PRED_FILE, mode="a", index=False, header=not os.path.exists(PRED_FILE))

        elapsed = (time.time() - t_start) / 60
        print(f"fold {fold_id + 1:2d}/{len(folds)} done  ({elapsed:.1f} min)", flush=True)

    print(f"\nSaved {FOLD_FILE} and {PRED_FILE}.  Next: python analyse_nested_cv.py")


if __name__ == "__main__":
    main()
