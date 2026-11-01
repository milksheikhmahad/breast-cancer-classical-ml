"""
Compare ALL models on the SAME stratified 80/20 split with the SAME 5-fold
stratified CV, using the metrics named in the project synopsis:
accuracy, precision, recall (sensitivity), F1, specificity, ROC-AUC,
confusion matrices, error analysis and feature importance.

Models : Logistic Regression, KNN, SVM, ID3, C4.5, CART, Random Forest, XGBoost
Tuning : GridSearchCV on the TRAIN set only (5-fold stratified, seed 42),
         model selected by F2 (recall-weighted) because a missed malignant
         tumour (false negative) is the costliest error.
Scaling: StandardScaler inside a Pipeline for LR / KNN / SVM (fit per CV fold,
         so there is no leakage). Tree models use raw features.
Test   : the held-out test set is used once, at the end.

Needs  : train_unscaled.csv, test_unscaled.csv   (python make_split.py)
Run    : python compare_all_models.py
Outputs: results/all_models_*.csv, results/all_models_best_params.json,
         results/figures/*.png
"""
import os
import json
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, roc_auc_score, roc_curve,
                             confusion_matrix, make_scorer)
from xgboost import XGBClassifier

from tree_models import ID3Classifier, C45Classifier   # from-scratch trees

SEED = 42
OUT = "results"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
sns.set_theme(style="whitegrid")


def score_of(model, X):
    """Continuous malignant score for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def wilson_ci(k, n, z=1.96):
    """95% Wilson interval for a proportion (used for recall)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)


def test_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    lo, hi = wilson_ci(tp, tp + fn)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y, pred),
        "F2": fbeta_score(y, pred, beta=2),
        "ROC-AUC": roc_auc_score(y, score),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Recall CI low": lo, "Recall CI high": hi,
    }


def build_models():
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "Logistic Regression": (
            scaled(LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEED)),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
             "clf__penalty": ["l1", "l2"],
             "clf__class_weight": [None, "balanced"]}),
        "KNN": (
            scaled(KNeighborsClassifier()),
            {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
             "clf__weights": ["uniform", "distance"],
             "clf__metric": ["euclidean", "manhattan"]}),
        "SVM": (
            scaled(SVC(random_state=SEED)),
            [{"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1, 10, 100]},
             {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100],
              "clf__gamma": ["scale", 0.001, 0.01, 0.1]}]),
        "ID3": (
            ID3Classifier(),
            {"n_bins": [3, 5, 8], "max_depth": [3, 5, 8, 12],
             "min_samples_split": [2, 10]}),
        "C4.5": (
            C45Classifier(),
            {"max_depth": [3, 5, 8, 12], "min_samples_split": [2, 10],
             "min_samples_leaf": [1, 3, 5]}),
        "CART": (
            DecisionTreeClassifier(criterion="gini", random_state=SEED),
            {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 10],
             "min_samples_leaf": [1, 3, 5], "ccp_alpha": [0.0, 0.005, 0.01]}),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 5, 10],
             "min_samples_leaf": [1, 3], "max_features": ["sqrt", "log2"]}),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [2, 3, 5],
             "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]}),
    }


def main():
    train = pd.read_csv("train_unscaled.csv")
    test = pd.read_csv("test_unscaled.csv")
    X_train, y_train = train.drop(columns="diagnosis"), train["diagnosis"].astype(int)
    X_test, y_test = test.drop(columns="diagnosis"), test["diagnosis"].astype(int)
    feats = X_train.columns.tolist()
    Xtr, Xte = X_train.values, X_test.values
    print(f"Train {Xtr.shape} | Test {Xte.shape} | malignant: "
          f"train {y_train.sum()} test {y_test.sum()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "f2": make_scorer(fbeta_score, beta=2), "roc_auc": "roc_auc"}

    test_rows, cv_rows, params, fitted = [], [], {}, {}
    preds = pd.DataFrame({"sample_idx": X_test.index, "y_true": y_test.values})

    for name, (est, grid) in build_models().items():
        gs = GridSearchCV(est, grid, cv=cv, scoring=scoring, refit="f2",
                          n_jobs=-1, return_train_score=False)
        gs.fit(Xtr, y_train.values)
        best = gs.best_estimator_
        fitted[name] = best
        params[name] = {k.replace("clf__", ""): (v if not isinstance(v, (np.integer, np.floating))
                                                 else v.item())
                        for k, v in gs.best_params_.items()}
        i = gs.best_index_
        cvr = {"Model": name}
        for m in scoring:
            cvr[f"CV {m} mean"] = gs.cv_results_[f"mean_test_{m}"][i]
            cvr[f"CV {m} std"] = gs.cv_results_[f"std_test_{m}"][i]
        cv_rows.append(cvr)

        pred = best.predict(Xte)
        score = score_of(best, Xte)
        test_rows.append({"Model": name, **test_metrics(y_test.values, pred, score)})
        preds[f"{name}_pred"] = pred
        preds[f"{name}_score"] = score
        print(f"[{name}] {params[name]}  CV F2={gs.best_score_:.4f}")

    tm = pd.DataFrame(test_rows).set_index("Model")
    cvm = pd.DataFrame(cv_rows).set_index("Model")
    tm.round(4).to_csv(f"{OUT}/all_models_test_metrics.csv")
    cvm.round(4).to_csv(f"{OUT}/all_models_cv_metrics.csv")
    preds.to_csv(f"{OUT}/all_models_predictions.csv", index=False)
    with open(f"{OUT}/all_models_best_params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=========== TEST SET ===========")
    print(tm[["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC",
              "TN", "FP", "FN", "TP"]].round(4))
    print("\n=========== 5-FOLD CV (train) ===========")
    print(cvm[[c for c in cvm.columns if "mean" in c]].round(4))

    # ---------------- error analysis ----------------
    names = list(fitted)
    wrong = pd.DataFrame({n: (preds[f"{n}_pred"] != preds["y_true"]).astype(int)
                          for n in names})
    err = pd.DataFrame({"sample_idx": preds["sample_idx"], "y_true": preds["y_true"],
                        "n_models_wrong": wrong.sum(axis=1),
                        "models_wrong": wrong.apply(
                            lambda r: ", ".join(r.index[r == 1]), axis=1)})
    err["error_type"] = np.where(err["y_true"] == 1, "False negative (missed malignant)",
                                 "False positive")
    err = err[err["n_models_wrong"] > 0].sort_values("n_models_wrong", ascending=False)
    key = ["radius_mean", "area_mean", "concave points_mean", "texture_mean",
           "radius_worst", "concave points_worst"]
    err = err.merge(X_test[key].reset_index().rename(columns={"index": "sample_idx"}),
                    on="sample_idx", how="left")
    err.round(4).to_csv(f"{OUT}/all_models_error_analysis.csv", index=False)
    print("\n=========== ERROR ANALYSIS (samples missed by >=1 model) ===========")
    print(err[["sample_idx", "error_type", "n_models_wrong", "models_wrong"]]
          .to_string(index=False))

    # ---------------- feature importance ----------------
    fi = {}
    for n in ["Random Forest", "XGBoost", "CART", "C4.5", "ID3"]:
        fi[n] = pd.Series(fitted[n].feature_importances_, index=feats)
    lr = fitted["Logistic Regression"].named_steps["clf"]
    coef = pd.Series(lr.coef_[0], index=feats)
    fi["Logistic Regression (|coef|, normalised)"] = coef.abs() / coef.abs().sum()
    fi_df = pd.DataFrame(fi)
    fi_df.round(4).to_csv(f"{OUT}/all_models_feature_importance.csv")
    coef.round(4).rename("standardised_coefficient").to_csv(f"{OUT}/logistic_regression_coefficients.csv")

    # ---------------- figures ----------------
    # 1. confusion matrices
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, n in zip(axes.ravel(), names):
        cm = confusion_matrix(y_test, preds[f"{n}_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        ax.set_title(f"{n}\nFN={cm[1, 0]}  FP={cm[0, 1]}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_confusion_matrices.png", dpi=200); plt.close()

    # 2. ROC curves
    plt.figure(figsize=(6.5, 5.5))
    for n in names:
        fpr, tpr, _ = roc_curve(y_test, preds[f"{n}_score"])
        plt.plot(fpr, tpr, label=f"{n} ({tm.loc[n, 'ROC-AUC']:.3f})", lw=1.6)
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
    plt.title("ROC curves (test set), AUC in brackets"); plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_roc_curves.png", dpi=200); plt.close()

    # 3. metric comparison
    cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]
    ax = tm[cols].plot.bar(figsize=(11, 5), width=0.8)
    ax.set_ylim(0.85, 1.01); ax.set_ylabel("Score"); ax.set_xlabel("")
    ax.set_title("Test-set performance of all models")
    plt.xticks(rotation=25, ha="right"); plt.legend(ncol=6, fontsize=8, loc="lower center")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_metric_comparison.png", dpi=200); plt.close()

    # 4. feature importance (top 10)
    show = ["Random Forest", "XGBoost", "CART", "Logistic Regression (|coef|, normalised)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, n in zip(axes, show):
        fi_df[n].nlargest(10)[::-1].plot.barh(ax=ax, color="#2a6f97")
        ax.set_title(n.split(" (")[0] + ": top 10", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_feature_importance.png", dpi=200); plt.close()

    # 5. CV recall vs test recall
    comp = pd.DataFrame({"CV recall (train, 5-fold)": cvm["CV recall mean"],
                         "Test recall": tm["Recall"]})
    ax = comp.plot.bar(figsize=(9, 4.5), yerr=[cvm["CV recall std"], np.zeros(len(tm))])
    ax.set_ylim(0.8, 1.02); ax.set_ylabel("Recall"); ax.set_xlabel("")
    ax.set_title("Cross-validated vs test recall")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(f"{FIG}/fig_cv_vs_test_recall.png", dpi=200); plt.close()

    print(f"\nSaved tables to ./{OUT}/ and figures to ./{FIG}/")


if __name__ == "__main__":
    main()
