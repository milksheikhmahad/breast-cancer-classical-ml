"""
Breast cancer classification with ID3, C4.5, CART, XGBoost and Random Forest.
Data   : train_unscaled.csv / test_unscaled.csv  (run make_split.py first;
         diagnosis: 1 = malignant, 0 = benign)

Install:  pip install numpy pandas scikit-learn xgboost matplotlib seaborn
Run:      python tree_models.py
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def entropy(y):
    counts = np.bincount(y)
    p = counts[counts > 0] / len(y)
    return -np.sum(p * np.log2(p))


def _entropy2(neg, pos):
    """Entropy of many two-class nodes at once, from their class counts."""
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    n = neg + pos
    with np.errstate(divide="ignore", invalid="ignore"):
        p0, p1 = neg / n, pos / n
        t0 = np.where(p0 > 0, p0 * np.log2(np.where(p0 > 0, p0, 1)), 0.0)
        t1 = np.where(p1 > 0, p1 * np.log2(np.where(p1 > 0, p1, 1)), 0.0)
    return -(t0 + t1)


def leaf_proba(y, n_classes=2):
    return np.bincount(y, minlength=n_classes) / len(y)


# ----------------------------------------------------------------------
# ID3  (categorical features, multiway splits, information gain)
# Continuous features are discretised into bins first, because ID3
# only works on categorical attributes.
# ----------------------------------------------------------------------
class ID3Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_bins=5, max_depth=6, min_samples_split=5):
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        # Quantile bin edges with linear interpolation, computed directly with NumPy
        # so results do not depend on the scikit-learn version
        # (KBinsDiscretizer changed its default quantile method in scikit-learn 1.9).
        self.bin_edges_ = []
        for j in range(X.shape[1]):
            edges = np.percentile(X[:, j], np.linspace(0, 100, self.n_bins + 1))
            keep = np.ediff1d(edges, to_begin=np.inf) > 1e-8   # drop zero-width bins
            self.bin_edges_.append(edges[keep])
        Xb = self._discretise(X)
        self.importances_ = np.zeros(X.shape[1])
        self.tree_ = self._build(Xb, y, list(range(X.shape[1])), 0)
        s = self.importances_.sum()
        self.feature_importances_ = self.importances_ / s if s > 0 else self.importances_
        return self

    def _build(self, X, y, features, depth):
        proba = leaf_proba(y)
        if (len(np.unique(y)) == 1 or not features or depth >= self.max_depth
                or len(y) < self.min_samples_split):
            return {"leaf": True, "proba": proba}

        base = entropy(y)
        best_gain, best_f = 0.0, None
        for f in features:
            rem = 0.0
            for v in np.unique(X[:, f]):
                m = X[:, f] == v
                rem += m.mean() * entropy(y[m])
            gain = base - rem
            if gain > best_gain:
                best_gain, best_f = gain, f

        if best_f is None or best_gain <= 1e-9:
            return {"leaf": True, "proba": proba}

        self.importances_[best_f] += best_gain * len(y)
        children = {}
        rest = [f for f in features if f != best_f]   # ID3: attribute used once per path
        for v in np.unique(X[:, best_f]):
            m = X[:, best_f] == v
            children[v] = self._build(X[m], y[m], rest, depth + 1)
        return {"leaf": False, "feature": best_f, "children": children, "proba": proba}

    def _predict_one(self, x, node):
        while not node["leaf"]:
            nxt = node["children"].get(x[node["feature"]])
            if nxt is None:               # unseen category -> use this node's majority
                break
            node = nxt
        return node["proba"]

    def _discretise(self, X):
        X = np.asarray(X, dtype=float)
        Xb = np.zeros(X.shape, dtype=int)
        for j, edges in enumerate(self.bin_edges_):
            Xb[:, j] = np.searchsorted(edges[1:-1], X[:, j], side="right")
        return Xb

    def predict_proba(self, X):
        Xb = self._discretise(X)
        return np.array([self._predict_one(x, self.tree_) for x in Xb])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ----------------------------------------------------------------------
# C4.5  (continuous features via binary threshold splits, gain ratio)
# Uses the C4.5 rule: among attributes whose gain is >= average gain,
# pick the one with the highest gain ratio. Pre-pruning via depth /
# min_samples_leaf (instead of C4.5's pessimistic post-pruning).
# ----------------------------------------------------------------------
class C45Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, max_depth=6, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.importances_ = np.zeros(X.shape[1])
        self.tree_ = self._build(X, y, 0)
        s = self.importances_.sum()
        self.feature_importances_ = self.importances_ / s if s > 0 else self.importances_
        return self

    def _best_threshold(self, x, y, base):
        # Vectorised search over all candidate thresholds: class counts on each
        # side come from cumulative sums, so every split is scored in one pass.
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(y)
        # candidate thresholds: midpoints where the value changes
        idx = np.where(xs[1:] != xs[:-1])[0]
        idx = idx[(idx + 1 >= self.min_samples_leaf) & (n - idx - 1 >= self.min_samples_leaf)]
        if len(idx) == 0:
            return None
        pos_cum = np.cumsum(ys)
        total_pos = pos_cum[-1]
        nl = idx + 1
        nr = n - nl
        pos_l = pos_cum[idx]
        pos_r = total_pos - pos_l
        gains = base - (nl / n) * _entropy2(nl - pos_l, pos_l) - (nr / n) * _entropy2(nr - pos_r, pos_r)
        b = int(np.argmax(gains))            # first maximum, as in a sequential scan
        i = idx[b]
        gain, thr, pl = gains[b], (xs[i] + xs[i + 1]) / 2, nl[b] / n
        split_info = -(pl * np.log2(pl) + (1 - pl) * np.log2(1 - pl))
        ratio = gain / split_info if split_info > 0 else 0
        return gain, ratio, thr

    def _build(self, X, y, depth):
        proba = leaf_proba(y)
        if (len(np.unique(y)) == 1 or depth >= self.max_depth
                or len(y) < self.min_samples_split):
            return {"leaf": True, "proba": proba}

        base = entropy(y)
        cands = []
        for f in range(X.shape[1]):
            r = self._best_threshold(X[:, f], y, base)
            if r is not None and r[0] > 1e-9:
                cands.append((f, *r))          # (feature, gain, ratio, thr)
        if not cands:
            return {"leaf": True, "proba": proba}

        avg_gain = np.mean([c[1] for c in cands])
        good = [c for c in cands if c[1] >= avg_gain] or cands
        f, gain, ratio, thr = max(good, key=lambda c: c[2])

        self.importances_[f] += gain * len(y)
        m = X[:, f] <= thr
        return {"leaf": False, "feature": f, "thr": thr, "proba": proba,
                "left": self._build(X[m], y[m], depth + 1),
                "right": self._build(X[~m], y[~m], depth + 1)}

    def _predict_one(self, x, node):
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["thr"] else node["right"]
        return node["proba"]

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in np.asarray(X, dtype=float)])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def main():
    # ---- load the shared stratified 80/20 split (create it with make_split.py)
    train = pd.read_csv("train_unscaled.csv")
    test = pd.read_csv("test_unscaled.csv")
    X_train, y_train = train.drop(columns="diagnosis"), train["diagnosis"].astype(int)
    X_test, y_test = test.drop(columns="diagnosis"), test["diagnosis"].astype(int)
    feature_names = X_train.columns.tolist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, "
          f"malignant share train={y_train.mean():.3f} test={y_test.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- models + hyperparameter grids
    # (trees do not need feature scaling)
    # scoring='recall' because false negatives (missed malignancies) matter most
    searches = {
        "ID3": (ID3Classifier(),
                {"n_bins": [3, 5, 8], "max_depth": [3, 5, 8],
                 "min_samples_split": [2, 10]}),
        "C4.5": (C45Classifier(),
                 {"max_depth": [3, 5, 8], "min_samples_split": [2, 10],
                  "min_samples_leaf": [1, 3, 5]}),
        "CART": (DecisionTreeClassifier(criterion="gini", random_state=SEED),
                 {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 10],
                  "min_samples_leaf": [1, 3, 5], "ccp_alpha": [0.0, 0.005, 0.01]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                          {"n_estimators": [100, 300], "max_depth": [None, 5, 10],
                           "min_samples_leaf": [1, 3], "max_features": ["sqrt", "log2"]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
                    {"n_estimators": [100, 300], "max_depth": [2, 3, 5],
                     "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]}),
    }

    results, fitted = [], {}
    for name, (est, grid) in searches.items():
        gs = GridSearchCV(est, grid, cv=cv, scoring="recall", refit=True, n_jobs=1)
        gs.fit(X_train.values, y_train.values)
        fitted[name] = gs.best_estimator_
        print(f"\n[{name}] best params: {gs.best_params_}  (CV recall={gs.best_score_:.4f})")
        row = evaluate(name, gs.best_estimator_, X_test.values, y_test.values)
        row["CV Recall"] = gs.best_score_
        results.append(row)

    res = pd.DataFrame(results).set_index("Model")
    pd.set_option("display.width", 200)
    print("\n================ TEST SET RESULTS ================")
    print(res.round(4))
    res.to_csv("tree_models_metrics.csv")

    # ---- confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for ax, (name, m) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, m.predict(X_test.values))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("tree_confusion_matrices.png", dpi=150)

    # ---- feature importance (top 10 per model)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (name, m) in zip(axes, fitted.items()):
        imp = pd.Series(m.feature_importances_, index=feature_names).nlargest(10)
        imp[::-1].plot.barh(ax=ax)
        ax.set_title(f"{name}: top 10 features")
    plt.tight_layout()
    plt.savefig("tree_feature_importance.png", dpi=150)

    # ---- metric comparison chart
    res[["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]].plot.bar(
        figsize=(10, 5), ylim=(0.8, 1.01))
    plt.title("Model comparison (test set)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("tree_models_comparison.png", dpi=150)

    print("\nSaved: tree_models_metrics.csv, tree_confusion_matrices.png, "
          "tree_feature_importance.png, tree_models_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
