"""
Analyse the repeated nested CV results produced by nested_cv_comparison.py.

Statistics
  * Mean and standard deviation of every metric over the 50 outer folds.
  * 95% confidence intervals and paired tests use the Nadeau-Bengio correction
    (variance x (1/J + n_test/n_train)), because CV folds share training data
    and are not independent; ignoring this makes differences look significant
    far too often (Nadeau & Bengio 2003; Bouckaert & Frank 2004).
  * Each model is compared with the best model (highest mean F2) using the
    corrected repeated k-fold t-test, with Holm correction for 7 comparisons.
  * Average rank per fold (1 = best) and a Friedman test, as a descriptive check.
  * "Per pass" numbers: each of the 10 repeats tests every one of the 569
    samples exactly once, so FN/FP per pass = missed malignant / false alarms
    out of 212 malignant and 357 benign cases.

Run   : python analyse_nested_cv.py
Output: nested_results/summary_*.csv, nested_results/figures/*.png
"""
import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

IN = "nested_results"
FIG = os.path.join(IN, "figures")
os.makedirs(FIG, exist_ok=True)
sns.set_theme(style="whitegrid")

ORDER = ["Logistic Regression", "KNN", "SVM", "ID3", "C4.5", "CART", "Random Forest", "XGBoost"]
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity",
           "f1", "f2", "mcc", "roc_auc", "pr_auc"]
LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "precision": "Precision",
          "recall": "Recall", "specificity": "Specificity", "f1": "F1", "f2": "F2", "mcc": "MCC",
          "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def corrected_ci(values, ratio, conf=0.95):
    """Mean and Nadeau-Bengio corrected confidence interval."""
    values = np.asarray(values, dtype=float)
    j = len(values)
    m = values.mean()
    se = np.sqrt((1 / j + ratio) * values.var(ddof=1))
    t = stats.t.ppf((1 + conf) / 2, j - 1)
    # all metrics here are bounded above by 1
    return m, m - t * se, min(1.0, m + t * se)


def corrected_ttest(a, b, ratio):
    """Corrected repeated k-fold t-test on paired per-fold scores a and b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / j + ratio) * var)
    p = 2 * stats.t.sf(abs(t), j - 1)
    return t, p


def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    adjusted = {}
    running = 0.0
    for i in range(m):
        name = names[i]
        val = min(1.0, (m - i) * pvals[name])
        running = max(running, val)
        adjusted[name] = running
    return adjusted


def main():
    fm = pd.read_csv(os.path.join(IN, "fold_metrics.csv"))
    oof = pd.read_csv(os.path.join(IN, "oof_predictions.csv"))
    models = []
    for name in ORDER:
        if name in set(fm["model"]):
            models.append(name)
    n_folds = fm["fold"].nunique()
    n_repeats = fm["repeat"].nunique()
    # every model must have every fold, otherwise the comparison is not paired
    for name in models:
        assert fm[fm["model"] == name]["fold"].nunique() == n_folds, name

    n = oof[oof["model"] == models[0]].groupby("repeat")["sample"].count().iloc[0]
    n_test = n / (n_folds / n_repeats)
    ratio = n_test / (n - n_test)            # n_test / n_train, about 0.25
    print(f"{len(models)} models x {n_folds} outer folds ({n_repeats} repeats), "
          f"n_test/n_train = {ratio:.3f}")

    # ---------- 1. mean, std and corrected CI per metric ----------
    rows = []
    for name in models:
        sub = fm[fm["model"] == name].sort_values("fold")
        row = {"Model": name}
        for m in METRICS:
            mean, lo, hi = corrected_ci(sub[m], ratio)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = sub[m].std(ddof=1)
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi
        # errors per full pass over the data (each repeat = one pass)
        per_pass = sub.groupby("repeat")[["fn", "fp", "tn", "tp"]].sum()
        row["fn_per_pass"] = per_pass["fn"].mean()
        row["fn_per_pass_std"] = per_pass["fn"].std(ddof=1)
        row["fp_per_pass"] = per_pass["fp"].mean()
        row["fp_per_pass_std"] = per_pass["fp"].std(ddof=1)
        row["fit_seconds_per_fold"] = sub["fit_seconds"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("Model")
    summary.round(4).to_csv(os.path.join(IN, "summary_metrics.csv"))

    # ---------- 2. ranks per fold ----------
    rank_rows = []
    for m in ["f2", "mcc", "recall", "roc_auc"]:
        wide = fm.pivot(index="fold", columns="model", values=m)[models]
        ranks = wide.rank(axis=1, ascending=False, method="average")
        chi, p = stats.friedmanchisquare(*[wide[c].values for c in models])
        r = {"metric": m, "friedman_chi2": chi, "friedman_p": p}
        for c in models:
            r[c] = ranks[c].mean()
        rank_rows.append(r)
    ranks_df = pd.DataFrame(rank_rows).set_index("metric")
    ranks_df.round(4).to_csv(os.path.join(IN, "summary_ranks.csv"))

    # ---------- 3. each model vs the best model (by mean F2) ----------
    best = summary["f2_mean"].idxmax()
    test_rows = []
    for m in ["f2", "recall", "mcc", "roc_auc", "accuracy"]:
        wide = fm.pivot(index="fold", columns="model", values=m)
        raw = {}
        tvals = {}
        for name in models:
            if name == best:
                continue
            t, p = corrected_ttest(wide[best], wide[name], ratio)
            raw[name] = p
            tvals[name] = t
        adj = holm(raw)
        for name in raw:
            test_rows.append({"metric": m, "best_model": best, "other_model": name,
                              "mean_difference": wide[best].mean() - wide[name].mean(),
                              "t": tvals[name], "p_raw": raw[name], "p_holm": adj[name],
                              "significant_at_0.05": adj[name] < 0.05})
    tests = pd.DataFrame(test_rows)
    tests.round(5).to_csv(os.path.join(IN, "summary_tests_vs_best.csv"), index=False)

    # full pairwise matrix for MCC (corrected t-test, Holm over all 28 pairs)
    wide = fm.pivot(index="fold", columns="model", values="mcc")
    raw = {}
    for i in range(len(models)):
        for k in range(i + 1, len(models)):
            raw[(models[i], models[k])] = corrected_ttest(wide[models[i]], wide[models[k]], ratio)[1]
    adj = holm(raw)
    pmat = pd.DataFrame(np.nan, index=models, columns=models)
    for (a, b), p in adj.items():
        pmat.loc[a, b] = p
        pmat.loc[b, a] = p
    pmat.round(4).to_csv(os.path.join(IN, "summary_pairwise_mcc_pvalues.csv"))

    # ---------- 4. hyperparameter stability ----------
    hp_rows = []
    for name in models:
        counts = Counter(fm[fm["model"] == name]["best_params"])
        params, k = counts.most_common(1)[0]
        hp_rows.append({"Model": name, "most_chosen_params": params,
                        "chosen_in_folds": k, "of_folds": n_folds,
                        "distinct_settings": len(counts)})
    pd.DataFrame(hp_rows).to_csv(os.path.join(IN, "summary_hyperparameters.csv"), index=False)

    # ---------- 5. hardest samples ----------
    oof["wrong"] = (oof["y_pred"] != oof["y_true"]).astype(int)
    hard = oof.groupby(["sample", "y_true"])["wrong"].mean().reset_index()
    hard = hard.rename(columns={"wrong": "error_rate_all_models"})
    per_model = oof.pivot_table(index="sample", columns="model", values="wrong", aggfunc="mean")
    hard = hard.merge(per_model[models], left_on="sample", right_index=True)
    data = pd.read_csv("cleaned_data.csv")
    hard = hard.merge(data[["radius_mean", "area_mean", "concave points_mean", "perimeter_worst"]],
                      left_on="sample", right_index=True)
    hard = hard.sort_values("error_rate_all_models", ascending=False)
    hard.round(3).to_csv(os.path.join(IN, "summary_hard_samples.csv"), index=False)

    # ---------- print ----------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = pd.DataFrame(index=models)
    for m in METRICS:
        show[LABELS[m]] = (summary[f"{m}_mean"].map("{:.3f}".format) + " ± "
                           + summary[f"{m}_std"].map("{:.3f}".format))
    show["FN/pass (of 212)"] = summary["fn_per_pass"].map("{:.1f}".format)
    show["FP/pass (of 357)"] = summary["fp_per_pass"].map("{:.1f}".format)
    print("\n=== Repeated nested CV (mean ± std over folds) ===")
    print(show.to_string())
    print("\n=== Average rank per fold (1 = best) ===")
    print(ranks_df.round(2).to_string())
    print(f"\n=== Corrected t-tests vs best model ({best}), Holm-adjusted ===")
    print(tests[["metric", "other_model", "mean_difference", "p_holm", "significant_at_0.05"]]
          .round(4).to_string(index=False))
    print("\n=== Most frequently chosen hyperparameters ===")
    print(pd.DataFrame(hp_rows).to_string(index=False))
    print("\n=== Hardest samples ===")
    print(hard.head(8)[["sample", "y_true", "error_rate_all_models", "radius_mean", "area_mean"]]
          .to_string(index=False))

    # ---------- figures ----------
    palette = dict(zip(models, sns.color_palette("tab10", len(models))))
    # (a) per-fold distributions
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=False)
    for ax, m in zip(axes, ["recall", "f2", "mcc", "roc_auc"]):
        sns.boxplot(data=fm, x="model", y=m, hue="model", legend=False, order=models, palette=palette, ax=ax,
                    fliersize=2, linewidth=0.8)
        ax.set_title(f"{LABELS[m]} per outer fold ({n_folds} folds)", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(LABELS[m])
        ax.tick_params(axis="x", rotation=60, labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "nested_boxplots.png"), dpi=200)
    plt.close()

    # (b) mean with corrected 95% CI
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m in zip(axes, ["recall", "f2", "mcc"]):
        means = summary.loc[models, f"{m}_mean"]
        lo = means - summary.loc[models, f"{m}_ci_low"]
        hi = summary.loc[models, f"{m}_ci_high"] - means
        ax.errorbar(means.values, range(len(models)), xerr=[lo.values, hi.values],
                    fmt="o", color="#2a6f97", ecolor="#7aa6c2", capsize=3)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.invert_yaxis()
        ax.set_title(f"{LABELS[m]}: mean and corrected 95% CI", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "nested_ci.png"), dpi=200)
    plt.close()

    # (c) pooled confusion matrices, averaged per pass
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, name in zip(axes.ravel(), models):
        sub = fm[fm["model"] == name]
        cm = np.array([[sub["tn"].sum(), sub["fp"].sum()],
                       [sub["fn"].sum(), sub["tp"].sum()]]) / n_repeats
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["B", "M"], yticklabels=["B", "M"])
        ax.set_title(f"{name} (FN={cm[1, 0]:.1f}, FP={cm[0, 1]:.1f})", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "nested_confusion_per_pass.png"), dpi=200)
    plt.close()

    # (d) pairwise MCC significance
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(pmat.astype(float), annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=0.2,
                cbar_kws={"label": "Holm-adjusted p (MCC)"}, mask=np.eye(len(models), dtype=bool))
    plt.title("Pairwise corrected t-tests on MCC (green = significant)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "nested_pairwise_mcc.png"), dpi=200)
    plt.close()
    print(f"\nSaved tables in ./{IN}/ and figures in ./{FIG}/")


if __name__ == "__main__":
    main()
