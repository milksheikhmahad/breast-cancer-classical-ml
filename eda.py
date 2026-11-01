"""
Exploratory data analysis for cleaned_data.csv (WDBC).
Run: python eda.py
Outputs: results/eda_summary.csv, results/eda_high_correlations.csv,
         results/figures/eda_*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)
sns.set_theme(style="whitegrid")
df = pd.read_csv("cleaned_data.csv")
X = df.drop(columns="diagnosis")
label = df["diagnosis"].map({0: "Benign", 1: "Malignant"})

print("shape:", df.shape, "| missing:", int(df.isna().sum().sum()),
      "| duplicates:", int(df.duplicated().sum()))
print(label.value_counts().to_string())

X.describe().T.round(4).to_csv("results/eda_summary.csv")

# class balance
plt.figure(figsize=(4, 3.6))
ax = sns.countplot(x=label, order=["Benign", "Malignant"], palette=["#2a6f97", "#c0392b"])
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.xlabel(""); plt.ylabel("Samples"); plt.title("Class balance")
plt.tight_layout(); plt.savefig("results/figures/eda_class_balance.png", dpi=200); plt.close()

# distributions of 6 informative features
feats = ["radius_mean", "texture_mean", "area_mean", "concavity_mean",
         "concave points_mean", "smoothness_mean"]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, f in zip(axes.ravel(), feats):
    sns.histplot(data=df.assign(Class=label), x=f, hue="Class", kde=True, bins=30,
                 palette={"Benign": "#2a6f97", "Malignant": "#c0392b"}, ax=ax)
    ax.set_title(f, fontsize=10)
plt.tight_layout(); plt.savefig("results/figures/eda_histograms.png", dpi=200); plt.close()

# box plots (standardised so one axis fits all 10 'mean' features)
mean_cols = [c for c in X.columns if c.endswith("_mean")]
Z = (X[mean_cols] - X[mean_cols].mean()) / X[mean_cols].std()
long = Z.assign(Class=label).melt(id_vars="Class", var_name="feature", value_name="z")
long["feature"] = long["feature"].str.replace("_mean", "")
plt.figure(figsize=(11, 4.6))
sns.boxplot(data=long, x="feature", y="z", hue="Class", showfliers=True, fliersize=2,
            palette={"Benign": "#2a6f97", "Malignant": "#c0392b"})
plt.xticks(rotation=35, ha="right"); plt.ylabel("Standardised value"); plt.xlabel("")
plt.title("Box plots of the ten mean-value features by class")
plt.tight_layout(); plt.savefig("results/figures/eda_boxplots.png", dpi=200); plt.close()

# correlation matrix
corr = X.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .7},
            xticklabels=True, yticklabels=True)
plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6)
plt.title("Feature correlation matrix")
plt.tight_layout(); plt.savefig("results/figures/eda_correlation.png", dpi=200); plt.close()

# highly correlated pairs
iu = np.triu_indices_from(corr, k=1)
pairs = pd.DataFrame({"feature_a": corr.index[iu[0]], "feature_b": corr.columns[iu[1]],
                      "correlation": corr.values[iu]})
hi = pairs[pairs["correlation"].abs() > 0.9].sort_values("correlation", ascending=False)
hi.round(4).to_csv("results/eda_high_correlations.csv", index=False)
print(f"pairs with |r| > 0.90: {len(hi)}")
print(hi.head(8).to_string(index=False))

# outliers (IQR rule, per feature)
q1, q3 = X.quantile(.25), X.quantile(.75)
iqr = q3 - q1
out = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum()
print("features with most IQR outliers:\n", out.sort_values(ascending=False).head(5).to_string())
print("saved EDA figures in results/figures/")
