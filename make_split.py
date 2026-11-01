"""
Create the shared stratified 80/20 train/test split used by EVERY model
(Logistic Regression notebook, tree models, and compare_all_models.py).

Input : cleaned_data.csv
Output: train_unscaled.csv, test_unscaled.csv   (same columns as the input)

Run once:  python make_split.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

df = pd.read_csv("cleaned_data.csv")
train, test = train_test_split(df, test_size=0.2, stratify=df["diagnosis"],
                               random_state=SEED)
train.to_csv("train_unscaled.csv", index=False)
test.to_csv("test_unscaled.csv", index=False)
print(f"train_unscaled.csv: {train.shape}, malignant share {train['diagnosis'].mean():.3f}")
print(f"test_unscaled.csv : {test.shape}, malignant share {test['diagnosis'].mean():.3f}")
