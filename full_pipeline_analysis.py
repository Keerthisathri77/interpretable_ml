"""
FULL DATA + MODEL + INTERPRETABILITY PIPELINE

This script:
1. Loads Breast Cancer dataset
2. Performs basic data engineering analysis
3. Visualizes statistics
4. Trains Random Forest
5. Evaluates performance
6. Generates feature importance
7. Generates SHAP explanations
8. Saves all outputs as files

Everything is saved locally so you can inspect outputs clearly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Create output folder
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("\n=== DATASET INFO ===")
print("Shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# Save feature names
pd.DataFrame(X.columns, columns=["Feature Names"]).to_csv("outputs/feature_names.csv", index=False)

# ---------------------------------------------------------
# 2. DATA ENGINEERING CHECKS
# ---------------------------------------------------------

print("\n=== NULL CHECK ===")
print(X.isnull().sum())

# Summary statistics
summary = X.describe()
summary.to_csv("outputs/summary_statistics.csv")

print("\nSummary statistics saved.")

# ---------------------------------------------------------
# 3. CORRELATION HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(12,10))
corr = X.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("Correlation heatmap saved.")

# ---------------------------------------------------------
# 4. FEATURE DISTRIBUTION (Top 5 Important Later)
# ---------------------------------------------------------

# Plot first 5 features distribution
for col in X.columns[:5]:
    plt.figure()
    sns.histplot(X[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/distribution_{col}.png")
    plt.close()

print("Feature distributions saved.")

# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\nModel accuracy:", accuracy)

# Save classification report
report = classification_report(y_test, model.predict(X_test), output_dict=True)
pd.DataFrame(report).transpose().to_csv("outputs/classification_report.csv")

# Confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test))

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Confusion matrix saved.")

# ---------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------

importances = model.feature_importances_

feat_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

feat_importance_df.to_csv("outputs/feature_importance.csv", index=False)

plt.figure(figsize=(8,6))
plt.barh(feat_importance_df["feature"][:10], feat_importance_df["importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("outputs/random_forest_importance.png")
plt.close()

print("Feature importance saved.")

# ---------------------------------------------------------
# 7. SHAP EXPLANATION FOR ONE INSTANCE
# ---------------------------------------------------------

explainer = shap.TreeExplainer(model)
instance = X_test.iloc[[0]]

shap_values = explainer.shap_values(instance)

shap_array = np.array(shap_values)

# shape: (1, 30, 2)
shap_vals = shap_array[0, :, 1]

shap_df = pd.DataFrame({
    "feature": X.columns,
    "shap_value": shap_vals
})

shap_df["abs"] = shap_df["shap_value"].abs()
shap_df = shap_df.sort_values("abs", ascending=False)

shap_df.to_csv("outputs/shap_values_instance.csv", index=False)

plt.figure(figsize=(8,6))
plt.barh(shap_df["feature"][:10], shap_df["shap_value"][:10])
plt.axvline(0, color='black')
plt.gca().invert_yaxis()
plt.title("Top SHAP Contributions (Instance Level)")
plt.tight_layout()
plt.savefig("outputs/shap_instance_plot.png")
plt.close()

print("SHAP explanation saved.")

print("\nALL OUTPUTS SAVED INSIDE /outputs FOLDER")