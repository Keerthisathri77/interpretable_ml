"""
Clean LIME vs SHAP comparison
Produces:
- Clean printed feature contributions
- Simple matplotlib bar chart
- No messy HTML
"""

import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# --------------------------
# 1. Load Data
# --------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 2. Train Model
# --------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

# --------------------------
# 3. Select Instance
# --------------------------

instance = X_test.iloc[[0]]

print("\nTrue label:", y_test.iloc[0])
print("Predicted probabilities:", model.predict_proba(instance)[0])


# ============================================================
# LIME
# ============================================================

print("\n=== LIME Explanation ===")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["malignant", "benign"],
    mode="classification",
    discretize_continuous=False
)

lime_exp = explainer_lime.explain_instance(
    instance.values[0],
    model.predict_proba,
    num_features=10
)

lime_results = lime_exp.as_list()

print("\nTop LIME features:")
for feature, weight in lime_results:
    print(f"{feature:35s} {weight:.4f}")


# ============================================================
# SHAP
# ============================================================

print("\n=== SHAP Explanation ===")

explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(instance)

print("SHAP shape:", np.array(shap_values).shape)

shap_array = np.array(shap_values)

# Your case: (1, 30, 2)
# We want: first sample, all features, positive class (index 1)

shap_vals = shap_array[0, :, 1]

base_value = explainer_shap.expected_value[1]

# Create DataFrame
shap_df = pd.DataFrame({
    "feature": X.columns,
    "shap_value": shap_vals
})

shap_df["abs"] = shap_df["shap_value"].abs()
shap_df = shap_df.sort_values("abs", ascending=False).head(10)

print("\nTop SHAP features:")
print(shap_df[["feature", "shap_value"]])

# Clean visualization
plt.figure(figsize=(8,6))
plt.barh(shap_df["feature"], shap_df["shap_value"])
plt.axvline(0, color='black')
plt.title("Top SHAP Feature Contributions")
plt.xlabel("SHAP value (impact on class 1)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("shap_barplot.png")
plt.close()

print("\nSHAP bar plot saved as shap_barplot.png")

# --------------------------
# Clean Visualization
# --------------------------

plt.figure(figsize=(8,6))
plt.barh(shap_df["feature"], shap_df["shap_value"])
plt.axvline(0, color='black')
plt.title("Top SHAP Feature Contributions")
plt.xlabel("SHAP value (impact on prediction)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("shap_barplot.png")
plt.close()

print("\nSHAP bar plot saved as shap_barplot.png")