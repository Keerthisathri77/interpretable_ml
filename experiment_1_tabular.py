"""
Experiment 1: LIME vs SHAP on Breast Cancer Dataset

Goal:
- Train a nonlinear model
- Explain one prediction using LIME
- Explain same prediction using SHAP
- Compare results

Dataset:
Breast Cancer (sklearn)
"""

import numpy as np
import pandas as pd
import shap
import lime
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
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 2. Train Model
# --------------------------

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

# --------------------------
# 3. Select Instance
# --------------------------

instance = X_test.iloc[0]

print("True label:", y_test.iloc[0])
print("Predicted probability:", model.predict_proba([instance])[0])

# --------------------------
# 4. LIME Explanation
# --------------------------

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns,
    class_names=["malignant", "benign"],
    mode="classification"
)

lime_exp = explainer_lime.explain_instance(
    instance.values,
    model.predict_proba,
    num_features=10
)

print("LIME explanation:")
print(lime_exp.as_list())

# --------------------------
# 5. SHAP Explanation
# --------------------------

explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(instance)

print("SHAP values for class 1 (benign):")
for feature, value in zip(X.columns, shap_values[1]):
    print(f"{feature}: {value}")