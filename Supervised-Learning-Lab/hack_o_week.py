"""
SIT-N Hack-O-Week – Week 7 & 8
Supervised Learning Lab
Author: Chhatrapati
Batch: 2024-28 | Semester: 5
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# ============================================================
# PART 1: REGRESSION
# Dataset: Diabetes (sklearn built-in)
# Models: Linear, Polynomial, Ridge, Lasso
# ============================================================
print("=" * 60)
print("PART 1: REGRESSION - Diabetes Dataset")
print("=" * 60)

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Polynomial Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ]),
}

reg_results = []
for name, model in reg_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    reg_results.append({
        "Model": name,
        "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 4),
        "R2": round(r2_score(y_test, pred), 4)
    })

reg_df = pd.DataFrame(reg_results)
print(reg_df.to_string(index=False))

# ============================================================
# PART 2: CLASSIFICATION
# Dataset: Breast Cancer (sklearn built-in)
# Models: Logistic Regression, KNN
# ============================================================
print("\n" + "=" * 60)
print("PART 2: CLASSIFICATION - Breast Cancer Dataset")
print("=" * 60)

Xc, yc = load_breast_cancer(return_X_y=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)

clf_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "KNN (k=5)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
}

clf_results = []
for name, model in clf_models.items():
    model.fit(Xc_train, yc_train)
    pred = model.predict(Xc_test)
    clf_results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(yc_test, pred), 4),
        "Precision": round(precision_score(yc_test, pred), 4),
        "Recall": round(recall_score(yc_test, pred), 4),
        "F1": round(f1_score(yc_test, pred), 4)
    })

clf_df = pd.DataFrame(clf_results)
print(clf_df.to_string(index=False))

print("\n" + "=" * 60)
print("DONE - All 6 models trained and evaluated")
print("=" * 60)