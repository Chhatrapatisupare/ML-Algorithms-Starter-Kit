# Supervised Learning Lab

**SIT-N Hack-O-Week – Week 7 & 8 Submission**  
**Student:** Chhatrapati  
**Batch:** 2024-28 | **Semester:** 5th

---

## Overview

A single Python script that trains, evaluates, and compares **6 supervised learning algorithms** on two standard datasets.

---

## Tasks Covered

### Regression
- Linear Regression
- Polynomial Regression (degree 2)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)

### Classification
- Logistic Regression
- K-Nearest Neighbors (KNN, k=5)

---

## Datasets

| Task | Dataset | Source |
|---|---|---|
| Regression | Diabetes progression | sklearn.datasets.load_diabetes |
| Classification | Breast Cancer Wisconsin | sklearn.datasets.load_breast_cancer |

Both are built into scikit-learn - no external download required.

---

## Project Structure

    Supervised-Learning-Lab/
    |-- hack_o_week.py       # Main script
    |-- results.txt          # Sample output
    |-- requirements.txt     # Dependencies
    |-- README.md            # This file
    |-- .gitignore           # Git ignore rules
    |-- venv/                # Virtual environment (not pushed)

---

## Setup and Run

### 1. Clone the repository
    git clone <your-repo-url>
    cd Supervised-Learning-Lab

### 2. Create and activate virtual environment
    python -m venv venv
    venv\Scripts\activate

### 3. Install dependencies
    pip install -r requirements.txt

### 4. Run the script
    python hack_o_week.py

---

## Results

### Regression (Diabetes Dataset)

| Model | RMSE | R2 |
|---|---|---|
| Linear Regression | 53.8534 | 0.4526 |
| Polynomial Regression | 55.6420 | 0.4156 |
| Ridge Regression | 53.7775 | 0.4541 |
| Lasso Regression | 53.7087 | 0.4555 |

### Classification (Breast Cancer Dataset)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| KNN (k=5) | 0.9561 | 0.9589 | 0.9722 | 0.9655 |

---

## Key Observations

- Regression: Ridge and Lasso slightly outperform plain Linear Regression by reducing overfitting. Polynomial degree=2 overfits the small diabetes dataset.
- Classification: Logistic Regression outperforms KNN on this dataset, likely because the classes are close to linearly separable after feature scaling.
- Preprocessing: All models use StandardScaler inside a Pipeline to prevent data leakage.

---

## Tech Stack

- Python 3.13
- NumPy, pandas
- scikit-learn
- Pipeline API for clean preprocessing

---

## Author

**Chhatrapati**  
5th Semester, Batch 2024-28  
Symbiosis Institute of Technology, Nagpur
