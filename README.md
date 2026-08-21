# Overview
Predicts heart attack risk from clinical vitals (age, heart rate, BP, blood sugar, CK-MB, Troponin) using ML classifiers -- Random Forest achieves 98.7% accuracy.

# Heart Attack Prediction 🫀

A machine learning project that predicts the likelihood of a heart attack based on a patient's clinical parameters, using classical ML classifiers trained on medical vitals.

## Overview

The model takes in key cardiac and metabolic indicators — age, gender, heart rate, blood pressure, blood sugar, CK-MB, and Troponin levels — and classifies the outcome as **positive** or **negative** for a heart attack event.

## Dataset

- **1,319 patient records**, 9 features
- Features: `Age`, `Gender`, `Heart rate`, `Systolic blood pressure`, `Diastolic blood pressure`, `Blood sugar`, `CK-MB`, `Troponin`
- Target: `Result` (heart attack positive/negative)

## Workflow

1. **Preprocessing** — checked for missing values, clipped negative values, removed outliers using IQR and Z-score methods
2. **EDA & Visualization** — line charts (heart rate by age), bar charts (CK-MB by result), histograms (age distribution), pie charts (gender split), and a correlation heatmap
3. **Encoding** — label-encoded the target variable
4. **Modeling** — 80/20 train-test split with feature scaling (StandardScaler), then trained four classifiers:

| Model | Accuracy |
|---|---|
| **Random Forest** | **98.76%** |
| Decision Tree | 98.35% |
| SVM (linear) | 84.30% |
| Logistic Regression | 82.64% |

## Key Result

Tree-based ensemble methods (Random Forest, Decision Tree) significantly outperformed linear models (SVM, Logistic Regression) on this dataset, suggesting non-linear relationships between vitals like CK-MB/Troponin and heart attack risk.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `SciPy`

## How to Run

```bash
git clone <repo-url>
cd heart-attack-prediction
pip install -r requirements.txt
jupyter notebook Heart_Attack_Prediction.ipynb
```

## Future Improvements

- Hyperparameter tuning (GridSearchCV) for Random Forest
- Cross-validation instead of a single train-test split
- Feature importance analysis to identify top clinical predictors
- Handle class imbalance if present in the target variable
