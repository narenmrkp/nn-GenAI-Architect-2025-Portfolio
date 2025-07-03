📈 Module-3: Supervised Learning – Regression Models
📁 GitHub Folder: ML-Module-3-Regression/

📘 1. Overview of Regression
Regression is a Supervised ML technique used to predict continuous numerical values.

🔧 Common Regression Models:
Model	Use Case	Notes
🔹 Linear Regression	House Price Prediction	Straight-line fit
🔹 Polynomial Regression	Curve-like patterns	Overfitting risk
🔹 Ridge / Lasso	Avoid overfitting	Adds regularization

🎯 Mini Project: House Price Prediction
Dataset: Boston Housing (from sklearn.datasets)
Target: Predict MEDV (Median House Value)

# 📈 Module-3: Regression Models – House Price Prediction

This module introduces common **regression algorithms** using the **Boston Housing dataset**.

## 🧠 Models Covered
- ✅ Linear Regression
- ✅ Polynomial Regression
- ✅ Ridge & Lasso Regression

## 📘 Dataset: Boston Housing
- 506 entries, 13 features (e.g., crime rate, distance to employment centers)
- Target: `MEDV` – Median House Price

## ⚙️ Tools Used
- `sklearn.datasets`, `LinearRegression`, `PolynomialFeatures`, `Ridge`, `Lasso`
- `r2_score`, `mean_squared_error` for evaluation

## 📊 Output Summary
| Model | R² Score |
|-------|----------|
| Linear | 0.73 |
| Polynomial | 0.79 |
| Ridge | 0.74 |
| Lasso | 0.72 |

## 📸 Screenshots
![Data Preview](screenshots/data-head.png)  
![True vs Predicted](screenshots/true-vs-pred.png)

---

## 🧠 Insights
- Polynomial model performs better on non-linear patterns
- Ridge prevents overfitting via regularization
- Lasso can reduce feature set by shrinking coefficients to zero

## 🚀 Next Module:
➡ [Module-4: Classification Models – KNN, SVM, RandomForest](../ML-Module-4-Classification/)
