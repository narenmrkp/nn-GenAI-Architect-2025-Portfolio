📈 Module-3: Regression Models – House Price Prediction (California Housing)
📁 GitHub Folder: ML-Module-3/

📘 1. Overview of Regression
Regression is a Supervised ML technique used to predict continuous numerical values.

🔧 Common Regression Models:
Model	Use Case	Notes
🔹 Linear Regression	House Price Prediction	Straight-line fit
🔹 Polynomial Regression	Curve-like patterns	Overfitting risk
🔹 Ridge / Lasso	Avoid overfitting	Adds regularization

📘 1. About the Dataset
The California Housing dataset (from sklearn.datasets) contains ~20,000 rows with features like:
MedInc, HouseAge, AveRooms, AveBedrms, Population, Latitude, Longitude
Target: Median House Value (MedHouseVal)
Ideal for regression tasks

# 📈 Module-3: Regression Models – California Housing Prediction

Explore regression techniques using the **California Housing** dataset (~20K rows).

## Models
- **Linear Regression**
- **Polynomial Regression (degree=2)**
- **Ridge Regression**
- **Lasso Regression**

## Dataset Features
Includes features like `MedInc`, `HouseAge`, `AveRooms`, and more.

## Evaluation Metrics
- **R² Score** (higher is better)
- **Mean Squared Error** (lower is better)

## 📊 Results Summary
| Model | R² Score | MSE |
|-------|----------|-----|
| Linear | 0.60 | 0.48 |
| Polynomial | 0.64 | 0.43 |
| Ridge | 0.61 | 0.47 |
| Lasso | 0.59 | 0.49 |

## 📸 Screenshots
Uploaded
---

## Insights
- Polynomial model captures non-linear trends best.
- Ridge and Lasso add stability via regularization.
- Lasso may reduce feature weights to zero.

## ➡ Next:
Module-4: Supervised Learning – **Classification Models** (KNN, SVM, Random Forest)

