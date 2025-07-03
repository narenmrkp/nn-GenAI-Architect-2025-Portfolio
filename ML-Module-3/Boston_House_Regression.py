# 🏠 House Price Prediction – Regression Models

# 🔽 Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 🔽 Load Dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')

# 🔽 Data Exploration
print("Shape:", X.shape)
print(X.head())

# 🔽 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 📌 2. Polynomial Regression (degree=2)
poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

# 📌 3. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 📌 4. Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# 🔽 Compare Models
print("\n📊 Model Performance:")
print(f"Linear R2: {r2_score(y_test, y_pred_lr):.2f}")
print(f"Poly R2: {r2_score(y_test, y_pred_poly):.2f}")
print(f"Ridge R2: {r2_score(y_test, y_pred_ridge):.2f}")
print(f"Lasso R2: {r2_score(y_test, y_pred_lasso):.2f}")

# 🔽 Visualization: True vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_lr, label="Linear", alpha=0.7)
plt.scatter(y_test, y_pred_poly, label="Polynomial", alpha=0.7)
plt.scatter(y_test, y_pred_ridge, label="Ridge", alpha=0.7)
plt.scatter(y_test, y_pred_lasso, label="Lasso", alpha=0.7)
plt.plot([0, 50], [0, 50], 'k--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("📈 True vs Predicted House Prices")
plt.legend()
plt.show()
