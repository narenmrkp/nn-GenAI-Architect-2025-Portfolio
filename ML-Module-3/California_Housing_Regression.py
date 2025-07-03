# 🏠 California Housing – Regression Models

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 🔽 Load Dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print("Dataset shape:", X.shape)
X.head()

# 🔽 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# 🔧 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 1. Linear Regression
lr = LinearRegression().fit(X_train_scaled, y_train)
y_lr = lr.predict(X_test_scaled)

# 📌 2. Polynomial Regression (degree=2)
poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly.fit(X_train_scaled, y_train)
y_poly = poly.predict(X_test_scaled)

# 📌 3. Ridge Regression
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
y_ridge = ridge.predict(X_test_scaled)

# 📌 4. Lasso Regression
lasso = Lasso(alpha=0.1).fit(X_train_scaled, y_train)
y_lasso = lasso.predict(X_test_scaled)

# 🔽 Model Performance
results = {
    "Model": ["Linear", "Polynomial (deg2)", "Ridge", "Lasso"],
    "R² Score": [
        r2_score(y_test, y_lr),
        r2_score(y_test, y_poly),
        r2_score(y_test, y_ridge),
        r2_score(y_test, y_lasso),
    ],
    "MSE": [
        mean_squared_error(y_test, y_lr),
        mean_squared_error(y_test, y_poly),
        mean_squared_error(y_test, y_ridge),
        mean_squared_error(y_test, y_lasso),
    ]
}
print(pd.DataFrame(results))

# 🔽 True vs Predicted Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_lr, label="Linear", alpha=0.5)
plt.scatter(y_test, y_poly, label="Polynomial", alpha=0.5)
plt.scatter(y_test, y_ridge, label="Ridge", alpha=0.5)
plt.scatter(y_test, y_lasso, label="Lasso", alpha=0.5)
plt.plot([0, 5], [0, 5], 'k--', linewidth=2)
plt.xlabel("Actual Median Value")
plt.ylabel("Predicted")
plt.legend()
plt.title("🏠 Actual vs Predicted House Values")
plt.show()
