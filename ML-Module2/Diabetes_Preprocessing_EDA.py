# 🩺 Diabetes Prediction – Preprocessing & EDA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 🔽 1. Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigree","Age","Outcome"]
df = pd.read_csv(url, header=None, names=columns)
df.head()

# 🔽 2. Missing Value Investigation
print("Null counts:\n", df.isnull().sum())
# Note: zeros represent missing in some features:
for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
    print(col, "zeros:", (df[col]==0).sum())

# 🔽 3. Replace zeros with median for missing values
for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
    median = df[df[col]!=0][col].median()
    df[col] = df[col].replace(0, median)

# 🔽 4. Normalize features
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df)

# 🔽 5. EDA – Outcome Distribution
sns.countplot(x='Outcome', data=df)
plt.title("Outcome Distribution (0=No Diabetes, 1=Diabetes)")
plt.show()

# 🔽 6. Correlation Heatmap
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("🔥 Feature Correlation Heatmap")
plt.show()

# 🔽 7. Final Data Preview
df.head()
