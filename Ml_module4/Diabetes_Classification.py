# 🤖 Full Code: Module-4 – Diabetes Prediction (Classification Models)
# 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 📥 Load Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigree","Age","Outcome"]
df = pd.read_csv(url, names=cols)

# 🧼 Replace zeros with median (for valid columns)
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col] = df[col].replace(0, df[col].median())

# 🎯 Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 🔀 Split + Scale
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Train Models
# 🤖 Initialize Models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# 📊 Train + Evaluate Each Model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n🔹 {name} Classification Report")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

# 📈 Final ROC Curve
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📈 ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 📊 Final Model Comparison Table
# 📊 Model Comparison Summary Table
summary = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    summary.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

summary_df = pd.DataFrame(summary).sort_values(by="Accuracy", ascending=False)
summary_df.reset_index(drop=True, inplace=True)

# 📋 Display with colors
summary_df.style.background_gradient(cmap="Blues").format(precision=3)

# 💾 Save All Models as .pkl
# 💾 Save Trained Models
with open("model_knn.pkl", "wb") as f:
    pickle.dump(models["KNN"], f)

with open("model_svm.pkl", "wb") as f:
    pickle.dump(models["SVM"], f)

with open("model_tree.pkl", "wb") as f:
    pickle.dump(models["Decision Tree"], f)

with open("model_rf.pkl", "wb") as f:
    pickle.dump(models["Random Forest"], f)

# 💾 Save the scaler too
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
