# 📁 Project: Diabetes Prediction with Advanced Tuning
# 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 📥 Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigree","Age","Outcome"]
df = pd.read_csv(url, names=cols)

# 🧼 Replace zeros in selected columns
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col] = df[col].replace(0, df[col].median())

# 🎯 Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 🔀 Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
--------------------------------------------------------------------------------------------------
# 🔧 GridSearchCV + Cross Validation for SVM & RandomForest
# 🔍 SVM Hyperparameter Tuning
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid_svm = GridSearchCV(SVC(probability=True), svm_params, cv=5, scoring='accuracy')
grid_svm.fit(X_train, y_train)
svm_best = grid_svm.best_estimator_

# 🌲 Random Forest Tuning
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, None]
}
grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)
rf_best = grid_rf.best_estimator_

print("✅ Best SVM:", grid_svm.best_params_)
print("✅ Best RF:", grid_rf.best_params_)
---------------------------------------------------------------------------------------------------
# 🧠 Ensemble Models (Voting + Stacking)
# 🤝 Voting Classifier (Hard Voting)
voting_model = VotingClassifier(estimators=[
    ('svm', svm_best),
    ('rf', rf_best),
    ('lr', LogisticRegression())
], voting='soft')  # use 'soft' for probability averaging
voting_model.fit(X_train, y_train)

# 🔗 Stacking Classifier
stack_model = StackingClassifier(estimators=[
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier())
], final_estimator=LogisticRegression())
stack_model.fit(X_train, y_train)
-------------------------------------------------------------------------------------------------
# 📊 Evaluation of All Models
models = {
    "SVM (Tuned)": svm_best,
    "Random Forest (Tuned)": rf_best,
    "VotingClassifier": voting_model,
    "StackingClassifier": stack_model
}

summary = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    summary.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    # 📉 Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # 📈 ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

# 🧾 Final ROC Plot
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📈 ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 📋 Display Table
summary_df = pd.DataFrame(summary).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
summary_df.style.background_gradient(cmap="Blues").format(precision=3)
-------------------------------------------------------------------------------------------------------------------------
# 💾 Save All Tuned/Ensemble Models
with open("svm_tuned.pkl", "wb") as f:
    pickle.dump(svm_best, f)

with open("rf_tuned.pkl", "wb") as f:
    pickle.dump(rf_best, f)

with open("voting_model.pkl", "wb") as f:
    pickle.dump(voting_model, f)

with open("stack_model.pkl", "wb") as f:
    pickle.dump(stack_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

