# MODULE-8: Explainable AI (SHAP & LIME)
# -------------------------------------
# 🧠 Goal: Predict diabetes + explain predictions using SHAP & LIME

# 📦 Install packages
!pip install shap lime --quiet
-------------------------------------------------------------------------
# 📌 Step 1: Imports
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from lime import lime_tabular
import warnings
warnings.filterwarnings("ignore")
----------------------------------------------------------------------
# 📌 Step 2: Load Dataset (Pima Indians Diabetes Dataset)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
df.head()
---------------------------------------------------------------------
# 📌 Step 3: Data Split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
---------------------------------------------------------------------
# 📌 Step 4: Model Training – Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
-----------------------------------------------------------------------------------
# 📌 Step 5: SHAP Explanation – Global & Local
# ✅ SHAP FIXED VERSION
import shap

# Use new explainer interface
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Plot summary (global)
shap.summary_plot(shap_values, X_test)
---------------------------------------------------------------------------------
# ✅ Best for Colab, forces plot as image
shap.force_plot(
    base_value = explainer.expected_value[class_index],
    shap_values = shap_values.values[sample_idx, :, class_index],
    features = X_test.iloc[sample_idx],
    matplotlib=True
)
---------------------------------------------------------------------------
# ✅ STEP 7: LIME Explanation for the same sample
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['No Diabetes', 'Diabetes'],
    mode='classification'
)

lime_exp = explainer_lime.explain_instance(
    data_row=X_test.iloc[sample_idx],
    predict_fn=model.predict_proba
)

lime_exp.show_in_notebook(show_table=True)
-----------------------------------------------------------------------
# Save SHAP values to CSV if needed
# Extract SHAP values for class 1 (index 1)
shap_values_class1 = shap_values.values[:, :, 1]

# Convert to DataFrame and save
pd.DataFrame(shap_values_class1, columns=X.columns).to_csv("shap_values_class1.csv", index=False)


# Save model for reuse (optional)
import joblib
joblib.dump(model, "rf_diabetes_model.pkl")
---------------------------------------------------------------------------------
################################### End of Module-8(Project) ###################
