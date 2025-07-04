📘 Module-8: Explainable AI – SHAP & LIME
🔍 Build trust in ML predictions with visual explanations
🎯 Focus: Diabetes prediction using SHAP & LIME interpretability tools

📌 Overview
In this project, we build a Random Forest classifier to predict diabetes and apply SHAP and LIME to interpret how the model makes decisions — both globally and locally.

📊 Dataset
Name: Pima Indians Diabetes Dataset
Features: 8 medical indicators (e.g., Glucose, BMI, Age)
Target: Outcome (1 = Diabetes, 0 = No Diabetes)
Source: Kaggle / Plotly dataset

🔧 Tools & Libraries Used
| Tool             | Purpose                        |
| ---------------- | ------------------------------ |
| `RandomForest`   | ML Model                       |
| `SHAP`           | Global & local explanations    |
| `LIME`           | Instance-specific explanations |
| `Matplotlib`     | Visualization                  |
| `Pandas`/`NumPy` | Data handling                  |

🚀 Steps Covered
Load and explore the dataset
Train a Random Forest classifier
Generate accuracy metrics
Use SHAP for:
  Global feature importance (summary plot)
  Local explanation (force plot)
Use LIME for:
  Explaining individual predictions

📈 Model Performance
| Metric    | Value (may vary) |
| --------- | ---------------- |
| Accuracy  | ✅ \~76%          |
| Precision | ✅ Balanced       |
| Recall    | ✅ Balanced       |

🔷 SHAP Force Plot (Local)
Explains how each feature pushed prediction for a single row
<img src="https://shap.readthedocs.io/en/latest/_images/force_plot.png" width="700"/>

🟢 LIME Explanation
Tabular breakdown of a prediction’s local reasoning
<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/lime_tabular.png" width="500"/>

💾 Files in This Module
Module-8-ExplainableAI/
├── shap_lime_diabetes.ipynb      # Full working notebook
├── shap_values_class1.csv        # SHAP values (class 1 only)
├── rf_diabetes_model.pkl         # Trained model
├── README.md                     # This file (copy-paste below)

🧠 What You’ll Learn
✅ How SHAP explains models at global + local levels
✅ How LIME generates interpretable approximations for specific predictions
✅ Why interpretability is essential in real-world ML systems

🔜 What’s Next?
Continue to:

👉 Module-9: ML Deployment with FastAPI + Docker
Deploy this model as a REST API and serve real-time predictions!

🙌 Credits
Created by: Nandyala Narendra
Guided via: ML + XAI Series (Modules 1–10)
License: MIT
