🔧 Module-5: Model Tuning & Ensemble Learning
🎯 Goal: Improve accuracy by optimizing hyperparameters and combining models.

🧠 ML Module-5: Hyperparameter Tuning & Ensemble Learning – Diabetes Prediction 🧪
🔍 Optimizing classification performance using GridSearchCV, Cross-Validation, and Ensemble Models like Voting & Stacking.

📌 Project Overview
This project focuses on improving prediction accuracy using advanced machine learning techniques:

✅ Hyperparameter tuning with GridSearchCV
✅ Cross-validation to reduce overfitting
✅ Powerful ensemble models like:
  🤝 VotingClassifier (soft voting)
  🔗 StackingClassifier (meta-learning)
✅ Model Evaluation via:
  Accuracy, Precision, Recall, F1-Score
  ROC-AUC and Confusion Matrix

📁 Dataset Used
📄 Pima Indians Diabetes Dataset
Source: UCI / Brownlee GitHub
Total Records: 768 | Features: 8 | Target: Outcome (0/1)

⚙️ Models Trained & Tuned:
| 🔧 Model             | ⬆️ Tuning Method      | 🧠 Description                          |
| -------------------- | --------------------- | --------------------------------------- |
| `SVC` (SVM)          | GridSearchCV + CV(5)  | Finds best `C`, `kernel`                |
| `Random Forest`      | GridSearchCV + CV(5)  | Tunes `n_estimators`, `max_depth`       |
| `VotingClassifier`   | Soft voting ensemble  | Combines SVM, RF, Logistic Regression   |
| `StackingClassifier` | Meta-learner (LogReg) | Combines KNN + DecisionTree base models |

📊 Performance Summary:
| Model                 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------------------- | -------- | --------- | ------ | -------- | ------- |
| ✅ VotingClassifier    | `0.79`   | `0.77`    | `0.74` | `0.75`   | `0.83`  |
| 🔗 StackingClassifier | `0.78`   | `0.76`    | `0.73` | `0.74`   | `0.81`  |
| SVM (Tuned)           | `0.76`   | `0.75`    | `0.71` | `0.73`   | `0.80`  |
| RF (Tuned)            | `0.75`   | `0.74`    | `0.69` | `0.71`   | `0.78`  |

📌 Best performance achieved with VotingClassifier!

📈 Visualizations (Uploaded)
🔍 Metric	📸 Screenshot
Confusion Matrix – SVM	
Confusion Matrix – RF	
Confusion Matrix – Voting	
Confusion Matrix – Stacking	
ROC Curve Comparison	
Model Comparison Table	

💾 Saved Models
All models and scaler are saved as .pkl files:
models/
├── svm_tuned.pkl
├── rf_tuned.pkl
├── voting_model.pkl
├── stack_model.pkl
├── scaler.pkl
Use this to reload models for deployment or inference.

🚀 Tech Stack
scikit-learn
GridSearchCV, VotingClassifier, StackingClassifier
pandas, matplotlib, seaborn
.pkl file saving for production

🏁 How to Run
# Clone this repo or open in Colab
!pip install scikit-learn
!python Diabetes_Tuned_Classification.ipynb

🤝 Author
Nandyala Narendra
🔗 GitHub | 🔗 LinkedIn

📢 Recruiter Note
This project demonstrates my expertise in:
  Practical model improvement using tuning
  Model selection based on multiple metrics
  Ensemble learning & robust evaluation
  Deployable models using .pkl files
