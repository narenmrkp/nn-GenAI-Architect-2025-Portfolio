🧹✨ Module-2: Data Preprocessing & Exploratory Data Analysis (EDA)
📁 GitHub Folder: ML-Module-2-Data-Preprocessing/

📘 1. Why Preprocessing Matters?
Raw data is incomplete, inconsistent, and noisy. Before feeding it to ML models, we need to:

Handle missing values

Encode categorical variables

Scale/normalize numerical features

Reduce irrelevant data

🔧 2. Techniques Used
Step	Description	Tools
🔍 Missing Value Handling	Replace/remove NAs	SimpleImputer, dropna()
🔢 Encoding Categorical	Convert text to numbers	LabelEncoder, OneHotEncoder
📏 Feature Scaling	Normalize numeric values	MinMaxScaler, StandardScaler
✅ Feature Selection	Keep only useful features	Correlation, SelectKBest

# 🧹 Module-2: Data Preprocessing & EDA – Pima Diabetes Dataset

This notebook covers **data cleaning**, **missing value handling**, **feature scaling**, and **exploratory analysis** on the Pima Indians Diabetes dataset.

## 📚 What You'll Learn
- Handling missing data via median replacement (zeros in key features)
- Normalizing features with Min-Max scaling
- Visualizing outcome distribution
- Understanding feature relationships through correlation heatmap

## 🗂️ Dataset Info
- **Source**: UCI / UCI via GitHub :contentReference[oaicite:11]{index=11}  
- **Records**: 768  
- **Features**: 8 numeric attributes + 1 binary target (`Outcome`)

## ⚙️ Preprocessing Steps
1. Identify zeros in features where zero isn't valid.
2. Replace zeros with median values.
3. Scale all features between 0 and 1.

## 📊 EDA Highlights
- Balanced class distribution—roughly half positive, half negative.
- Correlation patterns (like strong glucose–outcome relationship)

## 📸 Screenshots
Uploaded
---

## 🧭 Next Steps
After preprocessing, Module 3 will focus on **Regression models** (linear, polynomial, ridge) and a **mini-project**: predicting continuous targets or adapting this dataset for classification models.

