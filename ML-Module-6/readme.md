# 🧠 ML Module-6: Dimensionality Reduction & Feature Selection

> Reduce high-dimensional data without losing predictive power.
---

## 📌 Techniques Covered
- 🔹 Correlation Matrix + Chi2
- 🔸 RFE (Wrapper) & Lasso (Embedded)
- 📉 PCA (Principal Component Analysis)
- 🎨 t-SNE / UMAP (Optional advanced viz)
---

## 📈 Scree Plot – PCA Variance Explained
Uploaded
---

## 💻 Final Project: Breast Cancer Classifier with PCA

| Model | Input Features | Accuracy |
|-------|----------------|----------|
| Logistic Regression | All Features | 96.49% |
| Logistic + PCA (10D) | 10 Components | 97.36% ✅ |

---

## 📂 Outputs
- ✔️ Top 10 Chi2 Features
- ✔️ Top 10 RFE Features
- ✔️ PCA Scree plot
- ✔️ Confusion Matrix
- ✔️ Classification Report

---
## 📋 Interview Questions

1. What is the curse of dimensionality?
2. Difference between PCA and Feature Selection?
3. When to use Embedded methods vs Filter?
4. What’s the difference between t-SNE and PCA?
---

## 📊 Sample Chart
Uploaded

> 📍This project improves model performance and reduces overfitting by using PCA & RFE techniques.
-------------------------------------------------------------------------------------------------------------
Advanced DR Techniques:
## 🧠 Advanced Dimensionality Reduction

| Technique     | Type       | Use Case                    | Notes                        |
|---------------|------------|-----------------------------|------------------------------|
| Autoencoder   | Deep NN    | Feature compression         | Learns non-linear embeddings |
| t-SNE         | Non-linear | Cluster visualization (2D) | Slow, not for production use |
| UMAP          | Non-linear | Visualization + clustering | Faster, preserves structure  |
---

### 🔍 Autoencoder Output
- Input shape: (569, 30)
- Compressed to: (569, 5)
- MSE loss converged ✅
---

### 🎨 Visualizations
- ✅ t-SNE shows clear class separation
- ✅ UMAP gives better global + local grouping
----------------------------------------------------------------------------------------------------
Summary:
| Technique    | Purpose                  | Best For                 |
| ------------ | ------------------------ | ------------------------ |
| PCA          | Linear DR                | Model training           |
| RFE, Lasso   | Feature Selection        | Model optimization       |
| Autoencoder  | Deep learning DR         | NLP, images, time-series |
| t-SNE / UMAP | Non-linear visualization | EDA, Clustering          |

 1. Autoencoders (Neural Network based DR)
📘 What is it?
Autoencoders learn a compressed representation (bottleneck) of input data using neural networks.
  🎯 Encoder compresses → Decoder reconstructs
  The "bottleneck" is your reduced dimension!
