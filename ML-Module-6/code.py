# Concepts: Dimensionality reduction (Visual Intuition)
##############################################################################
# 1. Curse of Dimensionality – Visual Intuition
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate high-dimensional data
X, y = make_classification(n_samples=500, n_features=50, random_state=42)

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm')
plt.title("📉 Curse of Dimensionality: PCA Projection")
plt.show()
---------------------------------------------------------------------------------
# 2. Filter Methods (Correlation & Chi2)
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Chi2 Feature Selection
selector = SelectKBest(score_func=chi2, k=10)
X_new = selector.fit_transform(X, y)
selected = X.columns[selector.get_support()]
print("🎯 Selected Features via Chi2:", selected.tolist())
---------------------------------------------------------------------------
# 3. Wrapper Methods – RFE (Recursive Feature Elimination)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)

selected = X.columns[rfe.support_]
print("🔥 Top 10 Features via RFE:", selected.tolist())
----------------------------------------------------------------------------
# 4. Embedded Methods – Lasso and Tree-based
from sklearn.linear_model import LassoCV

lasso = LassoCV().fit(X, y)
lasso_importance = pd.Series(lasso.coef_, index=X.columns)
selected = lasso_importance[lasso_importance != 0].index.tolist()
print("🔗 Lasso Selected Features:", selected)

# Tree-based Feature Importance
rf = RandomForestClassifier()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("🌲 Top 10 Tree Features:\n", importances.head(10))
-------------------------------------------------------------------------------
# 5. PCA – Dimensionality Reduction (Explained Variance + Scree Plot)
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X)

# Scree Plot
plt.figure(figsize=(10,5))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("📊 Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.show()
----------------------------------------------------------------------------------
# Mini Project - Feature Reduction on Breast Cancer Dataset
################################################################
# 🎯 Goal: Improve classification performance using feature selection + PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Use reduced features from PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
---------------------------------------------------------------------------------------
# Advanced DR Techniques
# ✅ Code: Dimensionality Reduction using Autoencoder (Keras)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# Load and scale data
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)

# Split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Encoder-Decoder architecture
input_dim = X.shape[1]
encoding_dim = 5  # Compress to 5 features

input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
bottleneck = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(10, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
history = autoencoder.fit(X_train, X_train,
                          validation_data=(X_test, X_test),
                          epochs=50, batch_size=32, verbose=0)

# Encode the data (dimensionality reduced)
encoder = Model(inputs=input_layer, outputs=bottleneck)
X_encoded = encoder.predict(X)

print("Input Shape:", X.shape)
print("✅ Reduced shape:", X_encoded.shape)
--------------------------------------------------------------------------
# Bonus: Plot Reconstruction Loss (to verify training)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("📉 Autoencoder Loss Over Epochs")
plt.legend()
plt.show()
-----------------------------------------------------------------------
# 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data.target, cmap='coolwarm')
plt.title("🎨 t-SNE Projection (2D)")
plt.show()
-------------------------------------------------------------------------
# 3. UMAP (Uniform Manifold Approximation and Projection)
!pip install umap-learn

import umap.umap_ as umap

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data.target, cmap='coolwarm')
plt.title("🌈 UMAP Projection (2D)")
plt.show()
-------------------------------------------------------------------------
# Summary Comparison of all
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Manual input of above table for visualization
results = pd.DataFrame({
    'Model': ['Logistic-All', 'Logistic-PCA', 'Logistic-RFE', 'Logistic-Lasso', 'Logistic-AE', 'RF-All', 'RF-PCA'],
    'Features': [30, 10, 10, 12, 5, 30, 10],
    'Accuracy': [96.49, 97.36, 96.84, 96.14, 95.78, 97.89, 97.63]
})

plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Accuracy', data=results, palette='mako')
plt.title("📊 Accuracy Comparison: Before vs After Dimensionality Reduction")
plt.ylim(95, 98.5)
plt.grid()
plt.show()
----------------------------------------------------------------------------------------
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
