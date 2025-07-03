# ✅ Fashion MNIST Classification — Full ML Pipeline
#   🎯 Target: Classify images into 10 clothing types (T-shirt, Trouser, etc.)
# 🔰 Step 1: Preprocess — Reshape, Normalize
import numpy as np

# Flatten the 28x28 images to 784
X_train_flat = X_train.reshape(-1, 28 * 28)
X_test_flat = X_test.reshape(-1, 28 * 28)

# Normalize pixel values
X_train_flat = X_train_flat / 255.0
X_test_flat = X_test_flat / 255.0
---------------------------------------------------------------
# 📊 Step 2: EDA — Class Distribution, Visual Samples
import matplotlib.pyplot as plt
import seaborn as sns

# Class labels
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plot 10 images
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(labels[y_train[i]], fontsize=8)
    plt.axis('off')
plt.suptitle("Sample Fashion MNIST Images")
plt.show()

# Class distribution
sns.countplot(y_train)
plt.title("Class Distribution")
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.show()
----------------------------------------------------------------------
# ⚙️ Step 3: Train-Test Confirmation (Already Done)
print("X_train:", X_train_flat.shape, "| y_train:", y_train.shape)
print("X_test :", X_test_flat.shape, "| y_test :", y_test.shape)
-------------------------------------------------------------------------
# 🤖 Step 4: Baseline Model — Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

lr = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train_flat, y_train)
y_pred_lr = lr.predict(X_test_flat)

print("✅ Logistic Accuracy:", accuracy_score(y_test, y_pred_lr))
---------------------------------------------------------------------------
# 🌲 Step 5: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_flat, y_train)
y_pred_rf = rf.predict(X_test_flat)

print("✅ RF Accuracy:", accuracy_score(y_test, y_pred_rf))
----------------------------------------------------------------------------
# ⚡ Step 6: XGBoost Classifier
!pip install xgboost --quiet
from xgboost import XGBClassifier

xgb = XGBClassifier(objective='multi:softmax', num_class=10, eval_metric='mlogloss')
xgb.fit(X_train_flat[:10000], y_train[:10000])  # ✅ Only first 10k rows for speed
y_pred_xgb = xgb.predict(X_test_flat)

print("✅ XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
----------------------------------------------------------------------------------
# 📈 Step 7: Final Comparison Table
def summarize_model(name, y_pred):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1-Weighted": round(f1_score(y_test, y_pred, average='weighted'), 4)
    }

import pandas as pd

results = pd.DataFrame([
    summarize_model("Logistic Regression", y_pred_lr),
    summarize_model("Random Forest", y_pred_rf),
    summarize_model("XGBoost (10k rows)", y_pred_xgb),
])

print("📊 🔚 Final Model Comparison:")
print(results)
------------------------------------------------------------------------------------
# 🔁 Bonus-1: 📉 Autoencoder for Feature Compression (from 784 ➝ 64)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Input layer
input_dim = X_train_flat.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoding
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)

# Decoding
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train autoencoder
autoencoder.fit(X_train_flat, X_train_flat, epochs=5, batch_size=256, shuffle=True, validation_split=0.2)

# Extract compressed features
encoder = Model(input_layer, encoded)
X_train_compressed = encoder.predict(X_train_flat)
X_test_compressed = encoder.predict(X_test_flat)

print("✅ Compressed feature shape:", X_train_compressed.shape)
--------------------------------------------------------------------------------------------
# 🧠 Use Compressed Features in Classifier (e.g., XGBoost)
xgb_compressed = XGBClassifier(objective='multi:softmax', num_class=10, eval_metric='mlogloss')
xgb_compressed.fit(X_train_compressed[:10000], y_train[:10000])
y_pred_xgb_compressed = xgb_compressed.predict(X_test_compressed)

print("✅ XGBoost (Autoencoded) Accuracy:", accuracy_score(y_test, y_pred_xgb_compressed))
----------------------------------------------------------------------------------------------
# 📊 Bonus-2: Confusion Matrix Visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix for XGBoost (compressed)
cm = confusion_matrix(y_test, y_pred_xgb_compressed)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("📉 Confusion Matrix — XGBoost (Autoencoded Features)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
--------------------------------------------------------------------------------------------
