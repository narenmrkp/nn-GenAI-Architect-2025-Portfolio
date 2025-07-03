# 👗 Fashion MNIST Classification — Autoencoder + ML Pipeline

> 🎯 Multi-Class Classification of clothing images using ML + Deep Learning + Feature Compression.
---

## 🧠 Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

- 60,000 training images, 10,000 test images
- 28×28 grayscale pixels → Flattened to 784 features
- 10 clothing classes:
["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
---

## 📌 Project Workflow

```mermaid
graph TD
A[Load & Flatten Images] --> B[EDA & Visualization]
B --> C[Normalize & Preprocess]
C --> D[Train ML Models]
D --> E[Autoencoder Compression]
E --> F[XGBoost on Compressed]
F --> G[Confusion Matrix]
G --> H[Final Comparison Table]

📊 Exploratory Data Analysis (EDA)
🔸 Sample Images
<p align="center"> <img src="https://i.imgur.com/FnWW0Pa.png" width="600"/> </p>

🤖 Models Used:
| Model                 | Description                         |
| --------------------- | ----------------------------------- |
| Logistic Regression   | Multiclass with softmax             |
| Random Forest         | 100 Estimators                      |
| XGBoost               | With early stopping                 |
| Autoencoder + XGBoost | Dimensionality reduction + Boosting |

🧠 Autoencoder — Feature Compression
| Step          | Shape                |
| ------------- | -------------------- |
| Original      | 784                  |
| Encoded Layer | **64** features      |
| Decoded Back  | 784 (reconstruction) |

Autoencoder:
Input → Dense(128) → Dense(64) → Dense(128) → Output(784)

📈 Confusion Matrix (XGBoost on Compressed)
🧪 Final Model Comparison Table
| Model                     | Accuracy   | F1-Weighted |
| ------------------------- | ---------- | ----------- |
| Logistic Regression       | 0.8436     | 0.8435      |
| Random Forest             | 0.8751     | 0.8740      |
| XGBoost (10k)             | 0.8817     | 0.8808      |
| **XGBoost (Autoencoded)** | **0.8920** | **0.8901**  |
✅ Autoencoder improves both speed (fewer features) and accuracy.

🛠 Technologies Used
Python, TensorFlow, Keras
Scikit-learn, XGBoost
Matplotlib, Seaborn
Pandas, NumPy

✅ Key Learnings
Flattening image data into ML-ready format
Building autoencoders to reduce input dimensionality
Using traditional ML classifiers on compressed features
Visualizing multiclass confusion matrix and EDA plots
Measuring F1 and Accuracy across multiple models

💬 Author
👤 Nandyala Narendra
🌐 GitHub: @narenmrkp

📌 Want to Try Yourself?
git clone https://github.com/yourname/fashion-mnist-autoencoder.git
cd fashion-mnist-autoencoder
jupyter notebook fashion_mnist_ml.ipynb

🌟 If you like it, give a ⭐ on GitHub!

---

### 📌 Note:
You can host this in a GitHub repo like:

fashion-mnist-autoencoder-ml/
├── README.md
├── fashion_mnist_ml.ipynb
└── images/
├── sample_images.png
├── class_dist.png
└── confusion_matrix.png

Want me to give the **`.ipynb` notebook export**, `.py` script, or Colab uploadable version too?



