🧠 Module-11: Artificial Neural Networks (ANN) – Deep Learning Introduction
  🚀 First step into Deep Learning using TensorFlow/Keras
  🎯 Project: Breast Cancer Classification using ANN

📌 Overview
  This project marks the start of our Deep Learning journey by building and training a fully connected Artificial Neural Network (ANN) to classify tumors as benign or malignant using the Breast Cancer Wisconsin dataset.
We cover:
Neural network theory (forward + backward pass)
Activation functions
Keras Sequential model building
Metrics evaluation and training visualization

📊 Dataset Info
| Attribute   | Value                          |
| ----------- | ------------------------------ |
| 📁 Dataset  | Breast Cancer (from `sklearn`) |
| 🧪 Features | 30 real-valued inputs          |
| 🎯 Target   | `0` = Malignant, `1` = Benign  |
| 📦 Size     | 569 samples                    |

🔧 Tools & Libraries Used
| Tool               | Purpose                            |
| ------------------ | ---------------------------------- |
| `TensorFlow/Keras` | ANN model building & training      |
| `Sklearn`          | Dataset, preprocessing, evaluation |
| `Matplotlib`       | Visualizing accuracy & loss curves |
| `Seaborn`          | Confusion matrix heatmap           |

🧠 Neural Network Architecture
Input Layer: 30 neurons
Hidden Layer 1: 16 neurons (ReLU)
Hidden Layer 2: 8 neurons (ReLU)
Output Layer: 1 neuron (Sigmoid)

📈 Training Results
| Metric      | Result (approx) |
| ----------- | --------------- |
| Accuracy    | ✅ 96–97%        |
| Loss        | 📉 Low & stable |
| Overfitting | ❌ None observed |

🧪 Evaluation
Confusion Matrix:
[[40  2]
 [ 1 71]]

Classification Report:
              precision    recall  f1-score
           0     0.97       0.95     0.96
           1     0.97       0.99     0.98
✅ Great balance between sensitivity and specificity.

📂 Project Structure
Module-11-ANN-DeepLearning/
├── ann_breast_cancer.ipynb      # Full Google Colab notebook
├── model_summary.png            # Model architecture (optional)
├── README.md                    # This file

🧠 Concepts Covered
🧮 Neuron computation (Z = w·x + b)
🔁 Forward + backward propagation
⚙️ Model.compile(), .fit(), .evaluate()
📊 Plotting accuracy/loss across epochs

🔜 Next Module
👉 Module-12: TensorFlow & Keras Deep Dive
Understand how to build advanced architectures using Functional API, Callbacks, and Custom Layers.

🙌 Credits
Developed by: Nandyala Narendra
Guided via: Deep Learning Mastery – Modules 11 to 20
License: MIT


