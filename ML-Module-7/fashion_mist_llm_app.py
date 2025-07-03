# fashion_mnist_llm_app.py
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Fashion MNIST
def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X = X.reshape((X.shape[0], -1)) / 255.0  # Flatten + normalize
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Autoencoder definition
def get_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, output_layer)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Train and evaluate models
def train_models():
    X_train, X_test, y_train, y_test = load_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Base Logistic
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    # Autoencoder + XGBoost
    autoencoder, encoder = get_autoencoder(X_train.shape[1])
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=256, verbose=0)
    X_train_enc = encoder.predict(X_train_scaled)
    X_test_enc = encoder.predict(X_test_scaled)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train_enc, y_train)
    y_pred_xgb = xgb.predict(X_test_enc)

    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost (AE)'],
        'Accuracy': [
            np.mean(y_pred_log == y_test),
            np.mean(y_pred_rf == y_test),
            np.mean(y_pred_xgb == y_test)
        ]
    })

    cm = confusion_matrix(y_test, y_pred_xgb)
    return results, cm, y_test, y_pred_xgb

# Plot confusion matrix
def plot_cm(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (XGBoost AE)")
    plt.tight_layout()
    plt.savefig("confusion.png")
    return "confusion.png"

# LangChain LLM Summary using Groq
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
output_parser = StrOutputParser()
summary_template = PromptTemplate.from_template(
    """
    You're an AI assistant. Analyze this model performance table and confusion matrix insight:

    Performance Table:
    {performance}

    Confusion Matrix Key Insight: {insight}

    Summarize performance, misclassifications, and suggest future improvements in 5-6 lines.
    """
)
summary_chain = summary_template | llm | output_parser

def generate_summary(perf_df, cm):
    perf_txt = perf_df.to_markdown(index=False)
    confusion_issues = "Model confuses similar classes like Shirt vs T-shirt, or Coat vs Pullover."
    return summary_chain.invoke({
        "performance": perf_txt,
        "insight": confusion_issues
    })

# Gradio UI
def full_pipeline():
    perf, cm, y_true, y_pred = train_models()
    cm_path = plot_cm(cm)
    summary = generate_summary(perf, cm)
    return perf, cm_path, summary

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# 👗 Fashion MNIST Classifier with Autoencoder + LLM Summary

🔹 Logistic | 🔹 Random Forest | 🔹 Autoencoder + XGBoost

✅ Powered by Groq LLM for automatic performance summary
""")
    btn = gr.Button("🚀 Run Full Pipeline")
    out_df = gr.Dataframe(label="📊 Model Comparison")
    out_img = gr.Image(label="📌 Confusion Matrix")
    out_text = gr.Textbox(label="🧠 LLM Summary", lines=6)
    btn.click(fn=full_pipeline, outputs=[out_df, out_img, out_text])

# To launch in Colab:
# demo.launch(share=True)
```

---

### ✅ Features Included

- Full **ML pipeline**: Logistic, RF, XGBoost + Autoencoder  
- **Confusion matrix plot**  
- **LLM integration via Groq** to auto-generate **natural summary**  
- Gradio UI with one-click execution

---

### 🧪 To Run in Google Colab

1. Upload your Groq API key via `.env` or inline `os.environ['GROQ_API_KEY']`
2. Install dependencies:
   ```bash
   pip install gradio scikit-learn xgboost tensorflow matplotlib seaborn langchain langchain-groq
   ```
3. Launch the app:
   ```python
   demo.launch(share=True)
   ```

---

