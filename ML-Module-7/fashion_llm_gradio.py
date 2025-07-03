# 📌 Fashion MNIST ML + Autoencoder + Groq LLM Summary + PDF Export
!pip install -q gradio scikit-learn xgboost tensorflow matplotlib seaborn langchain langchain-groq fpdf2

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
from fpdf import FPDF
import os

# 🔐 Set your GROQ API key
os.environ['GROQ_API_KEY'] = 'gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# 🔹 Load Fashion MNIST
def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X = X.reshape((X.shape[0], -1)) / 255.0
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Autoencoder setup
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

# 🔹 Train models
def train_models():
    X_train, X_test, y_train, y_test = load_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)

    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

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

# 🔹 Confusion matrix plot
def plot_cm(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (XGBoost AE)")
    plt.tight_layout()
    path = "/content/confusion.png"
    plt.savefig(path)
    return path

# 🔹 Groq LLM Summary
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
output_parser = StrOutputParser()
summary_template = PromptTemplate.from_template(
    """
    You're an AI assistant. Analyze this model performance and confusion insights:

    📊 Model Performance:
    {performance}

    🔍 Confusion Summary:
    {insight}

    Give a 5-6 line summary explaining which model is better, what the confusion matrix shows, and recommendations.
    """
)
summary_chain = summary_template | llm | output_parser

def generate_summary(perf_df, cm):
    perf_txt = perf_df.to_markdown(index=False)
    insight = "Confusion shows close class misclassifications like Pullover vs Coat or Shirt vs T-shirt."
    return summary_chain.invoke({"performance": perf_txt, "insight": insight})

# 🔹 Create PDF report
def create_pdf(perf_df, cm_path, summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(40, 40, 100)
    pdf.cell(0, 10, "📘 Fashion MNIST ML Report", ln=True, align='C')

    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0,0,0)
    pdf.multi_cell(0, 10, "\n🧠 LLM Summary:\n" + summary_text)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "\n📊 Model Performance", ln=True)

    pdf.set_font("Arial", '', 12)
    for i, row in perf_df.iterrows():
        pdf.cell(0, 8, f"{row['Model']}: Accuracy = {row['Accuracy']:.4f}", ln=True)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "\n📌 Confusion Matrix", ln=True)

    pdf.image(cm_path, x=30, w=150)

    save_path = "/content/Fashion_MNIST_Report.pdf"
    pdf.output(save_path)
    return save_path

# 🔹 Full pipeline
def full_pipeline():
    perf, cm, y_true, y_pred = train_models()
    cm_path = plot_cm(cm)
    summary = generate_summary(perf, cm)
    pdf_path = create_pdf(perf, cm_path, summary)
    return perf, cm_path, summary, pdf_path

# 🔹 Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# 👗 Fashion MNIST Project with Autoencoder + Groq LLM Summary + 📄 PDF Export""")
    run_btn = gr.Button("🚀 Run Full ML Pipeline")
    df_out = gr.Dataframe(label="📊 Accuracy Comparison Table")
    img_out = gr.Image(label="📌 Confusion Matrix Image")
    txt_out = gr.Textbox(label="🧠 LLM Summary", lines=6)
    file_out = gr.File(label="📄 Download Full PDF Report")

    run_btn.click(fn=full_pipeline, outputs=[df_out, img_out, txt_out, file_out])

demo.launch(share=True)
