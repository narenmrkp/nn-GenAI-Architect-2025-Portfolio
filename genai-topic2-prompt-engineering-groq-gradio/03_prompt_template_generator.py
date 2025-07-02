✅ Project 3: Prompt Template Generator (Structured Reusable Templates)
---------------------------------------------------------------------------------
# ✅ Prompt Template Generator - Build Prompts Dynamically with Gradio + Groq
!pip install -q langchain langchain-groq gradio

import os, gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# 🔑 API Key
os.environ["GROQ_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

llm = ChatGroq(temperature=0.5, model_name="llama3-8b-8192")

TEMPLATES = {
    "Summarize Text": "Summarize the following passage in 3 bullet points:\n\n{text}",
    "Translate to French": "Translate the following English text to French:\n\n{text}",
    "Explain Like I'm 5": "Explain the following in very simple words:\n\n{text}",
    "List Pros & Cons": "List the pros and cons of the following:\n\n{text}"
}

def generate(task, text):
    template = PromptTemplate.from_template(TEMPLATES[task])
    chain = LLMChain(llm=llm, prompt=template)
    return chain.run(text=text)

gr.Interface(
    fn=generate,
    inputs=[
        gr.Dropdown(list(TEMPLATES.keys()), label="Select Task"),
        gr.Textbox(lines=5, label="Enter Input Text")
    ],
    outputs=gr.Textbox(label="LLM Response"),
    title="🧩 Prompt Template Generator (Groq LLaMA)",
    description="Select a task and auto-generate structured prompts with outputs"
).launch()
