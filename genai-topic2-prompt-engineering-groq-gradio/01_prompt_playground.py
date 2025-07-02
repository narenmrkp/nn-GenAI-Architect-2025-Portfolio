✅ Project 1: Prompt Playground (Zero, Few, CoT Comparison)
----------------------------------------------------------------
# ✅ Prompt Playground - Compare Zero-shot, Few-shot, Chain-of-Thought with Groq LLaMA + Gradio
!pip install -q langchain langchain-groq gradio

import os, gradio as gr
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔑 Insert your Groq API key here
os.environ["GROQ_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

def generate_prompt(user_input, prompt_type):
    few_shot_examples = "Q: What is the capital of France?\nA: Paris\n\nQ: What is the capital of Italy?\nA: Rome\n\n"
    cot_instruction = "Answer step-by-step.\n\n"

    if prompt_type == "Zero-Shot":
        return user_input
    elif prompt_type == "Few-Shot":
        return few_shot_examples + f"Q: {user_input}\nA:"
    elif prompt_type == "Chain-of-Thought":
        return cot_instruction + user_input
    return user_input

def respond(user_input, prompt_type):
    final_prompt = generate_prompt(user_input, prompt_type)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"))
    return chain.run({"input": final_prompt})

gr.Interface(
    fn=respond,
    inputs=[
        gr.Textbox(label="Your Question", placeholder="E.g. Why is the sky blue?"),
        gr.Dropdown(["Zero-Shot", "Few-Shot", "Chain-of-Thought"], label="Prompt Type")
    ],
    outputs=gr.Textbox(label="LLM Output"),
    title="🧠 Prompt Playground - Groq LLaMA3",
    description="Compare prompt styles: Zero-Shot vs Few-Shot vs Chain-of-Thought"
).launch()
