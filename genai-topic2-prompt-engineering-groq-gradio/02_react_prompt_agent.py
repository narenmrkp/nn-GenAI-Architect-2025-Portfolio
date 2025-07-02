✅ Project 2: ReAct Prompt Agent (Tool Use + Reasoning)
-------------------------------------------------------------
# ✅ ReAct Prompt Agent (Groq only - No external Python Tool needed)
!pip install -q langchain langchain-groq gradio

import os, gradio as gr
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq

# 🔑 Set your Groq API Key
os.environ["GROQ_API_KEY"] = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# ✅ Calculator Tool (Safe eval)
def calculator_tool(input_text):
    try:
        result = eval(input_text)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    Tool(name="Calculator", func=calculator_tool, description="Performs basic math like 24*19+13")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

def run_agent(question):
    try:
        return agent.run(question)
    except Exception as e:
        return f"❌ Error: {str(e)}"

gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(label="Ask Anything", placeholder="E.g. What is (24*19)+13?"),
    outputs=gr.Textbox(label="Agent Response"),
    title="🧠 ReAct Prompt Agent (Groq)",
    description="ReAct-based Agent using Groq LLaMA that performs reasoning and tool calling"
).launch()
