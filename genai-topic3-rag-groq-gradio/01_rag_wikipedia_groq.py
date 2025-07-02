✅ Project 1: RAG with WikipediaRetriever (Groq + Gradio)
------------------------------------------------------------
# 📚 Wikipedia-based RAG using LangChain + Groq + Gradio
!pip install -q langchain langchain-groq langchain-community wikipedia gradio

import os, gradio as gr
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.retrievers import WikipediaRetriever

# 🔐 Set Groq API Key
os.environ["GROQ_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 🧠 LLM
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# 🔎 Wikipedia Retriever
retriever = WikipediaRetriever(top_k_results=3, doc_content_chars_max=1500)

# 🔁 RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 🎯 Gradio UI Function
def ask_wiki(query):
    result = qa_chain(query)
    answer = result["result"]
    sources = "\n".join(
        [f"📄 {doc.metadata['title']}" for doc in result["source_documents"]]
    )
    return f"🧠 **Answer**:\n{answer}\n\n📚 **Sources**:\n{sources}"

gr.Interface(
    fn=ask_wiki,
    inputs=gr.Textbox(label="Ask a Question (Wikipedia RAG)", placeholder="What is Generative AI?"),
    outputs=gr.Markdown(),
    title="📖 Wikipedia RAG with Groq",
    description="Asks Groq LLaMA3 to answer using real-time Wikipedia context"
).launch()
