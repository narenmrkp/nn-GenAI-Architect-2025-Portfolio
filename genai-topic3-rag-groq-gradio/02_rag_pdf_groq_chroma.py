✅ Project 2: RAG with PDF Upload + Chroma Vector DB
------------------------------------------------------
# 📄 PDF-based RAG using Groq + Chroma + LangChain + Gradio
!pip install -q langchain langchain-groq langchain-community chromadb unstructured pdfminer.six gradio

import os, tempfile, gradio as gr
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔐 Groq Key
os.environ["GROQ_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# 📄 Load, Split, Embed PDF → Chroma Vectorstore
def create_pdf_rag(file):
    with tempfile.TemporaryDirectory() as tempdir:
        loader = UnstructuredPDFLoader(file.name)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(chunks, embedding=HuggingFaceEmbeddings(), persist_directory=tempdir)
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
        return qa_chain

qa_chain = None  # Global chain object

def process_pdf(file):
    global qa_chain
    qa_chain = create_pdf_rag(file)
    return "✅ PDF processed successfully! You can now ask questions."

def ask_pdf(query):
    if not qa_chain:
        return "❌ Please upload and process a PDF first."
    result = qa_chain.run(query)
    return f"📄 Answer: {result}"

with gr.Blocks(title="PDF RAG with Groq") as demo:
    gr.Markdown("## 📄 PDF Retrieval-Augmented Generation using Groq + LangChain")
    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Process PDF")
    query_input = gr.Textbox(label="Ask from PDF")
    answer_output = gr.Textbox(label="Answer")

    upload_btn.click(process_pdf, inputs=file_input, outputs=answer_output)
    query_input.submit(ask_pdf, inputs=query_input, outputs=answer_output)

demo.launch()
