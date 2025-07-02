✅ Project 3: Hybrid RAG (PDF + Wikipedia Combined)
-----------------------------------------------------
# 🔀 Hybrid RAG: Combine PDF and Wikipedia for Better Answers
!pip install -q langchain langchain-groq langchain-community wikipedia chromadb unstructured pdfminer.six gradio

import os, tempfile, gradio as gr
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["GROQ_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# 📄 Load PDF → Chroma → Retriever
def load_pdf_retriever(file):
    loader = UnstructuredPDFLoader(file.name)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, embedding=HuggingFaceEmbeddings())
    return vectordb.as_retriever()

# 🧠 Custom Stuff Chain for combining docs
def hybrid_chain(pdf_retriever):
    wiki_retriever = WikipediaRetriever(top_k_results=2)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the below context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    doc_chain = StuffDocumentsChain(llm=llm, prompt=prompt)

    def hybrid_run(query):
        docs_pdf = pdf_retriever.get_relevant_documents(query)
        docs_wiki = wiki_retriever.get_relevant_documents(query)
        all_docs = docs_pdf + docs_wiki
        return doc_chain.run(input_documents=all_docs, question=query)
    
    return hybrid_run

pdf_retriever = None
hybrid_qa = None

def process_file(file):
    global pdf_retriever, hybrid_qa
    pdf_retriever = load_pdf_retriever(file)
    hybrid_qa = hybrid_chain(pdf_retriever)
    return "✅ Hybrid Retriever Ready!"

def ask_combined(query):
    if not hybrid_qa:
        return "❌ Please upload PDF first."
    return hybrid_qa(query)

with gr.Blocks(title="Hybrid RAG") as demo:
    gr.Markdown("## 🔀 Hybrid RAG: Combine PDF and Wikipedia using Groq")
    with gr.Row():
        file = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Load Hybrid")
    question = gr.Textbox(label="Ask from PDF + Wikipedia")
    answer = gr.Textbox(label="Answer")

    upload_btn.click(process_file, inputs=file, outputs=answer)
    question.submit(ask_combined, inputs=question, outputs=answer)

demo.launch()
