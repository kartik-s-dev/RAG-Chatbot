import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chunk_data import chunk_documents
from load_data import load_pdf

load_dotenv()

def build_store():
    docs = load_pdf()
    chunks = chunk_documents(docs)
    
    # Free local embedding model (no API key required for embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("Vector store built and saved to ./chroma_db")
    return vectordb

if __name__ == "__main__":
    build_store()