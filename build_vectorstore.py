import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def build_store_from_pdf(pdf_path, persist_dir="chroma_db_temp"):
    # Clear old vector store directory to avoid mixing document chunks
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vector store successfully built from {pdf_path} into ./{persist_dir}")
    return vectordb

if __name__ == "__main__":
    build_store_from_pdf("data/source.pdf")