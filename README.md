# RAG Chatbot - Document Q&A with LangChain, ChromaDB & Groq (Llama 3.3)

An end-to-end Retrieval-Augmented Generation (RAG) system that allows users to query PDF documents and get accurate, context-grounded answers.

## Architecture & Data Flow
1. **PDF Ingestion**: `load_data.py` extracts text using `PyPDFLoader`.
2. **Chunking**: `chunk_data.py` splits documents into overlapping passages.
3. **Embeddings & Vector Store**: `build_vectorstore.py` generates embeddings using HuggingFace (`all-MiniLM-L6-v2`) and persists them in `ChromaDB`.
4. **Retrieval**: `retriever.py` fetches the top relevant chunks for any user question.
5. **Generation**: `rag_chain.py` uses Groq's `llama-3.3-70b-versatile` model to answer based strictly on retrieved context.
6. **Web Interface**: `main_ui.py` provides an interactive Streamlit UI.

## Tech Stack
* Python 3.13
* LangChain & LangChain Community
* ChromaDB (Vector Store)
* HuggingFace Transformers (Embeddings)
* Groq API (Llama 3.3 70B LLM)
* Streamlit (UI Frontend)

## Setup Instructions

### 1. Clone & Setup Environment
```bash
git clone <your-github-repo-url>
cd RAG-Chatbot
python -m venv venv
venv\Scripts\activate   # On Windows