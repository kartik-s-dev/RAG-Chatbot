# RAG Chatbot - Chat with Your PDF (LangChain, ChromaDB, Gemini)

A Retrieval-Augmented Generation (RAG) app. Upload a PDF, ask questions, and get answers based on the retrieved parts of that document.

Live demo: https://kartik-rag-chatbot.streamlit.app

## How it works
1. **Upload**: `main_ui.py` (Streamlit) takes a PDF upload.
2. **Chunking and embeddings**: `build_vectorstore.py` splits the PDF (chunk size 1500, overlap 300) and embeds it with HuggingFace `all-MiniLM-L6-v2`.
3. **Vector store**: chunks are stored in Chroma (temporary, per upload).
4. **Retrieval**: top 6 similar chunks are fetched for each question.
5. **Answer**: `rag_chain.py` sends the question and chunks to a Google Gemini model. If one model fails, it tries the next.
6. **Sources**: the app shows the retrieved chunks (first 200 characters each).

## Tech stack
* Python 3.13
* LangChain
* ChromaDB
* HuggingFace sentence-transformers (embeddings)
* Google Gemini (LLM)
* Streamlit (UI)

## Run locally
```bash
git clone https://github.com/kartik-s-dev/RAG-Chatbot.git
cd RAG-Chatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your Google API key:

```
GOOGLE_API_KEY=your_key_here
```

Then run:

```bash
streamlit run main_ui.py
```

## Limitations
* Answers can be wrong if the retrieved chunks miss the answer.
* Scanned (image-only) PDFs are not supported.
* Each upload builds a new temporary index.