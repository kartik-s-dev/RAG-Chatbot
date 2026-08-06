import streamlit as st
import tempfile
import shutil
import os
from dotenv import load_dotenv
from build_vectorstore import build_store_from_pdf
from rag_chain import build_chain_from_store

load_dotenv()

st.set_page_config(page_title="RAG Document Chatbot", page_icon="🤖")
st.title("🤖 RAG Document Chatbot")
st.write("Upload any PDF and ask questions about it.")

# Session state initialization
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar Clear Chat Button
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing PDF... this may take a moment"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            persist_dir = "chroma_db_temp"
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)

            vectordb = build_store_from_pdf(tmp_path, persist_dir=persist_dir)
            st.session_state.chain = build_chain_from_store(vectordb)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # Reset chat for new document
            
            os.remove(tmp_path)
            st.success(f"'{uploaded_file.name}' processed successfully. Ask your questions below!")

# Chat Interface
if st.session_state.chain is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask a question about the uploaded document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"query": user_query})
                answer = result["result"]
                st.write(answer)
                
                with st.expander("View source chunks used"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.caption(f"Source {i+1}")
                        st.text(doc.page_content[:200])
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF above to start chatting.")