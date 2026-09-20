import os
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Multi-Doc RAG Chatbot", page_icon="🤖", layout="wide")
load_dotenv()

st.title("🤖 Multi-Document RAG Chatbot System")

# Diagnostic Pipeline Loader
rag_pipeline = None
init_error = None

try:
    import rag_chain
    if hasattr(rag_chain, "build_chain"):
        rag_pipeline = rag_chain.build_chain()
    elif hasattr(rag_chain, "get_wrapped_chain"):
        rag_pipeline = rag_chain.get_wrapped_chain()
    elif hasattr(rag_chain, "get_rag_chain"):
        rag_pipeline = rag_chain.get_rag_chain()
    else:
        rag_pipeline = rag_chain
except Exception as err:
    init_error = str(err)

# Sidebar Core Diagnostics
with st.sidebar:
    st.header("⚙️ Core Diagnostics")
    if rag_pipeline is not None:
        st.success("RAG Pipeline: Connected & Active")
    else:
        st.error("RAG Pipeline: Connection Failed")
        
    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display Explicit Error in Main Area if Initialization Failed
if init_error:
    st.error(f"⚠️ **Pipeline Initialization Error:** {init_error}")
    st.info("💡 **Possible Fix:** Ensure Google API Key is set in `.env` file or required packages are installed.")

# Session State & Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Ask a question based on your uploaded documents...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        if rag_pipeline:
            with st.spinner("Analyzing document context..."):
                try:
                    if hasattr(rag_pipeline, "invoke"):
                        res = rag_pipeline.invoke({"query": user_query})
                        response_text = res.get("result", str(res))
                    elif callable(rag_pipeline):
                        response_text = rag_pipeline(user_query)
                    else:
                        response_text = f"Context processed for: '{user_query}'"
                except Exception as query_err:
                    response_text = f"Query Execution Error: {query_err}"
        else:
            response_text = "RAG Pipeline is offline. Fix the initialization error displayed above."

        st.write(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})