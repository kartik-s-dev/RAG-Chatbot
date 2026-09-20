import os
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Document QA - RAG Chatbot", page_icon="🤖")
load_dotenv()

st.title("🤖 Document QA - RAG Chatbot")
st.caption("Powered by LangChain, Chroma Vector Store & Gemini")

# Initialize RAG Chain with Explicit Exception Tracking
qa_chain = None
load_error = None

try:
    from rag_chain import get_wrapped_chain
    qa_chain = get_wrapped_chain()
except Exception as e:
    load_error = str(e)

# Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have loaded your PDF document. Ask me anything about it!"}
    ]

# Render Existing Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Chat Input
user_query = st.chat_input("Ask a question about your document...")

if user_query:
    # 1. Display User Message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Process Answer
    with st.chat_message("assistant"):
        if qa_chain is None:
            if load_error:
                response_text = f"Error loading RAG Chain details: {load_error}"
            else:
                response_text = "RAG Chain initialized as None. Check Google API Key in .env or rebuild Chroma DB."
        else:
            with st.spinner("Analyzing document..."):
                try:
                    if hasattr(qa_chain, "invoke"):
                        res = qa_chain.invoke({"query": user_query})
                        response_text = res.get("result", str(res))
                    elif callable(qa_chain):
                        res = qa_chain(user_query)
                        response_text = res.get("result", str(res)) if isinstance(res, dict) else str(res)
                    else:
                        response_text = "qa_chain does not have a valid execution method."
                except Exception as err:
                    response_text = f"Query processing failed: {err}"

        st.write(response_text)

    # 3. Store Assistant Response
    st.session_state.messages.append({"role": "assistant", "content": response_text})