import streamlit as st
from rag_chain import build_chain

# Page setup
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Document QA - RAG Chatbot")
st.caption("Powered by Llama 3.3 (Groq) & LangChain Chroma Vector Store")

# Initialize RAG Chain once
@st.cache_resource
def load_rag_chain():
    return build_chain()

try:
    chain = load_rag_chain()
except Exception as e:
    st.error(f"Error loading RAG Chain: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have loaded your PDF document. Ask me anything about it!"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Query Input
if user_query := st.chat_input("Ask a question about your document..."):
    # Display user input
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate & Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating answer..."):
            try:
                response = chain.invoke(user_query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating answer: {e}")