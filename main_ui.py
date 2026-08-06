import streamlit as st
from rag_chain import get_wrapped_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 Document QA - RAG Chatbot")
st.caption("Powered by LangChain, Chroma Vector Store & Gemini")

@st.cache_resource
def load_rag_chain():
    return get_wrapped_chain()

try:
    chain = load_rag_chain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I have loaded your PDF document. Ask me anything about it!"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                res = chain.invoke({"query": prompt})
                response = res["result"]
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Error loading RAG Chain: {e}")