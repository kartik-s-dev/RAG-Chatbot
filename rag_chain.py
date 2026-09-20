import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def build_chain_from_store(vectordb):
    api_key = os.getenv("GOOGLE_API_KEY")
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    class WorkingRAGChain:
        def __init__(self, retriever, api_key):
            self.retriever = retriever
            self.api_key = api_key

        def invoke(self, input_data):
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            docs = self.retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""Context:
{context_text}

Question: {query}

Instructions:
- Answer using only the information in the context above.
- Reply in the SAME language and script as the question. If the question is in English, reply in English. If the question is in Hinglish (Hindi words written in Roman/English letters), reply in Hinglish using Roman letters only — do NOT switch to Devanagari script. If the question is in Hindi (Devanagari script), reply in Hindi (Devanagari script).

Answer:"""
            
            # Explicit Google API full resource strings
            models_to_try = [
                "models/gemini-2.5-flash",
                "models/gemini-3.6-flash",
                "models/gemini-3.1-flash-lite"
            ]
            
            last_err = None
            for m in models_to_try:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=m, 
                        google_api_key=self.api_key, 
                        temperature=0.2
                    )
                    res = llm.invoke(prompt)

                    # Extract clean text regardless of response format (string or structured blocks)
                    if isinstance(res.content, str):
                        answer_text = res.content
                    elif isinstance(res.content, list):
                        answer_text = " ".join(
                            block.get("text", "") for block in res.content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        answer_text = str(res.content)

                    return {"result": answer_text, "source_documents": docs}
                except Exception as e:
                    last_err = e
                    continue
            
            raise last_err

    return WorkingRAGChain(retriever, api_key)


def get_wrapped_chain(persist_dir="chroma_db_temp"):
    """
    Loads the vector store that build_vectorstore.py created,
    and returns a ready-to-use RAG chain. Called by main_ui.py.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return build_chain_from_store(vectordb)