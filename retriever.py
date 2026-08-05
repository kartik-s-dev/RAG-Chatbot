from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_retriever(k=4):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vectordb.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.invoke("KARTIK")
    for i, r in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(r.page_content[:200])