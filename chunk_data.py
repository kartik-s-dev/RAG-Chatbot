from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_data import load_pdf

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    docs = load_pdf()
    chunks = chunk_documents(docs)
    if chunks:
        print(chunks[0].page_content)