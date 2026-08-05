from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path="data/source.pdf"):
    loader = PyPDFLoader(path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

if __name__ == "__main__":
    docs = load_pdf()
    if docs:
        print(docs[0].page_content[:300])