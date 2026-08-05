from rag_chain import build_chain

def main():
    print("--- RAG Chatbot Ready ---")
    print("Type 'exit' to quit.\n")
    chain = build_chain()
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue
        try:
            response = chain.invoke(query)
            print(f"\nBot: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()