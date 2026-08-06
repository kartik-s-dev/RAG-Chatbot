import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class DynamicChainWrapper:
    def __init__(self, vectordb):
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if groq_api_key:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
            
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        
        template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Keep the answer accurate and grounded in context.

Context:
{context}

Question:
{question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
    def invoke(self, inputs):
        query = inputs.get("query") or inputs.get("input") or inputs.get("question")
        res = self.chain.invoke(query)
        docs = self.retriever.invoke(query)
        return {
            "result": res,
            "source_documents": docs
        }

def build_chain_from_store(vectordb):
    return DynamicChainWrapper(vectordb)