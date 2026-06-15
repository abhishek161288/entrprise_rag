from langchain.tools import tool

from app.rag.rag_pipeline import (
    ask
)

@tool
def ask_rag(question: str) -> str:

    """
    Search Company policies and answer questions using RAG system. 

   """
    print(f"Received question for RAG: {question}")
    response = ask(question)
    print(f"RAG response: {response}")
    return response