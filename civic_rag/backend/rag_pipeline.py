# Handles vector store and retrieval-augmented generation (RAG) pipeline

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import civic_rag.config as config
import os
from typing import List, Any
from civic_rag.backend.chatmodels import get_agent
from civic_rag.backend.utils import load_vector_store


def answer_query(question: str) -> str:
    """Answer a query using the LangGraph-based protest guidance system."""
    agent = get_agent()
    
    # Create initial state with the user's question
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "context": "",
        "web_results": "",
        "rag_results": "",
        "final_answer": ""
    }
    
    # Run the graph
    result = agent.invoke(initial_state)
    
    # Return the final answer from the state
    return result.get("final_answer", "I apologize, but I couldn't generate a response. Please try again.")
    
    # Legacy QA chain approach (commented out, kept for reference)
    # qa_chain = get_qa_chain()
    # result = qa_chain.invoke({"query": question})
    # return result.get("result", str(result))

