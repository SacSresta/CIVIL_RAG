# Handles vector store and retrieval-augmented generation (RAG) pipeline

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import civic_rag.config as config
import os
from typing import List, Any
from civic_rag.backend.chatmodels import get_prompted_llm,get_agent, load_vector_store


def get_qa_chain():
    vectordb = load_vector_store()
    retriever = vectordb.as_retriever()
    llm, prompt = get_prompted_llm()  # Get LLM and prompt instead of agent
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def answer_query(question: str) -> str:
    # Option 1: Use the agent directly (recommended)
    agent = get_agent()
    result = agent.invoke({"input": question})
    return result.get("output", str(result))
    
    # Option 2: Use QA chain with prompted LLM (uncomment to use)
    # qa_chain = get_qa_chain()
    # result = qa_chain.invoke({"query": question})
    # return result.get("result", str(result))

