# chatmodels.py
# Defines a custom prompt template and a function to get a prompted LLM for use in RAG or other chains.

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.tools import BraveSearch
import civic_rag.config as config
import os
from typing import List, Any
def build_vector_store(docs: List[Any], persist_directory: str = config.CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


def load_vector_store(persist_directory: str = config.CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

@tool 
def rag_search(query: str) -> str:
    """Searches the RAG vector store for relevant information."""
    try:
        vectordb = load_vector_store()
        retriever = vectordb.as_retriever()
        docs = retriever.invoke(query)  # Updated from get_relevant_documents
        return "\n".join([doc.page_content for doc in docs[:3]])  # Return top 3 results
    except Exception as e:
        return f"RAG search failed: {e}"

@tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date information using BraveSearch."""
    try:
        searcher = BraveSearch.from_api_key(api_key='BSAZanmMIarkoXHk2K2C2ynx4YDs0UW')
        results = searcher.run(query)
        # Optional: limit results to top 3 if results is a list of strings
        if isinstance(results, list):
            return "\n".join(results[:3])
        return str(results)
    except Exception as e:
        return f"Web search failed: {e}"
# Define your custom prompt template
CUSTOM_CHAT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant for protest guidance. Use the following context to answer the user's question.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
)

# Define your custom prompt template for the agent
AGENT_PROMPT = """
You are a protest guidance assistant. When a user asks a question:
1. First, use the web search tool to find the current protest condition in Nepal.
2. Then, combine this with your knowledge to answer the user's query in a helpful, concise, and actionable way.
"""

# Define a web search tool (using DuckDuckGo as an example)
def get_prompted_llm():
    llm = ChatGroq(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        temperature=0.8,
        api_key=config.GROQ_API_KEY
    )
    return llm, CUSTOM_CHAT_PROMPT

def get_agent():
    llm = ChatGroq(
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.8,
        api_key=config.GROQ_API_KEY
    )
    tools = [web_search, rag_search]  # Add more tools as needed
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="chat-zero-shot-react-description",
        verbose=True,
        max_iterations=10,  # Stop after 5 iterations to prevent infinite loops
        max_execution_time=60,  # Stop after 60 seconds
        agent_kwargs={
            "prefix": AGENT_PROMPT
        },
        return_intermediate_steps=True,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )
    return agent
