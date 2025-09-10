"""
Tool utilities for RAG search and web search functionality.
"""

from langchain_core.tools import tool
from langchain_community.tools import BraveSearch
from .utils import load_vector_store


@tool 
def rag_search(query: str) -> str:
    """Searches the RAG vector store for relevant information about protest guidance."""
    try:
        vectordb = load_vector_store()
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"RAG search failed: {e}"


@tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date information about Nepal protests using BraveSearch."""
    try:
        searcher = BraveSearch.from_api_key(api_key='BSAZanmMIarkoXHk2K2C2ynx4YDs0UW')
        results = searcher.run(query)
        if isinstance(results, list):
            return "\n".join(results[:5])
        return str(results)
    except Exception as e:
        return f"Web search failed: {e}"
