"""
Vector store utilities and database operations for the protest guidance system.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import civic_rag.config as config
import os
import shutil
from typing import List, Any


def build_vector_store(docs: List[Any], persist_directory: str = config.CHROMA_DIR):
    """Build and persist a vector store from documents."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    # Note: persist() is no longer needed in newer versions of Chroma
    # The vector store is automatically persisted to the directory
    print(f"✅ Vector store created with {len(docs)} documents")
    return vectordb


def add_documents_to_vector_store(docs: List[Any], persist_directory: str = config.CHROMA_DIR):
    """Add new documents to existing vector store."""
    try:
        # Load existing vector store
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Add new documents
        vectordb.add_documents(docs)
        # Note: persist() is no longer needed in newer versions of Chroma
        
        print(f"✅ Added {len(docs)} documents to vector store")
        return vectordb
    except Exception as e:
        print(f"❌ Error adding documents: {e}")
        # If vector store doesn't exist, create a new one
        print("🔄 Creating new vector store...")
        return build_vector_store(docs, persist_directory)


def update_vector_store_from_directory(directory: str = config.DATA_DIR):
    """Update vector store with all PDFs from a directory."""
    from civic_rag.backend.ingestion import ingest_all_pdfs_in_directory
    
    print(f"📂 Processing PDFs from: {directory}")
    docs = ingest_all_pdfs_in_directory(directory)
    
    if docs:
        vectordb = add_documents_to_vector_store(docs)
        print(f"✅ Vector store updated with {len(docs)} document chunks")
        return vectordb
    else:
        print("⚠️ No PDFs found in directory")
        return None


def clear_and_rebuild_vector_store(directory: str = config.DATA_DIR):
    """Clear existing vector store and rebuild from scratch."""
    from civic_rag.backend.ingestion import ingest_all_pdfs_in_directory
    
    # Remove existing vector store
    if os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)
        print("🗑️ Cleared existing vector store")
    
    # Rebuild from directory
    print(f"🔄 Rebuilding vector store from: {directory}")
    docs = ingest_all_pdfs_in_directory(directory)
    
    if docs:
        vectordb = build_vector_store(docs)
        print(f"✅ Rebuilt vector store with {len(docs)} document chunks")
        return vectordb
    else:
        print("⚠️ No PDFs found to rebuild vector store")
        return None


def load_vector_store(persist_directory: str = config.CHROMA_DIR):
    """Load an existing vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb


def get_vector_store_info():
    """Get information about the current vector store."""
    try:
        vectordb = load_vector_store()
        collection = vectordb._collection
        count = collection.count()
        
        print("📊 Vector Store Information")
        print("=" * 30)
        print(f"📍 Location: {config.CHROMA_DIR}")
        print(f"📄 Total documents: {count}")
        print(f"🔍 Embedding model: {config.EMBEDDING_MODEL}")
        
        return {"count": count, "location": config.CHROMA_DIR}
    except Exception as e:
        print(f"❌ Error accessing vector store: {e}")
        return None
