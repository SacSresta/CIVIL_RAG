# Handles PDF and directory ingestion, chunking, and metadata attachment

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import os
import civic_rag.config as config


def ingest_pdf(pdf_path: str, metadata: Dict[str, Any]) -> List[Any]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)
    for doc in split_docs:
        doc.metadata.update(metadata)
    return split_docs


def ingest_all_pdfs_in_directory(directory: str = config.DATA_DIR, metadata: Dict[str, Any] = None) -> List[Any]:
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)
    if metadata:
        for doc in split_docs:
            doc.metadata.update(metadata)
    return split_docs
