# Handles PDF and directory ingestion, chunking, and metadata attachment

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import os
import civic_rag.config as config
from pathlib import Path


def ingest_pdf(pdf_path: str, metadata: Dict[str, Any]) -> List[Any]:
    """Ingest a single PDF file with error handling."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        split_docs = splitter.split_documents(docs)
        for doc in split_docs:
            doc.metadata.update(metadata)
        print(f"✅ Successfully processed: {os.path.basename(pdf_path)}")
        return split_docs
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(pdf_path)}: {e}")
        return []


def ingest_all_pdfs_in_directory(directory: str = config.DATA_DIR, metadata: Dict[str, Any] = None) -> List[Any]:
    """Ingest all PDFs in directory with error handling for individual files."""
    all_docs = []
    
    # Get all PDF files in directory
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {directory}")
        return []
    
    print(f"📂 Found {len(pdf_files)} PDF files to process...")
    
    # Process each PDF individually
    for pdf_file in pdf_files:
        try:
            # Check if file is empty
            if pdf_file.stat().st_size == 0:
                print(f"⚠️ Skipping empty file: {pdf_file.name}")
                continue
            
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            if not docs:
                print(f"⚠️ No content extracted from: {pdf_file.name}")
                continue
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE, 
                chunk_overlap=config.CHUNK_OVERLAP
            )
            split_docs = splitter.split_documents(docs)
            
            # Add metadata
            if metadata:
                for doc in split_docs:
                    doc.metadata.update(metadata)
            
            # Add filename to metadata
            for doc in split_docs:
                doc.metadata['source_file'] = pdf_file.name
            
            all_docs.extend(split_docs)
            print(f"✅ Processed: {pdf_file.name} ({len(split_docs)} chunks)")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            continue
    
    print(f"🎯 Total processed: {len(all_docs)} document chunks from {len(pdf_files)} files")
    return all_docs
