#!/usr/bin/env python3
"""
Quick fix: Rebuild vector store excluding problematic files
"""

import os
import shutil
from pathlib import Path

def quick_rebuild():
    print("🚀 Quick Vector Store Rebuild")
    print("=" * 40)
    
    try:
        import civic_rag.config as config
        from civic_rag.backend.utils import build_vector_store
        from civic_rag.backend.ingestion import ingest_pdf
        
        # Remove existing vector store
        if os.path.exists(config.CHROMA_DIR):
            shutil.rmtree(config.CHROMA_DIR)
            print("🗑️ Cleared existing vector store")
        
        # Get all PDFs and process them individually
        pdf_files = list(Path(config.DATA_DIR).glob("*.pdf"))
        print(f"📂 Found {len(pdf_files)} PDF files")
        
        all_docs = []
        successful_files = []
        
        for pdf_file in pdf_files:
            try:
                # Check file size
                if pdf_file.stat().st_size == 0:
                    print(f"⚠️ Skipping empty file: {pdf_file.name}")
                    continue
                
                # Try to process the PDF
                docs = ingest_pdf(str(pdf_file), {"source": pdf_file.name})
                
                if docs:
                    all_docs.extend(docs)
                    successful_files.append(pdf_file.name)
                    print(f"✅ {pdf_file.name}: {len(docs)} chunks")
                else:
                    print(f"⚠️ No content from: {pdf_file.name}")
                    
            except Exception as e:
                print(f"❌ Failed to process {pdf_file.name}: {e}")
                continue
        
        if all_docs:
            print(f"\n🔄 Building vector store with {len(all_docs)} chunks from {len(successful_files)} files...")
            vectordb = build_vector_store(all_docs)
            
            print(f"\n✅ Success! Vector store rebuilt with:")
            for filename in successful_files:
                print(f"   - {filename}")
            
            if len(successful_files) < len(pdf_files):
                failed_count = len(pdf_files) - len(successful_files)
                print(f"\n⚠️ Note: {failed_count} files were skipped due to errors")
                
        else:
            print("❌ No documents were successfully processed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_rebuild()
