#!/usr/bin/env python3
"""
Vector Store Management Utility
Use this script to update your vector database when you add new data.
"""

import sys
import os
from pathlib import Path

def check_pdf_files(directory):
    """Check PDF files for issues and report problems."""
    try:
        import pypdf
    except ImportError:
        print("⚠️ pypdf not available for file checking, proceeding anyway...")
        return 0
        
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    print("🔍 Checking PDF files for issues...")
    problematic_files = []
    
    for pdf_file in pdf_files:
        try:
            # Check if file is empty
            if pdf_file.stat().st_size == 0:
                print(f"⚠️ Empty file: {pdf_file.name}")
                problematic_files.append(pdf_file)
                continue
            
            # Try to open with pypdf to check for corruption
            with open(pdf_file, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                if len(pdf_reader.pages) == 0:
                    print(f"⚠️ No pages found: {pdf_file.name}")
                    problematic_files.append(pdf_file)
                else:
                    print(f"✅ Valid PDF: {pdf_file.name} ({len(pdf_reader.pages)} pages)")
                    
        except Exception as e:
            print(f"❌ Corrupted file: {pdf_file.name} - {e}")
            problematic_files.append(pdf_file)
    
    if problematic_files:
        print(f"\n⚠️ Found {len(problematic_files)} problematic files:")
        for file in problematic_files:
            print(f"   - {file.name}")
        
        choice = input("\nDo you want to move problematic files to a backup folder? (y/N): ").strip().lower()
        if choice == 'y':
            backup_dir = Path(directory) / "problematic_files"
            backup_dir.mkdir(exist_ok=True)
            
            for file in problematic_files:
                backup_path = backup_dir / file.name
                file.rename(backup_path)
                print(f"📁 Moved {file.name} to problematic_files/")
            
            print("✅ Problematic files moved to backup folder")
    
    return len(problematic_files)

def main():
    print("🗂️ Vector Store Management Utility")
    print("=" * 50)
    
    try:
        from civic_rag.backend.utils import (
            update_vector_store_from_directory,
            clear_and_rebuild_vector_store,
            add_documents_to_vector_store,
            get_vector_store_info
        )
        from civic_rag.backend.ingestion import ingest_pdf
        import civic_rag.config as config
        
        print("📍 Data Directory:", config.DATA_DIR)
        print("📍 Vector Store Directory:", config.CHROMA_DIR)
        print()
        
        # Check if data directory exists
        if not os.path.exists(config.DATA_DIR):
            print(f"❌ Data directory not found: {config.DATA_DIR}")
            print("💡 Create the directory and add PDF files to it")
            return
            
        # Count PDFs in data directory
        pdf_files = list(Path(config.DATA_DIR).glob("*.pdf"))
        print(f"📄 Found {len(pdf_files)} PDF files in data directory:")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        print()
        
        if len(pdf_files) == 0:
            print("⚠️ No PDF files found in data directory")
            print("💡 Add PDF files to process and try again")
            return
        
        # Check for problematic files
        problematic_count = check_pdf_files(config.DATA_DIR)
        if problematic_count > 0:
            print(f"\n⚠️ Fixed {problematic_count} problematic files")
            # Recount after cleanup
            pdf_files = list(Path(config.DATA_DIR).glob("*.pdf"))
            print(f"📄 Remaining valid PDF files: {len(pdf_files)}")
            
        # Check if vector store exists
        vector_store_exists = os.path.exists(config.CHROMA_DIR)
        print(f"🗄️ Vector store exists: {'Yes' if vector_store_exists else 'No'}")
        print()
        
        # Ask user what to do
        print("Choose an option:")
        print("1. Add new documents to existing vector store")
        print("2. Clear and rebuild vector store from scratch")
        print("3. Update with all PDFs (recommended for new data)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🔄 Adding new documents to existing vector store...")
            vectordb = update_vector_store_from_directory()
            if vectordb:
                print("✅ Vector store updated successfully!")
            
        elif choice == "2":
            confirm = input("⚠️ This will delete all existing data. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                print("\n🔄 Clearing and rebuilding vector store...")
                vectordb = clear_and_rebuild_vector_store()
                if vectordb:
                    print("✅ Vector store rebuilt successfully!")
            else:
                print("❌ Operation cancelled")
                
        elif choice == "3":
            print("\n🔄 Updating vector store with all PDFs...")
            vectordb = update_vector_store_from_directory()
            if vectordb:
                print("✅ Vector store updated successfully!")
                
        elif choice == "4":
            print("👋 Goodbye!")
            
        else:
            print("❌ Invalid choice")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
