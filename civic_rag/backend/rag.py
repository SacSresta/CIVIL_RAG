
# This module delegates ingestion and RAG pipeline to dedicated scripts for clarity.
from .ingestion import ingest_pdf, ingest_all_pdfs_in_directory
from .rag_pipeline import answer_query
from .chatmodels import build_vector_store, load_vector_store
# You can add orchestration or legacy compatibility functions here if needed.

