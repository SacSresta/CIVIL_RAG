# Civic RAG Project

This project implements a Retrieval-Augmented Generation (RAG) pipeline with a Streamlit frontend and SQLite query persistence.

## Structure
- `backend/`: Core logic (RAG, database)
- `frontend/`: Streamlit UI
- `data/`: PDF/data storage
- `queries.db`: SQLite database (auto-created)
- `requirements.txt`: Python dependencies
- `config.py`: Configuration

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run civic_rag/frontend/app.py`

