# Configuration loader for Civic RAG
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# RAG/Embedding Config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Data/Vector Store Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHROMA_DIR = os.path.join(BASE_DIR, 'chroma_db')
DB_PATH = os.path.join(BASE_DIR, 'queries.db')
