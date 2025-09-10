# CIVIL_RAG 🏛️

An AI-powered civic research and protest guidance system for Nepal, providing comprehensive analysis of political, economic, and social aspects of civil movements and protests.

## 🌟 Features

- **Multi-Modal Analysis**: Parallel processing of economic, political, and social dimensions
- **RAG-Powered Research**: Vector database with protest guidance documents and historical context
- **Real-Time Web Search**: Live data integration using BraveSearch API
- **Safety-First Approach**: Prioritizes citizen safety and legal compliance
- **LangGraph Architecture**: Advanced workflow orchestration with parallel node execution
- **Interactive Web Interface**: Streamlit-based chat interface for easy access

## 🏗️ Architecture

### Core Components

- **LangGraph Workflow**: Parallel processing nodes for comprehensive analysis
- **Vector Database**: Chroma-based document storage with HuggingFace embeddings
- **Multi-LLM Support**: ChatGroq integration with configurable models
- **Document Processing**: Automated PDF ingestion and chunking pipeline

### System Flow

```
User Question → Parallel Search Nodes → Data Merging → Multi-Aspect Analysis → Safety & Legal Review → Comprehensive Response
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Groq API Key
- BraveSearch API Key (optional, for web search)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SacSresta/CIVIL_RAG.git
   cd CIVIL_RAG
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment variables**
   Create a `.env` file in the `civic_rag` directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```

5. **Set up the vector database**
   ```bash
   python update_vector_store.py
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
civic_rag/
├── backend/
│   ├── chatmodels.py      # LangGraph workflow and analysis nodes
│   ├── utils.py           # Vector store utilities
│   ├── tools_utils.py     # RAG and web search tools
│   └── ingestion.py       # PDF processing pipeline
├── data/                  # PDF documents for RAG
├── vector_db/            # Chroma vector database
└── config.py             # Configuration management

scripts/
├── app.py                # Streamlit web interface
├── update_vector_store.py # Vector database management
└── quick_rebuild.py      # Quick vector store rebuild utility
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | API key for ChatGroq LLM | Yes |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API token for embeddings | Yes |

### Model Configuration

The system uses configurable LLM models through ChatGroq. Default model: `meta-llama/llama-4-maverick-17b-128e-instruct`

### Vector Database

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Database**: ChromaDB with persistent storage
- **Chunk Size**: 1000 characters with 200 character overlap

## 📚 Usage

### Web Interface

1. Start the application: `streamlit run app.py`
2. Navigate to `http://localhost:8501`
3. Ask questions about protests, civil rights, or civic guidance

### Programmatic Usage

```python
from civic_rag.backend.chatmodels import run_protest_guidance

# Get guidance on a specific topic
response = run_protest_guidance("What are my rights during a peaceful protest?")
print(response)
```

### Vector Database Management

```bash
# Update vector store with new documents
python update_vector_store.py

# Quick rebuild (bypasses problematic files)
python quick_rebuild.py

# Check vector store status
python -c "from civic_rag.backend.utils import get_vector_store_info; get_vector_store_info()"
```

## 🛠️ Development

### Adding New Documents

1. Place PDF files in the `civic_rag/data/` directory
2. Run the update script: `python update_vector_store.py`
3. Choose option 1 to update the existing vector store

### Customizing Analysis Nodes

The system uses modular analysis nodes that can be extended:

- `economic_analysis_node`: Economic impact analysis
- `political_analysis_node`: Political landscape analysis
- `social_analysis_node`: Social and cultural impact analysis
- `safety_analysis_node`: Safety recommendations
- `legal_analysis_node`: Legal rights and implications

### Error Handling

The system includes comprehensive error handling for:
- Corrupted PDF files
- API rate limits
- Token overflow issues
- Vector database connectivity

## 🔒 Safety & Legal Compliance

- **Safety First**: All recommendations prioritize citizen safety
- **Legal Awareness**: Provides guidance on constitutional rights and legal limitations
- **Factual Analysis**: Maintains neutrality and factual accuracy
- **Emergency Contacts**: Includes relevant emergency and legal aid contacts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system provides informational guidance only and should not be considered as legal advice. Always consult with qualified legal professionals for specific legal matters. The developers are not responsible for any actions taken based on the system's recommendations.

## 🙏 Acknowledgments

- LangChain community for the RAG framework
- ChromaDB for vector storage capabilities
- HuggingFace for embedding models
- Groq for LLM API services
- The open-source community for various dependencies

## 📞 Support

For support, issues, or questions:
- Open an issue on GitHub
- Contact: [Your Contact Information]

---

**Made with ❤️ for civic engagement and democratic participation in Nepal**
