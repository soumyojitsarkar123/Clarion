# Intelligent Document Structure and Knowledge Mapping System

A production-ready backend system for document structure extraction, concept hierarchy detection, and semantic relationship modeling using Retrieval-Augmented Generation (RAG) and Large Language Models.

## Features

- **Document Ingestion**: Supports PDF and DOCX file uploads with text extraction
- **Structure-Aware Chunking**: Semantic document segmentation preserving headings and sections
- **Embedding Generation**: Local embeddings using SentenceTransformers with FAISS vector store
- **Knowledge Mapping**: Extracts concepts and detects semantic relationships (prerequisite, definition, explanation, cause-effect, example-of, etc.)
- **Structured Summaries**: Hierarchical summaries organized by conceptual hierarchy
- **RAG Querying**: Semantic search with LLM-powered responses

## Architecture

```
Clarion-Backend/
├── main.py                 # Application entry point
├── routers/               # API route handlers
│   ├── upload.py          # Document upload endpoints
│   ├── analyze.py         # Analysis endpoints
│   ├── knowledge_map.py   # Knowledge map retrieval
│   └── query.py           # RAG querying
├── services/              # Business logic
│   ├── document_service.py
│   ├── chunking_service.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── knowledge_map_service.py
│   └── summary_service.py
├── models/                # Pydantic models
├── utils/                 # Utilities
├── vectorstore/           # FAISS management
└── data/                  # Data storage
```

## Requirements

- Python 3.10+
- OpenAI API key (for LLM features)

## Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings, especially OPENAI_API_KEY
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --reload
   ```

   Or with Python:
   ```bash
   python main.py
   ```

5. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs

## API Endpoints

### Upload Document
```
POST /upload
- Body: file (PDF or DOCX)
- Returns: document_id
```

### Analyze Document
```
POST /analyze/{document_id}
- Performs chunking, embedding, and knowledge map building
- Returns: chunk_count, concept_count
```

### Get Knowledge Map
```
GET /knowledge-map/{document_id}
- Returns: main_topics, subtopics, concepts, relations, dependencies
```

### Query Document
```
POST /query/{document_id}
- Body: { "query": "your question", "top_k": 5 }
- Returns: retrieval results and LLM-generated response
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL_NAME` | SentenceTransformer model | all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Max words per chunk | 512 |
| `CHUNK_OVERLAP` | Overlap between chunks | 50 |
| `OPENAI_MODEL` | LLM model for knowledge mapping | gpt-4 |
| `DEFAULT_TOP_K` | Default retrieval results | 5 |

## Usage Example

```python
import requests

# 1. Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
doc_id = response.json()["document_id"]

# 2. Analyze document
response = requests.post(f"http://localhost:8000/analyze/{doc_id}")
print(response.json())

# 3. Get knowledge map
response = requests.get(f"http://localhost:8000/knowledge-map/{doc_id}")
knowledge_map = response.json()

# 4. Query document
response = requests.post(
    f"http://localhost:8000/query/{doc_id}",
    json={"query": "What are the main concepts?", "top_k": 5}
)
print(response.json()["response"])
```

## License

MIT
