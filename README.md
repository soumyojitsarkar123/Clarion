# Clarion - Intelligent Document Intelligence System

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Clarion-black)](https://github.com/soumyojitsarkar123/Clarion)

**Enterprise-grade document intelligence system for knowledge extraction, semantic analysis, and RAG-powered question answering**

[Overview](#overview) • [Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [API](#api-endpoints) • [Configuration](#configuration) • [Contributing](CONTRIBUTING.md)

</div>

---

## 🎯 Overview

**Clarion** is a production-ready document intelligence platform engineered to transform unstructured documents into actionable knowledge. By combining advanced semantic analysis with modern Large Language Models, Clarion enables organizations to extract structured insights, build knowledge graphs, and implement retrieval-augmented generation (RAG) systems at scale.

### 🎯 Perfect For:
- 📚 Enterprise document processing and archival systems
- 🔬 Research data extraction and analysis  
- 📖 Knowledge base creation and management
- 💼 Business intelligence and document analytics
- 🎓 Educational content analysis and structuring
- 📑 Legal document review and analysis
- 💡 Custom knowledge graph creation

### 💡 Key Value Propositions:
- ✨ **No Vendor Lock-in** - Use OpenAI or local LLMs (Ollama)
- 🔒 **Privacy-First** - Keep data local, no cloud uploads required
- ⚡ **Production-Ready** - Async processing, scalable architecture
- 🧠 **Intelligent** - Semantic understanding of document structure
- 🎨 **Beautiful UI** - Professional web interface with visualizations
- 🔧 **Configurable** - Adjust all parameters for your needs

---

## ✨ Core Features

### 📄 Document Processing Pipeline
- **Multi-Format Support**: PDF, DOCX with extensibility for other formats
- **Intelligence-Based Chunking**: Structure-aware semantic segmentation preserving document hierarchy
- **Configurable Parameters**: 
  - Adjustable chunk size (default: 512 tokens)
  - Overlap between chunks (default: 50 tokens)
  - Minimum chunk size thresholds (default: 100 tokens)
- **Batch Processing**: Asynchronous document handling with background job queue
- **Smart Text Extraction**: Intelligent text cleaning and preprocessing with noise removal
- **Support for Large Documents**: Handles documents up to 50MB

### 🧠 Knowledge Extraction & Analysis
- **Concept Identification**: Automatic detection and indexing of key concepts using transformer models
- **Semantic Relationships** (6+ types):
  - Hierarchical relationships (parent-child, generalization-specialization)
  - Causal relationships (cause-effect)
  - Comparative relationships (better than, similar to)
  - Associative relationships (related to)
  - Reference relationships (cites, references)
  - Temporal relationships (before, after, during)
- **Entity Recognition**: Named entity recognition and linking
- **Semantic Similarity Analysis**: Vector-based similarity computation using FAISS
- **Hierarchy Generation**: Automatic concept hierarchy tree construction
- **Concept Frequency Analysis**: Importance scoring based on occurrence patterns

### 🕸️ Knowledge Graphs
- **Interactive Visualization**: Real-time knowledge graph rendering with Cytoscape.js
- **Graph Traversal**: Navigate through concept relationships with bidirectional links
- **Export Capabilities**: Export graph data in JSON, CSV, and GraphML formats
- **Customizable Layouts**: 
  - Force-directed layout (recommended)
  - Hierarchical layout
  - Circular layout
  - Grid layout
- **Node and Edge Filtering**: Filter by concept type, relationship strength
- **Statistics & Metrics**: Network statistics, centrality measures, clustering coefficients
- **Performance Optimized**: Handles graphs with 1000+ nodes efficiently

### 💬 RAG (Retrieval-Augmented Generation)
- **Semantic Search**: Find relevant document chunks using vector similarity
- **LLM Integration**: Support for both OpenAI (GPT-4, GPT-3.5) and local models (Ollama, Llama 2, Mistral, Qwen)
- **Context-Aware Responses**: Answers grounded in document content with source citations
- **Multiple LLM Support**: Seamlessly switch between cloud and local LLM providers
- **Configurable Retrieval**: 
  - Adjustable number of retrieved chunks (default: 5)
  - Configurable similarity thresholds (default: 0.5)
  - Custom prompt templates
- **Citation Management**: Automatic source attribution with confidence scores
- **Temperature Control**: Adjust response creativity (0=deterministic, 1=creative)

### 📊 Summarization & Analysis
- **Multi-Level Summaries**: 
  - Chunk-level summaries
  - Section summaries
  - Document-level summaries
- **Abstractive Summarization**: LLM-powered summary generation (not just extraction)
- **Key Points Extraction**: Automatic extraction and ranking of document key points
- **Section-Based Analysis**: Per-section summaries and hierarchical analysis
- **Custom Summary Templates**: Adjust summary format and focus areas

### 🚀 Production-Ready Architecture
- **Async Processing**: Non-blocking document processing with FastAPI
- **Background Job Queue**: Asynchronous task handling for long-running operations (up to N concurrent)
- **SQLite Database**: Reliable local persistence with transaction support and ACID guarantees
- **FAISS Vector Store**: High-performance semantic search with CPU optimization
- **CORS Support**: Secure cross-origin requests for web clients with configurable origins
- **Health Monitoring**: Built-in system health checks and detailed status endpoints
- **Error Handling**: Comprehensive error handling with meaningful messages
- **Logging**: Structured logging with configurable verbosity levels
- **Session Management**: Persistent session data with automatic cleanup

### 🎨 Professional Web Interface
- **Modern UI**: Clean, responsive interface built with vanilla JavaScript and HTML5
- **Real-Time Updates**: Polling-based status updates for document processing (configurable interval)
- **Drag-Drop Upload**: Intuitive file upload with visual feedback and drag-drop zones
- **Status Dashboard**: Real-time system status and statistics display
- **Knowledge Graph Visualization**: Interactive graph exploration with zoom, pan, search
- **LLM Configuration**: Switch between LLM providers in the UI without restart
- **Document Management**: List, delete, and manage uploaded documents
- **Query Interface**: Integrated RAG query interface with history
- **System Monitoring**: View active jobs, storage usage, model information
- **Responsive Design**: Works on desktop, tablet, and mobile devices

---

## 🛠️ Technology Stack

### Backend
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | FastAPI | 0.109+ | Modern async Python web framework |
| **Server** | Uvicorn | 0.27+ | ASGI server for FastAPI |
| **Data Validation** | Pydantic | 2.5+ | Type hints and validation |
| **Settings** | Pydantic Settings | 2.1+ | Environment configuration |
| **Database** | SQLite | Built-in | Local persistent storage |
| **Async DB** | aiosqlite | 0.19+ | Non-blocking database access |
| **Vector Search** | FAISS | 1.7.4+ | Semantic similarity search |
| **Embeddings** | Sentence-Transformers | 2.2+ | Text-to-vector conversion |
| **LLM Integration** | OpenAI Python | 1.10+ | Cloud LLM provider |
| **HTTP Client** | requests | 2.31+ | HTTP operations |
| **Document Processing** | PyPDF2 | 3.0+ | PDF text extraction |
| **Document Processing** | python-docx | 1.1+ | DOCX support |
| **Environment** | python-dotenv | 1.0+ | .env file loading |
| **Data Science** | NumPy | 1.24+ | Numerical operations |
| **Visualization** | Matplotlib | 3.8+ | Graph generation |
| **Utilities** | psutil | 5.9+ | System monitoring |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Vanilla JavaScript (ES6+) | No framework dependencies |
| **Markup** | HTML5 | Semantic structure |
| **Styling** | CSS3 (Modern) | Responsive design |
| **Visualization** | Cytoscape.js | Graph rendering |
| **API Communication** | Fetch API | REST client |
| **DOM Manipulation** | Vanilla Dom API | No jQuery dependency |

---

## ⚡ Quick Start (5 Minutes)

### Prerequisites
- **Python 3.10+** (recommended: 3.11 or 3.12)
- **pip** package manager
- **8GB+ RAM** (for optimal embedding performance)
- **2GB+ Disk Space** (for vectorstore and processing)
- **Windows/macOS/Linux** (any OS with Python support)

### Step 1: Clone Repository
```bash
git clone https://github.com/soumyojitsarkar123/Clarion.git
cd Clarion
```

### Step 2: Backend Setup (3 minutes)
```bash
cd Clarion-Backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment
```bash
# Copy example configuration
cp .env.example .env

# Edit .env file with your settings
# (Only OPENAI_API_KEY required for OpenAI, optional for Ollama)
```

### Step 4: Start Backend
```bash
python main.py
```

**Expected Output:**
```
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Start Frontend (new terminal)
```bash
cd frontend
python -m http.server 8080
```

**Expected Output:**
```
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...
```

### Step 6: Access Application
- **Frontend**: http://localhost:8080
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### ✅ Quick Verification
1. Open http://localhost:8080
2. Click "Check System Status" button
3. Verify all services show as "ready" ✅

---

## 🏗️ System Architecture

### Overall Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Port 8080)                      │
│           HTML5 + CSS3 + Vanilla JavaScript                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Upload Interface  │  Graph Visualization  │  Dashboard │  │
│  │  - Drag & Drop     │  - Interactive Graph  │  - Status  │  │
│  │  - File List       │  - Zoom & Pan        │  - Stats   │  │
│  │  - Status Monitor  │  - Search            │  - Config  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST with CORS
┌──────────────────────▼──────────────────────────────────────┐
│                  FastAPI Backend (Port 8000)                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ API Router Layer (8 routers)                           │  │
│  │                                                         │  │
│  ├─ upload_router          ─ File upload & management    │  │
│  ├─ analyze_router         ─ Start document analysis     │  │
│  ├─ knowledge_map_router   ─ Extract concepts & relations│  │
│  ├─ query_router           ─ RAG querying interface     │  │
│  ├─ summary_router         ─ Generate summaries         │  │
│  ├─ graph_router           ─ Graph operations           │  │
│  ├─ dataset_router         ─ Dataset management         │  │
│  └─ status_router          ─ System health & monitoring │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Service Layer (9 services)                             │  │
│  │                                                         │  │
│  ├─ document_service              Document CRUD & metadata│  │
│  ├─ chunking_service              Semantic text splitting│  │
│  ├─ embedding_service             Text-to-vector & storage│  │
│  ├─ knowledge_map_service         Concept extraction    │  │
│  ├─ retrieval_service             Semantic search & RAG │  │
│  ├─ summary_service               Summary generation    │  │
│  ├─ background_service            Async job processing  │  │
│  ├─ relation_dataset_service      Model training        │  │
│  └─ processing_pipeline           Orchestration         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Model Layer (9 Pydantic models)                        │  │
│  │                                                         │  │
│  ├─ document.py          Document metadata & status      │  │
│  ├─ chunk.py             Text chunk representation       │  │
│  ├─ embedding.py         Vector & embedding data         │  │
│  ├─ knowledge_map.py     Concepts & relationships        │  │
│  ├─ graph.py             Graph structure (nodes, edges)  │  │
│  ├─ summary.py           Summary data models             │  │
│  ├─ retrieval.py         Query & retrieval data          │  │
│  ├─ response.py          API response formats            │  │
│  └─ (additional models)                                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Utility Layer                                          │  │
│  │                                                         │  │
│  ├─ config.py            Settings & environment          │  │
│  ├─ logger.py            Logging setup                   │  │
│  ├─ sqlite.py            Database utilities              │  │
│  ├─ file_handler.py      File operations                 │  │
│  └─ graph_store.py       Graph persistence               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└────────────────┬──────────────────┬──────────────────┬────┘
                 │                  │                  │
      ┌──────────▼─┐    ┌──────────▼─┐    ┌──────────▼─┐
      │  SQLite DB │    │ FAISS Store│    │ LLM APIs   │
      │            │    │            │    │            │
      │ Tables:    │    │ Vectors:   │    │ Providers: │
      │ documents  │    │ embeddings │    │ - OpenAI   │
      │ chunks     │    │ indices    │    │ - Ollama   │
      │ metadata   │    │ similarity │    │ - Others   │
      │ graphs     │    │ search     │    │            │
      └────────────┘    └────────────┘    └────────────┘
```

### Data Flow Pipeline
```
1. Document Upload
   │
   ├─> File Validation (PDF/DOCX check)
   ├─> Storage in Disk
   └─> Database Entry Created
        │
        ▼
2. Text Extraction
   │
   ├─> PyPDF2 (for PDFs)
   ├─> python-docx (for DOCX)
   └─> Metadata Extraction (title, author, etc.)
        │
        ▼
3. Preprocessing
   │
   ├─> Text Cleaning
   ├─> Whitespace Normalization
   ├─> Encoding Fix
   └─> Noise Removal
        │
        ▼
4. Semantic Chunking
   │
   ├─> Structure Detection (headings, sections)
   ├─> Intelligent Splitting (preserve boundaries)
   ├─> Overlap Addition (context preservation)
   └─> Chunk Storage (SQLite)
        │
        ├──────┬──────────────────┬────────────────┬────────────┐
        │      │                  │                │            │
        ▼      ▼                  ▼                ▼            ▼
5a. Embedding    5b. Knowledge Map  5c. Summary   5d. Graph    5e. Analysis
    │                │                 │          │            │
    ├─> Text      ├─> Concept      ├─> Chunk   ├─> Nodes   ├─> Stats
    │  Processing │  Identification │  Summary  │  Creation │  Metrics
    │
    ├─> Vector    ├─> Relationship ├─> Section └─> Edges   └─> Quality
    │  Generation │  Detection        │ Summary    Creation    Scores
    │
    └─> FAISS     ├─> Hierarchy     └─> Full
       Storage      │  Creation         Document
                    │                   Summary
                    └─> Storage
        │
        ▼
6. Final Storage
   │
   ├─> SQLite (text, metadata)
   ├─> FAISS (vectors for search)
   └─> Status Update (completed)
        │
        ▼
7. Query & Retrieval
   │
   ├─> Semantic Search (FAISS)
   ├─> Chunk Retrieval
   ├─> LLM Processing
   └─> Answer Generation

```

---

## 🔌 API Endpoints (Complete Reference)

### Document Upload & Management

#### POST `/upload` - Upload Document
```http
Endpoint: POST http://localhost:8000/upload
Content-Type: multipart/form-data

Request:
  file: <PDF or DOCX file>

Response (200):
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "research_paper.pdf",
  "file_size": 2048576,
  "upload_time": "2024-02-28T10:30:00Z",
  "status": "uploaded"
}

Errors:
  400: Invalid file type
  413: File too large
  500: Server error
```

#### GET `/upload/list` - List All Documents
```http
Endpoint: GET http://localhost:8000/upload/list

Response (200):
{
  "documents": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "document.pdf",
      "file_size": 2048576,
      "upload_time": "2024-02-28T10:30:00Z",
      "status": "analyzed",
      "chunk_count": 15,
      "processing_time_seconds": 45.2
    }
  ],
  "total": 1,
  "total_size_mb": 2.0
}
```

### Document Analysis

#### POST `/analyze/{document_id}` - Start Analysis
```http
Endpoint: POST http://localhost:8000/analyze/{document_id}

Response (202):
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Analysis started",
  "job_id": "job-550e8400-e29b-41d4-a716-446655440000"
}
```

#### GET `/status/{document_id}` - Get Document Status
```http
Endpoint: GET http://localhost:8000/status/{document_id}

Response (200):
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "research_paper.pdf",
  "status": "analyzed",
  "progress": 100,
  "chunks": 15,
  "embedding_status": "completed",
  "summary_status": "completed",
  "knowledge_map_status": "completed",
  "processing_started": "2024-02-28T10:30:00Z",
  "processing_completed": "2024-02-28T10:31:15Z",
  "processing_time_seconds": 75.5
}
```

### Knowledge Extraction

#### GET `/knowledge-map/{document_id}` - Get Knowledge Map
```http
Endpoint: GET http://localhost:8000/knowledge-map/{document_id}

Response (200):
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "concepts": [
    {
      "id": "concept_001",
      "name": "Machine Learning",
      "type": "topic",
      "frequency": 12,
      "relevance_score": 0.95,
      "first_mentioned": 150,
      "last_mentioned": 2500
    }
  ],
  "relationships": [
    {
      "source": "concept_001",
      "target": "concept_002",
      "type": "hierarchical",
      "label": "is_a",
      "strength": 0.87,
      "evidence_count": 4
    }
  ],
  "concept_count": 27,
  "relationship_count": 43,
  "hierarchy": {
    "root_concepts": ["concept_001", "concept_003"],
    "levels": 4,
    "tree_structure": {}
  }
}
```

#### GET `/graph/{document_id}` - Get Knowledge Graph (Visualization)
```http
Endpoint: GET http://localhost:8000/graph/{document_id}

Response (200):
{
  "nodes": [
    {
      "id": "concept_001",
      "label": "Machine Learning",
      "size": 40,
      "color": "#FF6B6B",
      "importance": 0.95,
      "frequency": 12,
      "type": "topic"
    }
  ],
  "edges": [
    {
      "id": "edge_001",
      "source": "concept_001",
      "target": "concept_002",
      "label": "related_to",
      "weight": 0.87,
      "curved": true
    }
  ],
  "styles": {
    "node_colors": {},
    "layout": "cose"
  }
}
```

### Query & Retrieval (RAG)

#### POST `/query` - RAG Query
```http
Endpoint: POST http://localhost:8000/query
Content-Type: application/json

Request:
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What are the main concepts discussed?",
  "top_k": 5,
  "llm_model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 500
}

Response (200):
{
  "question": "What are the main concepts discussed?",
  "answer": "Based on the document, the main concepts are Machine Learning, Deep Learning, Neural Networks, Data Processing, and Model Training. These are foundational concepts that build upon each other...",
  "sources": [
    {
      "chunk_id": "chunk_001",
      "content": "Machine Learning is a subset of AI...",
      "similarity_score": 0.92,
      "chunk_index": 2
    }
  ],
  "confidence": 0.88,
  "retrieval_time_ms": 234,
  "llm_time_ms": 2100
}
```

### Summarization

#### GET `/summary/{document_id}` - Get Document Summary
```http
Endpoint: GET http://localhost:8000/summary/{document_id}

Response (200):
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_summary": "This research paper explores Machine Learning techniques...",
  "section_summaries": {
    "Introduction": "The introduction establishes...",
    "Main Content": "The main content discusses...",
    "Conclusion": "In conclusion, this paper..."
  },
  "key_points": [
    "Key point 1: Machine Learning has revolutionized...",
    "Key point 2: Deep Learning models achieve...",
    "Key point 3: Neural networks require..."
  ],
  "summary_length": 450,
  "key_points_count": 5
}
```

### System Status & Health

#### GET `/health` - Health Check
```http
Endpoint: GET http://localhost:8000/health

Response (200):
{
  "status": "healthy",
  "timestamp": "2024-02-28T10:35:00Z",
  "services": {
    "database": "ok",
    "embeddings": "ready",
    "graph_engine": "ready",
    "background_processor": "ready",
    "llm": "ready",
    "vectorstore": "ok"
  },
  "active_jobs": 0,
  "total_documents": 5,
  "uptime_seconds": 3600
}
```

#### GET `/system-status` - Detailed System Status
```http
Endpoint: GET http://localhost:8000/system-status

Response (200):
{
  "system": "running",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "python_version": "3.11.2",
  
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "status": "ready"
  },
  
  "embedding": {
    "model": "all-MiniLM-L6-v2",
    "device": "cpu",
    "status": "ready"
  },
  
  "database": {
    "type": "sqlite",
    "status": "ok",
    "size_mb": 15.2
  },
  
  "storage": {
    "total_documents": 5,
    "total_chunks": 87,
    "vectorstore_size_mb": 12.5,
    "database_size_mb": 15.2,
    "total_size_mb": 27.7
  },
  
  "performance": {
    "avg_chunking_time_ms": 2500,
    "avg_embedding_time_ms": 5000,
    "avg_query_time_ms": 2800
  }
}
```

---

## ⚙️ Configuration Guide

### Environment Variables (.env File)

```bash
# ============================================
# APPLICATION CONFIGURATION
# ============================================
APP_NAME=Intelligent Document Knowledge System
APP_VERSION=1.0.0
DEBUG=false

# ============================================
# DATABASE CONFIGURATION
# ============================================
DATABASE_URL=sqlite:///./data/clarion.db
SQLITE_TIMEOUT_SECONDS=30.0

# ============================================
# EMBEDDING MODEL CONFIGURATION
# ============================================
# Popular options:
# - all-MiniLM-L6-v2 (light, default)
# - all-mpnet-base-v2 (better quality)
# - all-distilroberta-v1 (fast)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu  # Options: cpu, cuda, mps

# ============================================
# DOCUMENT CHUNKING STRATEGY
# ============================================
# Larger chunks: Better for summaries, context
# Smaller chunks: Better for detailed analysis
CHUNK_SIZE=512          # Most appropriate: 256-1024
CHUNK_OVERLAP=50        # Usually 10% of chunk size
MIN_CHUNK_SIZE=100

# ============================================
# LLM CONFIGURATION - Choose ONE Provider
# ============================================

# OPTION 1: OpenAI (Cloud-based)
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4              # Options: gpt-4, gpt-3.5-turbo
OPENAI_TEMPERATURE=0.3          # Range: 0-1 (lower = deterministic)
OPENAI_MAX_TOKENS=2000          # Maximum response length

# OR OPTION 2: Ollama (Local - Recommended for privacy)
OLLAMA_MODEL=qwen3:latest       # Popular: qwen3, llama2, mistral
OLLAMA_API_BASE=http://localhost:11434/v1
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000

# ============================================
# RETRIEVAL & SEARCH CONFIGURATION
# ============================================
DEFAULT_TOP_K=5                 # Chunks to retrieve for context
SIMILARITY_THRESHOLD=0.5        # Minimum relevance score (0-1)
ALLOW_LEGACY_PICKLE_LOADING=false

# ============================================
# FILE UPLOAD CONFIGURATION
# ============================================
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=[".pdf", ".docx"]

# ============================================
# CORS CONFIGURATION (For Frontend Access)
# ============================================
CORS_ALLOWED_ORIGINS=["http://localhost:8080", "http://127.0.0.1:8080"]
CORS_ALLOW_CREDENTIALS=false

# ============================================
# DIRECTORY CONFIGURATION
# ============================================
DATA_DIR=./data
VECTORSTORE_DIR=./data/vectorstore
LOGS_DIR=./logs
```

### Configuration Parameters Explained

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `CHUNK_SIZE` | 512 | 256-2048 | Tokens per chunk (larger = more context, slower) |
| `CHUNK_OVERLAP` | 50 | 0-200 | Overlapping tokens (maintains context across chunks) |
| `MIN_CHUNK_SIZE` | 100 | 50-500 | Minimum chunk size (smaller = more chunks) |
| `DEFAULT_TOP_K` | 5 | 1-20 | Chunks retrieved for RAG (more = slower but comprehensive) |
| `SIMILARITY_THRESHOLD` | 0.5 | 0-1 | Minimum relevance score (higher = stricter matching) |
| `OPENAI_TEMPERATURE` | 0.3 | 0-1 | Response creativity (0=deterministic, 1=creative) |
| `OPENAI_MAX_TOKENS` | 2000 | 100-4000 | Maximum response length |
| `EMBEDDING_BATCH_SIZE` | 32 | 8-128 | Batch size for embedding (larger = faster on GPU) |
| `MAX_FILE_SIZE_MB` | 50 | 10-500 | Maximum uploadable file size |

### Configuration Presets

#### 🚀 Fast Mode (For Quick Responses)
```bash
CHUNK_SIZE=256
CHUNK_OVERLAP=25
DEFAULT_TOP_K=3
SIMILARITY_THRESHOLD=0.6
OPENAI_TEMPERATURE=0.2
```

#### 🎯 Balanced Mode (Recommended)
```bash
CHUNK_SIZE=512
CHUNK_OVERLAP=50
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.5
OPENAI_TEMPERATURE=0.3
```

#### 📚 Comprehensive Mode (For Detailed Analysis)
```bash
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
DEFAULT_TOP_K=10
SIMILARITY_THRESHOLD=0.3
OPENAI_TEMPERATURE=0.7
```

#### 🔒 Privacy Mode (Local Only)
```bash
OLLAMA_MODEL=qwen3:latest      # Use local model
OLLAMA_API_BASE=http://localhost:11434/v1
CORS_ALLOWED_ORIGINS=["http://localhost:8080"]  # Local only
ENCODING_DEVICE=cpu             # Use CPU for privacy
```

---

## 📚 Usage Examples

### Example 1: Upload and Analyze (cURL)
```bash
# Step 1: Upload document
RESPONSE=$(curl -s -X POST -F "file=@document.pdf" http://localhost:8000/upload)
DOC_ID=$(echo $RESPONSE | jq -r '.document_id')
echo "Uploaded: $DOC_ID"

# Step 2: Start analysis
curl -s -X POST http://localhost:8000/analyze/$DOC_ID

# Step 3: Check progress
curl -s http://localhost:8000/status/$DOC_ID | jq '.progress'

# Step 4: Get results
curl -s http://localhost:8000/knowledge-map/$DOC_ID | jq '.concepts'
```

### Example 2: Query Document with Python
```python
import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
DOCUMENT_ID = "550e8400-e29b-41d4-a716-446655440000"

# Upload document
with open("research_paper.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
    doc_id = response.json()["document_id"]
    print(f"✓ Uploaded: {doc_id}")

# Start analysis
requests.post(f"{BASE_URL}/analyze/{doc_id}")
print("✓ Analysis started")

# Wait for completion
while True:
    status = requests.get(f"{BASE_URL}/status/{doc_id}").json()
    progress = status.get("progress", 0)
    print(f"Progress: {progress}%")
    
    if progress == 100:
        break
    time.sleep(2)

# Query document
query_response = requests.post(
    f"{BASE_URL}/query",
    json={
        "document_id": doc_id,
        "question": "What are the main findings?",
        "top_k": 5,
        "llm_model": "gpt-4"
    }
)

result = query_response.json()
print(f"\nQuestion: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")

# Show sources
print("\nSources:")
for source in result['sources']:
    print(f"  - {source['content'][:100]}... (similarity: {source['similarity_score']:.2f})")
```

### Example 3: Batch Processing Multiple Documents
```bash
#!/bin/bash

# Process all PDFs in a directory
for file in documents/*.pdf; do
    echo "Processing: $file"
    
    # Upload
    response=$(curl -s -F "file=@$file" http://localhost:8000/upload)
    doc_id=$(echo $response | jq -r '.document_id')
    echo "  Uploaded ID: $doc_id"
    
    # Analyze
    curl -s -X POST "http://localhost:8000/analyze/$doc_id" > /dev/null
    
    # Wait for completion
    while true; do
        status=$(curl -s "http://localhost:8000/status/$doc_id" | jq -r '.progress')
        [ "$status" = "100" ] && break
        sleep 1
    done
    
    # Export results
    curl -s "http://localhost:8000/knowledge-map/$doc_id" > "results/${doc_id}_map.json"
    curl -s "http://localhost:8000/summary/$doc_id" > "results/${doc_id}_summary.json"
    
    echo "  ✓ Completed"
done
```

### Example 4: Using Different LLM Providers
```python
import requests

BASE_URL = "http://localhost:8000"
DOC_ID = "your_document_id"

# Query with OpenAI
openai_response = requests.post(
    f"{BASE_URL}/query",
    json={
        "document_id": DOC_ID,
        "question": "Summarize the main points",
        "llm_model": "gpt-4"
    }
)
print("OpenAI Answer:", openai_response.json()["answer"])

# Query with Ollama (local)
# First, ensure Ollama is running: ollama serve
ollama_response = requests.post(
    f"{BASE_URL}/query",
    json={
        "document_id": DOC_ID,
        "question": "Summarize the main points",
        "llm_model": "qwen3:latest"  # Or any Ollama model
    }
)
print("Ollama Answer:", ollama_response.json()["answer"])
```

---

## 🎛️ Advanced Configuration

### Custom Embedding Models

```bash
# Smaller, faster (light documents)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Larger, better quality (detailed analysis)
EMBEDDING_MODEL_NAME=all-mpnet-base-v2

# Fast, good for large batches
EMBEDDING_MODEL_NAME=all-distilroberta-v1

# Specialized for code
EMBEDDING_MODEL_NAME=CodeBERT
```

### GPU Acceleration

```bash
# Enable CUDA (NVIDIA GPUs)
EMBEDDING_DEVICE=cuda

# Enable MPS (Apple Silicon)
EMBEDDING_DEVICE=mps

# Check available devices
python -c "import torch; print(torch.cuda.is_available())"
```

### Tuning for Performance

**For Large Documents (100+ pages):**
```bash
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
EMBEDDING_BATCH_SIZE=64
EMBEDDING_DEVICE=cuda
DEFAULT_TOP_K=10
```

**For Many Documents (100+):**
```bash
EMBEDDING_BATCH_SIZE=128
SQLITE_TIMEOUT_SECONDS=60
CORS_ALLOW_CREDENTIALS=true
```

**For Limited Resources (Laptop, Server):**
```bash
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=8
CHUNK_SIZE=256
DEFAULT_TOP_K=3
```

---

## 📊 Performance Benchmarks

### Processing Times (8GB RAM, CPU)

| Task | Small Doc (5MB) | Medium (15MB) | Large (45MB) |
|------|---|---|---|
| Upload | < 1s | 2-3s | 5-7s |
| Text Extract | 1-2s | 3-5s | 8-12s |
| Chunking | 1s | 2-3s | 5-8s |
| Embedding | 5-10s | 15-30s | 40-60s |
| Knowledge Map | 3-5s | 8-12s | 20-30s |
| Summary | 2-4s | 5-10s | 15-20s |
| RAG Query | 1-3s | 2-5s | 3-7s |
| **Total** | **15-30s** | **40-70s** | **100-150s** |

### Optimization Tips

1. **Use GPU**: 3-5x speedup with CUDA
2. **Increase Batch Size**: Better throughput for multiple documents
3. **Local LLM**: Eliminates API latency
4. **Adjust Chunk Size**: Smaller = more chunks but faster processing of each
5. **Pre-download Models**: Avoid first-run delays

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | find ":8000"  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Use different port
uvicorn main:app --port 8001
```

### Embedding Model Not Downloading
```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Clear cache if corrupted
rm -rf ~/.cache/huggingface
```

### Out of Memory (OOM)
```bash
# You likely have too many concurrent embeddings
# Solutions:
# 1. Reduce EMBEDDING_BATCH_SIZE
EMBEDDING_BATCH_SIZE=8

# 2. Process documents one at a time
# 3. Use smaller embedding model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

### Ollama Connection Issues
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Or add to systemd (Linux)
systemctl enable ollama
systemctl start ollama
```

### Database Locked
```bash
# SQLite timeout issue
# Solution: Increase timeout in .env
SQLITE_TIMEOUT_SECONDS=60

# Or restart the application
# Kill all Python processes
pkill -f "python main.py"
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines (Black, isort)
- Git workflow (fork → branch → PR)
- Testing requirements (pytest)
- Commit message format
- Code review process

### Quick Contribution Steps
```bash
# 1. Fork and clone
git clone https://github.com/your-username/Clarion.git

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes locally
# (follow code style guide)

# 4. Test your changes
pytest tests/

# 5. Commit and push
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature

# 6. Create Pull Request on GitHub
```

### Areas for Contribution
- 🐛 Bug fixes and issue resolution
- ✨ New LLM provider support  
- 📄 Additional document format support (PPTX, TXT, etc.)
- 🎨 UI/UX improvements
- 🧪 Test coverage
- 📚 Documentation improvements
- 🚀 Performance optimizations
- 🔒 Security enhancements

---

## 📈 Roadmap

### v1.1.0 (Near Future)
- [ ] Batch document processing
- [ ] Export to PDF/JSON with formatting
- [ ] Advanced graph analytics
- [ ] Custom relationship types
- [ ] Document search full-text index

### v1.2.0 (RoadmapQ2)
- [ ] Multi-document comparative analysis
- [ ] User authentication & sessions
- [ ] Document versioning & history
- [ ] Advanced caching strategies
- [ ] WebSocket real-time updates

### v2.0.0 (Future)
- [ ] Distributed processing (Celery + Redis)
- [ ] GraphQL API option
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Advanced ML models (fine-tuning)
- [ ] Integration with PKM tools

---

## 📄 License

MIT License - Full details in [LICENSE](LICENSE)

**You are free to:**
- Use commercially
- Modify and distribute
- Use in private/closed projects

**You must:**
- Include copyright and license notice

---

## 🙏 Acknowledgments

### Core Technologies
- [FastAPI](https://fastapi.tiangolo.com/) - Amazing async web framework
- [Sentence-Transformers](https://www.sbert.net/) - State-of-the-art embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [Cytoscape.js](https://cytoscape.org/) - Professional graph visualization

### LLM Providers
- [OpenAI](https://openai.com/) - GPT-4 and GPT-3.5 models
- [Ollama](https://ollama.ai/) - Local LLM support

### Document Processing
- [PyPDF2](https://github.com/py-pdf/PyPDF2) - PDF handling
- [python-docx](https://python-docx.readthedocs.io/) - DOCX support

---

## 📞 Support

### Documentation
- 📖 [README.md](README.md) - Project overview
- 📋 [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- 📚 [docs/](docs/) - Technical documentation
- 🎨 [frontend/README.md](frontend/README.md) - Frontend guide

### Getting Help
- 🐛 [Report Issues](https://github.com/soumyojitsarkar123/Clarion/issues)
- 💬 [Discussions](https://github.com/soumyojitsarkar123/Clarion/discussions)
- 📧 [Contact](mailto:soumyojitsarkar123@gmail.com)

### Connect
- ⭐ Star the repository to show support
- 🍴 Fork to contribute
- 👀 Watch for updates

---

<div align="center">

## 🚀 Ready to Get Started?

**[Clone Repository](#quick-start) • [View Documentation](#documentation) • [Report Issues](https://github.com/soumyojitsarkar123/Clarion/issues)**

---

Made with ❤️ by the Clarion Community

[⬆ Back to Top](#clarion---intelligent-document-intelligence-system)

</div>
