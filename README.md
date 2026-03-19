# Clarion - Intelligent Document Analysis and Knowledge Graph System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109%2B-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/Embeddings-BAAI%2Fbge--large--en--v1.5-orange" alt="BAAI">
  <img src="https://img.shields.io/badge/LLM-Ollama%2FQwen3-purple" alt="Ollama">
</div>

## 📌 Abstract

**Clarion** is a research-oriented document intelligence platform designed to bridge the gap between unstructured text and structured knowledge representations. Developed for academic research, Clarion implements a novel pipeline that dynamically constructs interactive **Knowledge Graphs** and performs **Retrieval-Augmented Generation (RAG)** entirely on local hardware, ensuring absolute data privacy.

The system is engineered to extract hierarchical concepts, map semantic relationships, and provide context-aware query responses using state-of-the-art open-source Large Language Models (LLMs) and dense retrieval embedding models.

## 🔬 Research Objectives

1. **Automated Knowledge Extraction**: Transform raw PDFs and DOCX files into structured semantic graphs using zero-shot NLP concept identification.
2. **Contextual RAG**: Improve traditional vector search reliability by augmenting semantic dense retrieval with topological graph structures.
3. **Data Privacy & Localization**: Demonstrate that high-quality, enterprise-grade semantic analysis can execute solely locally using optimized quantized models.
4. **Dataset Generation for ML**: Automatically construct curated, structured JSON relation datasets from unstructured source texts for future model fine-tuning and academic validation.

## 🏗️ Methodology & Architecture

### 1. NLP & Dense Embedding Pipeline
Text extraction preserves document hierarchy, followed by semantic chunking (default 512 tokens with 50-token overlap). Vector embeddings are generated using **`BAAI/bge-large-en-v1.5`** (1024 dimensions) via `sentence-transformers` to maximize semantic resolution. Vectors are indexed using FAISS for low-latency similarity search.

### 2. Knowledge Graph Construction
Clarion utilizes a local **Ollama** LLM backend (`qwen3:latest` by default) to perform deep semantic parsing, identifying:
- **Concepts**: Distinct nodes representing specific terms, individuals, or theoretical ideas.
- **Relations**: Directed edges mapping interactions (e.g., hierarchical, causal, associative, temporal).

Relationships are structurally persisted in an asynchronous SQLite database, maintaining the topological integrity of the extracted knowledge.

### 3. Web Interface & Visualization
The frontend utilizes vanilla JavaScript and **Cytoscape.js** to render interactive force-directed graph layouts. This enables researchers to visually traverse concept hierarchies and evaluate relationship strengths dynamically, bridging the gap between raw data and interpretable knowledge networks.

## 📂 Repository Structure & Key Modules

The project separates the heavy analytical backend from a lightweight, dependency-free visualization frontend to maintain high performance and research reproducibility.

```text
Clarion/
├── Clarion-Backend/
│   ├── core/                        # Core analysis, evaluation, and LLM framework
│   │   ├── benchmarking/            # Performance & ablation scripts
│   │   ├── evaluation/              # RAG hallucination detection
│   │   └── llm/                     # Provider logic (glm_provider.py)
│   ├── graph/                       # Graph data models and hierarchical builders
│   ├── models/                      # Pydantic schemas (Document, Chunk, Graph, etc.)
│   ├── prompts/                     # Zero-shot extraction prompt templates
│   ├── routers/                     # FastAPI controllers
│   │   ├── analyze.py               # Document processing triggers
│   │   ├── query.py                 # RAG search endpoint
│   │   └── graph.py                 # Graph data endpoint
│   ├── services/                    # logic & ML pipelines
│   │   ├── chunking_service.py      # Semantic text segmentation
│   │   ├── embedding_service.py     # BAAI Dense vector generation
│   │   ├── knowledge_map_service.py # Zero-shot concept extraction
│   │   ├── rag_audit_service.py     # Response tracing module
│   │   └── retrieval_service.py     # FAISS semantic search
│   ├── utils/                       # Core utilities (config.py, logger.py, sqlite.py)
│   ├── export_dataset.py            # Clean JSON dataset generation script
│   └── main.py                      # FastAPI application entry point
└── frontend/
    ├── app.js                       # Logic for graph state management
    ├── index.html                   # Interactive research dashboard UI
    └── styles.css                   # Interface styling rules
```

## �🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- *Required model*: Execute `ollama run qwen3:latest` prior to starting the application backend.

### 1. Initialize Backend Environment
```bash
git clone https://github.com/soumyojitsarkar123/Clarion.git
cd Clarion/Clarion-Backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Research Parameters
Create a `.env` file in `Clarion-Backend/` with the optimized research configuration:

```ini
# Core LLM Architecture
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:latest
OLLAMA_API_BASE=http://localhost:11434/v1

# Embedding Architecture
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu

# Optimization Parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SQLITE_TIMEOUT_SECONDS=60.0
```

### 3. Launch System
**Start the Backend API:**
```bash
cd Clarion-Backend
python main.py
```

**Start the Frontend UI (in a new terminal):**
```bash
cd frontend
python -m http.server 8080
```
Navigate to **`http://localhost:8080`** to access the research dashboard.

## 📊 Dataset Export & Validation

For ongoing research validation and downstream ML model training, Clarion logs extracted relationships internally. To export a sanitized, peer-review-ready JSON dataset (containing concepts, relationships, LLM confidence scores, and source context chunks):

```bash
cd Clarion-Backend
python export_dataset.py
```
*The structured dataset will be generated at `Clarion-Backend/data/research_dataset_export.json`.*

---
*This repository and system architecture is developed as a college research initiative focusing on the intersection of semantic Knowledge Graphs, advanced RAG architectures, and localized LLM execution.*
