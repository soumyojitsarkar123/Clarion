# APPENDIX 1
## CODE & SNIPPET

### A1.1. Introduction
This chapter contains essential Python code snippets for implementing the **Clarion: Intelligent Document Structure and Knowledge Mapping System**. 
The implementation leverages **FastAPI** for the backend, **Sentence Transformers** for semantic embeddings, **FAISS** for vector retrieval, and Large Language Models (OpenAI, Gemini, Ollama) for knowledge extraction and summarization.
Code snippets are presented in modular sections reflecting the system's architecture.

---

### A1.2. Importing Required Libraries
The system integrates various libraries for web services, document processing, and AI orchestration.

**Table 1 - Core Backend & AI Libraries**
```python
import logging
import uuid
import re
import sqlite3
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import matplotlib.pyplot as plt
```

---

### A1.3. Document Loading and Preprocessing
#### A1.3.1. Structure-Aware Document Chunking
This section implements semantic segmentation of documents, ensuring context is preserved across chunks.

**Table 2 - Semantic Chunking Implementation**
```python
def _chunk_text(self, text: str, section_title: Optional[str], start_index: int) -> List[Chunk]:
    words = text.split()
    chunks = []
    
    # Process text in overlapping windows
    for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
        word_slice = words[i:i + self.chunk_size]
        chunk_text = ' '.join(word_slice)
        
        if len(chunk_text.strip()) < self.min_chunk_size:
            continue
        
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            section_title=section_title,
            content=chunk_text,
            position_index=start_index + i,
            word_count=len(word_slice)
        )
        chunks.append(chunk)
    return chunks
```

#### A1.3.2. Automated Heading Detection
Regex patterns are utilized to identify structural landmarks in the raw document text.

**Table 3 - Heading Detection Logic**
```python
def _detect_headings(self, text: str) -> List[Tuple[str, int]]:
    heading_patterns = [
        r'^(#{1,6})\s+(.+)$',         # Markdown
        r'^(\d+\.)+\s+(.+)$',        # Numbered
        r'^(?:CHAPTER|SECTION)\s+\d+', # Structured
    ]
    headings = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for pattern in heading_patterns:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                headings.append((line.strip(), i))
                break
    return headings
```

---

### A1.4. Large Language Model Integration
#### A1.4.1. Multi-Provider LLM Interface
A unified interface allows the system to switch between local execution (Ollama) and cloud APIs (OpenAI/Gemini).

**Table 4 - LLM Interface Configuration**
```python
class LLMInterface:
    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
```

#### A1.4.2. Concept Extraction Logic
Prompt engineering is used to distill document chunks into atomic knowledge concepts.

**Table 5 - Knowledge Concept Extraction**
```python
def extract_concepts(self, chunk_text: str) -> List[Dict[str, str]]:
    prompt = f"""Identify the most significant knowledge concepts in this text:
    Text: {chunk_text[:3000]}
    Return a JSON array of concepts with 'name' and 'definition'."""
    
    response = self.generate(prompt)
    return self._extract_json_fragment(response)
```

---

### A1.5. Knowledge Map Construction
#### A1.5.1. Relationship Detection
The system identifies semantic connections between concepts using contextual co-occurrence and LLM verification.

**Table 6 - Relationship Detection**
```python
def detect_all_relations(self, concepts: List[str], context: str) -> List[Dict[str, Any]]:
    concepts_str = ", ".join([f'"{c}"' for c in concepts])
    prompt = f"""Map out connections between these concepts: [{concepts_str}]
    Context: {context[:2500]}
    Return JSON array: [{{"from_concept": "X", "to_concept": "Y", "relation_type": "part-of"}}]"""
    
    response = self.generate(prompt)
    return self._extract_json_fragment(response)
```

#### A1.5.2. Heuristic Semantic Fallback
In scenarios where LLM services are offline, the system employs a deterministic heuristic for knowledge extraction.

**Table 7 - Heuristic Extraction Fallback**
```python
def _extract_concepts_heuristic(self, chunks: List[Chunk]) -> List[Concept]:
    # Extract noun phrases and frequent technical terms
    text = " ".join([c.content for c in chunks[:5]])
    words = re.findall(r'\b[A-Z][a-z]{3,}\b', text) # Simple Proper Noun Heuristic
    counter = Counter(words)
    return [Concept(name=w, definition="Extract heuristically") for w, _ in counter.most_common(10)]
```

---

### A1.6. RAG & Retrieval System
#### A1.6.1. FAISS Semantic Search
Vector index implementation for high-speed retrieval of relevant document context.

**Table 8 - Vector Store Retrieval**
```python
def retrieve(self, document_id: str, query: str, top_k: int = 5) -> List[RetrievalResult]:
    vector_store = self.embedding_service.get_vector_store(document_id)
    query_embedding = self.embedding_service.embed_query(query)
    
    # Perform similarity search in FAISS index
    search_results = vector_store.search(query_embedding, top_k)
    
    results = []
    for chunk_id, distance in search_results:
        score = 1.0 / (1.0 + distance)
        results.append(RetrievalResult(chunk_id=chunk_id, score=score))
    return results
```

#### A1.6.2. Retrieval Augmented Context Building
Formatting retrieved knowledge into a coherent prompt context for the LLM.

**Table 9 - Context Reconstruction**
```python
def retrieve_with_context(self, document_id: str, query: str) -> Tuple[List[RetrievalResult], str]:
    results = self.retrieve(document_id, query)
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"[Source {i}]: {result.content}")
    
    return results, "\n\n".join(context_parts)
```
