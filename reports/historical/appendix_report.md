# Appendix: Algorithmic Architecture, Mathematical Formulations, and Empirical Verification

**Author**: Soumyojit Sarkar  
**Affiliation**: Clarion Document Intelligence Project  
**System Repository**: Clarion Research Platform  

### Abstract & Scope
This appendix provides the formal algorithmic implementations, mathematical specifications, and empirical graph topological metrics for **Clarion** (Intelligent Document Structure, Knowledge Graph Generation, and Dataset Construction System). The software architecture integrates **FastAPI** for service orchestration, **Sentence-Transformers** (`BAAI/bge-large-en-v1.5`) and deterministic fallback hashing for vector representations, **FAISS** (`IndexFlatL2`) for dense similarity search, **NetworkX** for directed knowledge graph construction and topological analytics, and multi-provider Large Language Models (Ollama, OpenAI, DeepSeek, Gemini) for concept and relation extraction. All code listings and mathematical formulations presented herein correspond strictly to the production implementation in `Clarion-Backend/`.

---

## A1.1. Document Ingestion and Structure-Aware Chunking Service

### A1.1.1. Core System Dependencies
Table 1 specifies the foundational libraries employed across backend data ingestion, graph analysis, and AI orchestration.

**Table 1 — Core Backend, Scientific, and AI Libraries**
```python
import hashlib
import json
import logging
import math
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Set
import uuid

import faiss
from fastapi import FastAPI
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
```

### A1.1.2. Structure-Aware Sliding Window Chunking
Document chunking operates over a sequence of clean words $W = (w_1, w_2, \dots, w_N)$. To maintain consistent semantic density across technical and mathematical text, window boundaries are evaluated directly in word counts rather than raw character counts. 

Given window capacity $C_{\text{size}} = 512$ words, step overlap $C_{\text{overlap}} = 50$ words, and minimum chunk threshold $M_{\text{min\_chunk\_size}} = 100$ words:
$$\text{step} = \max(1, \text{self.chunk\_size} - \text{self.chunk\_overlap}) = \max(1, 512 - 50) = 462 \text{ words}$$

The verified implementation addresses three critical discrepancies from earlier prototype drafts:
1. **Step size calculation**: Defined as `step = max(1, self.chunk_size - self.chunk_overlap) = 462 words`, avoiding invalid zero-stride loops if parameters overlap.
2. **Word count threshold**: Evaluates `len(chunk_words) < self.min_chunk_size` over the slice words (removing the dimensionally invalid character length check `len(chunk_text.strip()) < self.min_chunk_size`, where a 100-character string corresponds to only $\approx 15\text{--}20$ words).
3. **Residual merging**: Implements residual merging by appending trailing residual text to preceding chunks via `chunks[-1]["text"] = chunks[-1]["text"] + " " + " ".join(remainder_words)` (or `chunks[-1].content += ' ' + ' '.join(remainder_words)` in the Pydantic model) rather than a naive `continue` discard that silently omitted trailing document sections. Multi-chunk documents subsequently undergo bidirectional merging via `_merge_undersized_chunks`, bounded by $1.25 \times C_{\text{size}} = 640$ words.

**Table 2 — Structure-Aware Semantic Chunking Implementation**
```python
def _chunk_text(
    self,
    text: str,
    section_title: Optional[str],
    start_index: int,
) -> List[Chunk]:
    """
    Chunk a block of text using a word-level sliding window with residual merging.
    
    Evaluates word counts directly rather than raw character counts to preserve
    semantic density and prevent trailing data loss.
    """
    words = text.split()
    if not words:
        return []

    # If text is smaller than chunk size, emit a single complete chunk
    if len(words) <= self.chunk_size:
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="",
            section_title=section_title,
            content=text,
            position_index=start_index,
            word_count=len(words)
        )
        return [chunk]

    chunks: List[Chunk] = []
    step = max(1, self.chunk_size - self.chunk_overlap)  # 512 - 50 = 462 words
    last_covered_idx = 0

    for i in range(0, len(words), step):
        word_slice = words[i:i + self.chunk_size]
        chunk_text = ' '.join(word_slice)

        # Word count threshold check: avoid creating isolated micro-chunks
        if len(word_slice) < self.min_chunk_size:
            if chunks:
                remainder_words = words[last_covered_idx:]
                if remainder_words:
                    # Residual merging: append trailing text to preceding chunk
                    chunks[-1].content += ' ' + ' '.join(remainder_words)
                    chunks[-1].word_count = len(chunks[-1].content.split())
                    last_covered_idx = len(words)
            continue

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="",
            section_title=section_title,
            content=chunk_text,
            position_index=start_index + i,
            word_count=len(word_slice)
        )
        chunks.append(chunk)
        last_covered_idx = min(len(words), i + len(word_slice))

    return chunks
```

### A1.1.3. Automated Structural Heading Detection
Document headings are extracted using regular expressions to preserve hierarchical topic trees and section metadata.

**Table 3 — Heading Detection Logic**
```python
def _detect_headings(self, text: str) -> List[Tuple[str, int]]:
    heading_patterns = [
        r'^(#{1,6})\s+(.+)$',           # Markdown headings (# through ######)
        r'^(\d+\.)+\s+(.+)$',          # Numbered decimal outlines (e.g., 1.2.3)
        r'^(?:CHAPTER|SECTION)\s+\d+', # Formal publication section labels
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

## A1.2. Knowledge Extraction: Language Model Orchestration

### A1.2.1. Multi-Provider LLM Abstraction Layer
The LLM execution layer dynamically wraps local engines (Ollama) and cloud APIs (OpenAI, DeepSeek, Gemini) via a unified client adapter interface.

**Table 4 — Multi-Provider LLM Interface Configuration**
```python
class LLMInterface:
    def __init__(self):
        self.provider = settings.llm_provider
        if self.provider == "ollama":
            self.model = settings.ollama_model
            self.api_base = settings.ollama_api_base
            self.api_key = "ollama"
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
            temperature=settings.llm_temperature
        )
        return response.choices[0].message.content
```

### A1.2.2. Concept and Entity Extraction
Document chunks are parsed into atomic academic concepts consisting of canonical identifiers, definitions, and source chunk provenance.

**Table 5 — Knowledge Concept Extraction**
```python
def extract_concepts(self, chunk_text: str) -> List[Dict[str, str]]:
    prompt = f"""Identify the most significant knowledge concepts in this text:
Text: {chunk_text[:3000]}
Return a JSON array of concepts with 'name' and 'definition'."""
    
    response = self.generate(prompt)
    return self._extract_json_fragment(response)
```

### A1.2.3. Directional Relationship Extraction
Concept pairs within local and section contexts are evaluated for directional semantic dependencies (`prerequisite_of`, `part_of`, `explains`, `leads_to`).

**Table 6 — Relationship Detection**
```python
def detect_all_relations(self, concepts: List[str], context: str) -> List[Dict[str, Any]]:
    concepts_str = ", ".join([f'"{c}"' for c in concepts])
    prompt = f"""Map out connections between these concepts: [{concepts_str}]
Context: {context[:2500]}
Return JSON array: [{{"from_concept": "X", "to_concept": "Y", "relation_type": "part-of"}}]"""
    
    response = self.generate(prompt)
    return self._extract_json_fragment(response)
```

### A1.2.4. Deterministic Heuristic Concept Extraction Fallback
When external LLM providers are unreachable, Clarion falls back to deterministic lexical extraction, computing phrase frequencies over normalized technical tokens.

**Table 7 — Heuristic Extraction Fallback**
```python
def _extract_concepts_heuristic(self, chunks: List[Chunk]) -> List[Concept]:
    """
    Offline fallback: count normalized n-gram phrase candidates,
    filtering out stopword-heavy and non-topical patterns.
    """
    phrase_counter = Counter()
    for chunk in chunks:
        clean_text = self._prepare_text_for_concepts(chunk.content)
        for phrase in self._extract_concept_candidates(clean_text):
            phrase_counter[phrase] += 1
    return [
        Concept(
            id=str(uuid.uuid4()),
            name=self._display_concept_name(phrase),
            chunk_ids=self._find_chunks_for_concept(phrase, chunks)
        )
        for phrase, _ in phrase_counter.most_common(12)
    ]
```

---

## A1.3. Vector Embedding and Dense Retrieval Service

### A1.3.1. Mathematical Formulation: Unit L2 Normalization and Distance Inversion
Dense retrieval employs the BAAI `bge-large-en-v1.5` transformer model (1024 embedding dimensions, 512 subword context window) \cite{xiao2023cpack} with FAISS `IndexFlatL2` nearest-neighbor indexing \cite{johnson2019billion}.

In the verified implementation, dense embeddings are unit L2 normalized upon generation (`normalize_embeddings=True` in live encoding and $\|v\|_2$ vector division in fallback hashing), guaranteeing that embeddings are unit L2 normalized (||q||_2 = ||v||_2 = 1.0):
$$\|q\|_2 = 1.0, \quad \|v\|_2 = 1.0$$

FAISS `IndexFlatL2` computes the squared Euclidean ($L_2$) distance $d^2$:
$$d^2(q, v) = \|q - v\|_2^2 = \sum_{j=1}^d (q_j - v_j)^2 = \|q\|_2^2 + \|v\|_2^2 - 2 \langle q, v \rangle = 2 - 2 \cos(q, v)$$

Because the cosine similarity $\cos(q, v) \in [-1, 1]$, the squared Euclidean distance is strictly bounded:
$$d^2(q, v) \in [0, 4]$$

The similarity score $s(q, v)$ is computed via monotonic inversion, bounding squared L2 distance in [0, 4] and calibrating similarity $s = 1/(1+d^2)$ in [0.20, 1.00]:
$$s(q, v) = \frac{1}{1 + d^2(q, v)}$$

Under unit L2 normalization, this formulation strictly calibrates similarity scores into the closed range:
$$s \in \left[\frac{1}{1 + 4}, \frac{1}{1 + 0}\right] = [0.20, 1.00]$$

- When vectors are collinear and identical ($\cos(q, v) = 1$): $d^2 = 0 \implies s = 1.00$.
- When vectors are orthogonal ($\cos(q, v) = 0$): $d^2 = 2 \implies s = \frac{1}{3} \approx 0.3333$.
- When vectors are diametrically opposite ($\cos(q, v) = -1$): $d^2 = 4 \implies s = 0.20$.

*Impact of Normalization*: If embeddings were unnormalized ($\|v\|_2 \approx 8.5\text{--}12.0$), $d^2$ routinely exceeds $50.0$, severely compressing similarity scores into the degenerate interval $[0.009, 0.06]$ and invalidating threshold filtering.

**Table 8 — FAISS Vector Retrieval and Bounded Distance-Derived Ranking Score**
```python
def retrieve(self, document_id: str, query: str, top_k: int = 5) -> List[RetrievalResult]:
    """
    Search indexed document chunks using unit L2-normalized embeddings and FAISS IndexFlatL2.
    Calibrates squared L2 distance into similarity s in [0.20, 1.00].
    """
    k = top_k or self.default_top_k
    vector_store = self.embedding_service.get_vector_store(document_id)
    if not vector_store:
        return []

    # Query embedding is unit L2-normalized: ||q||_2 = 1.0
    query_embedding = self.embedding_service.embed_query(query)

    # FAISS IndexFlatL2 search returns squared Euclidean distance: d^2 = ||q - v||_2^2 in [0, 4]
    search_results = vector_store.search(query_embedding, k * 2)

    results = []
    chunks = {c.chunk_id: c for c in self.chunking_service.get_chunks(document_id)}

    for chunk_id, distance in search_results:
        if chunk_id not in chunks:
            continue
        chunk = chunks[chunk_id]
        
        # Bounded monotonic distance-derived ranking score: s = 1 / (1 + d^2)
        score = 1.0 / (1.0 + float(distance))
        if score < self.similarity_threshold:
            continue

        results.append(RetrievalResult(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=score,
            section_title=chunk.section_title,
            position_index=chunk.position_index,
        ))
        if len(results) >= k:
            break

    return results
```

---

## A1.4. Knowledge Graph Construction and Graph Theoretical Analytics

### A1.4.1. Graph Theoretical Formulations
Knowledge graphs in Clarion are constructed as NetworkX directed graphs $G = (V, E)$ \cite{hagberg2008exploring}, where $V$ denotes the set of vertices (document root, sections, units, topics, and extracted concepts) and $E$ denotes the set of directed edges (structural hierarchy edges and semantic relationship edges).

1. **Directed Graph Density**:
   The directed graph density $D_{\text{dir}} = |E| / (|V|(|V|-1))$ measures the proportion of active directed edges relative to the maximal possible number of directed pairs:
   $$D_{\text{dir}}(G) = \begin{cases} \frac{|E|}{|V|(|V| - 1)} & \text{if } |V| > 1 \\ 0.0 & \text{if } |V| \le 1 \end{cases}$$
   where $D_{\text{dir}} \in [0, 1]$.

2. **Total Average Degree**:
   In NetworkX, the node degree method `graph.degree()` sums in-degree and out-degree ($d_{\text{total}}(v) = d_{\text{in}}(v) + d_{\text{out}}(v)$). The total average degree $\bar{d} = 2|E| / |V|$ across all vertices is:
   $$\bar{d} = \frac{1}{|V|} \sum_{v \in V} d_{\text{total}}(v) = \frac{1}{|V|} \sum_{v \in V} (d_{\text{in}}(v) + d_{\text{out}}(v)) = \frac{2|E|}{|V|}$$
   This total average degree $\bar{d}$ ($d_{\text{bar}} = 2|E|/|V|$) is exactly twice the directional average degree ($\bar{d}_{\text{dir}} = |E| / |V|$).

### A1.4.2. Synchronized Empirical Graph Parameters
Empirical audits across persisted knowledge graph artifacts in `Clarion-Backend/data/graphs/` and exported candidate relation datasets in `data/datasets/` reveal precise graph metrics, resolving previous reporting conflations:

1. **BSM 301 Mathematics-III (`65430b13-d3d6-49b8-8386-f34762f0ae9c.json`)**:
   - Canonical BSM 301 graph parameters: |V|=10, |E|=43 (9 structural, 34 semantic), distinguishing graph edges from the 34 exported relation candidate records.
   - Total Vertices: $|V| = 10$.
   - Total Graph Edges: $|E| = 43$ (9 structural hierarchy edges and 34 semantic edges).
   - Directed Graph Density:
     $$D_{\text{dir}} = \frac{43}{10 \times 9} = \frac{43}{90} \approx 0.4778 \quad (47.78\%)$$
   - Total Average Degree:
     $$\bar{d} = \frac{2 \times 43}{10} = 8.6$$
   - *Critical Disambiguation*: Candidate records and semantic graph edges happen to both equal 34 for this run, but they are distinct artifacts and must not be conflated. See `GRAPH_ARTIFACT_REGISTRY.md`.

2. **Thevenin's Theorem Evaluation Graphs**:
   - Synchronized Thevenin theorem graphs: Run 1 (|V|=16, |E|=16, D_dir=0.0667) and Run 2 (|V|=18, |E|=19, D_dir=0.0621).
   - **Run 1 (`827d8b9c-4444-4d35-ae6a-4a48dc81f0ec.json`)**:
     - Total Vertices: $|V| = 16$.
     - Total Graph Edges: $|E| = 16$ (15 structural hierarchy edges + 1 semantic relationship: `Impedance -> Complex Algebra`).
     - Directed Graph Density:
       $$D_{\text{dir}} = \frac{16}{16 \times 15} = \frac{16}{240} \approx 0.0667 \quad (6.67\%)$$
     - Total Average Degree: $\bar{d} = \frac{2 \times 16}{16} = 2.000$.
     - Exported Relation Candidates: 1 record.
   - **Run 2 (`fc5f2c9b-85ab-437b-8700-b23ae982ee08.json`)**:
     - Total Vertices: $|V| = 18$.
     - Total Graph Edges: $|E| = 19$ (17 structural hierarchy edges + 2 semantic relationships: `Impedance -> Complex Algebra`, `Impedance -> Phasors`).
     - Directed Graph Density:
       $$D_{\text{dir}} = \frac{19}{18 \times 17} = \frac{19}{306} \approx 0.0621 \quad (6.21\%)$$
     - Total Average Degree: $\bar{d} = \frac{2 \times 19}{18} \approx 2.1111$.
     - Exported Relation Candidates: 2 records.

**Table 9 — Graph Theoretical Analytics Implementation**
```python
class GraphAnalyzer:
    """
    Evaluates topological properties on NetworkX DiGraph instances.
    Implements directed graph density D_dir and total average degree d_bar.
    """
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def get_basic_metrics(self) -> Dict[str, Any]:
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()

        # Directed graph density: D_dir = |E| / (|V|(|V|-1)) for |V| > 1
        density = nx.density(self.graph) if n_nodes > 1 else 0.0

        metrics = {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "density": float(density),
        }

        if n_nodes > 0:
            # Degrees list contains total degree (in_degree + out_degree) per vertex
            degrees = [d for _, d in self.graph.degree()]
            # Total average degree: d_bar = 2|E| / |V|
            metrics["avg_degree"] = float(np.mean(degrees))
            metrics["max_degree"] = int(max(degrees))
            metrics["min_degree"] = int(min(degrees))
            metrics["median_degree"] = float(np.median(degrees))
        else:
            metrics.update({
                "avg_degree": 0.0,
                "max_degree": 0,
                "min_degree": 0,
                "median_degree": 0.0
            })

        return metrics
```

---

## A1.5. Context Assembly for Retrieval-Augmented Generation

Retrieved knowledge chunks and graph attributes are assembled into grounded context prompts for synthesis and question answering, maintaining strict source chunk provenance and section headers.

**Table 10 — Retrieval-Augmented Context Assembly**
```python
def retrieve_with_context(
    self,
    document_id: str,
    query: str,
    top_k: Optional[int] = None
) -> Tuple[List[RetrievalResult], str]:
    """
    Retrieve top-k chunks and format structured context for LLM generation.
    """
    results = self.retrieve(document_id, query, top_k)
    context_parts = []
    
    for i, result in enumerate(results, 1):
        header = f"[Chunk {i}]"
        if result.section_title:
            header += f" ({result.section_title})"
        context_parts.append(f"{header}\n{result.content}")

    return results, "\n\n".join(context_parts)
```

---

## A1.6. References and Institutional Acknowledgements

### Institutional Acknowledgements
The authors acknowledge the open-source scientific software community and foundational contributors behind the core dependencies enabling this research, including **NetworkX** (graph modeling and algorithms), **FAISS** (dense nearest-neighbor indexing), **Sentence-Transformers** (dense representation models), **FastAPI** (asynchronous backend architecture), and **Scikit-learn** (probabilistic calibration and evaluation harness).

### References
- **Cohen (1960)**: Cohen, J. "A coefficient of agreement for nominal scales." *Educational and Psychological Measurement*, 20(1), 37–46. \cite{cohen1960coefficient}
- **Fleiss (1971)**: Fleiss, J. L. "Measuring nominal scale agreement among many raters." *Psychological Bulletin*, 76(5), 378–382. \cite{fleiss1971measuring}
- **Guo et al. (2017)**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On calibration of modern neural networks." *ICML 2017*, PMLR 70:1321–1330. \cite{guo2017calibration}
- **Hagberg et al. (2008)**: Hagberg, A., Swart, P., & Schult, D. "Exploring network structure, dynamics, and function using NetworkX." *Proceedings of the 7th Python in Science Conference (SciPy 2008)*, pp. 11–15. \cite{hagberg2008exploring}
- **Johnson et al. (2019)**: Johnson, J., Douze, M., & Jégou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 7(3), 535–547. \cite{johnson2019billion}
- **Landis & Koch (1977)**: Landis, J. R., & Koch, G. G. "The measurement of observer agreement for categorical data." *Biometrics*, 33(1), 159–174. \cite{landis1977measurement}
- **Platt (1999)**: Platt, J. "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." *Advances in Large Margin Classifiers*, 10(3), 61–74. \cite{platt1999probabilistic}
- **Xiao et al. (2023)**: Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. "C-Pack: Packaged Resources to Advance General Chinese Embedding." *arXiv preprint arXiv:2309.07597*. \cite{xiao2023cpack}
