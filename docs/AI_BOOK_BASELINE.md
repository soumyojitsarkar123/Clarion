# AI_Book.pdf Empirical Baseline Telemetry Report

**Document ID**: `53faa553-512e-42cf-9a5d-c3559c8fa170`  
**Measurement Type**: Development Baseline (Pre-Optimization Snapshot)  
**Status**: Authoritative Reference Baseline for Milestone M0–M4  
**Date Captured**: 2026-10-09T05:55:00Z  
**Classification Scope**: Development Measurement Only — Canonical Benchmark Registries Untouched

---

## 1. Executive Summary & Core Baseline Metrics

This document records the empirical baseline telemetry captured from the unoptimized run of Russell & Norvig's *Artificial Intelligence: A Modern Approach* (`AI_Book.pdf`).

The initial pipeline execution completed with extensive runtime degradation, micro-chunk fragmentation, heuristic fallback domination, front-matter contamination, and heuristic misclassification.

### Core Metrics Summary Table

| Metric / Dimension | Baseline Value | Units / Description | Evidence Source |
| :--- | :--- | :--- | :--- |
| **Document Filename** | `AI_Book.pdf` | Plain PDF Document | SQLite `documents` |
| **File Size** | 21,133,797 | Bytes (~20.15 MB) | SQLite `documents` |
| **Page Count** | 1,151 | Pages | Extracted via PyPDF2 |
| **Total Word Count** | 552,827 | Words | SQLite `documents` |
| **Total Chunks** | 3,131 | Chunks | SQLite `chunks` |
| **Min Chunk Size** | **1** | Words (Micro-chunk failure) | SQLite `chunks` |
| **Mean Chunk Size** | 178.72 | Words | SQLite `chunks` |
| **Max Chunk Size** | 611 | Words | SQLite `chunks` |
| **Micro-chunks (<100 words)** | **1,444 (46.12%)** | Under target threshold | SQLite `chunks` |
| **Micro-chunks (<20 words)** | **628 (20.06%)** | Severe micro-fragments | SQLite `chunks` |
| **Embedding Model** | `BAAI/bge-large-en-v1.5` | 335M parameters (1.34 GB) | `config.py` / `clarion.log:5774` |
| **Embedding Device** | **`cpu`** | CPU execution (`torch+cpu`) | `clarion.log:6651` |
| **Embedding Duration** | **3,642.15s (~60.70 min)** | 99.3% of total pipeline runtime | `clarion.log:6549` |
| **Embedding Throughput** | **0.8597 chunks/sec** | ~51.6 chunks / minute | 3,131 chunks / 3,642.15s |
| **LLM Model Configured** | `qwen3.5:4b` | Configured in `.env` | `.env:48`, `clarion.log:6587` |
| **LLM Availability** | **`UNAVAILABLE`** | HTTP 500 error on Ollama | `clarion.log:6590, 6603` |
| **Fallback Usage** | **`100% FALLBACK USED`** | Extractive unigram heuristics | `clarion.log:6552, 6591` |
| **Concepts Extracted** | 12 | 12 concepts (front-matter noisy) | SQLite `knowledge_maps` |
| **Candidate Relations** | 40 | Co-occurrence candidates | SQLite `knowledge_maps` |
| **Graph Nodes / Edges** | 20 nodes / 59 edges | Density: 0.1553 | `data/graphs/` / Cytoscape |
| **Document Classification** | **`"assessment"` (INCORRECT)** | Misclassified as exam/test | SQLite `summaries` |
| **Summary Sections** | 2,684 | Headings/sections captured | SQLite `summaries` |
| **Total Pipeline Duration** | **3,668.08s (~61.13 min)** | Full run wall time | `clarion.log:5765-6571` |

---

## 2. Chunking Quality & Micro-Chunk Proliferation

The chunking engine generated **3,131 chunks**, characterized by an extreme distribution skew:
- **46.12%** (1,444 chunks) have fewer than 100 words.
- **20.06%** (628 chunks) have fewer than 20 words, with the minimum chunk size reaching **1 word**.

### Chunk Size Distribution

| Word Count Interval | Chunk Count | Percentage |
| :--- | :--- | :--- |
| 1-19 words (micro-fragments) | 628 | 20.06% |
| 20-49 words (short fragments) | 378 | 12.07% |
| 50-99 words (sub-target) | 438 | 13.99% |
| 100-249 words (moderate) | 770 | 24.59% |
| 250-499 words (full chunks) | 560 | 17.89% |
| 500+ words (overflow) | 357 | 11.4% |

### Root Causes Diagnosed
1. **False Section Detection from Running Headers**: Repeating headers (`"Artificial Intelligence: A Modern Approach"`, chapter titles, page numbers) triggered `_is_heading_line`, creating artificial 1–5 word sections.
2. **Broken Merge Logic**: The merge predicate in `chunking_service.py` failed when `prev_chunk` was near maximum size, leaving trailing section remnants stranded as micro-chunks.

---

## 3. Embedding Performance & Hardware Telemetry

- **Configured Model**: `BAAI/bge-large-en-v1.5` (Canonical, research-grade model, 1024-dimensional embeddings).
- **Compute Device**: `cpu`.
  - An NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) is present in the host environment.
  - However, the virtual environment `.venv_linux` contains `torch==2.14.1+cpu`. As a result, `torch.cuda.is_available()` resolved to `False`.
- **Runtime**: **3,642.15 seconds (60.70 minutes)**.
- **Throughput**: **0.8597 chunks/second**.

### Pipeline Stage Timing Breakdown

| Stage | Duration (s) | Share of Total Runtime |
| :--- | :--- | :--- |
| Embedding | 3642.15s | 99.29% |
| Ingestion | 2.45s | 0.07% |
| Chunking | 2.14s | 0.06% |
| Evaluation | 1.35s | 0.04% |
| Mapping | 0.54s | 0.01% |
| Graph_building | 0.04s | 0.0% |
| Hierarchy | 0.0s | 0.0% |

---

## 4. LLM Service, Provider Diagnostics & Fallback Execution

- **Configured Model**: `qwen3.5:4b`
- **Provider Status**: `UNAVAILABLE`
- **Error Description**: `Ollama HTTP 500 error: 'CreateFile E:\model\manifests-v2\ollama.com\library\qwen3.5\4b: The path cannot be traversed because it contains an untrusted mount point.'`
- **Fallback Execution**: `100% FALLBACK USED`
- **Downstream Consequences**:
  - The pipeline fell back to heuristic concept extraction (`_extract_concepts_heuristic`).
  - Sampling was restricted to `chunks[:16]`, which concentrated extraction entirely within the book's front matter (title, copyright, dedication, preface, and contributor list).
  - Top-level application status deceptive: reported `COMPLETED` despite 100% LLM failure.

---

## 5. Information Extraction & Knowledge Graph Baseline

### Extracted Concepts (12)
```python
['Book', 'David', 'These', 'Editor', 'John', 'Peter', 'Learning', 'Peter Norvig', 'Some', 'Ernest Davis', 'Sebastian Thrun', 'Planning']
```
**Quality Assessment**:
- **Front-Matter / Contributor Noise**: `"Book"`, `"David"`, `"These"`, `"Editor"`, `"John"`, `"Peter"`, `"Some"`.
- **Domain Concepts**: `"Learning"`, `"Planning"`, `"Peter Norvig"`, `"Ernest Davis"`, `"Sebastian Thrun"`.
- Notice that 7 of 12 concepts (58.3%) are editorial noise or pronouns due to unigram extraction without adequate stopwords and front-matter sampling bias.

### Candidate Relations (40 Candidate Relations)
All 40 relations were extracted via heuristic co-occurrence in the first 20 chunks. All were assigned relation type `explanation`.

**Sample Candidate Relations**:
- **Book** -> `explanation` -> **David** (confidence: 0.61, evidence: *"Don Barker, Tony Barrett, James Newton Bass, Don Beal,Howard Beck, Wolfgang Bibel, John Binder, Larr..."*)
- **Book** -> `explanation` -> **These** (confidence: 0.69, evidence: *"The concepts ofbelief state (a set of possible worlds) and state estimation (maintaining the belief ..."*)
- **Book** -> `explanation` -> **Editor** (confidence: 0.53, evidence: *"Upper Saddle River Boston Columbus San Francisco New York Indianapolis London Toronto Sydney Singapo..."*)
- **Book** -> `explanation` -> **John** (confidence: 0.61, evidence: *"Don Barker, Tony Barrett, James Newton Bass, Don Beal,Howard Beck, Wolfgang Bibel, John Binder, Larr..."*)
- **Book** -> `explanation` -> **Peter** (confidence: 0.78, evidence: *"Credits: Stan Honda/Getty (Kasparaov), Library of Congress (Bayes), NASA (Mars rover), National Muse..."*)

### Knowledge Graph Metrics
- **Nodes**: 20 (1 document node, 7 topic nodes, 12 concept nodes)
- **Edges**: 59
- **Density**: 0.1553
- **Status**: Dominated by co-occurrence edges between editorial names and generic terms.

---

## 6. Document Classification & Summary Quality

### Document Classification
- **Result**: **`"assessment"`**
- **Ground Truth Expected**: `textbook` / `reference`
- **Root Cause**: Overly sensitive question-mark heuristic (_looks_like_assessment: '?' >= 3). Textbook exercises and question marks triggered false assessment categorization.

### Overall Summary Output (Verbatim)
> *"This document is an assessment-style document focused on state, one, agent, and but. It presents question-driven material that asks the reader to interpret, compare, and apply state, one, and agent. Boxall, Ronen Braf-man, John Bresina, Gerhard Brewka, Selmer Bringsjord, Carla Brodley, Chris Brown, EmmaBrunskill, Wilhelm Burger, Lauren Burka, Carlos Bustamante, Joao Cachopo, Murray Camp-bell, Norman Carver,."*

**Quality Assessment**:
1. Categorizes an 1,151-page world-standard AI textbook as an *"assessment-style document"* due to question marks in practice exercises.
2. The summary concludes with an unedited contributor list from the acknowledgments section (*"Boxall, Ronen Braf-man, John Bresina, Gerhard Brewka, Selmer Bringsjord, Carla Brodley..."*), because the sentence informativeness heuristic prioritized high unique-word counts found in rosters.

---

## 7. Research Data Quarantine & Scientific Integrity Attestation

- **Quarantined Directories**:
  - `data/benchmarks/CANONICAL_BENCHMARK_REGISTRY.json` — **UNTOUCHED (Hash Verified)**
  - `data/benchmarks/bench_*.json` — **UNTOUCHED (3 Canonical/Historical Files Preserved)**
  - `Clarion-Backend/data/ground_truth/` — **UNTOUCHED (Annotation Protocol & Queues Preserved)**
- **Scope Confirmation**: This document serves strictly as a development measurement artifact for comparison against optimized milestones (M1–M4). No research gold standards, annotation queues, or canonical evaluation baselines were modified.
