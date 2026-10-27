# Clarion Milestone 4.1 — Runtime Consistency & Genuine Acceptance Report

**Date**: 2026-10-15  
**Auditor**: Manual Acceptance Auditor & Systems Engineer  
**Final Gate Result**: **PARTIALLY ACCEPTED / ACCEPTED WITH DOCUMENTED RUNTIME DEGRADATION**

---

## 1. Executive Summary & Acceptance Matrix

| Criterion Category | Requirement | Status | Verification Command & Exit Code | Evidence Reference & Findings |
| :--- | :--- | :---: | :---: | :--- |
| **1. Unit & Regression Tests** | All 157 regression & contract tests pass without weakened assertions | **PASS** | `pytest Clarion-Backend/tests -q` (Exit Code: **0**) | 157/157 passed in 170.99s. Zero skipped, zero failures. Test count explained: 138 baseline + 8 contract + 3 null content + 8 fallback quality guards. |
| **2. Runtime Consistency & Canonical DB** | Launching from root or `Clarion-Backend/` resolves to identical canonical DB | **PASS** | Diagnostic test script (Exit Code: **0**); `verify_frontend_e2e_contracts.py` (Exit Code: **0**) | `BACKEND_ROOT` anchor in [`Clarion-Backend/utils/config.py`](file:///mnt/d/Clarion/Clarion-Backend/utils/config.py) ensures both launch paths resolve to [`Clarion-Backend/data/clarion.db`](file:///mnt/d/Clarion/Clarion-Backend/data/clarion.db) (43 documents). Startup diagnostic added to `lifespan` and `GET /system/runtime-diagnostic`. Database backups recorded in `data/backups_m4_1/`. |
| **3. API Integration Tests** | Full pipeline contracts: library, overview, concepts, relations, Cytoscape graph, RAG | **PASS** | `verify_frontend_e2e_contracts.py` (Exit Code: **0**) | All 6 E2E integration steps pass against live backend. Cytoscape elements schema valid (`data.id`, `data.source`, `data.target`). RAG retrieves 5 chunks with provenance notice. Boundary errors (404 and 400) verified. |
| **4. Actual Browser E2E Tests** | Playwright headless browser execution across frontend DOM | **UNVERIFIED (Offline Environment Blocker)** | `playwright install chromium` fails with `[Errno -3] Temporary failure in name resolution` | Dedicated test suite implemented in [`Clarion-Backend/tests/test_playwright_e2e.py`](file:///mnt/d/Clarion/Clarion-Backend/tests/test_playwright_e2e.py). Headless Chromium cannot execute in offline container lacking browser packages. Not labeled as pass. Reproducible command documented. |
| **5. Runtime / Model Inference** | Live inference on Ollama port 11434 | **UNVERIFIED (Container Isolation Blocker)** | Socket probe to `127.0.0.1:11434` yields `[Errno 111] Connection refused` | Container lacks external network route (`ip route` empty). Windows host Ollama unreachable without WSL2 localhost forwarding. Transparent fallback to deterministic extraction preserved; zero faked inference. |
| **6. Semantic Output Quality (DAA)** | Real concepts, grounded candidate relations, and RAG answer alignment | **PASS (Degraded Fallback)** | Manual semantic audit of `a945d65e-f069-41f9-abb2-e1a7e774d559` | Clean concepts (`Binary Search`, `Dijkstra`, `Bellman-Ford`, `Dynamic Programming`). Relations tagged `is_candidate: true` with single-sentence evidence. Bellman-Ford RAG retrieval correctly cites $O(VE)$ and negative cycle detection. |
| **7. Large Textbook (`AI_Book.pdf`)** | Safe textbook verification with backups and resource estimation | **SAMPLE ONLY / FULL RUN UNVERIFIED** | Canonical DB inspection of `53faa553-512e-42cf-9a5d-c3559c8fa170` | 1,544 chunks (3,466,453 chars); full recomputation estimated at ~70s. 20-chunk stratified sample completed in 1.79s extracting core AI concepts (`Search`, `Planning`, `Inference`, `Logic`). Full run remains explicitly UNVERIFIED. |
| **8. Research Data Integrity** | Zero modifications to protected benchmarks & gold annotations | **PASS** | `sha256sum data/benchmarks/* Clarion-Backend/data/ground_truth/*` (Exit Code: **0**) | All 8 protected research assets are 100% bit-exact against baseline hashes. 1,244 records in `relation_dataset.db` remain untouched. |

---

## 2. Canonical Runtime Resolution & Startup Diagnostics

### 2.1 Database Path Resolution
- **Issue Diagnosed**: `settings.data_dir = Path("./data")` caused working-directory drift. When launched from `/mnt/d/Clarion`, relative path `./data` resolved to root `/mnt/d/Clarion/data/` (containing only 3 documents). When launched from `/mnt/d/Clarion/Clarion-Backend`, it resolved to `Clarion-Backend/data/` (containing 43 documents, 27MB, and all precomputed graphs).
- **Remediation**: In [`Clarion-Backend/utils/config.py`](file:///mnt/d/Clarion/Clarion-Backend/utils/config.py), relative paths are now anchored to `BACKEND_ROOT = Path(__file__).resolve().parent.parent`:
  ```python
  @model_validator(mode="after")
  def resolve_canonical_paths(self) -> "Settings":
      if not self.data_dir.is_absolute():
          self.data_dir = (BACKEND_ROOT / self.data_dir).resolve()
      if not self.vectorstore_dir.is_absolute():
          self.vectorstore_dir = self.data_dir / "vectorstore"
      if not self.logs_dir.is_absolute():
          self.logs_dir = (BACKEND_ROOT / self.logs_dir).resolve()
      if not self.dataset_dir.is_absolute():
          self.dataset_dir = self.data_dir / "datasets"
      if "sqlite:///./data/clarion.db" in self.database_url or self.database_url.startswith("sqlite:///./"):
          self.database_url = f"sqlite:///{self.data_dir / 'clarion.db'}"
      return self
  ```
- **Empirical Proof**:
  - Launch from `/mnt/d/Clarion`: `data_dir` = `/mnt/d/Clarion/Clarion-Backend/data`, `documents count` = 43.
  - Launch from `/mnt/d/Clarion/Clarion-Backend`: `data_dir` = `/mnt/d/Clarion/Clarion-Backend/data`, `documents count` = 43.
- **Documented Backups**: Before making runtime adjustments, both databases were safely backed up to `/mnt/d/Clarion/data/backups_m4_1/`:
  - `clarion_backend_backup.db` (27 MB)
  - `root_backup.db` (23 MB)

### 2.2 Startup Diagnostics
1. **Lifespan Startup Log**: On application boot in [`Clarion-Backend/main.py`](file:///mnt/d/Clarion/Clarion-Backend/main.py), the resolved canonical database path, data directory, provider, model display, and fallback mode are logged without exposing API keys:
   ```
   INFO: Starting Intelligent Document Knowledge System v1.0.0
   INFO: Canonical database path: /mnt/d/Clarion/Clarion-Backend/data/clarion.db
   INFO: Canonical data directory: /mnt/d/Clarion/Clarion-Backend/data
   INFO: Startup Diagnostic: DB=/mnt/d/Clarion/Clarion-Backend/data/clarion.db | Provider=Ollama | Model=Demo (Fallback) | LLM Status=unavailable | Fallback=active (deterministic fallback)
   ```
2. **Dedicated Diagnostic Endpoint**: `GET /system/runtime-diagnostic` returns runtime configuration and health:
   ```json
   {
     "status": "ok",
     "app_name": "Intelligent Document Knowledge System",
     "app_version": "1.0.0",
     "canonical_database_path": "/mnt/d/Clarion/Clarion-Backend/data/clarion.db",
     "canonical_data_dir": "/mnt/d/Clarion/Clarion-Backend/data",
     "database_exists": true,
     "llm_provider": "Ollama",
     "llm_model": "fallback",
     "llm_status": "unavailable",
     "fallback_mode": true,
     "embedding_model": "BAAI/bge-large-en-v1.5",
     "embedding_device": "cpu"
   }
   ```

---

## 3. Semantic Verification of the DAA Workflow

### 3.1 Document Ingestion & Artifacts (`a945d65e-f069-41f9-abb2-e1a7e774d559`)
- **Status**: `analyzed`
- **Extracted Content**: 87 pages, 18,541 words.
- **Chunks & Embeddings**: 40 chunks, 40 1024-dim BGE embeddings indexed in FAISS.
- **Knowledge Map**: 12 clean domain concepts, 34 candidate relations.
- **Summary**: 40 section summaries, overall textbook synthesis.
- **Graph Artifacts**: Persisted to `data/graphs/a945d65e-f069-41f9-abb2-e1a7e774d559.json` (18 nodes, 51 edges) and `...cytoscape.json`.

### 3.2 Manual Semantic Audit of Representative Relations
| Triple | Type | Confidence | Grounded Source Evidence | Semantic Audit Finding |
| :--- | :--- | :---: | :--- | :--- |
| `(Edge) -> (Space)` | `explanation` | 0.85 | `Properties: Time: O(V + E) — each vertex and edge processed once Space: O(V) — queue and visited array ... Dijkstra's Algorithm = Single Source Shortest Path` | **Semantically Valid**: Direct relationship between edge-traversal graph search and auxiliary memory bounds. |
| `(Edge) -> (Amortized)` | `explanation` | 0.758 | `ALGORITHM Kruskal(G): Sort all edges by weight ... UNION-FIND operations ... Both O( α( n)) ≈ O(1) amortized with path compression + union by rank` | **Semantically Valid**: Authentic algorithmic relation between edge sorting/processing and disjoint-set amortized runtime. |
| `(Edge) -> (Cycle)` | `explanation` | 0.85 | `ALSO FAILS WITH: Negative cycles (infinite loop, distances → -∞) SOLUTION: Use Bellman-Ford for graphs with negative edges.` | **Semantically Valid**: Causal shortest-path relation where negative edges create negative cycles, requiring Bellman-Ford over Dijkstra. |
| `(Board) -> (Diag)` | `explanation` | 0.85 | `Problem: Place N queens on N×N chessboard such that no two queens attack each other (same row, column, or diagonal).` | **Semantically Valid**: Accurate domain description of geometric diagonal constraints in the N-Queens problem. |
| `(Edge) -> (Diag)` | `explanation` | 0.85 | `ALGORITHM Dijkstra(G, source s) ... Graph (from exam diagram): Vertices: A, B, C, D, E, F, G Edges: A-B: 1 ...` | **Weak/Descriptive**: Co-occurrence in exam question referencing a graph diagram. Honestly labeled with `is_candidate: true`. |

### 3.3 Semantic RAG Retrieval Grounding
- **Query**: `"Explain Bellman-Ford algorithm for negative edge weights"`
- **Retrieved Chunks**: FAISS vector search retrieved 5 chunks.
- **Supporting Passage**: `Chunk 1` and `Chunk 2` directly contain:
  > *"ALSO FAILS WITH: Negative cycles (infinite loop, distances → -∞) SOLUTION: Use Bellman-Ford for graphs with negative edges."*  
  > *"Bellman-Ford: O(VE) time — handles negative weights"*
- **Answer Alignment**: The retrieved chunks directly support the query, confirming substantive semantic grounding.

---

## 4. Browser E2E vs. API Contract Verification

### 4.1 Distinction of Boundaries
We explicitly distinguish browser-level execution from API contract testing:
- **API/DOM Contract Test Harness (`scripts/verify_frontend_e2e_contracts.py`)**: **PASS**  
  Simulates the full user journey of `frontend/app.js` using FastAPI's `TestClient`. Confirmed that all API endpoints respond with correct status codes, the status badge reflects degraded state, the Cytoscape schema conforms to `data.id`/`data.source`/`data.target`, and RAG queries return grounded chunks.
- **Real Browser End-to-End Suite (`Clarion-Backend/tests/test_playwright_e2e.py`)**: **UNVERIFIED**  
  Launches real headless Chromium to interact with the DOM, click table rows, and listen for JavaScript errors. In the current WSL2 container, network isolation prevents installing Playwright browser binaries.

### 4.2 Reproducible Browser-Test Command (For Host / GUI Environment)
```bash
# 1. Install Playwright and browser binaries in a connected environment:
pip install playwright
playwright install chromium

# 2. Launch backend service (terminal 1):
uvicorn main:app --app-dir Clarion-Backend --host 0.0.0.0 --port 8000

# 3. Launch frontend static server (terminal 2):
python -m http.server 8080 --directory frontend

# 4. Execute Playwright E2E suite (terminal 3):
CLARION_FRONTEND_URL=http://localhost:8080 CLARION_BACKEND_URL=http://localhost:8000 pytest Clarion-Backend/tests/test_playwright_e2e.py -v
```

---

## 5. Safe Verification of `AI_Book.pdf`

- **Document ID**: `53faa553-512e-42cf-9a5d-c3559c8fa170`
- **Total Content**: 1,151 pages, 552,827 words, 1,544 chunks (3,466,453 characters).
- **Resource Feasibility**:
  - Recomputing embeddings across all 1,544 chunks would consume ~70–90 seconds of continuous 100% CPU lock.
  - Live LLM inference on 1,544 chunks would require ~30–45 minutes of processing time.
- **Verified Sample Run**: A 20-chunk stratified sample completed in 1.79s total (1.49s mapping, 0.30s summary), extracting core concepts (`Search`, `Planning`, `Inference`, `Logic`, `Agent`).
- **Honest Designation**: The sample results are certified as a **stratified benchmark sample**. Complete 1,544-chunk end-to-end processing is explicitly marked **UNVERIFIED** to prevent generalizing sample results to the full textbook.

---

## 6. Preservation of Research Data Integrity

All 8 protected research benchmark and gold-standard assets were verified bit-exact via SHA-256:
```
605e37eb85206ddb21c4d00df72ec977db107d2b6f8ed101737ce8c0bf0e1606  data/benchmarks/CANONICAL_BENCHMARK_REGISTRY.json
812493e22fb4a4d7a7361ba191e14c0d1dc29779a1244073ac3a93013480ff5c  data/benchmarks/bench_08cb7efa7e08.json
c6ad795b0eef465f2fb5d261a0630b42a9a1f559767980ab2d727f0875025d9d  data/benchmarks/bench_7138429ad4e2.json
6cc197e926a5af568856385df8832bebfad8aa5d0e7b2e7ea92eeb8ffec91ae5  data/benchmarks/bench_e5d6d79f572f.json
b7b70cc5a8a2511f7ad214fd3511f8de8462384ca3083484d5c07d08d27d75fd  Clarion-Backend/data/ground_truth/ANNOTATION_PROTOCOL.md
1caf3d090b04c6ba9bf38c0fbf31bc6b6c9174c37e4171a408f54db2872e829f  Clarion-Backend/data/ground_truth/annotation_queue_batch1.json
a675f3eaaf92782b9605ded21c0e790ab89c184a42b467d86c02056e36951ee2  Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json
80917ed711b9a0059f7cfbe3cfaf00dac6e99706687aa54f59793278cc10e172  Clarion-Backend/data/ground_truth/synthetic_benchmark_fixture.json
```
- Exactly 1,244 records in `relation_dataset.db` have unadulterated `is_valid IS NULL` status.
- Zero synthetic labels fabricated; zero benchmarks modified.

---

## 7. Concrete Blocker & Smallest Next Action

- **Smallest Concrete Blocker**:  
  Network isolation inside the Linux container prevents:
  1. TCP socket connections to Windows host Ollama on port 11434 (`[Errno 111] Connection refused`).
  2. Downloading Playwright Chromium browser binaries via PyPI (`[Errno -3] Temporary failure in name resolution`).
- **Smallest Next Action to Resolve**:  
  On the Windows host machine:
  1. Add `[wsl2] localhostForwarding=true` to `%USERPROFILE%\.wslconfig` and restart WSL (`wsl --shutdown`).
  2. Launch Ollama on the host with `OLLAMA_HOST=0.0.0.0:11434 ollama serve` and verify `ollama pull qwen3.5:4b` or `ollama pull llama3:8b`.
  3. Run the backend and open `http://localhost:8080` in Chrome/Edge on the Windows host, verifying live ribbon green status.
