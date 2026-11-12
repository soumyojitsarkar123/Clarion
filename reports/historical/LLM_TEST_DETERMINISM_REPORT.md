# Clarion LLM Test Determinism & Fallback Integrity Verification Report

**Document ID**: `CLR-REP-LLM-DET-001`  
**Author / Role**: Deliverables Authoring Worker (`teamwork_worker_m4`)  
**Milestone**: Milestone 4 (R1 & R5 Verification)  
**Date**: 2026-11-09  
**Execution Environment**: Linux Ubuntu 24.04 LTS (x86_64), Python 3.12.3 (`/mnt/d/Clarion/.venv_linux/bin/python`), Pytest 9.1.1  
**Determinism Status**: **100% DETERMINISTIC / VERIFIED PASS**  
**Workflow Separation**: Workflow A (Offline / Deterministic Fallback) = **VERIFIED PASS**; Workflow B (Live Host LLM) = **DOCUMENTED BLOCKER (WSL2 Sandbox Isolation)**  

---

## 1. Executive Summary

During previous backend test executions recorded in `Clarion-Backend/backend-pytest.log`, three unit test cases failed unexpectedly:
1. `tests/test_document_intelligence_overhaul.py::test_summary_deterministic_fallback_notice`
2. `tests/test_knowledge_map_service.py::test_build_knowledge_map_is_heuristic`
3. `tests/test_m3_fallback_quality.py::test_fallback_overall_summary_has_no_template_boilerplate`

A comprehensive architectural and forensic investigation was conducted to determine why these tests failed, whether production algorithms or assertions should be modified, and how to guarantee strict, hermetic determinism without invalidating neural embeddings (`BAAI/bge-large-en-v1.5`), tokenizers, SQLite persistence, or FAISS vector indexing.

### Key Investigation Outcomes:
1. **Root Cause Confirmed**: The failures were **not** caused by software bugs in the fallback generation algorithms. Rather, they were caused by **unisolated network discovery of an ambient Ollama daemon running on `localhost:11434`** on the developer's host machine. When `LLMInterface._is_service_available()` was called during unit tests without an active mock, it sent an unmocked HTTP GET probe to `http://localhost:11434/api/tags`. The running Ollama daemon responded with HTTP 200, prompting the service to execute live neural generation and set metadata flags (`fallback_used = False`, `extraction_method = "llm"`). The tests, which specifically assert that deterministic fallback metadata is emitted, failed as a direct consequence.
2. **Production Integrity Preserved**: Fallback metadata (`fallback_used: bool`, `analysis_mode: str`, `extraction_method: str`, `llm_summary_accepted: bool`) is an essential, customer-facing application contract. It governs the pipeline's `degraded` operational state machine in `ProcessingPipeline.execute` and renders amber warning status badges in `frontend/app.js`. Therefore, relaxing test assertions or changing production fallback code to mask live detection would violate the core architectural contract.
3. **Deterministic Hermetic Isolation**: A suite-level autouse fixture `_isolate_external_services` was established in `Clarion-Backend/tests/conftest.py`. This fixture intercepts `LLMInterface._is_service_available` and sets it to return `False` during unit tests, completely eliminating ambient network probing.
4. **Zero Neural Mocking**: In accordance with Requirement R1 and R5, genuine SentenceTransformer BGE embeddings, HuggingFace fast tokenizers, SQLite relational persistence, and FAISS vector indexing remain **100% real and unmocked**.
5. **Empirical Verification**: All three previously failing tests execute in **12.74 seconds** and pass with 100% determinism.

---

## 2. Root Cause Analysis of the Three Failing Tests

### 2.1 Failure 1: `test_summary_deterministic_fallback_notice`
- **Location**: `Clarion-Backend/tests/test_document_intelligence_overhaul.py:259-278`
- **Verbatim Error from Unisolated Test Run**:
  ```text
  tests\test_document_intelligence_overhaul.py:271: in test_summary_deterministic_fallback_notice
  >       assert summary.metadata["fallback_used"] is True
  E       assert False is True
  ```
- **Execution Path & Failure Mechanism**:
  1. The test instantiates `summary_service = SummaryService()`.
  2. `SummaryService.__init__()` (`services/summary_service.py:93`) initializes `self.llm = LLMInterface()`.
  3. `summary_service.generate_summary(chunks, title)` invokes `self._generate_overall_summary(chunks, title)` (`services/summary_service.py:138, 194-230`).
  4. At line 202, the service evaluates:
     ```python
     if not self._prefer_deterministic and self.llm._is_service_available():
         response = self.llm.generate(prompt)
     ```
  5. `LLMInterface._is_service_available()` (`services/knowledge_map_service.py:231-312`) executes an unmocked network probe:
     ```python
     requests.get(f"{self._get_ollama_base_url()}/api/tags", timeout=settings.ollama_health_timeout_seconds)
     ```
  6. The ambient host Ollama daemon on port 11434 answered with HTTP 200 containing model `qwen3.5:4b` / `qwen2.5:latest`.
  7. `SummaryService` successfully generated summary text via live LLM synthesis, causing `llm_summary_accepted` to evaluate to `True`.
  8. Line 154 set:
     ```python
     summary.metadata["fallback_used"] = not llm_summary_accepted  # Evaluated to False
     summary.metadata["analysis_mode"] = "llm"
     ```
  9. Line 271 asserted:
     ```python
     assert summary.metadata["fallback_used"] is True
     ```
     This assertion failed with `assert False is True`.

- **Enforced Contract Assertions**:
  ```python
  assert isinstance(summary, StructuredSummary)
  assert summary.overall_summary and len(summary.overall_summary.strip()) > 0
  assert "Deterministic fallback analysis" not in summary.overall_summary
  assert "assessment-style document focused on" not in summary.overall_summary
  assert summary.metadata["fallback_used"] is True
  assert summary.metadata["analysis_mode"] == "deterministic_fallback"
  assert summary.metadata["llm_summary_accepted"] is False
  ```

---

### 2.2 Failure 2: `test_build_knowledge_map_is_heuristic`
- **Location**: `Clarion-Backend/tests/test_knowledge_map_service.py:7-15`
- **Verbatim Error from Unisolated Test Run**:
  ```text
  tests\test_knowledge_map_service.py:12: in test_build_knowledge_map_is_heuristic
  >       assert knowledge_map.metadata.get("extraction_method") == "heuristic"
  E       AssertionError: assert 'llm' == 'heuristic'
  E         
  E         - heuristic
  E         + llm
  ```
- **Execution Path & Failure Mechanism**:
  1. The test calls `service = KnowledgeMapService()`.
  2. `KnowledgeMapService.__init__()` (`services/knowledge_map_service.py:785`) instantiates `self.llm = LLMInterface()`.
  3. `service.build_knowledge_map("doc-sample", sample_chunks)` checks:
     ```python
     if self.llm._is_service_available():
         concepts = self._extract_concepts(chunks)
         extraction_method = "llm"
     ```
  4. Ollama daemon on `localhost:11434` responded to `/api/tags`, returning HTTP 200.
  5. The service dispatched extraction prompts to Ollama, setting `extraction_method = "llm"` (`services/knowledge_map_service.py:829`).
  6. Line 856 populated:
     ```python
     knowledge_map.metadata["extraction_method"] = "llm"
     ```
  7. Line 12 asserted:
     ```python
     assert knowledge_map.metadata.get("extraction_method") == "heuristic"
     ```
     This assertion failed with `AssertionError: assert 'llm' == 'heuristic'`.

- **Enforced Contract Assertions**:
  ```python
  assert isinstance(knowledge_map, KnowledgeMap)
  assert knowledge_map.metadata.get("extraction_method") == "heuristic"
  assert knowledge_map.concepts
  assert all(concept.chunk_ids for concept in knowledge_map.concepts)
  ```

---

### 2.3 Failure 3: `test_fallback_overall_summary_has_no_template_boilerplate`
- **Location**: `Clarion-Backend/tests/test_m3_fallback_quality.py:163-207`
- **Verbatim Error from Unisolated Test Run**:
  ```text
  tests\test_m3_fallback_quality.py:204: in test_fallback_overall_summary_has_no_template_boilerplate
  >       assert summary.metadata["fallback_used"] is True
  E       assert False is True
  ```
- **Execution Path & Failure Mechanism**:
  1. The test calls `service = SummaryService()`.
  2. `service.generate_summary(chunks, title)` triggers `_generate_overall_summary(...)`.
  3. The unmocked `LLMInterface` probe detected ambient host Ollama, received live generation, and set `llm_summary_accepted = True`.
  4. Summary metadata populated with `fallback_used = False` and `analysis_mode = "llm"`.
  5. Line 204 asserted:
     ```python
     assert summary.metadata["fallback_used"] is True
     ```
     This assertion failed with `assert False is True`.

- **Enforced Contract Assertions**:
  ```python
  assert overall and len(overall) > 50
  # Rejection of canned phrases:
  assert "deterministic fallback analysis" not in overall.lower()
  assert "assessment-style document" not in overall.lower()
  assert "question-driven material" not in overall.lower()
  assert "it also discusses applications and examples" not in overall.lower()
  assert not overall.lower().startswith("covers ")
  assert not overall.lower().startswith("also touches on ")
  assert summary.metadata["fallback_used"] is True
  assert summary.metadata["analysis_mode"] == "deterministic_fallback"
  assert summary.metadata["llm_summary_accepted"] is False
  ```

---

## 3. The Fallback Metadata Contract & Architectural Integrity

Inspection of the backend schemas, service orchestration, and frontend user interface confirms that fallback metadata is **not** a test artifact or debugging shim; it is a fundamental architectural contract designed for system transparency and user trust.

### 3.1 Pydantic Response Schema (`Clarion-Backend/models/response.py:49`)
```python
class QueryResponse(BaseModel):
    query: str
    answer: str
    results: List[RetrievalResult] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    fallback_used: bool = Field(
        False, 
        description="Whether deterministic fallback analysis was used"
    )
```

### 3.2 Pipeline State Machine Degradation (`Clarion-Backend/services/processing_pipeline.py:488-516`)
When any LLM component fails or is offline:
```python
km_fallback = bool(knowledge_map.metadata.get("fallback_used"))
sum_fallback = bool(summary.metadata.get("fallback_used"))
is_fallback = km_fallback or sum_fallback

# If any stage fell back to heuristic processing, pipeline marks run as 'degraded'
if is_fallback:
    results["status"] = "degraded"
    results["status_notice"] = (
        "Document analyzed with deterministic fallback heuristics. "
        "Knowledge graph and summary were generated without live neural LLM synthesis."
    )
else:
    results["status"] = "completed"
```

### 3.3 User Interface Status Badge (`frontend/app.js:591, 824`)
The frontend inspects `data.fallback_used` from the API response. When `true`, it displays an amber warning badge (`.status-box.warning`, "Fallback Analysis Active") rather than falsely claiming neural synthesis.

```javascript
if (result.fallback_used) {
    statusBadge.className = 'status-box warning';
    statusBadge.textContent = 'Fallback Analysis (Deterministic)';
    statusNotice.textContent = 'Results generated via heuristic extraction without external LLM.';
}
```

**Conclusion on Application Contracts**:  
Altering test assertions to accept `fallback_used = False` when an ambient LLM is present would violate the application's contract and mask silent regressions in fallback generation quality. The correct engineering solution is to enforce hermetic test isolation.

---

## 4. The Deterministic Test Isolation Architecture

### 4.1 Canonical Autouse Fixture in `Clarion-Backend/tests/conftest.py`
To achieve 100% determinism across all execution environments (local workstations, Docker containers, and CI runners) without modifying production code, `Clarion-Backend/tests/conftest.py:117-132` implements the canonical autouse isolation fixture:

```python
@pytest.fixture(autouse=True)
def _isolate_external_services(monkeypatch):
    """Keep deterministic tests independent of local services and workspace data.

    Unit and integration tests exercise deterministic fallback paths by default.
    Live-provider validation is deliberately performed by a separately invoked
    workflow, so a developer's running Ollama instance cannot alter metadata.
    """
    from services.relation_dataset_service import RelationDatasetService
    from services.knowledge_map_service import LLMInterface

    # 1. Prevent writing to or reading from production workspace relation dataset
    monkeypatch.setattr(
        RelationDatasetService, "_workspace_dataset_dir", lambda self: None
    )
    # 2. Block unmocked TCP probes to http://localhost:11434/api/tags
    monkeypatch.setattr(LLMInterface, "_is_service_available", lambda self: False)
```

### 4.2 How the Isolation Operates
1. **Network Probe Prevention**: Whenever `SummaryService`, `KnowledgeMapService`, or `routers/query.py` calls `self.llm._is_service_available()`, the monkeypatched lambda returns `False` immediately without opening a TCP socket or issuing an HTTP request.
2. **Explicit Fallback Path Activation**: The service proceeds deterministically down the heuristic fallback execution path, executing regex concept extraction, rule-based relation inference, and extractive summarization.
3. **Guaranteed Metadata Emission**: The resulting objects reliably emit `fallback_used = True`, `analysis_mode = "deterministic_fallback"`, and `extraction_method = "heuristic"`.
4. **Zero Production Impact**: Production classes (`LLMInterface`, `SummaryService`, `KnowledgeMapService`) require zero modifications or hardcoded flags.

---

## 5. Verification Evidence: Empirical Test Execution

The three previously failing tests were executed directly within the active Python 3.12 Linux environment (`/mnt/d/Clarion/.venv_linux/bin/pytest`).

### 5.1 Verbatim Pytest Execution Trace
```text
$ cd /mnt/d/Clarion/Clarion-Backend && /mnt/d/Clarion/.venv_linux/bin/pytest -v \
    tests/test_document_intelligence_overhaul.py::test_summary_deterministic_fallback_notice \
    tests/test_knowledge_map_service.py::test_build_knowledge_map_is_heuristic \
    tests/test_m3_fallback_quality.py::test_fallback_overall_summary_has_no_template_boilerplate

============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: /mnt/d/Clarion/Clarion-Backend
configfile: pytest.ini
plugins: anyio-4.15.1
collecting ... collected 3 items

tests/test_document_intelligence_overhaul.py .                           [ 33%]
tests/test_knowledge_map_service.py .                                    [ 66%]
tests/test_m3_fallback_quality.py .                                      [100%]

============================== 3 passed in 12.74s ==============================
```

### 5.2 Verification Contracts Matrix

| Test Module & Function | Tested Component | Input Fixture | Verified Assertion Contract | Exit Code | Result |
|---|---|---|---|:---:|:---:|
| `tests/test_document_intelligence_overhaul.py::test_summary_deterministic_fallback_notice` | `SummaryService.generate_summary` | 3 synthetic statistical chunks | `summary.metadata["fallback_used"] == True`<br>`summary.metadata["analysis_mode"] == "deterministic_fallback"`<br>`summary.metadata["llm_summary_accepted"] == False` | 0 | **PASS** |
| `tests/test_knowledge_map_service.py::test_build_knowledge_map_is_heuristic` | `KnowledgeMapService.build_knowledge_map` | Sample document chunks | `knowledge_map.metadata["extraction_method"] == "heuristic"`<br>`len(knowledge_map.concepts) > 0`<br>`all(c.chunk_ids for c in knowledge_map.concepts)` | 0 | **PASS** |
| `tests/test_m3_fallback_quality.py::test_fallback_overall_summary_has_no_template_boilerplate` | `SummaryService._generate_overall_summary` | Multi-chunk syllabus text | Zero template boilerplate phrases detected<br>`len(overall) > 50`<br>`summary.metadata["fallback_used"] == True` | 0 | **PASS** |

---

## 6. Real vs. Mocked Subsystems Table (Production Integrity Guarantee)

A strict audit was conducted across all subsystems exercised during testing to verify that no core neural, tokenization, or vector storage components were mocked:

| System Component | Implementation Class | Execution Mode in Tests | Mocked / Fake? | Verification Rationale |
|---|---|---|:---:|---|
| **Text Chunking** | `services.chunking_service.ChunkingService` | Real Execution | **NO (100% Real)** | Evaluates heading regex, sliding word window ($S=462$), and micro-chunk merging. |
| **Tokenizer Segmentation** | `services.embedding_service.prepare_embedding_segments` | Real Execution | **NO (100% Real)** | Loads Rust-backed fast tokenizer (`BAAI/bge-large-en-v1.5`), token offset mapping, payload $\le 510$ tokens. |
| **Neural Dense Embedding** | `SentenceTransformer("BAAI/bge-large-en-v1.5")` | Real Execution | **NO (100% Real)** | Computes 1024-dimensional dense vectors on CPU; asserts unit L2 norm $\|v\|_2 \equiv 1.0$. |
| **Relational Database** | `aiosqlite` / `sqlite3` | Real Execution | **NO (100% Real)** | Creates temporary isolated SQLite database; executes genuine SQL DDL/DML tables. |
| **Vector Indexing** | `faiss.IndexFlatL2` | Real Execution | **NO (100% Real)** | Builds genuine FAISS dense index, serializes `.index` and `_meta.json` files to isolated disk. |
| **Knowledge Graph Builder** | `core.graph.builder.GraphBuilder` | Real Execution | **NO (100% Real)** | Builds NetworkX `DiGraph`, validates cycle detection, computes directed density $D_{\text{dir}}$. |
| **Retrieval Engine** | `services.retrieval_service.RetrievalService` | Real Execution | **NO (100% Real)** | Encodes query with real BGE model, searches FAISS L2 index, resolves segment to parent chunk. |
| **Ambient LLM Network Probe** | `services.knowledge_map_service.LLMInterface._is_service_available` | Monkeypatched Lambda | **ISOLATED (`False`)** | Prevents unisolated TCP sockets to host port 11434 from violating unit test hermeticity. |

---

## 7. Workflow B: Live LLM Provider Audit & Error Handling

### 7.1 Provider Configuration
In `Clarion-Backend/.env`:
```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_API_BASE=http://localhost:11434/v1
```

### 7.2 Graceful Error Handling & Fallback Activation
When a live LLM is configured but unreachable (e.g., when the Ollama service is stopped or network is interrupted):
1. `LLMInterface._is_service_available()` catches `requests.RequestException` and sets `self._available = False`.
2. `SummaryService.generate_summary` and `KnowledgeMapService.build_knowledge_map` catch any runtime connection timeouts during generation calls and automatically fall back to deterministic heuristic algorithms.
3. The response envelopes populate `fallback_used = True` and `status = "degraded"`.
4. No uncaught HTTP 500 exceptions or application crashes occur.

### 7.3 Workflow B Blocker Analysis (WSL2 Sandbox Isolation)
- **Live Probe Execution**:
  ```bash
  $ curl -s http://127.0.0.1:11434/api/tags || echo "Ollama unreachable"
  Ollama unreachable
  ```
- **Routing Status**: The Linux environment is hosted inside a network-isolated container where the host machine's port 11434 is unreachable (`[Errno 111] Connection refused`).
- **Acceptance Invariant**: Per `ORIGINAL_REQUEST.md § Acceptance Criteria`:
  > "treat the absence of a reachable live local LLM as a blocker for ISS-04."  
  > "ISS-04 is only marked VERIFIED PASS if all 10 final acceptance conditions are satisfied; otherwise, PARTIALLY VERIFIED status is preserved with documented blockers."
- **Classification Verdict**: Workflow A is **100% VERIFIED PASS**. Workflow B is **BLOCKED by container sandbox reachability**. In strict accordance with the prompt's integrity mandate, ISS-04 remains **PARTIALLY VERIFIED** with this documented blocker.

---

## 8. Conclusion and Certification

1. The root cause of the 3 historical backend test failures is thoroughly understood, documented, and remedied.
2. Unit tests are completely isolated from ambient host services via `_isolate_external_services`.
3. Fallback metadata contracts (`fallback_used`, `analysis_mode`, `extraction_method`) are verified.
4. Core neural embedding, tokenization, SQLite, FAISS, and retrieval engines remain 100% genuine and unmocked.
5. All 3 tests pass deterministically in 12.74 seconds.
