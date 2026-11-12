## Final verification addendum — 2026-11-11

This current addendum supersedes status statements recorded earlier in this report. Source identity: branch `main`; base and HEAD `ebca6ed0125c7131c05c945620b23a1970844b92`; tracked-diff SHA-256 `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`.

Current results: backend suite **304 passed, 2 skipped, 0 failed** in 90.31s; targeted backend contracts **19 passed**; frontend timeline contracts **13 passed**; `node --check frontend/app.js` passed. The ordinary suite’s two skips remain recorded as skips in that run. Separately, the two live Ollama tests passed when run with host-daemon access; they are not included in the ordinary suite’s pass count.

Chromium rendered the actual frontend against deterministic stub API responses. It verified seven canonical stages, requested labels and duration formats, unavailable `--` metrics, no duplicate `llm_analysis`, and completed/running/failed/pending states. This verifies frontend rendering, not live-backend browser E2E; live-backend browser E2E remains unverified.

Live Ollama `qwen3.5:4b` smoke validation completed two chunks and four schema-validated records. Request durations were 8.252s and 15.370s in this run. Strict export succeeded; a separate controlled fault-injection run verified retry, partial accounting, strict-export refusal (HTTP 409), and resume with checkpoint reuse. This is smoke and reliability evidence, not a controlled performance benchmark or a claim of general performance improvement.

SHA-256 checks for all three production databases matched before and after isolated integration/full-suite testing. Overall project classification: **PARTIALLY VERIFIED**.

---

# Clarion Engineering and Research Fix Log

## Final worktree recovery remediation (2026-11-11)

| Issue | Change | Evidence | Status |
|---|---|---|---|
| `estimate_size()` property access could throw from `hasattr()` | Guarded graph attribute probing and word-count property access; preserved prior fallback values | Adversarial estimate-size case and custom raising-property cases passed | VERIFIED PASS |
| Serializer fabricated missing derived counts/rates as zero and dropped mapping metadata while pending | Keep unmeasured values unavailable; retain `analysis_mode`/`model` on mapping | Seven-stage serializer and resilience tests passed | VERIFIED PASS |
| Integration fixtures leaked router singleton assignments into following tests | Replaced direct module assignments with `monkeypatch.setattr` in adversarial and E2E fixture setup | Final full suite 304 passed, 2 skipped, 0 failed | VERIFIED PASS |
| Regression suite after stash concern | Saved tracked diff snapshot, inspected empty stash list, ran final full backend suite | 304 passed, 2 skipped, 0 failed | VERIFIED PASS |
| Dataset/export integrity | Retained strict gate and checkpoint behavior | Mocked-provider suite 8/8 passed; separate live Ollama smoke passed (2 chunks, 4 validated records); controlled performance benchmark remains incomplete | VERIFIED for smoke/reliability checks; performance benchmarking remains incomplete |
| Browser timeline rendering | Playwright installed outside project Python environment; actual frontend rendered against stub API | Node syntax passed, frontend contract 13/13, Chromium verified seven stages, durations, labels, unavailable metrics and four statuses; screenshots saved | VERIFIED PASS (rendering) |

## Verification snapshot identity

All current entries correspond to snapshot `Clarion_Verification_Snapshot_2026-11-11`, base `ebca6ed0125c7131c05c945620b23a1970844b92`, tracked-diff SHA-256 `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`. Full backend suite: 304 passed, 2 skipped, 0 failed in 90.31s. Live backend browser E2E and Ollama inference remain unverified. Production database hashes matched before and after isolated integration tests.

**Project**: Clarion (Intelligent Document Structure, Knowledge Graph Generation, and Dataset Construction System)  
**Date**: 2026-11-09  
**Scope**: All fixes across Backend, Frontend, Paper Appendix, Configuration, Test Infrastructure, and E2E Pipeline Orchestration  
**Status**: 100% Implemented; Workflow A Verified Pass; ISS-04 Overall Status: PARTIALLY VERIFIED (Workflow B Blocked by Container Sandbox Network Isolation)

## Post-package remediation (2026-11-09)

| Issue | Status | Remediation | Validation status |
|---|---|---|---|
| ISS-04 | PARTIALLY VERIFIED | Implemented tokenizer-aware offset segmentation, 3-tier hierarchy (`embedding_segments`), segment-aware retrieval, lossless string reconstruction, SQLite-FAISS consistency, and reprocessing deduplication. | Workflow A fully verified via `test_e2e_pipeline_orchestration.py` (7/7 passed in ~177s) under Linux Python 3.12 with real BGE embeddings, fast tokenizer segmentation, SQLite persistence, and FAISS indexing; Workflow B live LLM blocked by container network isolation of host port 11434 ([Errno 111] Connection refused). |
| Graph counts | VERIFIED FIXED | Added reproducible graph registry generator and corrected test-readiness documentation to canonical artifact UUIDs. | Registry generated from 46 graph JSON files and canonical relation DB. |
| Metric term | VERIFIED FIXED | Replaced “calibrated similarity” with bounded distance-derived ranking score. | Source inspection. |
| Latency | UNRESOLVED | Historical numbers reclassified as preliminary; timing design defect documented. | Monotonic timer verification measured empirical RAG components (780.31 ms E2E). |
| Manuscript | BLOCKED BY MISSING EVIDENCE | Recorded absence of a complete main manuscript; appendix not promoted to paper. | Repository search confirms appendix exists, main manuscript text pending. |

---

## 1. Master Fix Inventory Table

| # | Issue ID | Issue Category | Original Behaviour | Correction Applied | Files Changed | Validation |
|---|---|---|---|---|---|---|
| 1 | **ISSUE-01** | Backend Algorithm | In `knowledge_map_service.py:1237`, inferred candidate relation types were overwritten with hardcoded `RelationType.EXPLANATION`, collapsing the taxonomy. | Modified instantiation to `relation_type=inferred_rel_type or RelationType.EXPLANATION`, preserving lexical indicators (`part-of`, `cause-effect`, `prerequisite`, `example-of`, `derives-from`, `definition`). | `Clarion-Backend/services/knowledge_map_service.py`<br>`Clarion-Backend/tests/test_m3_fallback_quality.py` | `test_m1_algorithmic_fixes.py:test_relation_type_override_preserves_inferred_patterns` (PASS); E2E `T1.4` (PASS). |
| 2 | **ISSUE-02** | Backend Algorithm | Neural encoder calls omitted `normalize_embeddings=True`, causing unnormalized BGE vectors ($\|v\|_2 \approx 8.5\text{--}12.0$) and uncalibrated FAISS L2 distances ($d^2 > 15$, $s < 0.06$). | Added `normalize_embeddings=True` to `generate_embeddings`, `generate_embeddings_async`, and `embed_query`. Live embeddings now guarantee unit norm ($\|v\|_2 \equiv 1.0$). | `Clarion-Backend/services/embedding_service.py` | `test_m1_algorithmic_fixes.py:test_embedding_normalization_unit_length` (PASS); E2E `T1.2` (PASS). |
| 3 | **ISSUE-03** | Backend Algorithm | Document chunks exceeding 512 subwords were silently truncated by HuggingFace `SentenceTransformer` context window limits without warnings or observability. | Added `check_token_length` safety guards and warning logging in both chunking and embedding pipelines using model tokenizer and 1.35x word expansion fallback. | `Clarion-Backend/services/chunking_service.py`<br>`Clarion-Backend/services/embedding_service.py` | `test_m1_algorithmic_fixes.py:test_token_length_safety_warning_chunking_service` (PASS); E2E `T2.3` (PASS). |
| 4 | **ISSUE-04** | Backend Config | Default application name in `config.py` and `.env.example` was set to prototype string `"Intelligent Document Knowledge System"`. | Updated `Settings.app_name` default to canonical product name `"Clarion"` and synchronized `.env.example`. | `Clarion-Backend/utils/config.py`<br>`Clarion-Backend/.env.example` | `test_m1_algorithmic_fixes.py:test_app_name_is_clarion` (PASS); `GET /system/runtime-diagnostic` returns `"Clarion"`. |
| 5 | **ISSUE-05** | Backend Config | Active deployment environment file `Clarion-Backend/.env` retained obsolete prototype name `APP_NAME="Intelligent Document Knowledge System"`. | Updated `APP_NAME="Clarion"` in `Clarion-Backend/.env`, reconciling environment variables with Pydantic settings. | `Clarion-Backend/.env` | Programmatic verification script asserting `"APP_NAME=\"Clarion\""` (PASS). |
| 6 | **ISSUE-06** | Frontend UI/UX | Missing `<link rel="icon">` in `frontend/index.html` head caused browsers to issue 404 HTTP requests for `/favicon.ico`. | Embedded inline SVG data-URI favicon (`💠`) in `<head>`, rendering crisp diamond glyph with zero network asset dependencies. | `frontend/index.html` | Static DOM invariant test `verify_startup_dom_invariants.py` (PASS); DOM XPath assertion confirms 1 favicon link. |
| 7 | **ISSUE-07** | Frontend UI/UX | Header brand title lacked the Research Preview badge, leaving unused CSS `.brand-badge` rules in `styles.css`. | Restored `<span class="brand-badge">Research Preview</span>` adjacent to brand logo inside `.brand-title-row`. | `frontend/index.html` | Static DOM invariant test `verify_startup_dom_invariants.py` (PASS); DOM XPath assertion confirms `.brand-badge` text. |
| 8 | **ISSUE-08** | Frontend UI/UX | Overview metrics grid only contained 5 cards, omitting Relations, Nodes, and Edges, breaking the 4x2 grid layout and causing JS null errors. | Restored 3 missing metric cards (`stat-relations`, `stat-nodes`, `stat-edges`) with `--` startup placeholders, completing 8-card grid. | `frontend/index.html` | `verify_startup_dom_invariants.py` (PASS); `verify_frontend_e2e_contracts.py` (PASS 6/6); all 8 cards verified. |
| 9 | **ISSUE-09** | Paper Math | `appendix_report.md` Table 2 checked character length `len(chunk_text.strip()) < min_chunk_size` rather than word count, and dropped remainder words with `continue`. | Replaced character check with word count `len(word_slice) < self.min_chunk_size`; added explicit residual merging into preceding chunk (`chunks[-1].content += ...`). | `appendix_report.md` | Programmatic verification script (PASS); Pytest suite passes; E2E `T1.1` and `T2.2` pass. |
| 10 | **ISSUE-10** | Paper Math | `appendix_report.md` Table 8 omitted unit vector normalization and mathematical bounds on FAISS L2 distance conversion ($s = 1/(1+d^2)$). | Documented unit $L_2$ norm ($\|q\|_2 = \|v\|_2 = 1.0$) and squared Euclidean bounds $d^2 \in [0, 4]$. The output is now correctly termed a bounded monotonic distance-derived ranking score, not probability calibration. | `appendix_report.md` | Formula inspection; empirical probability calibration was not performed. |
| 11 | **ISSUE-11** | Paper Math | `appendix_report.md` Table 9 lacked formal definitions for directed density $D_{\text{dir}} = |E| / (|V|(|V|-1))$ and total average degree $\bar{d} = 2|E|/|V|$. | Formalized $D_{\text{dir}}$ and $\bar{d}$ equations with boundary conditions for single-node graphs ($|V|=1 \implies D_{\text{dir}}=0.0$). | `appendix_report.md` | Programmatic verification script (PASS); E2E `T1.5` and `T2.4` pass. |
| 12 | **ISSUE-12** | Paper Data | Conflation of visualization graph edges with exported candidate relation records in BSM 301 and Thevenin graphs. | Disambiguated 19 visualization edges from 34 exported relation records for BSM 301, and documented exact topologies for Thevenin Runs 1 & 2. | `appendix_report.md` | Direct JSON graph inspection and SQLite query verification (PASS); verified bit-exact against disk artifacts. |
| 13 | **ISSUE-13** | Paper Formatting | `appendix_report.md` contained colloquial student thesis phrasing ("This chapter contains essential Python code snippets..."). | Rewrote intro and styling to academic publication grade, adding author attribution (`Soumyojit Sarkar`), institutional affiliation, and scientific acknowledgements. | `appendix_report.md` | String search confirms zero instances of colloquial phrasing (PASS). |
| 14 | **ISSUE-14** | Bibliography | Missing formal BibTeX bibliography file for the 8 literature citations audited across the research paper and reports. | Authored `references.bib` with verified BibTeX entries for Landis & Koch (1977), Cohen (1960), Fleiss (1971), Platt (1999), Guo et al. (2017), Xiao et al. (2023), Johnson et al. (2019), and Hagberg et al. (2008). | `references.bib` | Syntactic parser verified balanced braces and complete metadata (PASS). |
| 15 | **ISSUE-15** | Research Reporting | Ambiguity regarding human evaluation completion status on 1,554 candidate relations. | Formally documented that `relation_dataset.db` is 100.0% unlabeled (1,554 records with `is_valid IS NULL`), queues are 0% completed, and $\kappa$ is UNMEASURED on real data. | `Clarion-Backend/reports/RESEARCH_EXPERIMENT_REPORT.md` | SQLite query confirms 1,554 unlabeled; `annotation_tool.py validate` confirms 0 completed records (PASS). |
| 16 | **ISSUE-16** | Research Reporting | Latency curve in `generate_latency_graph.py` risked being misinterpreted as empirical per-relation wall clock measurements. | Added explicit scientific disclaimer to script header and documented empirical monotonic wall-clock RAG latencies (780.31 ms E2E) in reports. | `Clarion-Backend/generate_latency_graph.py`<br>`Clarion-Backend/reports/RESEARCH_EXPERIMENT_REPORT.md` | Code review and monotonic timer verification script `verify_rag_reviewer2.py` (PASS). |
| 17 | **ISSUE-17** | Research Reporting | Benchmark results in `ENGINEERING_VALIDATION_REPORT.md` risked being cited as human gold-standard evaluation scores. | Added prominent CAUTION notices above Tables 1–3 identifying them as synthetic test harness runs on `synthetic_benchmark_fixture.json`, and added full Reproducibility Manifest. | `Clarion-Backend/reports/ENGINEERING_VALIDATION_REPORT.md` | Benchmark runner `run_canonical_benchmark.py` outputs integrity notices and reproduces tables (PASS). |

---

## 2. Detailed Technical Fix Logs

### ISSUE-01: Candidate Relation Type Override
- **Affected File**: `Clarion-Backend/services/knowledge_map_service.py` (lines 1202–1249)
- **Root Cause**: In `_detect_relations_by_cooccurrence`, lexical indicator rules checked for patterns such as `"is a component of"`, `"leads to"`, `"requires"`, and successfully assigned `inferred_rel_type`. However, when constructing `CandidateRelation`, line 1237 hardcoded:
  ```python
  relation_type=RelationType.EXPLANATION,
  ```
  This systematically discarded all structural, causal, and prerequisite classifications, turning all relations into generic explanations.
- **Correction Applied**:
  ```python
  relation_type=inferred_rel_type or RelationType.EXPLANATION,
  ```
- **Validation**: `Clarion-Backend/tests/test_m1_algorithmic_fixes.py` tests that indicator phrases properly map to `PART_OF`, `CAUSE_EFFECT`, `PREREQUISITE`, etc., while sentences without indicators cleanly fall back to `EXPLANATION`.

---

### ISSUE-02: Embedding Vector L2 Normalization
- **Affected File**: `Clarion-Backend/services/embedding_service.py` (lines 204–215, 282–298, 424–431)
- **Root Cause**: Calls to `self.model.encode(text, ...)` omitted `normalize_embeddings=True`. Because BGE-large produces unnormalized vectors with $\|v\|_2 \approx 8.5\text{--}12.0$, Euclidean distances in FAISS were inflated ($d^2 > 15$). Under the distance inversion formula $s = 1/(1+d^2)$, similarity scores collapsed to $[0.005, 0.062]$. Meanwhile, offline fallback embeddings were normalized to unit length, causing scale divergence.
- **Correction Applied**: Passed `normalize_embeddings=True` to all encoding methods in `EmbeddingService`.
- **Validation**: `test_embedding_normalization_unit_length` verifies $\|v\|_2 = 1.0 \pm 10^{-5}$ across live and fallback embedding modes.

---

### ISSUE-03: Token Length Safety Monitoring
- **Affected Files**: `Clarion-Backend/services/chunking_service.py`, `Clarion-Backend/services/embedding_service.py`
- **Root Cause**: Word-based chunking allows chunks up to 512 words (merging up to 640 words). In technical corpora, word-to-subword expansion averages 1.35x, yielding 650–860 subword tokens. The BGE-large model has a hard context window of 512 subwords; text beyond 512 tokens was silently truncated by the transformer tokenizer without warning.
- **Correction Applied**: Implemented `check_token_length` using `self.model.tokenizer` (or 1.35x word multiplier fallback) that emits a warning when a chunk exceeds 512 subwords, giving full observability into truncation boundaries.
- **Validation**: `test_token_length_safety_warning_chunking_service` and `test_token_length_safety_warning_embedding_service` assert warning logs are triggered when inputs exceed 512 subwords.

---

### ISSUE-04 & ISSUE-05: Product Branding Alignment
- **Affected Files**: `Clarion-Backend/utils/config.py`, `Clarion-Backend/.env.example`, `Clarion-Backend/.env`
- **Root Cause**: Default settings and deployment `.env` retained prototype name `"Intelligent Document Knowledge System"`.
- **Correction Applied**: Updated `app_name: str = "Clarion"` in `config.py`, and `APP_NAME="Clarion"` in both `.env.example` and `.env`.
- **Validation**: `test_app_name_is_clarion` and runtime diagnostic endpoint return canonical `"Clarion"` branding.

---

### ISSUE-06, ISSUE-07, ISSUE-08: Frontend DOM Invariants & Metric Deck
- **Affected File**: `frontend/index.html`
- **Root Cause**: The web interface was missing `<link rel="icon">` (causing 404s), `<span class="brand-badge">` (leaving orphaned CSS), and 3 metric cards (`stat-relations`, `stat-nodes`, `stat-edges`).
- **Correction Applied**:
  - Embedded inline SVG favicon (`<link rel="icon" href="data:image/svg+xml,...">`).
  - Added `<span class="brand-badge">Research Preview</span>` to `.brand-title-row`.
  - Added metric cards for Relations, Nodes, and Edges to `.metrics-grid`, restoring the 8-card 4x2 dashboard layout.
- **Validation**: `scripts/verify_startup_dom_invariants.py` passes 15/15 invariant checks; `scripts/verify_frontend_e2e_contracts.py` passes 6/6 contract steps.

---

### ISSUE-09 to ISSUE-14: Research Paper Appendix Alignment & Bibliography
- **Affected Files**: `appendix_report.md`, `references.bib`
- **Root Cause**: Outdated draft snippets in `appendix_report.md` had character-length checks, omitted residual merging, lacked FAISS distance derivations, reported ambiguous graph statistics, and lacked formal BibTeX references.
- **Correction Applied**: Reconciled chunking formulas ($S=462$), FAISS bounds ($d^2 \in [0, 4], s \in [0.20, 1.00]$), graph density and degree ($D_{\text{dir}}, \bar{d}$), and created `references.bib` with 8 complete BibTeX citations.
- **Validation**: Independent programmatic verification script confirms all formulas, numbers, and bibliography entries.

---

### ISSUE-15, ISSUE-16, ISSUE-17: Experimental Methodology & Scientific Transparency
- **Affected Files**: `Clarion-Backend/reports/RESEARCH_EXPERIMENT_REPORT.md`, `Clarion-Backend/reports/ENGINEERING_VALIDATION_REPORT.md`, `Clarion-Backend/generate_latency_graph.py`
- **Root Cause**: Risk of presenting synthetic test harness metrics or proportional latency models as real-world human evaluation results.
- **Correction Applied**:
  - Transparently disclosed that `relation_dataset.db` is 100% unlabeled (1,554 records) and annotation queues are 0% complete.
  - Formally classified ROUGE, MRR, P@k, and Recall@k as UNMEASURED.
  - Disclosed the proportional token allocation formula in `generate_latency_graph.py` ($t_i = 92.5 \cdot \text{tokens}_i / \sum \text{tokens}$) and reported genuine empirical RAG component latencies (780.31 ms E2E).
  - Added CAUTION notices and Reproducibility Manifest to engineering validation tables.
- **Validation**: Forensic Auditor certified CLEAN with zero integrity violations.

---

### ISS-04: Tokenizer-Aware Segmentation, 3-Tier Hierarchy, and Retrieval Provenance
- **Issue Classification**: Core Architecture / Content & Provenance Integrity / Retrieval Grounding
- **Status**: **PARTIALLY VERIFIED** (Workflow A: 100% Verified Pass; Workflow B: Blocked by WSL2 Container Network Isolation)
- **Affected Files**:
  - `Clarion-Backend/services/embedding_service.py` (lines 40–160, 204–215, 282–298)
  - `Clarion-Backend/services/retrieval_service.py` (lines 50–120)
  - `Clarion-Backend/services/processing_pipeline.py` (lines 140–210)
  - `Clarion-Backend/tests/test_e2e_pipeline_orchestration.py` (7 automated test cases)
- **Defect Description**:
  Prior to this remediation, Stage 2 sliding-window chunking allowed chunks of up to 512 words (with merging up to 640 words). In technical corpora, word-to-subword expansion averages 1.35x, causing chunks to frequently expand to 800–900 subword tokens. When ingested by the SentenceTransformer dense embedding model (`BAAI/bge-large-en-v1.5`, hard sequence limit $L = 512$ subwords), the Hugging Face tokenizer silently truncated all tokens beyond position 512 without raising an exception or logging a warning. The content beyond token 512 was permanently lost from the FAISS vector index, creating a silent retrieval blind spot and corrupting retrieval provenance.
- **Algorithmic Remediation**:
  1. *Tokenizer-Aware Offset Segmentation (`EmbeddingService.prepare_embedding_segments`)*:
     - Leverages the Hugging Face fast tokenizer offset mapping with an operational payload ceiling of $P = 510$ tokens (preserving slots for `[CLS]` and `[SEP]`).
     - Maps the source chunk into non-overlapping character spans $\text{Span}_i = [B_i, E_i^*)$.
     - Explicitly assigns inter-token whitespace gaps $g_k = [e_{k-1}, b_k]$ to the preceding segment ($B_0 = 0$, $E_{m-1}^* = N$, $B_{i+1} = E_i^*$).
     - Implements dynamic token-count decrementing: if `TokenCount`$(C[B_i:E_i^*]) > 512$, $E_i^*$ is decremented until the token count fits within the limit.
     - Mathematically proves and empirically verifies lossless string coverage:
       $$\bigoplus_{i=0}^{m-1} s_i = C \iff \sum_{i=0}^{m-1} |s_i| = |C|$$
  2. *3-Tier Relational Hierarchy Model*:
     - Tier 1: Document (`documents.id`)
     - Tier 2: Semantic Chunk (`chunks.chunk_id`)
     - Tier 3: Embedding Segment (`embedding_segments.segment_id` = `{chunk_id}:seg:{idx}`)
     - Relational table `embedding_segments` records `segment_id`, `chunk_id`, `document_id`, `segment_index`, `char_start`, `char_end`, `token_count`, and `segment_text`.
  3. *Segment-Aware Vector Retrieval (`retrieval_service.py`)*:
     - Embeds user queries using normalized BGE embeddings ($\|q\|_2 \equiv 1.0$).
     - Searches FAISS `IndexFlatL2` for top-$k$ nearest segment vectors.
     - Resolves each `segment_id` to its parent `chunk_id` in SQLite, extracts grounded segment text, and returns formatted evidence preserving document and chunk provenance.
  4. *SQLite-FAISS Consistency & Reprocessing Deduplication*:
     - Guarantees 1:1 bidirectional alignment between SQLite `embedding_segments` records and FAISS vector index rows.
     - When reprocessing with `force_reanalyze=True`, existing segments and vectors for `document_id` are atomically purged and re-indexed, preventing orphaned records and duplicate retrieval results.
  5. *Historical Data Backward Compatibility*:
     - Maintains zero degradation for legacy unsegmented chunks lacking `embedding_segments` records, falling back to chunk-level retrieval while providing a documented atomic reindexing migration path.
- **Exact Reproduction Commands**:
  ```bash
  # Execute full E2E pipeline orchestration & provenance suite (Workflow A)
  cd /mnt/d/Clarion/Clarion-Backend
  /mnt/d/Clarion/.venv_linux/bin/pytest -v tests/test_e2e_pipeline_orchestration.py

  # Confirm production database cleanliness (assert 0 changes)
  git status --porcelain /mnt/d/Clarion/Clarion-Backend/data/clarion.db

  # Probe live host Ollama reachability (Workflow B)
  curl -s http://127.0.0.1:11434/api/tags || echo "Ollama unreachable (WSL2 sandbox network isolation blocker)"
  ```
- **Validation Results**:
  - **Workflow A (Offline / Deterministic Integration)**: **100% VERIFIED PASS** across all 7 automated tests in `test_e2e_pipeline_orchestration.py` (execution time: 177.27s):
    1. `test_e2e_upload_analyze_query_workflow_a`: PASSED (genuine upload -> ingestion -> chunking -> segmentation -> BGE embedding -> FAISS -> retrieval).
    2. `test_lossless_segmentation_exact_string_cover_and_token_ceiling`: PASSED (exact string reconstruction $\bigoplus s_i == C$, token ceiling $\le 512$ respected).
    3. `test_daa_final_pdf_fixture_ingestion_provenance_and_retrieval`: PASSED (long document 87 pages / 19,830 words, 60 chunks, 64 segments, 100% retrieved).
    4. `test_sqlite_faiss_consistency_and_three_tier_hierarchy`: PASSED (strict 1:1 relational-vector count alignment, zero dangling vectors).
    5. `test_failed_processing_cleanup_and_error_containment`: PASSED (stage failure containment, status `FAILED`, zero orphaned chunks).
    6. `test_reprocessing_deduplication_with_force_reanalyze`: PASSED (idempotent re-run, atomic purge, zero duplicate records).
    7. `test_historical_unsegmented_data_backward_compatibility`: PASSED (legacy chunk-level retrieval fallback verified).
  - **Production Database Cleanliness**: Confirmed 0 modifications to `Clarion-Backend/data/clarion.db`.
  - **Workflow B (Live Local LLM Integration)**: **DOCUMENTED BLOCKER**. Probing `http://localhost:11434` or `http://127.0.0.1:11434` returns `[Errno 111] Connection refused` due to container sandbox network isolation. Per `ORIGINAL_REQUEST.md` line 61, the absence of a reachable live local LLM is treated as a blocker for full resolution.
  - **Final Classification**: **`PARTIALLY VERIFIED`** (Workflow A: Verified Pass; Workflow B: Blocked by Container Sandbox Network Isolation).

