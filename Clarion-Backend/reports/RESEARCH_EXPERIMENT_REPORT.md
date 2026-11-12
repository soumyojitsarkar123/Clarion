# Clarion: Scientific Research Status & Human Annotation Protocol

**Document Type:** Empirical Research Protocol & Data Audit  
**Status:** `RESEARCH INFRASTRUCTURE READY — HUMAN ANNOTATION REQUIRED`  
**Date:** October 2026  
**Repository:** `Clarion-Backend`  
**Database Audited:** `Clarion-Backend/data/relation_dataset.db`

---

## CURRENT STATUS

**RESEARCH INFRASTRUCTURE READY — HUMAN ANNOTATION REQUIRED**

The research engineering implementation is complete. The next and only remaining scientific bottleneck is human annotation.

---

## REAL DATA STATUS

| Item | Count |
|------|-------|
| Candidate relations | **1,554** |
| Source documents | **44 distinct documents** |
| Human-annotated labels | **0 (zero)** |

All 1,554 `is_valid` fields are `NULL` (100.0%) in `relation_dataset.db`.

```sql
SELECT COUNT(*) FROM relation_dataset;                           -- 1,554 total candidates
SELECT COUNT(DISTINCT document_id) FROM relation_dataset;         -- 44 real documents
SELECT COUNT(*) FROM relation_dataset WHERE is_valid IS NOT NULL; -- 0 labeled records
```

> [!CAUTION]
> **Research gold dataset not yet available.**  
> It is a scientific integrity violation to report metrics derived from
> `synthetic_benchmark_fixture.json` as empirical research results.  
> No publication-grade claims regarding real-world relation precision, recall,
> or calibration error can be made until human annotations are completed and
> verified via inter-annotator agreement metrics.

---

## IMPLEMENTED (Engineering/Infrastructure — Verified)

All items below are **software-verified** — they work correctly on synthetic fixtures
or unlabeled real data. None constitute research findings.

| Component | File | Status |
|-----------|------|--------|
| Graph target-leakage prevention | `core/experiments/graph_features.py` | ✅ Implemented & tested |
| Document-level split isolation | `core/benchmarking/splitter.py` | ✅ Implemented & tested |
| Calibration infrastructure (Platt/Isotonic) | `core/evaluation/calibration.py` | ✅ Implemented & tested |
| Evidence verifier (lexical, NOT NLI) | `core/evaluation/evidence_verifier.py` | ✅ Terminology corrected |
| Multi-signal validator | `core/evaluation/multi_signal_validator.py` | ✅ Double-counting fixed |
| Benchmark harness (baselines + ablations) | `core/benchmarking/harness.py` | ✅ Implemented & tested |
| Paired document-cluster bootstrap CI | `core/benchmarking/harness.py` | ✅ Percentile bootstrap (B=2000) |
| CI validity fix (Wald → bootstrap) | `core/benchmarking/harness.py` | ✅ CI always bounded [min, max] |
| Research experiment lock | `core/benchmarking/harness.py` | ✅ Blocks real-data run if 0 labels |
| Annotation tool (export/validate/import) | `core/benchmarking/annotation_tool.py` | ✅ Human-readable README embedded |
| Inter-annotator agreement (κ) | `core/benchmarking/agreement.py` | ✅ Cohen's + Fleiss' κ mathematically implemented (UNMEASURED on real data; 0 empirical κ claimed) |
| Gold dataset contract | `core/benchmarking/gold_dataset_contract.py` | ✅ Field eligibility enforced |
| Annotation protocol | `data/ground_truth/ANNOTATION_PROTOCOL.md` | ✅ Gold contract + IAA status |
| Canonical benchmark CLI | `run_canonical_benchmark.py` | ✅ Git provenance + integrity notice |
| Provenance (git commit, versions) | `run_canonical_benchmark.py` | ✅ SHA-256 + software versions |
| Canonical synthetic benchmark | `data/benchmarks/bench_7138429ad4e2.json` | ✅ 5 seeds [42–46] (Synthetic Regression) |
| Primary annotation batch (200 records, seed=42) | `data/ground_truth/annotation_queue_batch1.json` | ✅ Exported & validated (0% completed; framework ready) |
| Independent overlap batch (50 records, seed=42) | `data/ground_truth/annotation_queue_batch1_overlap_b.json` | ✅ Exported & validated (0% completed; framework ready) |

---

## NOT YET POSSIBLE (Blocked — Human Annotation Required)

The following items **cannot be completed** without human-annotated gold labels:

- ❌ **Training a relation validity classifier on real data** — zero labeled records
- ❌ **Reporting real-world F1, precision, recall** — no gold standard
- ❌ **Reporting calibration error (ECE/MCE/Brier) on real data** — no gold standard
- ❌ **Testing H1 (Calibration)**: Platt ECE reduction vs. LLM confidence
- ❌ **Testing H2 (Hallucination Mitigation)**: FAR reduction ≥ 25%
- ❌ **Testing H3 (Orthogonal Synergy)**: Leave-one-signal-out ablation significance
- ❌ **Computing inter-annotator agreement (Cohen's κ, Fleiss' κ)** — unmeasured on real data (zero empirical kappa claimed; requires ≥2 annotators labeling the same batch)
- ❌ **Reporting Information Retrieval or Summarization metrics (P@k, Recall@k, MRR, ROUGE-N)** — formally UNMEASURED / UNTESTED (no ground-truth test collections or reference summaries exist)
- ❌ **Promoting annotation batch to ADJUDICATED GOLD** — requires completion + IAA ≥ 0.70
- ❌ **Any publication-grade empirical claim** — none can be made

---

## 1. Ground Truth Reality & Human Evaluation Status

A forensic, programmatic audit of the evaluation datasets and annotation assets was conducted across `Clarion-Backend/data/`:

### A. Candidate Relation Database Inventory
Direct SQL inspection of the canonical candidate relation database (`Clarion-Backend/data/relation_dataset.db`) establishes the following exact counts:
- **Total Candidate Records:** Exactly **1,554** candidate relations.
- **Source Documents:** Exactly **44** distinct documents.
- **Human Gold Annotations:** Exactly **0** human labels (**100.0%** of records have `is_valid IS NULL`).
- All 1,554 candidates were extracted automatically by the extraction pipeline and flagged as candidates for subject-matter expert review.

### B. Human Annotation Queue Batches
Programmatic inspection using `core/benchmarking/annotation_tool.py validate` reveals:
1. **Primary Human Annotation Queue (`Clarion-Backend/data/ground_truth/annotation_queue_batch1.json`)**:
   - Total Records: Exactly **200** candidate relations.
   - Sampling Strategy: Stratified across documents and relation types with fixed `random_seed: 42`.
   - Completion Status: **0 / 200 records completed (0.0% completion)**.
   - All `label`, `error_category`, `annotator_id`, and `adjudicated_evidence_text` fields are `null`.
2. **Independent Overlap Queue (`Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json`)**:
   - Total Records: Exactly **50** candidate relations.
   - Sampling Strategy: Stratified subset drawn from Batch 1 for independent second annotator review with fixed `random_seed: 42`.
   - Completion Status: **0 / 50 records completed (0.0% completion)**.
   - All `label`, `error_category`, `annotator_id`, and `adjudicated_evidence_text` fields are `null`.
3. **Synthetic Regression Fixture (`Clarion-Backend/data/ground_truth/synthetic_benchmark_fixture.json`)**:
   - Total Records: 20 synthetic records across 5 document clusters.
   - Purpose: Exclusively used for software contract regression tests (`SYNTHETIC_REGRESSION`).

**Scientific Assessment:**  
The human annotation infrastructure, web UI (`core/benchmarking/annotation_ui.py`), CLI validation tooling (`core/benchmarking/annotation_tool.py`), and gold promotion contract enforcement (`core/benchmarking/gold_dataset_contract.py`) are **100% operational**. However, human annotation trials are pending execution (**0% complete**).

### C. Inter-Annotator Agreement Reality
The inter-annotator agreement engine in `core/benchmarking/agreement.py` provides complete mathematical implementations for:
- Cohen's Kappa ($\kappa$) for two-rater nominal agreement:
  $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
- Fleiss' Generalized Kappa ($\kappa_{\text{fleiss}}$) for multi-rater agreement across $m \ge 3$ judges.
- Systematic disagreement extraction for expert committee adjudication.

**Empirical Reality:**  
Because zero human annotations are completed in `relation_dataset.db` and the batch queues, inter-annotator agreement is **UNMEASURED on real data**. **Zero empirical kappa ($\kappa = \text{UNMEASURED}$) is claimed.** Reporting any quantitative agreement score in research submissions without completed dual human annotations would constitute scientific fabrication.

---

## 2. Closed-Domain Human Annotation Protocol

To convert the 1,554 candidates into a publication-quality gold dataset, Clarion implements a formal adjudication protocol.

### A. Closed-Domain Grounding Rule
Annotators evaluate relation validity **strictly within the context of the source document**:
- **Allowed Evidence:** Statements explicitly present in the provided `chunk_context`.
- **Prohibited:** External world knowledge or common sense absent from the text. If concept $A$ is known in the real world to cause $B$, but the document does not assert this link, the relation must be marked `INVALID` (`UNSUPPORTED_ASSERTION`).

### B. Annotation Schema (3 Classes)
1. **`VALID`:** The source text explicitly and unambiguously supports the directional relation between `concept_a` and `concept_b`.
2. **`INVALID`:** The relation is contradicted, inverted, unsupported, or between spurious co-occurring entities.
3. **`AMBIGUOUS`:** The text mentions both concepts in a related context, but the exact semantic relation type or direction cannot be confirmed without unstated assumptions. (Ambiguous records are excluded from binary benchmark evaluation).

### C. Error Taxonomies for `INVALID` Relations
Every `INVALID` label must specify an error category:
- `UNSUPPORTED_ASSERTION`: The text discusses both concepts, but never asserts the claimed relationship.
- `DIRECTION_INVERSION`: The causal or prerequisite direction is inverted (e.g. text says $B$ requires $A$, but candidate asserts $A \to B$).
- `TYPE_MISMATCH`: A relationship exists, but the relation type is misclassified (e.g. `prerequisite` instead of `explanation`).
- `SPURIOUS_COOCCURRENCE`: The concepts appear in adjacent clauses or sentences without semantic connection.
- `NOISE_ENTITY`: One or both extracted concepts are garbled tokens, section headers, or non-entities.

---

## 3. Inter-Annotator Agreement Engine

**Agreement analysis is pending human annotation.**

Reliability of the human dataset must be mathematically quantified before model training:
1. **Cohen's Kappa ($\kappa$):** Computed for every pair of annotators on overlapping batches:
   $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
   Target: $\kappa \ge 0.70$ (Substantial Agreement under Landis & Koch, 1977).
2. **Fleiss' Kappa ($\kappa_{\text{fleiss}}$):** Computed across panels of $m \ge 3$ annotators.
3. **Disagreement Adjudication:** All records where annotators diverge are extracted via `extract_disagreements_for_adjudication()` and referred to a domain expert committee.

IAA tooling is implemented in `core/benchmarking/agreement.py` and cannot be exercised until at least two annotators complete annotations on the same batch.

---

## 4. Gold Dataset Contract

A record is **NOT eligible** for the gold dataset unless:

| Field | Requirement |
|-------|-------------|
| `source document` | Must exist and be traceable to `document_id` |
| `candidate` | Must exist in `relation_dataset.db` with a `record_id` |
| `evidence` | `adjudicated_evidence_text` must be non-empty |
| `label` | Must be `VALID`, `INVALID`, or `AMBIGUOUS` — NOT null |
| `annotator_id` | Must be a real, non-null annotator identifier |
| `annotation_version` | Must reference a valid `guidelines_version` |
| Field validation | Must pass `validate_annotation_batch()` schema checks |

### Promotion Pipeline
```
RAW ANNOTATIONS (annotation_queue_batch1.json)
        ↓  validate_annotation_batch() passes, completion = 100%
        ↓  dual-annotator batch has κ ≥ 0.70
ADJUDICATED GOLD (human_gold_v1.json)
        ↓  SHA-256 hash recorded in ExperimentManifest
        ↓  benchmark_type = "REAL_HUMAN_GOLD"
RESEARCH BENCHMARK (run_canonical_benchmark.py)
```

---

## 5. Information Retrieval & NLP Metrics: Formal Classification

In strict adherence to the project's empirical integrity principles (Rule 2: "No fabricated results"), all Information Retrieval (IR) and Natural Language Processing (NLP) summarization metrics are formally classified as follows:

| Metric Category | Specific Metrics | Repository Code Status | Ground Truth Fixtures | Scientific Classification |
|---|---|---|---|---|
| **Retrieval Precision** | Precision@k ($P@k$, $k \in \{1, 3, 5\}$) | Not implemented (0 loc) | None (no query-doc relevance sets) | **UNMEASURED / UNTESTED** |
| **Retrieval Recall** | Recall@k ($R@k$, $k \in \{1, 3, 5\}$) | Not implemented (0 loc) | None (no query-doc relevance sets) | **UNMEASURED / UNTESTED** |
| **Ranked Retrieval** | Mean Reciprocal Rank ($MRR$) | Not implemented (0 loc) | None (no ranked relevance judgements) | **UNMEASURED / UNTESTED** |
| **Summarization Overlap** | ROUGE-1, ROUGE-2, ROUGE-L | Not implemented (0 loc) | None (no human reference summaries) | **UNMEASURED / UNTESTED** |

### Scientific Boundary & Transparency Rationale:
1. **Absence of Ground-Truth Evaluation Sets**: Evaluating $P@k$, Recall@k, and $MRR$ requires curated information retrieval test collections consisting of queries matched to exhaustive binary or graded document relevance assessments (e.g. TREC/BEIR format). Clarion does not contain such collections.
2. **Absence of Reference Summaries**: Evaluating $ROUGE\text{-}N$ requires gold-standard human reference summaries for document sections. No reference summaries exist in `data/ground_truth/` or elsewhere.
3. **Zero Implementation in Codebase**: AST parsing and ripgrep searches across `Clarion-Backend/` confirm that `rouge`, `mrr`, `precision_at_k`, and `recall_at_k` functions do not exist in the codebase.
4. **Quarantine of Unsupported Claims**: Any academic manuscript or documentation asserting empirical quantitative scores for $P@k$, Recall@k, $MRR$, or ROUGE on Clarion is scientifically unfounded and must remain quarantined as **UNMEASURED**.

---

## 6. System Latency Claims & Empirical Profiling

Clarion's performance reporting distinguishes between analytical proportional batch allocation models and verified wall-clock microsecond component benchmarks.

### A. Proportional Batch Allocation Model (`generate_latency_graph.py`)
The exploratory script `Clarion-Backend/generate_latency_graph.py` generates a curve depicting latency as a function of token context size.
- **Implementation Mechanism**: The script inspects candidate relations from an aggregate batch processing session ($T_{\text{session}} = 92.5\text{ seconds}$) and allocates time strictly proportionally based on character-based token proxy count ($\text{tokens} = \text{len}(\text{chunk\_context}) / 4$):
  $$t_i = 92.5 \cdot \frac{\text{tokens}_i}{\sum_{j} \text{tokens}_j}$$
- **Analytical Scope**: As documented in the script's scientific header (lines 4–9), these values represent an **analytical scaling allocation**, NOT individual per-relation wall-clock latency measurements. They illustrate how computational budget scales across varying context lengths under uniform batch processing.

### B. Empirical Wall-Clock RAG Component Profiling
True wall-clock component latencies were measured empirically on an Ubuntu 24.04 Linux host using high-resolution monotonic microsecond timers (`time.perf_counter()`) over $N=5$ realistic academic queries on the canonical DAA textbook benchmark document (`a945d65e-f069-41f9-abb2-e1a7e774d559`, 40 indexed chunks):

| Pipeline Stage | Component / Method | Average Latency | Min Latency | Max Latency | Verification Method |
|---|---|:---:|:---:|:---:|---|
| **Query Embedding** | `BAAI/bge-large-en-v1.5` (1024-dim, CPU, warm) | **157.33 ms** | 152.42 ms | 168.61 ms | Monotonic timer around `embed_query()` |
| **Vector Search** | FAISS `IndexFlatL2` ($k=4$ over 40 vectors) | **0.10 ms** | 0.09 ms | 0.11 ms | Monotonic timer around `vstore.search()` |
| **Context Assembly** | `retrieval_service.retrieve_with_context()` | **497.82 ms** | 449.25 ms | 536.90 ms | Monotonic timer around ranking & assembly |
| **Synthesis (Fallback)** | Deterministic fallback extraction engine | **0.03 ms** | 0.02 ms | 0.08 ms | Monotonic timer around `sanitize_llm_text()` |
| **Full HTTP E2E** | FastAPI `POST /query/{doc_id}` | **780.31 ms** | 715.64 ms | 804.22 ms | Starlette TestClient roundtrip |

### C. Live LLM Inference Status
- **Environment Isolation**: In containerized environments with port 11434 network isolation (e.g. WSL2 container without host port forwarding), live Ollama LLM inference is unreachable (`[Errno 111] Connection refused`).
- **Resilient Fallback Guarantee**: The system automatically activates its deterministic fallback synthesis engine, which delivers verified sub-second response times (**780.31 ms** end-to-end HTTP response) with 100% text grounding and zero hallucination risk.

---

## 7. Evaluation Methodology & Pre-Registered Hypotheses

When the human gold dataset is imported, benchmarking will execute using:

### A. Document-Isolated Evaluation
- Splits clustered by `document_id` to guarantee zero test leakage.
- 5 repeated splits (seeds 42–46) with 60% train, 20% validation, 20% test.
- Thresholds calibrated strictly on validation splits.

### B. Statistical Significance Protocol
- **Paired Document-Cluster Bootstrap (2,000 iterations):** Resamples entire document clusters with replacement. Percentile bootstrap CI — always bounded within observed data range.
- Computes empirical 95% CIs and one-tailed $p$-values for:
  $$\Delta F_1 = F_{1,\text{calibrated}} - F_{1,\text{baseline}}$$
  $$\Delta \text{ECE} = \text{ECE}_{\text{calibrated}} - \text{ECE}_{\text{baseline}}$$
- Statistical significance claimed if $p < 0.05$ and $95\%\text{ CI}$ excludes zero.

### C. Pre-Registered Hypotheses (To Be Tested on Real Data)
1. **$H_1$ (Calibration):** Platt-calibrated multi-signal validation will achieve statistically significant ECE reduction ($\Delta \text{ECE} < -0.10, p < 0.05$) vs. uncalibrated LLM confidence.
2. **$H_2$ (Hallucination Mitigation):** Multi-signal evidence verification will reduce False Accept Rate (FAR) by ≥ 25% vs. generator confidence ($p < 0.05$).
3. **$H_3$ (Orthogonal Synergy):** Leave-one-signal-out ablation will show lexical evidence support and graph structural features provide orthogonal, non-redundant predictive power.

---

## 8. Operational Commands

```bash
# 1. Annotation batch already exported (200 candidates, seed=42):
#    Clarion-Backend/data/ground_truth/annotation_queue_batch1.json

# 2. Validate completed annotations:
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/core/benchmarking/annotation_tool.py validate \
  --file Clarion-Backend/data/ground_truth/annotation_queue_batch1.json

# 3. Validate overlap annotations:
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/core/benchmarking/annotation_tool.py validate \
  --file Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json

# 4. Import adjudicated labels into GroundTruthDataset:
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/core/benchmarking/annotation_tool.py import \
  --input Clarion-Backend/data/ground_truth/annotation_queue_batch1.json \
  --output-gold Clarion-Backend/data/ground_truth/human_gold_v1.json \
  --db-path Clarion-Backend/data/relation_dataset.db

# 5. Execute canonical leak-free benchmark on real gold dataset:
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/run_canonical_benchmark.py \
  --dataset Clarion-Backend/data/ground_truth/human_gold_v1.json \
  --runs 5 --seed 42

# 6. Synthetic regression ONLY (not a research result):
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/run_canonical_benchmark.py \
  --dataset Clarion-Backend/data/ground_truth/synthetic_benchmark_fixture.json \
  --runs 5 --seed 42
```
