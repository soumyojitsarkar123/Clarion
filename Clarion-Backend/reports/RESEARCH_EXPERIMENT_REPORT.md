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
| Candidate relations | **1,046** |
| Source documents | **32 distinct documents** |
| Human-annotated labels | **0 (zero)** |

All 1,046 `is_valid` fields are `NULL` in `relation_dataset.db`.

```sql
SELECT COUNT(*) FROM relation_dataset;                           -- 1,046 total candidates
SELECT COUNT(DISTINCT document_id) FROM relation_dataset;         -- 32 real documents
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
| Inter-annotator agreement (κ) | `core/benchmarking/agreement.py` | ✅ Cohen's + Fleiss' κ |
| Gold dataset contract | `core/benchmarking/gold_dataset_contract.py` | ✅ Field eligibility enforced |
| Annotation protocol | `data/ground_truth/ANNOTATION_PROTOCOL.md` | ✅ Gold contract + IAA status |
| Canonical benchmark CLI | `run_canonical_benchmark.py` | ✅ Git provenance + integrity notice |
| Provenance (git commit, versions) | `run_canonical_benchmark.py` | ✅ SHA-256 + software versions |
| Canonical synthetic benchmark | `data/benchmarks/bench_08cb7efa7e08.json` | ✅ 5 seeds [42–46] |
| Annotation batch (200 records, seed=42) | `data/ground_truth/annotation_queue_batch1.json` | ✅ Exported |

---

## NOT YET POSSIBLE (Blocked — Human Annotation Required)

The following items **cannot be completed** without human-annotated gold labels:

- ❌ **Training a relation validity classifier on real data** — zero labeled records
- ❌ **Reporting real-world F1, precision, recall** — no gold standard
- ❌ **Reporting calibration error (ECE/MCE/Brier) on real data** — no gold standard
- ❌ **Testing H1 (Calibration)**: Platt ECE reduction vs. LLM confidence
- ❌ **Testing H2 (Hallucination Mitigation)**: FAR reduction ≥ 25%
- ❌ **Testing H3 (Orthogonal Synergy)**: Leave-one-signal-out ablation significance
- ❌ **Computing inter-annotator agreement (Cohen's κ, Fleiss' κ)** — requires ≥2 annotators labeling the same batch
- ❌ **Promoting annotation batch to ADJUDICATED GOLD** — requires completion + IAA ≥ 0.70
- ❌ **Any publication-grade empirical claim** — none can be made

---

## 1. Ground Truth Reality: Current Real-Data Status

A rigorous database audit of `Clarion-Backend/data/relation_dataset.db` was conducted using direct SQL inspection. See table above.

---

## 2. Closed-Domain Human Annotation Protocol

To convert the 1,046 candidates into a publication-quality gold dataset, Clarion implements a formal adjudication protocol.

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

## 5. Evaluation Methodology & Pre-Registered Hypotheses

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

## 6. Operational Commands

```bash
# 1. Annotation batch already exported (200 candidates, seed=42):
#    Clarion-Backend/data/ground_truth/annotation_queue_batch1.json

# 2. Validate completed annotations:
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/core/benchmarking/annotation_tool.py validate \
  --file Clarion-Backend/data/ground_truth/annotation_queue_batch1.json

# 3. Import adjudicated labels into GroundTruthDataset:
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/core/benchmarking/annotation_tool.py import \
  --input Clarion-Backend/data/ground_truth/annotation_queue_batch1.json \
  --output-gold Clarion-Backend/data/ground_truth/human_gold_v1.json \
  --db-path Clarion-Backend/data/relation_dataset.db

# 4. Execute canonical leak-free benchmark on real gold dataset:
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/run_canonical_benchmark.py \
  --dataset Clarion-Backend/data/ground_truth/human_gold_v1.json \
  --runs 5 --seed 42

# 5. Synthetic regression ONLY (not a research result):
PYTHONPATH=Clarion-Backend .venv_linux/bin/python \
  Clarion-Backend/run_canonical_benchmark.py \
  --dataset Clarion-Backend/data/ground_truth/synthetic_benchmark_fixture.json \
  --runs 5 --seed 42
```
