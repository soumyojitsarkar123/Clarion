# Research Validation Report

**Date:** 2026-11-11 (Asia/Kolkata)  
**Initial Git identity:** branch `main`, HEAD `ebca6ed0125c7131c05c945620b23a1970844b92`. The worktree was already extensively modified with untracked evidence. A pre-research snapshot was created; no reset or destructive cleanup occurred.

## Findings

Clarion’s current PDF extraction and indexing path was exercised in a dedicated isolated directory. Ten of eleven PDFs had extractable text; the 1,151-page `AI_Book.pdf` returned no text and was rejected. The readable corpus produced 86 retained chunks, 170 tokenizer-safe embedding segments and 170 BGE vectors. All per-document SQLite segment/embedding and FAISS counts aligned: True. These are processing results, not evidence of retrieval relevance or answer quality.

## Scientific claim register

| Claim | Evidence and dataset | Metric / experiment | Result | Limitations | Status |
|---|---|---|---|---|---|
| PDF extraction and chunk counts reflect current code | `pdf_page_extraction.csv`, `pdf_chunk_counts.csv`; all 11 PDFs | Actual PyPDF2 extractor and chunk service methods; per-page count | 10 text-bearing PDFs; 86 retained chunks | No independent reference text; one source unextractable | VERIFIED for observed corpus |
| BGE segmentation meets sequence limit | `embedding_segment_token_lengths.csv`; 170 segments | Local fast BGE tokenizer, special tokens included | 170/170 <=512; BGE inference/index succeeded | One model/version and corpus | VERIFIED |
| Chunk size changes record counts | `chunking_configuration_sensitivity.csv` | 4 sizes x 2 overlaps; exact chunk/merge/segment code | Counts vary; larger chunks often create fewer raw chunks | No retrieval relevance/latency measurement | VERIFIED for counts only |
| Tokenizer segmentation improves relevance | No relevance labels | P@k, Recall@k, MRR not computable | No result | Query benchmark absent | NOT EVALUATED |
| Knowledge graph improves retrieval | No query relevance dataset | Paired retrieval experiment not run | No result | Frozen held-out queries absent | BLOCKED |
| Relation extraction quality is measurable | Canonical backend DB: 1562 candidates, 1562 unlabeled; queue exports 200/50 | Precision/recall/F1; kappa on 50 double labels | No human labels | Annotation work not completed | BLOCKED |
| Pipeline cost is characterized | Isolated run has no controlled timing samples | Median/p95, throughput, repeated runs absent | No result | Requires controlled benchmark | NOT EVALUATED |

## Corpus evidence

The only partially relevant PDF is the generic IR overview. Pages 39-40 describe a judgment-based evaluation procedure and P/R/F1 definitions; they contain no measured benchmark values. No source supplies Clarion results. Source-level classification and extraction limits are in `PDF_SOURCE_REGISTER.md` and `PDF_RESEARCH_EVIDENCE.md`.

## Readiness

Human relation annotation, independent retrieval judgments, graph/retrieval ablations and controlled performance measurement remain incomplete. A complete manuscript is not supported; `appendix_report.md` remains an appendix. Main manuscript and publication readiness are **BLOCKED** pending valid evaluation evidence and an author-approved contribution/venue scope. Do not claim novelty or improvements from this processing run.

## Repository state and reproduction

Cleanup was inventory-led. Reproducible root caches and two empty backend pytest temp directories were removed only after recording sizes/SHA-256, moving them to quarantine, and verifying hashes. No source PDF, unique evidence, database, validation environment, or application source was removed. The existing engineering baseline (304 passed, 2 skipped, prior targeted/frontend checks) remains a historical snapshot and was not rerun. Production database paths were not used for indexing.

The relationship annotation instructions and blank capture schema are in `research_artifacts/annotation/RELATION_ANNOTATION_PROTOCOL.md` and `annotation_template.csv`.

See `research_artifacts/scripts/` for extraction, chunk simulation, isolated indexing and chart scripts. Machine-readable outputs are under `research_artifacts/processed/`; the isolated FAISS/SQLite index is under `research_artifacts/isolated_pdf_index_2026-11-11/`.
