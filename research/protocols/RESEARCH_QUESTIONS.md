# Research questions and preregistered evaluation plan

**Protocol status:** proposed; no confirmatory retrieval or relation-quality results exist. Freeze query sets, split, software/model versions, and analysis code before outcome inspection.

## RQ1 — Does source-coordinate provenance survive the document pipeline?

- **Motivation / gap:** Clarion currently retains chunk-local segment character spans and segment-to-chunk IDs. PDF page coordinates and a mapping across extraction/normalization are absent. A verifiable coordinate chain could make retrieved evidence auditable; novelty is unestablished and must be compared with source attribution/provenance systems.
- **Hypothesis:** A page-aware provenance implementation can map every retained segment to one or more valid page-local source intervals with no false interval, and can reconstruct normalized text only where transformations are lossless.
- **IV:** legacy pipeline versus provenance-aware pipeline; PDF type (text, mixed, scanned); normalization transform.
- **DV:** mapped-segment rate, interval validity, source-text agreement, unmapped/ambiguous rate, added storage and latency.
- **Unit/data:** segment nested within document; use all text-bearing internal PDFs plus manually checked page samples. Cluster uncertainty by document. Do not treat segments as independent.
- **Baselines:** current chunk-local coordinates; independent manual page/quote tracing.
- **Metrics:** exact source-span precision/recall against checked references; page assignment accuracy; coverage; character/token reconstruction rate; transform-specific ambiguity.
- **Statistics:** document-clustered bootstrap 95% CI; paired document-level comparison; report all document-level observations.
- **Acceptance:** every returned segment has an explicit page and source interval or is marked unmappable; zero silently false coordinates in audited references; all non-lossless normalization marked many-to-one/ambiguous. A contribution claim additionally requires comparative literature review and replicated evaluation.
- **Limitations:** text extraction offsets are parser coordinates, not visual glyph coordinates; reading order, ligatures, tables, OCR and multi-column layouts can defeat exact character mapping.

## RQ2 — How do chunking choices affect evidence retrieval and traceability?

- **Hypothesis:** chunk size/overlap affect retrieval and evidence coverage; direction is unknown.
- **IV:** 256/384/512/768-word target, overlap 0/50, plus current section/minimum/tail merge behavior; exact variants must be recorded.
- **DV:** Recall@k, MRR, nDCG@k (graded judgments), source-span coverage/duplication, segment count, latency.
- **Unit/data:** query-document judgments, queries grouped by source document. Current 11-PDF corpus is insufficient for broad generalization; use a clearly separate licensed benchmark or curated, independently judged set.
- **Baselines:** BM25; BGE dense; hybrid; current configured retriever; reranker only if fixed and available.
- **Statistics:** paired query-level comparisons with document-clustered bootstrap; Holm correction across predeclared primary comparisons.
- **Acceptance:** at least 100 independently authored/judged queries from ≥10 source documents for internal pilot; report uncertainty and raw query results. This threshold is a pilot criterion, not a universal publication standard.
- **Limitations:** synthetic/query author leakage, small corpus, domain bias, and judged-pool incompleteness.

## RQ3 — Does graph augmentation improve evidence retrieval?

- **Hypothesis:** graph expansion may improve multi-hop evidence recall, with potentially lower precision; no directional result assumed.
- **IV:** dense/BM25/hybrid, with graph expansion off/on; identical corpus, query, budget, and reranker.
- **DV:** Recall@k, MRR, nDCG@k, evidence support and latency.
- **Unit/data:** query; source-document clustered; multi-hop questions with answer-support evidence. The existing internal corpus has no gold retrieval judgments.
- **Baseline:** strongest non-graph retriever tuned only on development split.
- **Statistics/acceptance:** paired clustered bootstrap and Holm correction; report practical effect and CI. Success only if predeclared primary metric improves without an unacceptable precision/support loss.
- **Limitations:** graph construction errors and dataset-specific entity linking.

## RQ4 — What is the quality of candidate relations and evidence validation?

- **Hypothesis:** human labels will reveal error modes; whether a validator improves precision at fixed recall is unknown.
- **IV:** raw extractor versus validation/filtering; annotator adjudication status.
- **DV:** relation precision/recall/F1, class-specific errors, evidence support, calibration only if calibrated against labels.
- **Unit/data:** candidate relation. Sample from the canonical 1,562-candidate DB with documented random seed and inclusion criteria. Existing 200/50 queues have complete overlap of overlap IDs with primary; they are not evidence of two completed independent annotation passes.
- **Baseline:** raw candidates and majority/adjudicated human label.
- **Statistics:** confusion matrix, Wilson intervals, paired bootstrap by source document; Cohen's kappa or Krippendorff's alpha on true independent overlap, with prevalence caveats.
- **Acceptance:** two independent annotators, blinded to system score, complete guideline and adjudication; no agreement statistic until labels exist.
- **Limitations:** current annotation category schema may not distinguish unsupported, incomplete, and contradictory evidence adequately.

## RQ5 — What resource cost does each pipeline stage incur?

- **Hypothesis:** no directional hypothesis; characterize costs under a frozen environment.
- **IV:** stage, document size/type, cold versus warm state.
- **DV:** wall latency, throughput, peak RSS/VRAM, index bytes, token counts.
- **Unit:** repeated run nested in document; report cold/warm separately.
- **Protocol:** ≥5 warm runs per document/configuration, ≥3 cold starts where feasible, monotonic clock, fixed hardware/software/model and concurrency; p50/p95 and raw samples.
- **Statistics:** descriptive quantiles and document-stratified bootstrap; do not infer causal speedups from uncontrolled runs.
- **Acceptance:** full environment manifest and raw timing samples. Existing smoke timings are not a benchmark.

## Common controls

Keep source-document splits disjoint; freeze code commit/hash, configuration, seeds, queries and judgments; maintain raw versus derived results; preregister one primary metric per RQ. A chunk is not an independent replicate when several chunks come from one document. Missing human participation, a valid benchmark, or GPU/host resource access is recorded as BLOCKED rather than imputed.
