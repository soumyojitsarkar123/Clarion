# Retrieval evaluation protocol

**Status: protocol defined; execution BLOCKED pending valid relevance judgments.**

## Scope and systems

Freeze an evaluation corpus, query set, document-level split, relevance judgments, commit/hash, and configuration before test evaluation. Compare BM25, BGE dense, fixed hybrid fusion, current Clarion retrieval, and optional reranking. Graph augmentation is a separate paired ablation; do not change multiple components simultaneously. Use the same corpus, filters, candidate budget, top-k, and split for every system.

## Dataset and judgments

Internal Clarion PDFs currently have no independently authored query/qrel set. Their heterogeneous educational content is a pilot corpus, not a publication-grade retrieval benchmark. Prefer official BEIR datasets for standardized IR comparison and a separate source-grounded long-document set such as Qasper for document QA, after verifying each dataset's license and task compatibility. Do not report a pooled cross-task score as one benchmark result. For internal evaluation, use document-held-out queries and have two assessors judge retrieved passages blind to system identity; retain qrels and adjudication.

## Metrics

- Precision@k = relevant retrieved items among first k divided by k.
- Recall@k = relevant retrieved items among first k divided by all judged relevant items for the query.
- Reciprocal rank = 1/rank of first relevant item (0 if none); MRR is its query mean.
- nDCG@k only when graded relevance is defined before annotation.
- Also report query-level source-document evidence coverage, duplicate/overlap rate, and provenance validity separately from relevance.

Define unjudged-item handling and empty-qrel exclusions before scoring. Report exact query counts and all per-query outcomes.

## Statistical plan

Query is paired unit; queries are clustered by source document. Report mean, median, SD, and document-clustered bootstrap 95% CI. Compare predeclared primary systems with paired permutation or bootstrap differences; use Holm correction for multiple planned comparisons. No pseudo-replication by treating multiple chunks from one source document as independent. Report negative and tied outcomes.

## Freeze/acceptance

Archive raw runs in TREC run format, qrels, configs, versions, seeds, and generation script. No query tuning on test labels. Acceptance requires no system-specific query leakage, documented dataset license, complete run/qrel files, and reproducible score regeneration. Current result status: **NOT EVALUATED**; existing retrieval dataset document is a protocol draft, not labeled ground truth.
