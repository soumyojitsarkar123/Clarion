# Chunking Forensic Audit

## Method

Every PDF was extracted by Clarion’s current `extract_text_from_pdf`. The count simulation calls the actual `ChunkingService` methods without saving. A separate isolated run calls upload, chunk, embedding and FAISS index services using `research_artifacts/isolated_pdf_index_2026-11-11/`; production data paths were not used.

A **raw chunk** is output from `_chunk_text`; a **retained chunk** is returned by `_merge_undersized_chunks` and persisted; an **embedding segment** is a token-offset split of a retained chunk; an **indexed vector** is a produced BGE embedding added to a FAISS index and mapped in SQLite.

**Table T2. Corpus extraction and chunk-stage counts.** Source: `research_artifacts/processed/pdf_chunk_counts.csv` plus isolated index results.

| PDF | Pages | Pages with / without text | Words | Sections | Raw chunks | Retained | Segments | Indexed vectors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AI_Book.pdf | 1151 | 0 / 1151 | 0 | 0 | 0 | 0 | 0 | 0 |
| DAA FINAL.pdf | 87 | 87 / 0 | 19,830 | 5 | 44 | 44 | 100 | 100 |
| UNIT II — SELF MANAGEMENT SKILLS.pdf | 24 | 24 / 0 | 1,235 | 1 | 3 | 3 | 5 | 5 |
| UNIT III — ENTREPRENEURSHIP SKILLS.pdf | 37 | 37 / 0 | 1,470 | 1 | 3 | 3 | 6 | 6 |
| UNIT III — SOFTWARE & AI WORKFLOW.pdf | 25 | 25 / 0 | 1,397 | 7 | 9 | 4 | 6 | 6 |
| UNIT IV Information Retrieval Overview.pdf | 43 | 43 / 0 | 3,053 | 1 | 7 | 7 | 13 | 13 |
| UNIT IV — NEGOTIATION SKILLS.pdf | 26 | 26 / 0 | 1,414 | 2 | 4 | 4 | 6 | 6 |
| UNIT IV — STORYTELLING, BRANDING & ETHICS.pptx.pdf | 29 | 29 / 0 | 1,085 | 4 | 5 | 3 | 4 | 4 |
| Unit-2 Data exploration and Vizualization.pdf | 10 | 10 / 0 | 3,014 | 1 | 7 | 7 | 13 | 13 |
| Unit-3 Vector Calculus.pdf | 32 | 32 / 0 | 1,974 | 1 | 5 | 5 | 9 | 9 |
| Unit-4 Advanced OLAP Concepts.pptx.pdf | 31 | 31 / 0 | 1,601 | 10 | 10 | 6 | 8 | 8 |

Totals: 1,495 pages; 10 PDFs with text; 86 retained chunks; 170 segments; 170 vectors. All per-document SQLite chunk/segment/embedding counts and FAISS counts align. The 1,151-page `AI_Book.pdf` had zero extracted text and was rejected for fewer than 20 words. OCR was not run.

## Extraction findings

The slide PDFs contain relatively sparse text (roughly 1.2k-3.1k extracted words each); their 3-10 retained chunks reflect extracted text volume, not page count. `DAA FINAL.pdf` yields 19,830 words and 44 retained chunks / 100 segments. The older 60-chunk report was not reproduced with current parser and code. `AI_Book.pdf` has no metadata, generated no text, and pypdf warned at page 1002 about reaching a decompression recovery limit. Rendered pages 1-2 were blank. Image-only versus malformed cannot be distinguished from available evidence.

## Mathematics and limits

Defaults: C=512 words, O=50, step S=C-O=462, minimum chunk size=100. The planning approximation `1 + ceil((N-C)/S)` is not the exact implementation: `_chunk_text` loops over starts spaced by S, skips a tail below the minimum and merges uncovered tail text into the previous chunk. `_merge_undersized_chunks` then merges small chunks with a neighbor, with a 1.25C bound and trailing backward merge. Sections restart chunking, so exact code output is authoritative. Detailed section counts are in `chunking_section_counts.csv`.

Observed descriptive density is 86 retained chunks / (36,073 extracted words / 1,000) = **2.384 retained chunks per 1,000 words** across the ten text-bearing PDFs. Extraction coverage is `extracted characters / independent reference characters`; no independent reference extraction exists, so it is NOT MEASURED. Source coverage would be `|union of original source intervals represented by segments| / source length`; duplication ratio would be `1 - unique source units covered / total represented units`. Neither is measured because reliable original source offsets are not retained across chunk merges/reindexing.

Chunk overlap means lengths cannot be summed to claim source coverage. The chunk service does not retain reliable source word offsets through merge/reindex operations, so source coverage and duplication ratio are NOT MEASURED. Segment character offsets are retained within each source chunk. The local fast tokenizer checked 170 segments including special tokens; maximum length was 512. This is token-limit evidence, not retrieval-quality evidence.

## Status

- Extraction, raw/retained/segment/vector counts: VERIFIED for this checkout and isolated corpus run.
- Token compliance: VERIFIED, 170/170 segments <=512 tokens.
- Independent-reference extraction coverage: NOT EVALUATED.
- Source coverage/duplication: NOT EVALUATED.
- Retrieval, latency, cache effects, OCR completeness: NOT EVALUATED or BLOCKED.

See `CHUNKING_EXPERIMENT_RESULTS.md` and the CSVs in `research_artifacts/processed/`.
