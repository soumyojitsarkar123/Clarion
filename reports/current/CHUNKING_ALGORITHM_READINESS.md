# Chunking algorithm readiness

**Status:** PARTIALLY VERIFIED as an operational algorithm; source-coordinate readiness NOT VERIFIED.

## Current algorithm

`Clarion-Backend/services/chunking_service.py` normalizes input, detects section headings, splits section text by whitespace-delimited words, uses configured `chunk_size=512`, `chunk_overlap=50`, `min_chunk_size=100`, and advances by 462 words. Undersized tails are merged; final `position_index` is assigned cumulatively by retained chunk word counts. BGE embedding segmentation subsequently uses its actual available tokenizer and special-token limit. The 512-word chunk target is not a 512-token embedding limit; embedding segments enforce a separate model input ceiling.

## Measured and unmeasured

Prior chunk configuration sensitivity includes four word sizes (256/384/512/768) and two overlap settings (0/50), applied to the captured readable corpus. It measures chunk/segment counts and token lengths, not relevance, evidence recall, quality, latency, or source coverage. The 100-word minimum/tail merge and section rules mean the usual sliding-window count formula is only approximate.

No paragraph/sentence-boundary or semantic chunker has been promoted. No measured basis supports changing defaults. Do not call chunking a model.

## Provenance and next gates

Embedding `char_start/end` are relative to a chunk. Extraction/normalization discard PDF page and source-coordinate mappings; merge/reindex also preclude treating `position_index` as a source offset. Therefore character/word source coverage and duplication cannot be reported reliably from the current pipeline. Implement a transform-aware source interval relation first, then test page boundary, sentence/table/math integrity and compare algorithms on identical documents with judged retrieval queries. Until that work, retain current chunk settings as a control and classify the algorithm as **PARTIALLY VERIFIED**.
