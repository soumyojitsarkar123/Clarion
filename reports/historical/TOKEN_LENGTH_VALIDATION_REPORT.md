# Token-Length Validation Report

## Status: PARTIALLY VERIFIED

`EmbeddingService` now obtains the loaded SentenceTransformer's `get_max_seq_length()` (falling back only to the model/tokenizer exposed value), reserves its actual special-token budget, and splits source chunks at tokenizer offset boundaries. Each embedding input is checked with special tokens included before encoding. The source chunk remains authoritative in `chunks`; deterministic segment IDs (`<source_chunk_id>:seg:<n>`) are recorded in `embedding_segments` with source IDs and character spans.

The new regression tests in `Clarion-Backend/tests/test_embedding_segments.py` cover exact-limit and over-limit inputs, exact source reconstruction, no overlap, and per-segment token ceilings.

## Verification record

- Static compilation: `C:\Users\VICTUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile services\embedding_service.py services\retrieval_service.py services\processing_pipeline.py` — passed on 2026-11-08.
- Runtime pytest: the focused embedding/segment suite passed (33 tests) in `.venv-validation`.
- Live tokenizer limit: verified as 512 tokens with `BAAI/bge-large-en-v1.5` on CPU; over-limit and 1,302-token inputs were segmented and embedded successfully.

## Scope boundary

When the SentenceTransformer is unavailable, deterministic hashing fallback remains unbounded and no claim of transformer token validation is made. A production rerun must load `BAAI/bge-large-en-v1.5`, ingest an over-limit source chunk, and inspect `embedding_segments` plus the generated vector index.
