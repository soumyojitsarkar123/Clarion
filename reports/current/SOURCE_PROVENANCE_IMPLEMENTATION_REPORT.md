# Source provenance implementation report

**Classification: PARTIALLY VERIFIED.** The persisted chain now carries PDF page text, extracted-text offsets, normalization mappings, chunk spans, model-specific segment/vector identity, retrieved evidence references, and frontend citation fields. Visual PDF coordinates, OCR/layout recovery, and live browser citation display are not established.

## Implemented

- PDF pages are extracted and persisted separately with one-based page numbers, page-local extracted text, document-level extracted-text start/end offsets, extraction method/version, and warnings. A page with no usable text has null offsets and an explicit unavailable warning.
- DOCX paragraphs and table rows retain source blocks and extracted-text offsets. PDF table structure, reading order in multicolumn pages, glyph coordinates, and header/footer removal are not inferred; warnings state that these were unavailable or not applied.
- Normalization offset maps are versioned as `clarion-normalization-offset-map-v2`. Copied runs map character-by-character; replacements map to the source token span; inserted and removed material is recorded. NFC normalization and line-end hyphen joining are applied before the map is built.
- Chunking stores disjoint source intervals, page splits, normalized intervals, raw chunk IDs, retained chunk ID, extraction/normalization/chunk versions, and content-to-source mappings. Segment projection keeps page-local intervals split rather than joining across a page boundary.
- `embedding_segments_v2` and `embedding_vector_map` store model ID/revision and vector ID → segment ID → retained chunk ID → raw chunk IDs → document/source intervals. Revisions are resolved from loaded model metadata or the local Hugging Face cache reference when available.
- FAISS index manifests include the model identity. Retrieval rejects an index unless its ordered vector IDs match the model-scoped SQLite map. The query API returns document name, page references, page-local/source offsets, segment and raw chunk identity; the frontend renders those fields and explicitly states when coordinates are unavailable.
- The configured defaults remain `qwen3.5:4b`, `BAAI/bge-large-en-v1.5`, and 512-word chunks.

## Verified limits

The PDF source positions refer to PyPDF2 extracted text, not page glyph locations. OCR-only pages, complex reading order, PDF table cells, and automatic running-header/footer removal remain unsupported or explicitly unverified. Legacy rows retain null provenance and old unversioned indexes are not loaded under a model identity until reindexed.

See [`SOURCE_PROVENANCE_INTEGRATION_TEST_REPORT.md`](SOURCE_PROVENANCE_INTEGRATION_TEST_REPORT.md) and the isolated evidence bundle at [`isolated-provenance-e2e-2026-11-12`](../../research_artifacts/isolated-provenance-e2e-2026-11-12/isolated_provenance_e2e_manifest.json).
