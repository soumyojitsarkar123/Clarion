# Source provenance integration test report

**Classification: PARTIALLY VERIFIED.** Isolated migrations and a real 87-page PDF → BGE → FAISS → source-span search passed. Live HTTP upload/query, frontend rendering in a browser, interrupted FAISS/database cross-resource recovery, and backup restore were not verified.

## Isolated test results

Command from `Clarion-Backend`:

```powershell
.\.venv-validation\Scripts\python.exe scripts\run_backend_tests.py tests\test_source_provenance_chain.py -q
```

Result in the final focused run: provenance test module passed. It covers fresh and legacy schemas, idempotent migration, injected transactional rollback, normalization replacement/deletion spans, page-split chunk-to-vector metadata, model-scoped retrieval citation fields, reindex-map replacement, deletion, and rejecting query embeddings from a model that does not match the selected FAISS index.

## Real PDF integration evidence

The script [`run_isolated_provenance_e2e.py`](../../research_artifacts/scripts/run_isolated_provenance_e2e.py) used `docs/DAA FINAL.pdf` and wrote only under `research_artifacts/isolated-provenance-e2e-2026-11-12/`.

- Source: 87 pages, all 87 with extracted text; source PDF SHA-256 `de44cb33892fea9906fe119d0fad697b51aee68c4f13949af7b18c8a3b7e3efe`.
- Pipeline: 44 retained chunks, 100 tokenizer-bounded segments, 100 BGE vectors, all 100 with source spans; FAISS search returned a segment mapped to pages 47–48.
- Model identity: `BAAI/bge-large-en-v1.5@d4aa6901d3a41ba39fb536a557fa166f842b0e09`.
- Page and page-local offset bounds were checked against the persisted extracted page text. The database and FAISS index SHA-256 values are recorded in the final verification report.
- This exercised the services directly with isolated storage, not the running FastAPI/HTML application. The returned passage excerpt was not visually compared with a rendered PDF page.

## Recovery and remaining tests

Fresh/legacy/interrupted schema migration behavior is covered by injected isolated SQLite failure. Reindex and deletion map behavior is covered with a fake index in unit tests and real FAISS creation/search was exercised in the integration script. FAISS write failure after SQLite commit, process termination between FAISS and SQLite updates, production migration rehearsal, and backup restore were not run. No production database was opened through Clarion/SQLite or modified; current files were raw-read for hashes and copied byte-for-byte into the recovery snapshot.
