# Live application end-to-end report

**Classification: PARTIALLY VERIFIED.** A real multi-page PDF passed the isolated extraction/chunk/BGE/FAISS/source-coordinate workflow. The real HTTP upload → frontend query → configured Qwen answer → rendered citation workflow was not run.

## Completed isolated workflow

Using `docs/DAA FINAL.pdf` (87 pages), direct Clarion services extracted and persisted page text, normalized with a v2 source map, generated 44 chunks and 100 BGE token-bounded segments, produced 100 real BGE vectors, wrote a FAISS index, and searched the index. All vectors mapped to source intervals and verified page-local bounds; the retrieved segment mapped to pages 47–48. BGE commit revision, input PDF SHA-256, DB/index manifests, and page offsets are recorded in [`isolated_provenance_e2e_manifest.json`](../../research_artifacts/isolated-provenance-e2e-2026-11-12/isolated_provenance_e2e_manifest.json).

The actual configured generation model remains `qwen3.5:4b`; the local Ollama manifest has a model-layer digest, but connection to `127.0.0.1:11434` is blocked by the host policy. The real provider did not generate an answer. FastAPI could not be exercised through its async TestClient because of the reproduced loop initialization stall. Missing evidence, failure injection for FAISS/database writes, and recovery from an interrupted upload were not verified end to end.

All integration storage was redirected to `research_artifacts/isolated-provenance-e2e-2026-11-12/data`; production Clarion DB files were not opened by the workflow.
