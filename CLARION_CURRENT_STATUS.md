# Clarion current status

**Updated:** 2026-11-12 (Asia/Kolkata)  
**Classification:** **PARTIALLY VERIFIED — NOT RELEASE READY**  
**Git:** `main` at `ebca6ed0125c7131c05c945620b23a1970844b92`; current user/engineering changes remain uncommitted.

## Latest engineering blocker-resolution run

- Source provenance now persists PDF extracted text by page, versioned normalization maps, disjoint chunk/segment source spans, model-scoped FAISS vector maps, retrieved citation fields, and frontend citation rendering. Coordinate meaning is extracted-text/page-local, not visual glyph coordinates.
- Isolated real PDF processing of `docs/DAA FINAL.pdf` passed: 87 pages, 44 chunks, 100 BGE segments/vectors, all vector maps had verifiable source intervals, and an isolated FAISS search returned intervals on pages 47–48.
- Final focused regression set: **21 passed**. Fresh/legacy/rollback migrations, token/provider boundaries, segmentation and citation tests are included.
- The full backend suite passed in Ubuntu 24.04.3 under WSL with Python 3.12.3: **324 passed, 2 skipped** (325 collected, 1 collection skip), 81.17 seconds. The pinned CPU validation environment passed `pip check`. The separate Windows run still stalls during asyncio socketpair initialization; a minimal reproduction is captured in the raw evidence.
- The matching Linux test command and environment are recorded in `research_artifacts/processed/linux-test-environment-2026-11-12.json`. The GitHub Actions workflow was updated to use CPU-only Torch, but the remote workflow was not run.
- Ollama model files exist locally, but loopback port 11434 is denied in this session. Two live Ollama tests skip; live provider generation and actual token usage remain unverified.
- No Python Playwright package or browser executable was available. Live frontend E2E and screenshots remain unverified.
- Experiments redirected document/chunk/vector storage to `research_artifacts/isolated-provenance-e2e-2026-11-12/data`. Production databases and model defaults were not changed. A pre-run database hash baseline was not captured.
- An isolated backup/restore and interrupted-index recovery rehearsal passed without accessing production data. Security review still identifies missing authentication and rate limiting; do not expose the unauthenticated server beyond a trusted local environment. Research quality gates remain blocked because no gold labels or valid retrieval qrels are available.
- Current source/worktree recovery material and byte-for-byte SQLite file copies are preserved at `research_artifacts/engineering-final-blocker-resolution-2026-11-12/` (directory snapshot, no new ZIP).

## Readiness decision

Software release readiness remains **NOT READY**. Research and publication readiness are separate and are not claimed. The existing readiness ZIP was preserved; no new distribution ZIP was created because the final verification gate did not pass.

## Current reports

See `reports/current/README.md` and the detailed reports in that folder. The final Linux full-suite capture is `research_artifacts/raw/backend-full-suite-linux-py312-final-2026-11-12.log`; gate, security, research audit, and recovery evidence are indexed in `research_artifacts/processed/final_gate_2026-11-12.json`. The isolated PDF/BGE/FAISS manifest is `research_artifacts/isolated-provenance-e2e-2026-11-12/isolated_provenance_e2e_manifest.json`.
