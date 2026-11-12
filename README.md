# Clarion

Clarion is a local-first document intelligence application with PDF/DOCX ingestion, text chunking, dense retrieval, relation candidates, knowledge maps, summaries, and an optional LLM-backed query path. It includes tools for relation annotation and benchmark experiments. The repository contains active software, source material, historical engineering evidence, and research artifacts; these are kept in separate locations.

## Readiness

Current classification: **PARTIALLY VERIFIED — NOT RELEASE READY**. The final backend suite passed under Ubuntu 24.04.3/WSL with Python 3.12.3: 324 passed and 2 skipped. The Windows test runner still has a reproducible asyncio socketpair initialization hang. Live provider generation and browser E2E remain unverified; security lacks authentication and rate limiting, and research quality evaluation remains blocked by the absence of gold labels and valid qrels. See [current status](CLARION_CURRENT_STATUS.md), [final gate evidence](research_artifacts/processed/final_gate_2026-11-12.json), and [software release readiness](reports/current/SOFTWARE_RELEASE_READINESS.md).

## Architecture

- `Clarion-Backend/`: FastAPI backend, SQLite persistence, FAISS indexing, retrieval, graph construction, LLM adapters, annotation and benchmark code.
- `frontend/`: static browser application.
- `docs/`: source PDFs and technical documents. Source paths are preserved for existing scripts.
- `research/`: current protocols, audits, machine-readable indexes, annotation guidance, and research draft/archive.
- `research_artifacts/`: run-specific raw and processed observations, figures, scripts, isolated indices, and recoverable snapshots.
- `reports/current/`: authoritative engineering readiness reports.
- `reports/historical/`: prior audit records, verification evidence, and recovery artifacts.
- `scripts/` and `tests/`: repository-level verification helpers and cross-project tests.

The default controls remain `BAAI/bge-large-en-v1.5` for dense embeddings and `qwen3.5:4b` for local generation. Chunk defaults remain 512 words, 50-word overlap, and 100-word minimum. These are configuration controls, not evidence that the settings are optimal.

## Run the backend

```powershell
cd Clarion-Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Configuration is supplied through environment variables or a local `.env` based on `.env.example`. Keep secrets and runtime databases local; do not commit them.

## Run backend tests

```powershell
cd Clarion-Backend
python -m pytest
```

The recorded complete run used Python 3.12.14 and an isolated pytest base directory. See `research_artifacts/raw/software_readiness_backend_full_suite.log`, the post-organization rerun log at `research_artifacts/raw/software_post_cleanup_backend_suite.log`, and [the verification report](reports/current/SOFTWARE_FINAL_VERIFICATION_REPORT.md). Live-provider tests require the configured local Ollama service and model. Browser E2E requires Playwright.

## Evidence and data handling

- Original PDFs remain in `docs/`.
- User and runtime databases remain under their existing `data/` locations; never run research indexing against production stores.
- Raw and processed research observations are separated under `research_artifacts/`.
- Model IDs, local digests, dependency versions, and smoke measurements are recorded in `research_artifacts/processed/`.
- A verified pre-clean snapshot preserves the worktree state before the report/archive organization. See `research_artifacts/engineering-frozen-presnapshot-2026-11-11/README.md`.

## Documentation map

- [Current status](CLARION_CURRENT_STATUS.md)
- [Current software reports](reports/current/README.md)
- [Historical reports and recovery evidence](reports/historical/README.md)
- [Research evidence index](research/FINAL_EVIDENCE_INDEX.md)
- [Repository structure](reports/current/FINAL_REPOSITORY_STRUCTURE.md)
