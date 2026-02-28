# Research Readiness Upgrade Report

## Scope
This pass reviewed the backend for publish-grade concerns: security, reproducibility, operational reliability, and data safety.

## High-Risk Weaknesses Found
1. Insecure deserialization paths
- Graph and vector metadata used pickle loading in runtime code paths.
- Impact: arbitrary code execution risk if persisted artifacts are tampered with.

2. Overly permissive CORS defaults
- API allowed `*` origins with credential support settings.
- Impact: weak production boundary and browser-side policy ambiguity.

3. SQLite lock contention risk under async/background load
- Multiple services opened default SQLite connections without busy timeout/WAL mode.
- Impact: intermittent `database is locked` failures in concurrent jobs.

4. Runtime import fragility
- `sentence_transformers` imported at module import time; app startup failed when optional ML dependency versions mismatched.
- Impact: whole API could fail to start due optional subsystem dependency.

5. Analysis API options not applied
- `run_evaluation`, `generate_hierarchy`, and sync wait options were defined but effectively ignored.
- Impact: misleading API behavior; poor experiment control.

6. Information leakage on server errors
- Several routers returned raw internal exception messages.
- Impact: exposes internals and complicates stable external API behavior.

## Upgrades Implemented
1. Safe graph and vector serialization
- Added `utils/graph_store.py` using NetworkX node-link JSON persistence.
- Switched pipeline graph read/write to JSON.
- Added guarded legacy pickle fallback behind `allow_legacy_pickle_loading` config (default `False`).
- Switched vector metadata from pickle to JSON with optional legacy fallback.

2. CORS hardening via configuration
- Replaced wildcard CORS defaults with configurable origin allowlist.
- Added `cors_allowed_origins` and `cors_allow_credentials` settings.

3. SQLite reliability improvements
- Added shared connector `utils/sqlite.py` applying:
  - `journal_mode=WAL`
  - `busy_timeout`
  - `foreign_keys=ON`
- Integrated connector in core services:
  - `document_service.py`
  - `chunking_service.py`
  - `embedding_service.py`
  - `knowledge_map_service.py`
  - `relation_dataset_service.py`
  - `background_service.py`

4. Startup resilience improvements
- Converted `services/__init__.py` to lazy imports.
- Deferred `sentence_transformers` import to runtime model-load path in `embedding_service.py`.

5. API behavior correctness improvements
- `analyze` endpoint now honors request controls:
  - stage skipping (`run_evaluation`, `generate_hierarchy`)
  - synchronous execution (`wait_for_completion`, `timeout_seconds`)

6. Error response hardening
- Replaced raw exception details with stable generic 500 messages in status/dataset/analyze routes.

7. Input validation improvements
- Added bounds to query payload:
  - `query` length limit
  - `top_k` range guard

8. Artifact readiness accuracy
- Status endpoint now checks real knowledge map existence and new graph artifact format.

## Validation Run
- `python -m compileall -q .` passed.
- Import checks passed:
  - `from main import app`
  - `from services.processing_pipeline import ProcessingPipeline`
  - `from services.background_service import BackgroundService`
  - `from services.relation_dataset_service import RelationDatasetService`

## Remaining Recommendations (Next Iteration)
1. Add test suite for:
- Pipeline stage skipping behavior
- Graph JSON backward compatibility
- SQLite concurrency under parallel jobs

2. Standardize exception taxonomy
- Replace broad `except Exception` blocks with typed failures where possible.

3. Add reproducibility metadata for experiments
- Persist model/version hashes, prompt versions, and run seeds with each analysis artifact.
