## Final verification addendum — 2026-11-11

This current addendum supersedes status statements recorded earlier in this report. Source identity: branch `main`; base and HEAD `ebca6ed0125c7131c05c945620b23a1970844b92`; tracked-diff SHA-256 `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`.

Current results: backend suite **304 passed, 2 skipped, 0 failed** in 90.31s; targeted backend contracts **19 passed**; frontend timeline contracts **13 passed**; `node --check frontend/app.js` passed. The ordinary suite’s two skips remain recorded as skips in that run. Separately, the two live Ollama tests passed when run with host-daemon access; they are not included in the ordinary suite’s pass count.

Chromium rendered the actual frontend against deterministic stub API responses. It verified seven canonical stages, requested labels and duration formats, unavailable `--` metrics, no duplicate `llm_analysis`, and completed/running/failed/pending states. This verifies frontend rendering, not live-backend browser E2E; live-backend browser E2E remains unverified.

Live Ollama `qwen3.5:4b` smoke validation completed two chunks and four schema-validated records. Request durations were 8.252s and 15.370s in this run. Strict export succeeded; a separate controlled fault-injection run verified retry, partial accounting, strict-export refusal (HTTP 409), and resume with checkpoint reuse. This is smoke and reliability evidence, not a controlled performance benchmark or a claim of general performance improvement.

SHA-256 checks for all three production databases matched before and after isolated integration/full-suite testing. Overall project classification: **PARTIALLY VERIFIED**.

---

# Dataset Generation Reliability Report

**Date:** 2026-11-11  
**Status:** VERIFIED PASS for mocked-provider contracts; live provider PARTIALLY VERIFIED

**Source snapshot:** `Clarion_Verification_Snapshot_2026-11-11`; branch `main`; base commit `ebca6ed0125c7131c05c945620b23a1970844b92`; tracked-diff SHA-256 `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`.

## Source audit

`Clarion-Backend/services/dataset_factory_service.py` implements chunk-level work, structured payload parsing and validation, bounded retries with backoff, SQLite checkpoints, resume of completed chunks, retry of failed chunks, attempt/failure telemetry, provenance, and partial-completion state. With `require_llm_for_export=True`, missing providers and incomplete/heuristic data are refused from strict gold export; deterministic fallback is only selected when strict LLM export is disabled.

## Execution evidence

Python 3.12.14 / pytest 9.2.0, `tests/test_dataset_factory.py`: **8 passed**. These tests use a mocked provider and verify provenance, retry telemetry, per-request timeout, JSON repair, checkpoint/resume, partial strict-gate refusal, no heuristic fallback under strict mode, and successful strict export when all chunks complete.

Live Ollama dataset smoke validation was performed after the initial snapshot-time unavailability observation: configured model `qwen3.5:4b` generated four schema-validated records from two chunks. Recorded request durations were 8.252s and 15.370s. Strict export succeeded. A separate controlled failure/resume experiment injected three transient failures, verified partial status and strict-export HTTP 409 refusal, then resumed while reusing the successful checkpoint and completed generation. This demonstrates a small live run and recovery behavior, not representative latency or inference-quality validation. The fixture-based tests do not write to production databases. The full backend suite passed 304 tests, skipped 2, and had no failures in 90.31s. Production database SHA-256 values matched before and after; comparison evidence is in `production_db_sha256_comparison.csv` in the snapshot.
