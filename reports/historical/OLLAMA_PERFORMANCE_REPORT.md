## Final verification addendum — 2026-11-11

This current addendum supersedes status statements recorded earlier in this report. Source identity: branch `main`; base and HEAD `ebca6ed0125c7131c05c945620b23a1970844b92`; tracked-diff SHA-256 `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`.

Current results: backend suite **304 passed, 2 skipped, 0 failed** in 90.31s; targeted backend contracts **19 passed**; frontend timeline contracts **13 passed**; `node --check frontend/app.js` passed. The ordinary suite’s two skips remain recorded as skips in that run. Separately, the two live Ollama tests passed when run with host-daemon access; they are not included in the ordinary suite’s pass count.

Chromium rendered the actual frontend against deterministic stub API responses. It verified seven canonical stages, requested labels and duration formats, unavailable `--` metrics, no duplicate `llm_analysis`, and completed/running/failed/pending states. This verifies frontend rendering, not live-backend browser E2E; live-backend browser E2E remains unverified.

Live Ollama `qwen3.5:4b` smoke validation completed two chunks and four schema-validated records. Request durations were 8.252s and 15.370s in this run. Strict export succeeded; a separate controlled fault-injection run verified retry, partial accounting, strict-export refusal (HTTP 409), and resume with checkpoint reuse. This is smoke and reliability evidence, not a controlled performance benchmark or a claim of general performance improvement.

SHA-256 checks for all three production databases matched before and after isolated integration/full-suite testing. Overall project classification: **PARTIALLY VERIFIED**.

---

# Ollama Performance Report

**Date:** 2026-11-11  
**Status:** Live smoke validation complete; controlled performance benchmark not completed

**Source snapshot:** `Clarion_Verification_Snapshot_2026-11-11`; branch `main`; base commit `ebca6ed0125c7131c05c945620b23a1970844b92`; tracked-diff SHA-256 `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`.

At the initial snapshot capture, the daemon was unavailable in the restricted test context and two live cases were skipped in the ordinary full-suite run. A separate execution with host-daemon access subsequently passed both live provider tests. The live dataset harness also completed two chunks and four schema-validated records. These checks establish provider/model availability and smoke-level functionality; they do not constitute a controlled performance benchmark, throughput study, or model-quality evaluation.

The full backend run was 304 passed, 2 skipped, 0 failed in 90.31s. The ordinary full-suite run recorded two skips; separately, the live provider test module passed 2/2 when run with host-daemon access. Mock-provider contracts remain separately identified in `DATASET_GENERATION_RELIABILITY_REPORT.md`.

A controlled benchmark remains future work: define repeated trials, warm/cold conditions, concurrency, token counts, latency distributions, throughput, quality criteria, and retained raw outputs before making comparative performance claims.
