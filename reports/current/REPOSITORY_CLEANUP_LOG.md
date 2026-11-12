# Repository cleanup action log

**Date:** 2026-11-11 (Asia/Kolkata)  
**Frozen baseline:** `research_artifacts/engineering-frozen-presnapshot-2026-11-11/` (tracked patch and untracked archive validated before organization).

## Completed

1. Kept the repository root limited to its README, project record, current status, contribution guidance, and license.
2. Moved 59 root-level reports and evidence artifacts into current/historical report shelves, research data, bibliography, recovery, and package archive locations; added navigation indexes.
3. Retained both ZIPs because their contents differ; both passed integrity validation. Moved rather than deleted them.
4. Removed the extracted documentation folder only after a full file-set/hash comparison against its retained ZIP. Removed the duplicate backend recovery patch only after matching its SHA-256 to the archived canonical patch.
5. Repaired moved Markdown references. Post-organization link check: 92 checked local links, 0 broken links (archived `.agents` handoff notes excluded because they contain regex/link-like examples).
6. Preserved source documents, databases, environments, model files, experiments, benchmark records, and recovery material.

## Verification after organization

- Earlier complete backend run on this code checkout: **309 passed, 2 skipped, 0 failed** in 84.23 seconds; see [`software_readiness_backend_full_suite.log`](../../research_artifacts/raw/software_readiness_backend_full_suite.log). Skips were Playwright unavailable and a relation database unavailable in the isolated test temp.
- A post-organization full-suite rerun did **not** complete: pytest collection encountered Windows `faiss._swigfaiss` DLL initialization failure in `test_adversarial_stress_r2_2.py` (and once in `test_e2e_pipeline_orchestration.py`). Direct `import faiss` succeeds; see [`software_post_cleanup_backend_suite.log`](../../research_artifacts/raw/software_post_cleanup_backend_suite.log). This is recorded as **NOT VERIFIED** for a fresh post-cleanup full-suite run, not as a passing result.
- Browser E2E remains **NOT VERIFIED** because Playwright is not installed.

## Final disposition

Cleanup and organization are complete. Release readiness remains **NOT READY**. See [current status](../../CLARION_CURRENT_STATUS.md) and [software release readiness](SOFTWARE_RELEASE_READINESS.md) for outstanding engineering gates.
- The five token-budget unit tests passed in isolation. A focused API query test stalled in the local test runtime and was interrupted; this path remains **NOT VERIFIED** in the fresh post-organization run.
