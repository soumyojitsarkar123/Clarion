# Software release readiness

**Decision: NOT READY.** Current status is **PARTIALLY VERIFIED**. Do not mark this checkout as a stable software release candidate yet.

## Gate

| Acceptance criterion | Status |
|---|---|
| PDF extracted-page and source-coordinate chain through BGE segments, FAISS, and retrieval evidence | **PARTIALLY VERIFIED**: isolated 87-page PDF workflow passes. |
| Frontend renders correct live source citations | **NOT VERIFIED**: no live browser run or screenshot. |
| Provider token budgets and output reservations | **PARTIALLY VERIFIED**: provider boundaries and focused tests pass; exact deployed Qwen token count/live usage not captured. |
| Actual runtime context and no silent response truncation | **PARTIALLY VERIFIED**: configured checks and truncation reporting are implemented; no live provider verification. |
| Model identity and FAISS compatibility | **PARTIALLY VERIFIED**: BGE revision recorded; FAISS map consistency checked in isolated workflow. |
| Complete backend regression on final source | **BLOCKED**: Windows async self-pipe stall after 14 passes; Ubuntu CI workflow has not yet run. |
| Upload-to-cited-answer in real browser application | **NOT VERIFIED**. |
| Additive migration, rollback, legacy compatibility | **PARTIALLY VERIFIED**: isolated tests pass; no production migration/restore rehearsal. |
| Security review, recovery and backup restore | **NOT VERIFIED**. |
| Recoverable current engineering state | **VERIFIED PASS**: final worktree patch, untracked files, and byte-for-byte DB copies are captured in `research_artifacts/engineering-final-blocker-resolution-2026-11-12/`. |

Configured defaults remain `qwen3.5:4b`, `BAAI/bge-large-en-v1.5`, and the current 512-word chunk baseline. No publication-readiness claim is made.

## Required to close the gate

1. Run `.github/workflows/backend-regression.yml` in a compatible Ubuntu 24.04/Python 3.12 runner and resolve every failure or skip decision.
2. Execute a real backend upload/query and real browser session with the configured models and isolated storage; compare displayed page/source spans with the original PDF and save screenshots.
3. Verify provider token usage/context and dataset/query end-to-end behavior with live requests.
4. Rehearse production-like migration rollback and database/FAISS restore without using live user data.
5. Retain the recorded recoverable engineering snapshot with this report set.
