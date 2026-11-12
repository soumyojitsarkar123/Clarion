# Software final verification report

**Final gate: NOT READY. Overall classification: PARTIALLY VERIFIED.** The targeted unit set and isolated real-PDF/BGE/FAISS pipeline pass. The complete regression suite, live backend workflow, browser citations, security review, and recovery restore are not verified.

## Source and environment

- Git HEAD: `ebca6ed0125c7131c05c945620b23a1970844b92`; source changes remain in the worktree and include pre-existing user changes. No commit was created.
- Windows x64; CPython 3.12.14; pytest 9.2.0; FastAPI 0.162.2; Starlette 1.10.0; HTTPX 0.28.1 / HTTPX2 2.16.0; NumPy 2.5.4; faiss-cpu 1.15.1; sentence-transformers 6.2.0; Transformers 5.22.2; PyTorch 2.15.0. Full freeze: [`backend-validation-requirements-2026-11-12.txt`](../../research_artifacts/processed/backend-validation-requirements-2026-11-12.txt).
- Defaults retained: generation `qwen3.5:4b`, embedding `BAAI/bge-large-en-v1.5`, 512-word chunk baseline. Local Qwen manifest model layer digest: `sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`. BGE cache revision: `d4aa6901d3a41ba39fb536a557fa166f842b0e09`.

## Verification results

| Area | Result | Evidence |
|---|---|---|
| Final focused regression | **PASS**: 21 tests | Provenance chain, provider/token budget, embedding segmentation, dataset oversized-input rejection and isolated migration, chunk warning behavior. |
| Backend collection | **PASS**: 322 collected, 1 skipped | The skip is `tests/test_playwright_e2e.py`: Playwright is unavailable. |
| Complete backend execution | **BLOCKED** | 14 tests passed, then test 15 `test_upload_invalid_extension_exe` stalled in AnyIO/asyncio Windows selector self-pipe socketpair initialization; terminated after 90 seconds. No full-suite pass/fail total. |
| Live Ollama tests | **SKIPPED**: 2 | `localhost:11434` denied/unavailable in this host. |
| Isolated real PDF → BGE → FAISS | **PASS** for service pipeline | 87 pages, 44 chunks, 100 segments/vectors; 100 source-mapped; query returned a page-bounded source interval. |
| Live FastAPI upload/query/LLM | **NOT VERIFIED** | TestClient event-loop path stalls; Ollama socket blocked. |
| Browser citations | **NOT VERIFIED** | No Playwright/browser executable; no screenshot. |
| Migration rollback | **PASS** in isolated SQLite | Fresh, legacy, idempotent, and injected rollback tests. Production migration not attempted. |
| Security/backup restore | **NOT VERIFIED** | No full security review, restore rehearsal, or production failure injection. |

## Data integrity and snapshot

Experiments used isolated storage at `research_artifacts/isolated-provenance-e2e-2026-11-12/data`. The production Clarion database was not opened through Clarion/SQLite or modified; its raw files were read for hashes and byte-copied into the recovery snapshot. Current production DB / WAL / SHM hashes are respectively `1AC7A955825B7551FC805B1AE74CCE101E5F14C9D4B4B0F1B0E741140536DCE6`, `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`, and `FD4C9FDA9CD3F9AE7C962B0DDF37232294D55580E1AA165AA06129B8549389EB`. A before-run baseline was not recorded, so these are traceability values, not a before/after comparison.

Isolated E2E DB SHA-256: `1232C2149A2C0959231A12CF763B3B9E870DAFB9EE542D990CDC19CD0B4B142B`. FAISS index SHA-256: `9E35CF44301C4DD052CBF470884804809B256789659A023CE098EE93159D9D89`. Vector metadata SHA-256: `B4D1607D420E2A46AB95D2EF7880681EFD40DB77D7D8DB2203DFC7F77AB0125D`.

The full suite log is [`backend-full-suite-final-2026-11-12.log`](../../research_artifacts/raw/backend-full-suite-final-2026-11-12.log). The complete current worktree and byte-for-byte SQLite files are preserved in [`engineering-final-blocker-resolution-2026-11-12`](../../research_artifacts/engineering-final-blocker-resolution-2026-11-12/README.md). Release readiness remains **NOT READY** until the full suite passes in a compatible runtime and live backend/browser citations are verified.
