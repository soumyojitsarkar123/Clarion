# Test runtime recovery report

**Classification: BLOCKED on this Windows host; Linux CI execution NOT VERIFIED.** Collection succeeds. Neither the Proactor nor the test-only Selector policy avoids the local socket-pair stall. A Linux GitHub Actions workflow now exists, but no authenticated remote run was available from this environment.

## Environment and reproduction

- Windows x64; CPython 3.12.14; pytest 9.2.0; AnyIO 4.16.0; FAISS CPU 1.15.1; NumPy 2.5.4.
- `scripts/run_backend_tests.py --collect-only -q -rs` collected 322 tests and reported one collection skip: `tests/test_playwright_e2e.py` because Playwright is not installed.
- Full command: `scripts/run_backend_tests.py --selector-loop -vv -o faulthandler_timeout=10`, bounded to 90 seconds. Fourteen tests passed. Execution stalled on test 15, `tests/test_adversarial_stress_r2_2.py::TestUploadIngressRobustness::test_upload_invalid_extension_exe`; the faulthandler stack entered Starlette `TestClient` → AnyIO `run_blocking_portal` → `asyncio.SelectorEventLoop._make_self_pipe` → `socket._fallback_socketpair` → `socket.accept`. The child was terminated at 90 seconds. No suite failure summary or final pass total is claimed.
- Direct `asyncio.run()` and a focused dataset-factory operation show the same blocked socket-pair initialization. Selecting `WindowsSelectorEventLoopPolicy` still reaches the same OS socket accept path. This is distinct from the prior FAISS collection/import-order issue.
- The two live Ollama tests were separately run and both skipped because the local daemon is inaccessible. `wsl.exe --status` had previously returned `Wsl/ E_ACCESSDENIED`.

## CI path

`.github/workflows/backend-regression.yml` runs the full suite on Ubuntu 24.04 / Python 3.12 using the captured pinned validation dependency manifest. Dependency compatibility is grounded in the current Windows environment freeze; Ubuntu wheel resolution and test execution have not been observed yet. The GitHub CLI has no usable authenticated session here, so the workflow was not pushed or executed remotely. Production networking behavior was not changed to work around the runner defect.

Full run output: [`backend-full-suite-final-2026-11-12.log`](../../research_artifacts/raw/backend-full-suite-final-2026-11-12.log).
