# FAISS and Windows pytest environment diagnosis

**Date:** 2026-11-12 (Asia/Kolkata)  
**Status:** Windows collection is reproducible with the launcher below. A full fresh backend execution remains **NOT VERIFIED** because the current Windows asyncio Proactor runtime stalls during dataset tests.

## Captured environment

- Interpreter: `Clarion-Backend/.venv-validation/Scripts/python.exe`, CPython 3.12.14, Windows x64.
- `faiss-cpu`: 1.15.1.
- NumPy: 2.5.4.
- pytest: 9.2.0.
- The repository root `.venv` and `.venv_linux` contain Linux venv metadata pointing at `/usr/bin`; they are not runnable as Windows venvs.
- `wsl.exe --status` returned `Wsl/ E_ACCESSDENIED`; the recorded Linux venv could not be exercised from this session.

## Findings

1. In fresh processes, `import faiss`, `import numpy; import faiss`, and importing application configuration before FAISS all succeed.
2. Pytest collection that imports FAISS from `test_adversarial_stress_r2_2.py` after other imports can fail in `faiss._swigfaiss` with `DLL load failed while importing _swigfaiss: A dynamic link library (DLL) initialization routine failed`. A collection attempt also emitted Windows fatal exception `0xc0000008` from the SWIG extension initializer.
3. Preloading FAISS before pytest initialization (`import faiss; import pytest; pytest.main(...)`) collected all 33 test modules successfully with normal pytest plugins enabled. This makes import order the demonstrated trigger; the evidence does not support reinstalling or downgrading NumPy/FAISS.
4. `scripts/run_backend_tests.py` preserves that native import order. Reproducible commands from `Clarion-Backend`:

   ```powershell
   .\.venv-validation\Scripts\python.exe scripts\run_backend_tests.py --collect-only -q
   .\.venv-validation\Scripts\python.exe scripts\run_backend_tests.py -q
   ```

5. Execution proceeds past the first 14 tests, then a focused dataset-factory test stalls in `asyncio.run()` while CPython 3.12.14 initializes `ProactorEventLoop`; the captured stack is `asyncio.runners.run -> windows_events.ProactorEventLoop -> _make_self_pipe -> socket._fallback_socketpair -> socket.accept`. The test's provider is an in-memory mock and has not yet been called. This is separate from FAISS collection and prevents claiming a fresh full-suite result for this Windows runtime.

## Resolution and limits

- Keep FAISS installed; no dependency changes are justified by the observations.
- Use the launcher for Windows collection and test runs so FAISS initializes before pytest plugins and test modules.
- Run full execution in the established Linux Python 3.12 environment once that environment is accessible. WSL access was denied in this run, so Linux is a documented strategy, not a current verification result.
- The launcher improves collection only. It does not resolve the Proactor socket-pair stall, certify the Visual C++ runtime, or prove FAISS indexing integration under this runtime.

## Current verification result

- Direct import order checks: **PASS**.
- Individual FAISS-affected file collection with preloaded FAISS: **PASS** (11 tests collected).
- Full backend collection with preloaded FAISS: **PASS** (33 modules collected).
- Full backend execution: **NOT VERIFIED**; interrupted after diagnosis of the independent asyncio initialization stall. No pass count is reported.
