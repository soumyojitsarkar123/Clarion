# Environment Recovery Report

## Status: PARTIALLY VERIFIED

The original `Clarion-Backend/.venv` was preserved. Its `pyvenv.cfg` points at a Microsoft Store Python 3.13 executable under `WindowsApps`; that executable returns `Access is denied`. The environment cannot be repaired in place without restoring that base interpreter.

A separate environment was created at `Clarion-Backend/.venv-validation` with Python 3.12.14 from the locally managed runtime. The project has no `pyproject.toml` or lockfile and `requirements*.txt` specify lower bounds, not pins. Therefore this is a reproducible command sequence, not a dependency-locked build.

`pip check` passed. Installation initially encountered Windows `WinError 32` locks while extracting package files; individual retries completed. Relevant installed versions: torch 2.15.0+cpu, sentence-transformers 6.2.0, transformers 5.22.2, faiss-cpu 1.15.1, numpy 2.5.4, pytest 9.2.0, FastAPI 0.162.2, Pydantic 2.14.3.

CUDA is unavailable (`torch.cuda.is_available() == False`), so validation used CPU. The existing BGE model was already available in the local Hugging Face cache.

## Reproduction

```powershell
C:\Users\VICTUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv --copies .venv-validation
.\.venv-validation\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Pin and lock dependencies before treating this as a durable development environment.
