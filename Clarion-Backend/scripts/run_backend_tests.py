"""Run backend pytest with FAISS initialized before pytest imports its plugins.

On the captured Windows runtime, importing FAISS after some pytest collection
imports can fail in the SWIG DLL initializer, while importing it first is stable.
This launcher keeps that native import order explicit and reproducible.
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import faiss  # noqa: F401  # must precede pytest/plugin and test imports
import pytest


if __name__ == "__main__":
    args = list(sys.argv[1:])
    if "--selector-loop" in args:
        args.remove("--selector-loop")
        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if sys.platform == "win32" and selector_policy is not None:
            asyncio.set_event_loop_policy(selector_policy())
        else:
            raise SystemExit("--selector-loop is available only on Windows")
    targets = []
    remaining = []
    for arg in args:
        candidate_text, separator, node = arg.partition("::")
        candidate = Path(candidate_text)
        if not arg.startswith("-") and (candidate.exists() or (BACKEND_DIR / candidate).exists()):
            target = candidate if candidate.exists() else BACKEND_DIR / candidate
            targets.append(str(target.resolve()) + (separator + node if separator else ""))
        else:
            remaining.append(arg)
    raise SystemExit(pytest.main([*(targets or [str(BACKEND_DIR / "tests")]), *remaining]))
