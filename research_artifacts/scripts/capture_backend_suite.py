"""Capture the complete backend regression attempt with a bounded timeout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend_dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    backend = args.backend_dir.resolve()
    root = Path(__file__).resolve().parents[2]
    log_path = root / "research_artifacts" / "raw" / "backend-full-suite-final-2026-11-12.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(backend / ".venv-validation" / "Scripts" / "python.exe"),
        "scripts/run_backend_tests.py", "--selector-loop", "-vv",
        "-o", "faulthandler_timeout=10",
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, cwd=backend, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=args.timeout_seconds,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        status = f"EXIT={result.returncode}"
        output = result.stdout + result.stderr
        code = result.returncode
    except subprocess.TimeoutExpired as error:
        status = f"TIMEOUT after {args.timeout_seconds}s; child process terminated"
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        error_text = error.stderr or ""
        if isinstance(error_text, bytes):
            error_text = error_text.decode("utf-8", errors="replace")
        output += error_text
        code = 124
    elapsed = time.perf_counter() - started
    log = (
        f"COMMAND: {' '.join(cmd)}\nCWD: {backend}\n"
        f"DURATION_SECONDS: {elapsed:.3f}\nSTATUS: {status}\n\n{output}"
    )
    log_path.write_text(log, encoding="utf-8")
    print(f"CAPTURE_LOG: {log_path}")
    print(log)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
