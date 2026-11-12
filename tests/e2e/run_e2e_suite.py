#!/usr/bin/env python3
"""Unified Clarion E2E Test Suite Standalone Runner.

Supports:
- Selective execution by tier (--tier {1,2,3,4,all})
- Verbose output (--verbose / -v)
- Pre- and post-execution verification of production database immutability
- Comprehensive test collection and result reporting
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
PROD_DB_PATH = BACKEND_DIR / "data" / "clarion.db"
PROD_DB_EXPECTED_SIZE = 34533376
PYTEST_BIN = ROOT_DIR / ".venv_linux" / "bin" / "pytest"
PYTHON_BIN = ROOT_DIR / ".venv_linux" / "bin" / "python"

TIER_MAP = {
    "1": ROOT_DIR / "tests" / "e2e" / "tier1_feature_coverage",
    "2": ROOT_DIR / "tests" / "e2e" / "tier2_boundary_corner",
    "3": ROOT_DIR / "tests" / "e2e" / "tier3_cross_feature",
    "4": ROOT_DIR / "tests" / "e2e" / "tier4_real_world_scenarios",
    "all": ROOT_DIR / "tests" / "e2e",
}


def get_db_fingerprint(db_path: Path):
    """Compute file size and SHA-256 hash of database file."""
    if not db_path.exists():
        return None, None
    stat = db_path.stat()
    sha256 = hashlib.sha256(db_path.read_bytes()).hexdigest()
    return stat.st_size, sha256


def verify_db_safety(stage: str, initial_size: int, initial_hash: str):
    """Verify that the production database has not mutated."""
    if not PROD_DB_PATH.exists():
        return
    current_size, current_hash = get_db_fingerprint(PROD_DB_PATH)
    if current_size != initial_size or current_size != PROD_DB_EXPECTED_SIZE:
        print(f"\n[FATAL] Production database size mutated during {stage}!")
        print(f"  Expected: {PROD_DB_EXPECTED_SIZE} bytes")
        print(f"  Initial:  {initial_size} bytes")
        print(f"  Current:  {current_size} bytes")
        sys.exit(2)
    if current_hash != initial_hash:
        print(f"\n[FATAL] Production database SHA-256 hash mutated during {stage}!")
        print(f"  Initial: {initial_hash}")
        print(f"  Current: {current_hash}")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Clarion E2E Test Suite Runner")
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Specify which test tier to execute (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose test output",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect and display tests without executing them",
    )
    parser.add_argument(
        "-k", "--keyword",
        type=str,
        default=None,
        help="Pytest expression keyword filter (-k)",
    )
    args = parser.parse_args()

    # Pre-execution DB fingerprint
    initial_size, initial_hash = get_db_fingerprint(PROD_DB_PATH)
    if initial_size is not None:
        if initial_size != PROD_DB_EXPECTED_SIZE:
            print(f"[WARNING] Production DB initial size {initial_size} differs from expected {PROD_DB_EXPECTED_SIZE}!")
        else:
            print(f"[SAFETY] Production clarion.db verified ({initial_size} bytes, SHA-256: {initial_hash[:12]}...)")

    target_dir = TIER_MAP[args.tier]
    if not target_dir.exists():
        print(f"[ERROR] Target tier path {target_dir} does not exist.")
        sys.exit(1)

    pytest_cmd = [
        str(PYTEST_BIN if PYTEST_BIN.exists() else "pytest"),
        str(target_dir),
    ]

    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")

    if args.collect_only:
        pytest_cmd.append("--collect-only")

    if args.keyword:
        pytest_cmd.extend(["-k", args.keyword])

    # Enforce offline flags and PYTHONPATH
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{BACKEND_DIR}:{ROOT_DIR}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    print("=" * 70)
    print(f" CLARION E2E TEST RUNNER — Tier: {args.tier.upper()} ({target_dir.name})")
    print(f" Command: {' '.join(pytest_cmd)}")
    print("=" * 70)

    start_time = time.time()
    proc = subprocess.run(pytest_cmd, env=env, cwd=str(ROOT_DIR))
    elapsed = time.time() - start_time

    # Post-execution DB verification
    if initial_size is not None:
        verify_db_safety("test session execution", initial_size, initial_hash)
        print(f"[SAFETY] Post-run check: Production clarion.db remains pristine (size: {initial_size} bytes).")

    print("-" * 70)
    status_str = "SUCCESS" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
    print(f" Run Completed in {elapsed:.2f}s — Result: {status_str}")
    print("=" * 70)

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
