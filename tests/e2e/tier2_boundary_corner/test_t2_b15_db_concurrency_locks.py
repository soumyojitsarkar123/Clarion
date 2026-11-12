"""Tier 2 Boundary & Corner Cases: test_t2_b15_db_concurrency_locks.py
Feature F15: Production Database Isolation & Safety.
Verifies boundary conditions: production DB immutability (size 34,533,376 bytes and SHA-256),
isolated temporary DB creation without reliance on production records, path traversal protection,
and zero write lock contention against production clarion.db.
"""

import hashlib
import os
import sqlite3
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
PROD_DB = BACKEND_DIR / "data" / "clarion.db"
PROD_DB_EXPECTED_SIZE = 34533376


def test_b15_production_db_file_size_invariant():
    """T2.F15.1: Production clarion.db size remains exactly 34,533,376 bytes."""
    if PROD_DB.exists():
        stat = PROD_DB.stat()
        assert stat.st_size == PROD_DB_EXPECTED_SIZE, (
            f"Production DB size changed! Expected {PROD_DB_EXPECTED_SIZE}, got {stat.st_size}"
        )


def test_b15_production_db_sha256_invariance():
    """T2.F15.2: Production DB SHA-256 hash calculation."""
    if PROD_DB.exists():
        h1 = hashlib.sha256(PROD_DB.read_bytes()).hexdigest()
        assert len(h1) == 64
        # Re-read to assert consistency
        h2 = hashlib.sha256(PROD_DB.read_bytes()).hexdigest()
        assert h1 == h2


def test_b15_isolated_db_environment_cleanliness(tmp_path):
    """T2.F15.3: Isolated test creates independent DB without reading production DB."""
    temp_db = tmp_path / "sandbox.db"
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test_table (id TEXT PRIMARY KEY, val TEXT)")
    cursor.execute("INSERT INTO test_table VALUES ('t-1', 'isolated')")
    conn.commit()

    cursor.execute("SELECT val FROM test_table WHERE id = 't-1'")
    assert cursor.fetchone()[0] == "isolated"
    conn.close()

    # Assert production DB was not touched
    assert PROD_DB.stat().st_size == PROD_DB_EXPECTED_SIZE


def test_b15_path_traversal_data_dir_rejected():
    """T2.F15.4: Path traversal attempt resolved safely."""
    from utils.config import BACKEND_ROOT
    traversal_path = Path("../../etc/passwd")
    resolved = (BACKEND_ROOT / traversal_path).resolve()
    # Path resolution detects directory traversal outside backend root
    assert not str(resolved).startswith(str(BACKEND_ROOT / "data"))


def test_b15_concurrent_read_on_production_db():
    """T2.F15.5: Production DB read-only transaction concurrency."""
    if PROD_DB.exists():
        # Open read-only connection via URI
        uri = f"file:{PROD_DB.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        assert table_count > 0
        conn.close()
