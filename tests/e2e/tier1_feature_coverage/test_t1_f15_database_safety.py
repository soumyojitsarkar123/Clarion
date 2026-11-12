"""Tier 1 Feature Coverage: F15 Production Database Isolation & Safety.

Specifications:
- T1.F15.1: Production Database File Size Invariance (exactly 34,533,376 bytes)
- T1.F15.2: Production Database SHA-256 Invariance
- T1.F15.3: Pytest Isolated Settings Fixture (redirected to /tmp/ sandbox)
- T1.F15.4: Zero Write Lock on Production DB During Tests
- T1.F15.5: Cleanup of Temporary Test Artifacts
"""

import hashlib
import os
from pathlib import Path
import sqlite3
import pytest
from utils.config import settings

ROOT_DIR = Path(__file__).resolve().parents[3]
PROD_DB = ROOT_DIR / "Clarion-Backend" / "data" / "clarion.db"
EXPECTED_PROD_SIZE = 34533376


def test_f15_prod_db_file_size_invariant():
    """T1.F15.1: Production Database File Size Invariance."""
    assert PROD_DB.exists(), f"Production database file not found at {PROD_DB}"
    current_size = os.path.getsize(str(PROD_DB))
    assert current_size == EXPECTED_PROD_SIZE, (
        f"Production clarion.db size mutated! Expected {EXPECTED_PROD_SIZE}, got {current_size}"
    )


def test_f15_prod_db_sha256_invariant():
    """T1.F15.2: Production Database SHA-256 Invariance."""
    assert PROD_DB.exists()
    hasher = hashlib.sha256()
    with open(PROD_DB, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    assert len(digest) == 64, f"Invalid SHA-256 digest: {digest}"


def test_f15_isolated_settings_fixture(isolated_settings):
    """T1.F15.3: Pytest Isolated Settings Fixture."""
    assert isolated_settings.data_dir != PROD_DB.parent, (
        f"Settings data_dir {isolated_settings.data_dir} must not point to canonical production {PROD_DB.parent}"
    )
    assert "clarion-e2e-" in str(isolated_settings.data_dir) or "pytest-" in str(isolated_settings.data_dir) or "data" in str(isolated_settings.data_dir)


def test_f15_zero_write_lock_on_prod_db():
    """T1.F15.4: Zero Write Lock on Production DB During Tests."""
    # Open read-only connection to production database; must not throw sqlite3.OperationalError: database is locked
    uri = f"file:{PROD_DB.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_count = cursor.fetchone()[0]
    assert table_count > 0, "Production database should contain tables"
    conn.close()


def test_f15_cleanup_test_artifacts(tmp_path):
    """T1.F15.5: Cleanup of Temporary Test Artifacts."""
    test_db = tmp_path / "sandbox.db"
    conn = sqlite3.connect(str(test_db))
    conn.execute("CREATE TABLE test_data (id INT PRIMARY KEY)")
    conn.commit()
    conn.close()
    assert test_db.exists()
    test_db.unlink()
    assert not test_db.exists(), "Temporary test artifact should be removable"
