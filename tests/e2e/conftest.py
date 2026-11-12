"""Root conftest for Clarion E2E Testing Suite.

Guarantees:
1. Session-level import-time storage redirection to isolated temporary directories (/tmp/clarion-e2e-XXXXXX).
2. Hard safety assertions checking Clarion-Backend/data/clarion.db stat (size 34,533,376 bytes and mtime) before and after test sessions.
3. Fixtures for isolated TestClient, temporary document generators, and isolated SQLite connections.
"""

import atexit
import io
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure offline operation
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROD_DB_PATH = BACKEND_DIR / "data" / "clarion.db"
PROD_DB_EXPECTED_SIZE = 34533376

# Capture production DB stat before any imports or operations
_INITIAL_PROD_STAT = PROD_DB_PATH.stat() if PROD_DB_PATH.exists() else None

# Session-level import-time storage redirection
_SESSION_ROOT = Path(tempfile.mkdtemp(prefix="clarion-e2e-"))
from utils.config import settings as _settings

_settings.data_dir = _SESSION_ROOT / "data"
_settings.vectorstore_dir = _SESSION_ROOT / "data" / "vectorstore"
_settings.logs_dir = _SESSION_ROOT / "logs"
_settings.dataset_dir = _SESSION_ROOT / "data" / "datasets"
_settings.database_url = f"sqlite:///{_settings.data_dir / 'clarion.db'}"

for _d in (
    _settings.data_dir,
    _settings.vectorstore_dir,
    _settings.logs_dir,
    _settings.dataset_dir,
):
    _d.mkdir(parents=True, exist_ok=True)


def assert_production_db_unmodified():
    """Hard invariant check: Clarion-Backend/data/clarion.db must remain untouched."""
    if not PROD_DB_PATH.exists():
        return
    current_stat = PROD_DB_PATH.stat()
    if _INITIAL_PROD_STAT:
        assert current_stat.st_size == _INITIAL_PROD_STAT.st_size == PROD_DB_EXPECTED_SIZE, (
            f"FATAL: Production DB size changed! Expected {PROD_DB_EXPECTED_SIZE}, "
            f"initial={_INITIAL_PROD_STAT.st_size}, current={current_stat.st_size}"
        )
        assert current_stat.st_mtime == _INITIAL_PROD_STAT.st_mtime, (
            f"FATAL: Production DB mtime changed! Initial={_INITIAL_PROD_STAT.st_mtime}, "
            f"current={current_stat.st_mtime}"
        )


atexit.register(assert_production_db_unmodified)


@pytest.fixture(scope="session", autouse=True)
def guard_production_db_session():
    """Assert production database is untouched across the entire test session."""
    assert_production_db_unmodified()
    yield
    assert_production_db_unmodified()
    try:
        shutil.rmtree(_SESSION_ROOT, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def guard_production_db_per_test():
    """Per-test guard asserting zero mutations to production DB."""
    assert_production_db_unmodified()
    yield
    assert_production_db_unmodified()


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Redirect storage to a per-test temporary directory."""
    temp_data = tmp_path / "data"
    temp_vectorstore = temp_data / "vectorstore"
    temp_logs = tmp_path / "logs"
    temp_datasets = temp_data / "datasets"
    temp_db = temp_data / "clarion.db"

    monkeypatch.setattr(_settings, "data_dir", temp_data)
    monkeypatch.setattr(_settings, "vectorstore_dir", temp_vectorstore)
    monkeypatch.setattr(_settings, "logs_dir", temp_logs)
    monkeypatch.setattr(_settings, "dataset_dir", temp_datasets)
    monkeypatch.setattr(_settings, "database_url", f"sqlite:///{temp_db}")

    for directory in (temp_data, temp_vectorstore, temp_logs, temp_datasets):
        directory.mkdir(parents=True, exist_ok=True)

    yield _settings


@pytest.fixture
def e2e_client(tmp_path, monkeypatch):
    """Provides isolated TestClient and validates production DB invariance."""
    temp_data = tmp_path / "data"
    temp_data.mkdir(parents=True, exist_ok=True)
    temp_db = temp_data / "clarion.db"

    monkeypatch.setattr(_settings, "data_dir", temp_data)
    monkeypatch.setattr(_settings, "vectorstore_dir", temp_data / "vectorstore")
    monkeypatch.setattr(_settings, "dataset_dir", temp_data / "datasets")
    monkeypatch.setattr(_settings, "logs_dir", tmp_path / "logs")
    monkeypatch.setattr(_settings, "database_url", f"sqlite:///{temp_db}")

    for d in (
        temp_data,
        temp_data / "vectorstore",
        temp_data / "datasets",
        tmp_path / "logs",
    ):
        d.mkdir(parents=True, exist_ok=True)

    from main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client


@pytest.fixture
def isolated_db_path(tmp_path):
    """Return path to a new, isolated test SQLite database."""
    db_file = tmp_path / "test_isolated.db"
    return db_file


@pytest.fixture
def isolated_sqlite_conn(tmp_path):
    """Return an active connection to an isolated test SQLite database."""
    db_file = tmp_path / "test_isolated.db"
    conn = sqlite3.connect(str(db_file))
    yield conn
    conn.close()


@pytest.fixture
def sample_text_document():
    """Factory fixture for creating sample text documents with metadata."""
    def _create(title="Sample Technical Document", content=None):
        if content is None:
            content = (
                "Introduction to Graph Theory and Algorithms.\n\n"
                "A graph G = (V, E) consists of a set of vertices V and edges E. "
                "In directed graphs, edges have orientation from source to target. "
                "Topological sorting provides a linear ordering of vertices for DAGs. "
                "Dijkstra algorithm computes shortest paths from a single source vertex. "
                "The time complexity is bounded by O((V + E) log V) with binary heaps."
            )
        return {
            "title": title,
            "content": content,
            "size": len(content.encode("utf-8")),
            "word_count": len(content.split()),
        }
    return _create


@pytest.fixture
def sample_docx_factory():
    """Factory fixture for creating in-memory DOCX bytes from paragraphs."""
    def _create_docx(paragraphs):
        from docx import Document
        doc = Document()
        for p in paragraphs:
            doc.add_paragraph(p)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    return _create_docx
