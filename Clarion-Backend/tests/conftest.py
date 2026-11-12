"""Shared pytest fixtures for the Clarion backend test suite.

Storage is redirected to a session-scoped temporary directory at import
time, BEFORE any service or router is imported, so singletons never bind
to the real data/ or logs/ trees.
"""

import io
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.config import settings as _settings

# Redirect storage paths before any test module imports services or routers.
_SESSION_ROOT = Path(tempfile.mkdtemp(prefix="clarion-tests-"))
_settings.data_dir = _SESSION_ROOT / "data"
_settings.vectorstore_dir = _SESSION_ROOT / "data" / "vectorstore"
_settings.logs_dir = _SESSION_ROOT / "logs"
_settings.dataset_dir = _SESSION_ROOT / "data" / "datasets"
_settings.database_url = f"sqlite:///{_settings.data_dir / 'clarion.db'}"

for _directory in (
    _settings.data_dir,
    _settings.vectorstore_dir,
    _settings.logs_dir,
    _settings.dataset_dir,
):
    _directory.mkdir(parents=True, exist_ok=True)

import pytest

_STORAGE_FIELDS = (
    "data_dir",
    "vectorstore_dir",
    "logs_dir",
    "dataset_dir",
    "database_url",
)


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Redirect storage to a per-test temporary directory.

    Services must be constructed inside the test after this fixture is
    applied, so they bind to the isolated paths.
    """
    _settings.data_dir = tmp_path / "data"
    _settings.vectorstore_dir = tmp_path / "data" / "vectorstore"
    _settings.logs_dir = tmp_path / "logs"
    _settings.dataset_dir = tmp_path / "data" / "datasets"
    _settings.database_url = f"sqlite:///{_settings.data_dir / 'clarion.db'}"

    for directory in (
        _settings.data_dir,
        _settings.vectorstore_dir,
        _settings.logs_dir,
        _settings.dataset_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    yield _settings

    # Restore the session-scoped storage after the test.
    _settings.data_dir = _SESSION_ROOT / "data"
    _settings.vectorstore_dir = _SESSION_ROOT / "data" / "vectorstore"
    _settings.logs_dir = _SESSION_ROOT / "logs"
    _settings.dataset_dir = _SESSION_ROOT / "data" / "datasets"
    _settings.database_url = f"sqlite:///{_settings.data_dir / 'clarion.db'}"


def make_docx_bytes(paragraphs):
    """Build in-memory DOCX bytes from a list of paragraph strings."""
    from docx import Document

    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def stats_docx_bytes():
    """DOCX bytes for a short statistics document."""
    return make_docx_bytes(STATS_TEXT.split("\n\n"))


STATS_TEXT = (
    "Hypothesis Testing\n\n"
    "Hypothesis testing is a statistical method used to make decisions about a "
    "population based on a random sample. The null hypothesis represents the "
    "default assumption, while the alternative hypothesis represents the claim "
    "being tested. A sampling distribution describes the distribution of the "
    "sample mean under repeated sampling. The standard normal variate is used "
    "to compute test statistics. The significance level controls the "
    "probability of a type one error. Confidence intervals provide a range of "
    "plausible values for the population parameter."
)


@pytest.fixture
def sample_chunks(isolated_settings):
    """Deterministic chunks for the statistics sample document."""
    from services.chunking_service import ChunkingService

    service = ChunkingService()
    return service.chunk_document("doc-sample", STATS_TEXT)


@pytest.fixture(autouse=True)
def _isolate_external_services(monkeypatch):
    """Keep deterministic tests independent of local services and workspace data.

    Unit/integration tests exercise deterministic fallback paths by default.
    Live-provider validation is deliberately performed by a separately invoked
    workflow, so a developer's running Ollama instance cannot alter metadata.
    """
    from services.relation_dataset_service import RelationDatasetService
    from services.knowledge_map_service import LLMInterface

    monkeypatch.setattr(
        RelationDatasetService, "_workspace_dataset_dir", lambda self: None
    )
    monkeypatch.setattr(LLMInterface, "_is_service_available", lambda self: False)
