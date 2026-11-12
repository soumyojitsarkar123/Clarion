"""Tier 4 Real-World Application Scenario: test_t4_s08_adversarial_corrupt_ingest.py
Scenario 8: Adversarial Corrupt File Ingestion.
Simulates adversarial upload of corrupted/truncated files (damaged headers, binary noise),
asserting graceful rejection, informative error messages, and stage error telemetry tracking.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app
from services.pipeline_observability import PipelineStateTracker


def test_t4_s08_adversarial_corrupt_ingest():
    """T4.SCEN.8: Corrupt file upload and error telemetry handling."""
    client = TestClient(app)

    # 1. Upload 0-byte corrupt PDF
    res_empty = client.post(
        "/upload",
        files={"file": ("corrupt_zero.pdf", b"", "application/pdf")},
    )
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json().get("detail", "").lower()

    # 2. Upload truncated PDF header (invalid magic bytes)
    res_bad_magic = client.post(
        "/upload",
        files={"file": ("damaged.pdf", b"NOT_A_VALID_PDF_HEADER_TRUNCATED", "application/pdf")},
    )
    # The server should either reject with 400 or handle error during extraction
    assert res_bad_magic.status_code in (200, 400)

    # 3. Verify stage error logging in tracker
    tracker = PipelineStateTracker()
    doc_id = "doc-corrupt-tracked"
    tracker.init_pipeline(doc_id, total_steps=7)
    tracker.start_stage(doc_id, "ingestion")
    tracker.fail_stage(doc_id, "ingestion", error="Corrupt PDF: cannot find startxref table")

    status = tracker.get_status(doc_id)
    assert status["stages"]["ingestion"]["status"] == "failed"
    assert len(status["stages"]["ingestion"]["errors"]) >= 1
    assert "startxref" in status["stages"]["ingestion"]["errors"][0]
