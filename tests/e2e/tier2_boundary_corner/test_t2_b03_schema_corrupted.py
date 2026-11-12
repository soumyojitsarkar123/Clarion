"""Tier 2 Boundary & Corner Cases: test_t2_b03_schema_corrupted.py
Feature F3: Authoritative 7-Stage Sequence & Telemetry Schema.
Verifies boundary conditions: non-existent document lookup, malformed/SQLi document parameter,
mid-pipeline stage cancellation, corrupted persisted analysis_report JSON, and large stage metrics.
"""

import json
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pipeline_observability import PipelineStateTracker, PIPELINE_STAGES
from services.analysis_report_service import AnalysisReportService
from fastapi.testclient import TestClient
from main import app


def test_b03_non_existent_document_lookup(e2e_client):
    """T2.F3.1: Non-existent document ID lookup."""
    res = e2e_client.get("/system/pipeline-status?document_id=00000000-0000-0000-0000-000000000000")
    # Returns 200 with not_found status or 404
    assert res.status_code in (200, 404)
    data = res.json()
    if res.status_code == 200:
        assert data.get("status") == "not_found" or data == {} or "not found" in data.get("message", "").lower()


def test_b03_malformed_uuid_document_parameter(e2e_client):
    """T2.F3.2: Malformed UUID / SQL injection attempt in document_id parameter."""
    sqli_param = "'; DROP TABLE documents; --"
    res = e2e_client.get(f"/system/pipeline-status?document_id={sqli_param}")
    # Must never crash with 500 or execute SQLi
    assert res.status_code in (200, 400, 404, 422)
    assert res.status_code != 500


def test_b03_mid_pipeline_stage_cancellation():
    """T2.F3.3: Mid-pipeline stage cancellation handling."""
    tracker = PipelineStateTracker()
    doc_id = "doc-cancel-123"
    tracker.init_pipeline(doc_id, total_steps=7)
    tracker.start_stage(doc_id, "ingestion")
    tracker.complete_stage(doc_id, "ingestion", input_size=1000, output_size=1000)
    tracker.start_stage(doc_id, "chunking")
    tracker.complete_stage(doc_id, "chunking", input_size=1000, output_size=5)
    tracker.start_stage(doc_id, "embedding")
    tracker.fail_stage(doc_id, "embedding", error="User cancelled pipeline execution")

    status = tracker.get_status(doc_id)
    stages = status["stages"]
    assert stages["ingestion"]["status"] == "completed"
    assert stages["chunking"]["status"] == "completed"
    assert stages["embedding"]["status"] == "failed"
    assert "cancelled" in stages["embedding"]["errors"][0].lower()
    # Downstream stages remained pending
    assert stages["mapping"]["status"] == "pending"
    assert stages["graph_building"]["status"] == "pending"
    assert stages["evaluation"]["status"] == "pending"
    assert stages["hierarchy"]["status"] == "pending"


def test_b03_corrupted_analysis_report_json(tmp_path):
    """T2.F3.4: Corrupted persisted analysis_report data in SQLite."""
    db_file = tmp_path / "test_report.db"
    report_service = AnalysisReportService(db_path=db_file)
    # Inject corrupted JSON string into analysis_reports table
    import sqlite3
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analysis_reports (document_id, data, created_at) VALUES (?, ?, ?)",
        ("doc-corrupted", '{"document_id": "doc-corrupted", "incomplete": [', "2026-11-11T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    # Attempting to load corrupted JSON should raise json.JSONDecodeError cleanly
    with pytest.raises(json.JSONDecodeError):
        report_service.get_report("doc-corrupted")


def test_b03_large_stage_metrics():
    """T2.F3.5: Extremely large stage metrics serialization."""
    tracker = PipelineStateTracker()
    raw_state = {
        "document_id": "doc-large-metrics",
        "status": "completed",
        "stages": {
            "ingestion": {"status": "completed", "input_size": 100_000_000, "output_size": 100_000_000},
            "chunking": {"status": "completed", "input_size": 100_000_000, "output_size": 50_000},
            "embedding": {"status": "completed", "input_size": 50_000, "output_size": 50_000, "duration_sec": 120.5},
            "mapping": {"status": "completed", "duration_sec": 300.0},
            "graph_building": {
                "status": "completed",
                "quality_metrics": {"node_count": 50_000, "edge_count": 200_000},
            },
            "evaluation": {"status": "completed", "input_size": 200_000, "output_size": 1_500},
            "hierarchy": {"status": "completed", "input_size": 250_000, "output_size": 12},
        },
    }
    formatted = tracker._format_pipeline_status(raw_state)
    stages = formatted["stages"]
    assert stages["graph_building"]["node_count"] == 50_000
    assert stages["graph_building"]["edge_count"] == 200_000
    # Must serialize cleanly to JSON string
    serialized = json.dumps(formatted)
    assert "50000" in serialized
    assert "200000" in serialized
