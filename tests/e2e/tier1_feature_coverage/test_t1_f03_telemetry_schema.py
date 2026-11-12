"""Tier 1 Feature Coverage: F3 Authoritative 7-Stage Sequence & Telemetry Schema.

Specifications:
- T1.F3.1: Sequential Stage Progression Order
- T1.F3.2: Top-level Status Schema Compliance
- T1.F3.3: Per-Stage Record Field Structure
- T1.F3.4: Cold-Recovery Report Completeness
- T1.F3.5: Telemetry Summary Metrics Parity
"""

import pytest
from services.pipeline_observability import (
    PIPELINE_STAGES,
    get_pipeline_state_tracker,
    PipelineAuditor,
)
from services.analysis_report_service import get_analysis_report_service


def test_f03_stage_progression_order():
    """T1.F3.1: Sequential Stage Progression Order."""
    expected_order = [
        "ingestion",
        "chunking",
        "embedding",
        "mapping",
        "graph_building",
        "evaluation",
        "hierarchy",
    ]
    assert PIPELINE_STAGES == expected_order, f"PIPELINE_STAGES order deviated: {PIPELINE_STAGES}"


def test_f03_top_level_status_schema(e2e_client):
    """T1.F3.2: Top-level Status Schema Compliance."""
    doc_id = "doc-t1-f03-top"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    assert "document_id" in payload and payload["document_id"] == doc_id
    assert "status" in payload and payload["status"] in ["pending", "running", "completed", "degraded", "failed"]
    assert "overall_progress" in payload and 0 <= payload["overall_progress"] <= 100
    assert "stages" in payload and isinstance(payload["stages"], dict)
    assert "metrics" in payload and isinstance(payload["metrics"], dict)
    assert "total_duration_sec" in payload


def test_f03_per_stage_record_structure(e2e_client):
    """T1.F3.3: Per-Stage Record Field Structure."""
    doc_id = "doc-t1-f03-stage-rec"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)

    auditor.start_stage("embedding", input_size=5, progress=35)
    auditor.complete_stage(
        "embedding",
        output_size=5,
        quality_metrics={"indexed": True},
        progress=50,
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    stage_rec = payload["stages"]["embedding"]
    required_keys = ["name", "status", "duration_sec", "input_size", "output_size", "quality_metrics"]
    for k in required_keys:
        assert k in stage_rec, f"Stage record missing required key: {k}"


def test_f03_cold_recovery_report_completeness(e2e_client):
    """T1.F3.4: Cold-Recovery Report Completeness."""
    doc_id = "doc-t1-f03-cold"
    report_service = get_analysis_report_service()

    report_payload = {
        "document_id": doc_id,
        "metadata": {"title": "Cold Recovery Document"},
        "analysis_date": "2026-11-11T05:00:00",
        "stages": {
            st: {
                "name": st,
                "status": "completed",
                "duration_sec": 0.1,
                "input_size": 100,
                "output_size": 10,
                "quality_metrics": {"verified": True},
            }
            for st in PIPELINE_STAGES
        },
        "metrics": {
            "words": 500,
            "pages": 2,
            "sections": 3,
            "chunks": 10,
            "concepts": 15,
            "candidate_relations": 8,
            "graph_nodes": 15,
            "graph_edges": 8,
        },
    }
    report_service.save_report(doc_id, report_payload)

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "completed"
    assert data["overall_progress"] == 100
    for st in PIPELINE_STAGES:
        assert data["stages"][st]["status"] == "completed"
        assert data["stages"][st]["duration_sec"] is not None


def test_f03_telemetry_summary_metrics_parity(e2e_client):
    """T1.F3.5: Telemetry Summary Metrics Parity."""
    doc_id = "doc-t1-f03-metrics"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    metrics = payload.get("metrics", {})
    expected_metric_keys = {
        "words",
        "pages",
        "sections",
        "chunks",
        "concepts",
        "candidate_relations",
        "graph_nodes",
        "graph_edges",
    }
    assert expected_metric_keys.issubset(set(metrics.keys())), (
        f"Missing expected metrics keys: {expected_metric_keys - set(metrics.keys())}"
    )
