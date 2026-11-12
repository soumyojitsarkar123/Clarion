"""Tier 1 Feature Coverage: F2 Eliminate Synthetic llm_analysis Row.

Specifications:
- T1.F2.1: Authoritative 7-Stage Key Set in Serializer
- T1.F2.2: Stage Duration Non-Duplication
- T1.F2.3: Frontend Timeline Row Count Parity
- T1.F2.4: Cold-Recovery Status Stage Parity
- T1.F2.5: Active Model Display Relocation
"""

import time
import pytest
from services.pipeline_observability import (
    PIPELINE_STAGES,
    PipelineStateTracker,
    PipelineAuditor,
    get_pipeline_state_tracker,
)
from services.analysis_report_service import get_analysis_report_service
from tests.e2e.test_helpers import simulate_render_pipeline_status


def test_f02_authoritative_7_stage_keys(e2e_client):
    """T1.F2.1: Authoritative 7-Stage Key Set in Serializer."""
    doc_id = "doc-t1-f02-keys"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    stages = payload.get("stages", {})
    expected_stages = {
        "ingestion",
        "chunking",
        "embedding",
        "mapping",
        "graph_building",
        "evaluation",
        "hierarchy",
    }
    assert set(stages.keys()) == expected_stages, f"Mismatch in stage keys: {set(stages.keys())}"
    assert "llm_analysis" not in stages, "Synthetic 'llm_analysis' stage must NOT be in stages"


def test_f02_stage_duration_non_duplication(e2e_client):
    """T1.F2.2: Stage Duration Non-Duplication."""
    doc_id = "doc-t1-f02-dur"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)

    # Complete mapping with a distinct known duration
    auditor.start_stage("mapping", input_size=10, progress=55)
    time.sleep(0.02)  # distinct duration
    auditor.complete_stage(
        "mapping",
        output_size=20,
        quality_metrics={"concept_count": 12, "relation_count": 8},
        progress=65,
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    payload = response.json()
    stages = payload.get("stages", {})

    mapping_dur = stages["mapping"].get("duration_sec")
    assert mapping_dur is not None and mapping_dur > 0

    # Ensure no other stage in stages has identical duration due to cloning
    for name, st in stages.items():
        if name != "mapping" and st.get("status") == "pending":
            assert st.get("duration_sec") is None, f"Stage {name} duration should be None when pending"


def test_f02_frontend_timeline_row_count_parity():
    """T1.F2.3: Frontend Timeline Row Count Parity."""
    # Feed status payload with 7 stages to simulate_render_pipeline_status
    sample_stages = {
        k: {"status": "completed", "duration_sec": 0.05, "input_size": 10, "output_size": 10}
        for k in PIPELINE_STAGES
    }
    rows = simulate_render_pipeline_status({"stages": sample_stages})
    assert len(rows) == 7, f"Expected exactly 7 rows in timeline table, got {len(rows)}"

    for row in rows:
        row_text = " ".join(row["cells"]).lower()
        assert "llm_analysis" not in row_text, f"Found forbidden llm_analysis in row: {row_text}"


def test_f02_cold_recovery_status_stage_parity(e2e_client, tmp_path):
    """T1.F2.4: Cold-Recovery Status Stage Parity."""
    doc_id = "doc-t1-f02-recovery"
    report_service = get_analysis_report_service()

    # Create dummy analysis report on disk to simulate cold recovery
    report_data = {
        "document_id": doc_id,
        "metadata": {"title": "Test Recovery Doc"},
        "analysis_date": "2026-11-11T05:00:00",
        "stages": {
            "ingestion": {"status": "completed", "duration_sec": 0.02, "input_size": 1000, "output_size": 950},
            "chunking": {"status": "completed", "duration_sec": 0.03, "input_size": 950, "output_size": 5},
            "embedding": {"status": "completed", "duration_sec": 0.04, "input_size": 5, "output_size": 5},
            "mapping": {"status": "completed", "duration_sec": 0.08, "input_size": 5, "output_size": 10},
            "graph_building": {"status": "completed", "duration_sec": 0.03, "input_size": 10, "output_size": 10},
            "evaluation": {"status": "completed", "duration_sec": 0.02, "input_size": 10, "output_size": 2},
            "hierarchy": {"status": "completed", "duration_sec": 0.02, "input_size": 10, "output_size": 2},
        },
        "metrics": {"words": 150, "pages": 1, "sections": 1, "chunks": 5, "concepts": 6, "candidate_relations": 4, "graph_nodes": 6, "graph_edges": 4},
    }
    report_service.save_report(doc_id, report_data)

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()
    stages = payload.get("stages", {})

    assert len(stages) == 7
    assert "llm_analysis" not in stages


def test_f02_active_model_display_relocation(e2e_client):
    """T1.F2.5: Active Model Display Relocation."""
    doc_id = "doc-t1-f02-model"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)
    auditor.start_stage("mapping", input_size=10, progress=55)
    auditor.complete_stage("mapping", output_size=15, progress=65)

    tracker.complete_pipeline(
        doc_id,
        is_degraded=False,
        analysis_mode="llm",
        active_model="qwen3.5:4b",
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    # Active model displayed in top-level payload, not as a synthetic stage name
    assert payload.get("active_model") == "qwen3.5:4b"
    assert "llm_analysis" not in payload.get("stages", {})
    mapping_stage = payload.get("stages", {}).get("mapping", {})
    assert mapping_stage.get("model") == "qwen3.5:4b"
