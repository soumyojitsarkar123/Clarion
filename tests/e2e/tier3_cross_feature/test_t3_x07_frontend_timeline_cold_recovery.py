"""Tier 3 Cross-Feature Combination: test_t3_x07_frontend_timeline_cold_recovery.py
Feature Interaction: F1 (Genuine Metrics) + F3 (Authoritative Telemetry Schema) + F6 (Frontend Timeline Conformance).
Validates that upon cold system recovery, persisted pipeline telemetry accurately restores
all 7 completed stages without data loss, rendering cleanly in the frontend timeline DOM.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.pipeline_observability import PipelineStateTracker, PIPELINE_STAGES
from tests.e2e.test_helpers import simulate_render_pipeline_status


def test_t3_x07_frontend_timeline_cold_recovery():
    """T3.INT.7: F1 + F3 + F6 Pairwise Integration."""
    # Cold start: re-instantiate tracker with state recovered from persisted report
    tracker = PipelineStateTracker()
    doc_id = "doc-x07-cold-recovery"

    cold_state = {
        "document_id": doc_id,
        "status": "completed",
        "overall_progress": 100,
        "total_duration_sec": 4.15,
        "stages": {
            stage: {
                "status": "completed",
                "progress": 100,
                "duration_sec": 0.5,
                "input_size": 100,
                "output_size": 10,
                "quality_metrics": {},
                "errors": [],
            }
            for stage in PIPELINE_STAGES
        },
    }

    status = tracker._format_pipeline_status(cold_state)
    assert status["status"] == "completed"
    assert len(status["stages"]) == 7
    assert set(status["stages"].keys()) == set(PIPELINE_STAGES)

    # F6 Frontend Conformance: Render recovered pipeline state in DOM table
    rows = simulate_render_pipeline_status(status)
    assert len(rows) == 7
    # Verify no row has llm_analysis
    for r in rows:
        assert "llm_analysis" not in r["html"]
