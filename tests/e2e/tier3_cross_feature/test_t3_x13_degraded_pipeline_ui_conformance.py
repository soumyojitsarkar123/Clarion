"""Tier 3 Cross-Feature Combination: test_t3_x13_degraded_pipeline_ui_conformance.py
Feature Interaction: F2 (No Synthetic llm_analysis Row) + F6 (Frontend Timeline Conformance) + F16 (Live vs Mock Transparency).
Validates that when the pipeline operates under deterministic fallback, the frontend status ribbon
displays Deterministic Fallback and the timeline table renders strictly 7 rows without duplicate llm_analysis.
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


def test_t3_x13_degraded_pipeline_ui_conformance():
    """T3.INT.13: F2 + F6 + F16 Pairwise Integration."""
    tracker = PipelineStateTracker()
    doc_id = "doc-x13-degraded-ui"

    raw = {
        "document_id": doc_id,
        "status": "degraded",
        "analysis_mode": "deterministic_fallback",
        "active_model": "Demo (Fallback)",
        "stages": {
            stage: {
                "status": "completed" if stage != "mapping" else "fallback",
                "duration_sec": 0.1,
                "input_size": 10,
                "output_size": 10,
                "quality_metrics": {},
                "errors": [],
            }
            for stage in PIPELINE_STAGES
        },
    }

    status = tracker._format_pipeline_status(raw)
    assert status["analysis_mode"] == "deterministic_fallback"
    assert status["stages"]["mapping"]["status"] == "fallback"
    assert "llm_analysis" not in status["stages"]

    # Render in simulated frontend DOM
    rows = simulate_render_pipeline_status(status)
    assert len(rows) == 7
    # Assert zero occurrences of llm_analysis across all table rows
    for r in rows:
        assert "llm_analysis" not in r["html"]
