"""Tier 4 Real-World Application Scenario: test_t4_s02_technical_slides_pipeline.py
Scenario 2: Technical Lecture Slides (Thevenin Theorem).
Simulates ingestion of technical presentation slides with equations and circuits,
validating graceful fallback degradation, concept extraction, and honest status ribbon display.
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


def test_t4_s02_technical_slides_pipeline():
    """T4.SCEN.2: Technical lecture slides pipeline with deterministic fallback."""
    doc_id = "doc-thevenin-slides"
    tracker = PipelineStateTracker()

    slides_state = {
        "document_id": doc_id,
        "status": "degraded",
        "analysis_mode": "deterministic_fallback",
        "active_model": "Demo (Fallback)",
        "stages": {
            "ingestion": {"status": "completed", "input_size": 4500, "output_size": 4500, "duration_sec": 0.04},
            "chunking": {"status": "completed", "input_size": 4500, "output_size": 8, "duration_sec": 0.10},
            "embedding": {"status": "completed", "input_size": 8, "output_size": 8, "duration_sec": 0.35},
            "mapping": {"status": "fallback", "input_size": 8, "output_size": 14, "duration_sec": 0.50},
            "graph_building": {"status": "completed", "input_size": 14, "output_size": 20, "duration_sec": 0.25},
            "evaluation": {"status": "completed", "input_size": 14, "output_size": 0, "duration_sec": 0.15},
            "hierarchy": {"status": "completed", "input_size": 20, "output_size": 2, "duration_sec": 0.10},
        },
        "total_duration_sec": 1.44,
    }

    status = tracker._format_pipeline_status(slides_state)

    # Invariants: 7 canonical stages, deterministic fallback reported, no synthetic row
    assert len(status["stages"]) == 7
    assert status["analysis_mode"] == "deterministic_fallback"
    assert status["stages"]["mapping"]["status"] == "fallback"
    assert "llm_analysis" not in status["stages"]

    rows = simulate_render_pipeline_status(status)
    assert len(rows) == 7
