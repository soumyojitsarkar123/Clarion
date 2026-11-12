"""Tier 2 Boundary & Corner Cases: test_t2_b02_synthetic_row_guards.py
Feature F2: Eliminate Synthetic llm_analysis Row.
Verifies boundary conditions: legacy reports with preexisting llm_analysis keys,
degraded pipelines under deterministic fallback, failed mapping isolation,
empty state handling, and zero-duration mapping metrics without synthetic injection.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pipeline_observability import PipelineStateTracker, PIPELINE_STAGES


def test_b02_legacy_report_preexisting_llm_analysis():
    """T2.F2.1: Legacy report with pre-existing llm_analysis key in raw state."""
    tracker = PipelineStateTracker()
    raw_state = {
        "document_id": "doc-legacy-001",
        "status": "completed",
        "stages": {
            "ingestion": {"status": "completed", "duration_sec": 0.05},
            "chunking": {"status": "completed", "duration_sec": 0.10},
            "embedding": {"status": "completed", "duration_sec": 0.20},
            "mapping": {"status": "completed", "duration_sec": 1.50},
            "llm_analysis": {"status": "completed", "duration_sec": 1.50, "synthetic": True},
            "graph_building": {"status": "completed", "duration_sec": 0.30},
            "evaluation": {"status": "completed", "duration_sec": 0.25},
            "hierarchy": {"status": "completed", "duration_sec": 0.15},
        },
        "total_duration_sec": 2.55,
    }
    formatted = tracker._format_pipeline_status(raw_state)
    stages = formatted["stages"]

    # Must contain strictly the 7 canonical stages
    assert set(stages.keys()) == set(PIPELINE_STAGES)
    assert "llm_analysis" not in stages
    assert len(stages) == 7


def test_b02_degraded_pipeline_no_synthetic_duplicate():
    """T2.F2.2: Degraded pipeline without LLM execution."""
    tracker = PipelineStateTracker()
    raw_state = {
        "document_id": "doc-degraded-002",
        "status": "degraded",
        "analysis_mode": "deterministic_fallback",
        "stages": {
            "ingestion": {"status": "completed", "duration_sec": 0.04},
            "chunking": {"status": "completed", "duration_sec": 0.08},
            "embedding": {"status": "completed", "duration_sec": 0.12},
            "mapping": {"status": "completed", "duration_sec": 0.45},
            "graph_building": {"status": "completed", "duration_sec": 0.20},
            "evaluation": {"status": "completed", "duration_sec": 0.15},
            "hierarchy": {"status": "completed", "duration_sec": 0.10},
        },
    }
    formatted = tracker._format_pipeline_status(raw_state)
    stages = formatted["stages"]

    # Mapping status coerced to fallback
    assert stages["mapping"]["status"] == "fallback"
    assert "llm_analysis" not in stages
    assert len(stages) == 7


def test_b02_failed_mapping_stage_isolation():
    """T2.F2.3: Failed mapping stage error isolation."""
    tracker = PipelineStateTracker()
    raw_state = {
        "document_id": "doc-fail-003",
        "status": "failed",
        "stages": {
            "ingestion": {"status": "completed", "duration_sec": 0.05},
            "chunking": {"status": "completed", "duration_sec": 0.05},
            "embedding": {"status": "completed", "duration_sec": 0.10},
            "mapping": {
                "status": "failed",
                "duration_sec": 5.0,
                "errors": ["Connection timeout to provider localhost:11434"],
            },
            "graph_building": {"status": "pending"},
            "evaluation": {"status": "pending"},
            "hierarchy": {"status": "pending"},
        },
    }
    formatted = tracker._format_pipeline_status(raw_state)
    stages = formatted["stages"]

    assert stages["mapping"]["status"] == "failed"
    assert len(stages["mapping"]["errors"]) == 1
    assert "llm_analysis" not in stages
    assert stages["graph_building"]["status"] == "pending"


def test_b02_empty_pipeline_state_object():
    """T2.F2.4: Empty pipeline state object formatting."""
    tracker = PipelineStateTracker()
    formatted = tracker._format_pipeline_status({})
    stages = formatted["stages"]

    assert len(stages) == 7
    assert set(stages.keys()) == set(PIPELINE_STAGES)
    for stage_key in PIPELINE_STAGES:
        assert stages[stage_key]["status"] == "pending"
        assert stages[stage_key]["duration_sec"] is None
        assert stages[stage_key]["progress"] == 0


def test_b02_zero_duration_mapping():
    """T2.F2.5: Zero-duration mapping stage (0.000s) preservation."""
    tracker = PipelineStateTracker()
    raw_state = {
        "document_id": "doc-zero-005",
        "status": "completed",
        "stages": {
            "ingestion": {"status": "completed", "duration_sec": 0.01},
            "chunking": {"status": "completed", "duration_sec": 0.02},
            "embedding": {"status": "completed", "duration_sec": 0.03},
            "mapping": {"status": "completed", "duration_sec": 0.0},
            "graph_building": {"status": "completed", "duration_sec": 0.04},
            "evaluation": {"status": "completed", "duration_sec": 0.05},
            "hierarchy": {"status": "completed", "duration_sec": 0.06},
        },
        "total_duration_sec": 0.21,
    }
    formatted = tracker._format_pipeline_status(raw_state)
    stages = formatted["stages"]

    assert stages["mapping"]["duration_sec"] == 0.0
    assert "llm_analysis" not in stages
    # Sum of stage durations strictly equals the 7 genuine stages
    durations = [st["duration_sec"] for st in stages.values() if st["duration_sec"] is not None]
    assert sum(durations) == pytest.approx(0.21, abs=1e-4)
