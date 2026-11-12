"""Tier 2 Boundary & Corner Cases: test_t2_b05_disambiguation_limits.py
Feature F5: Semantic Disambiguation Labels & Tooltips.
Verifies boundary conditions: 100% risk flags in evaluation, missing quality metrics,
zero hierarchy levels/chains, extreme concept label lengths (300 chars),
and simultaneous null semantic metric fields without layout corruption or NaN errors.
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

from services.pipeline_observability import PipelineStateTracker
from tests.e2e.test_helpers import simulate_render_pipeline_status


def test_b05_evaluation_100_percent_risk_flags():
    """T2.F5.1: Evaluation with 100% risk flags (all evaluated candidates flagged)."""
    tracker = PipelineStateTracker()
    raw = {
        "document_id": "doc-eval-100",
        "stages": {
            "evaluation": {
                "status": "completed",
                "input_size": 20,
                "output_size": 20,
                "duration_sec": 0.45,
                "quality_metrics": {
                    "candidate_relations_evaluated": 20,
                    "risk_flags_detected": 20,
                    "risk_ratio": 1.0,
                },
            }
        },
    }
    status = tracker._format_pipeline_status(raw)
    eval_stage = status["stages"]["evaluation"]
    assert eval_stage["input_size"] == 20
    assert eval_stage["output_size"] == 20
    assert eval_stage["quality_metrics"]["risk_flags_detected"] == 20

    # Ensure simulated render processes without error
    rows = simulate_render_pipeline_status(status)
    assert len(rows) >= 1


def test_b05_missing_quality_metrics_evaluation():
    """T2.F5.2: Missing quality metrics in evaluation stage."""
    tracker = PipelineStateTracker()
    raw = {
        "document_id": "doc-eval-missing",
        "stages": {
            "evaluation": {
                "status": "completed",
                "input_size": None,
                "output_size": None,
                "quality_metrics": {},
            }
        },
    }
    status = tracker._format_pipeline_status(raw)
    eval_stage = status["stages"]["evaluation"]
    assert eval_stage["input_size"] is None
    assert eval_stage["output_size"] is None
    assert eval_stage["quality_metrics"] == {}

    rows = simulate_render_pipeline_status(status)
    assert len(rows) >= 1


def test_b05_hierarchy_zero_levels_zero_chains():
    """T2.F5.3: Hierarchy stage with 0 graph elements and 0 hierarchy levels."""
    tracker = PipelineStateTracker()
    raw = {
        "document_id": "doc-hier-zero",
        "stages": {
            "hierarchy": {
                "status": "completed",
                "input_size": 0,
                "output_size": 0,
                "duration_sec": 0.05,
                "quality_metrics": {
                    "graph_elements": 0,
                    "hierarchy_levels": 0,
                },
            }
        },
    }
    status = tracker._format_pipeline_status(raw)
    hier_stage = status["stages"]["hierarchy"]
    assert hier_stage["input_size"] == 0
    assert hier_stage["output_size"] == 0

    rows = simulate_render_pipeline_status(status)
    assert len(rows) >= 1


def test_b05_extreme_concept_label_length():
    """T2.F5.4: Extremely long concept label (300 chars) in metrics."""
    tracker = PipelineStateTracker()
    long_label = "Concept_" + ("A" * 292)
    raw = {
        "document_id": "doc-long-concept",
        "stages": {
            "mapping": {
                "status": "completed",
                "quality_metrics": {
                    "primary_concept": long_label,
                },
            }
        },
    }
    status = tracker._format_pipeline_status(raw)
    assert status["stages"]["mapping"]["quality_metrics"]["primary_concept"] == long_label
    rows = simulate_render_pipeline_status(status)
    assert len(rows) >= 1


def test_b05_simultaneous_null_semantic_fields():
    """T2.F5.5: Simultaneous null values in all semantic fields."""
    tracker = PipelineStateTracker()
    raw = {
        "document_id": "doc-all-null",
        "stages": {
            "evaluation": {"status": "pending", "input_size": None, "output_size": None, "quality_metrics": None},
            "hierarchy": {"status": "pending", "input_size": None, "output_size": None, "quality_metrics": None},
        },
    }
    status = tracker._format_pipeline_status(raw)
    for st_name in ("evaluation", "hierarchy"):
        st = status["stages"][st_name]
        assert st["status"] == "pending"
        assert st["input_size"] is None
        assert st["output_size"] is None
