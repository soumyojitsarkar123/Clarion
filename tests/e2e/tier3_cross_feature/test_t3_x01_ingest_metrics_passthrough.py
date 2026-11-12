"""Tier 3 Cross-Feature Combination: test_t3_x01_ingest_metrics_passthrough.py
Feature Interaction: F1 (Genuine Metrics Passthrough) + F2 (Eliminate Duplicate Row).
Validates that stage telemetry accurately reflects genuinely recorded input and output metrics
across all 7 pipeline stages, while strictly eliminating any duplicate llm_analysis row
and preventing double-counting of stage durations.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pipeline_observability import PipelineStateTracker, PIPELINE_STAGES


def test_t3_x01_ingest_metrics_passthrough():
    """T3.INT.1: F1 + F2 Pairwise Integration."""
    tracker = PipelineStateTracker()
    doc_id = "doc-x01-metrics-passthrough"

    # Simulate genuine pipeline execution across 7 stages
    raw_state = {
        "document_id": doc_id,
        "status": "completed",
        "stages": {
            "ingestion": {"status": "completed", "input_size": 2500, "output_size": 2500, "duration_sec": 0.05},
            "chunking": {"status": "completed", "input_size": 2500, "output_size": 6, "duration_sec": 0.12, "quality_metrics": {"chunk_count": 6}},
            "embedding": {"status": "completed", "input_size": 6, "output_size": 6, "duration_sec": 0.40, "quality_metrics": {"chunk_count": 6}},
            "mapping": {"status": "completed", "input_size": 6, "output_size": 15, "duration_sec": 2.10},
            "graph_building": {"status": "completed", "input_size": 15, "output_size": 22, "duration_sec": 0.35, "quality_metrics": {"node_count": 10, "edge_count": 12}},
            "evaluation": {"status": "completed", "input_size": 12, "output_size": 1, "duration_sec": 0.28, "quality_metrics": {"candidate_relations_evaluated": 12, "risk_flags_detected": 1}},
            "hierarchy": {"status": "completed", "input_size": 22, "output_size": 3, "duration_sec": 0.15, "quality_metrics": {"graph_elements": 22, "hierarchy_levels": 3}},
        },
        "total_duration_sec": 3.45,
    }

    formatted = tracker._format_pipeline_status(raw_state)
    stages = formatted["stages"]

    # F2: Invariant: Exactly 7 stages, zero synthetic llm_analysis row
    assert len(stages) == 7
    assert set(stages.keys()) == set(PIPELINE_STAGES)
    assert "llm_analysis" not in stages

    # F1: Passthrough metrics check
    assert stages["ingestion"]["input_size"] == 2500
    assert stages["chunking"]["output_size"] == 6
    assert stages["chunking"]["chunk_count"] == 6
    assert stages["embedding"]["throughput_chunks_sec"] > 0
    assert stages["graph_building"]["node_count"] == 10
    assert stages["graph_building"]["edge_count"] == 12
    assert stages["evaluation"]["input_size"] == 12
    assert stages["hierarchy"]["output_size"] == 3

    # Non-duplication duration invariant: mapping duration must not be double-counted
    stage_durations = [st["duration_sec"] for st in stages.values() if st["duration_sec"] is not None]
    assert sum(stage_durations) == pytest.approx(3.45, abs=1e-4)
