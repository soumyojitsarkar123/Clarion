"""
Adversarial Challenge Test Suite for Milestone M1:
Authoritative Telemetry Schema, Invariant Checks, Fallback Truthfulness,
Throughput Stress-Testing, and E2E JSON Schema Conformance.

Authored by: Challenger 2 (Milestone M1)
"""

import asyncio
import json
import math
import threading
import time
from typing import Any, Dict

import pytest

from services.pipeline_observability import (
    PIPELINE_STAGES,
    PipelineStateTracker,
    PipelineAuditor,
    estimate_size,
)
from services.analysis_report_service import AnalysisReportService
from services.processing_pipeline import ProcessingPipeline
from routers import system
from utils.config import settings


# ============================================================================
# 1. INVARIANT CHECKS: STAGE KEYS GUARANTEE & EXTRA KEY STRIPPING
# ============================================================================

def test_stages_keys_strictly_match_pipeline_stages_clean():
    """Verify clean initialization guarantees exact stage keys in order."""
    tracker = PipelineStateTracker()
    doc_id = "test-doc-keys-clean"
    tracker.start_pipeline(doc_id)

    status = tracker.get_status(doc_id)
    stages = status["stages"]

    assert list(stages.keys()) == PIPELINE_STAGES
    assert len(stages) == 7
    assert "llm_analysis" not in stages


def test_stages_keys_adversarial_injection_and_missing_stages():
    """Verify that alien/synthetic keys are stripped and missing keys are synthesized as pending."""
    tracker = PipelineStateTracker()
    doc_id = "test-doc-corrupted"

    # Inject pathological state directly into tracker internal state
    tracker._state[doc_id] = {
        "document_id": doc_id,
        "status": "running",
        "analysis_mode": "llm",
        "stages": {
            # Only two legitimate stages
            "ingestion": {
                "name": "ingestion",
                "status": "completed",
                "duration_sec": 0.05,
                "input_size": 1000,
                "output_size": 950,
            },
            "mapping": {
                "name": "mapping",
                "status": "completed",
                "duration_sec": 0.5,
                "input_size": 10,
                "output_size": 15,
            },
            # Pathological extra keys that should NEVER appear in output
            "llm_analysis": {"name": "llm_analysis", "duration_sec": 0.5},
            "synthetic_pipeline_duplicate": {"foo": "bar"},
            "__proto__": {"injected": True},
            "random_stage_xyz": {"status": "completed"},
        },
    }

    formatted = tracker.get_status(doc_id)
    stages = formatted["stages"]

    # Invariant: stages.keys() must strictly match PIPELINE_STAGES in exact order
    assert list(stages.keys()) == PIPELINE_STAGES
    assert len(stages) == 7

    # Alien keys must be completely purged
    assert "llm_analysis" not in stages
    assert "synthetic_pipeline_duplicate" not in stages
    assert "__proto__" not in stages
    assert "random_stage_xyz" not in stages

    # Missing stages (chunking, embedding, graph_building, evaluation, hierarchy)
    # must be filled in with default pending values
    for missing_stage in ["chunking", "embedding", "graph_building", "evaluation", "hierarchy"]:
        st = stages[missing_stage]
        assert st["name"] == missing_stage
        assert st["status"] == "pending"
        assert st["duration_sec"] is None
        assert st["input_size"] is None
        assert st["output_size"] is None
        assert st["errors"] == []
        assert st["quality_metrics"] == {}


def test_stages_keys_empty_raw_stages():
    """Verify behavior when raw stages dictionary is completely empty."""
    tracker = PipelineStateTracker()
    doc_id = "test-doc-empty-stages"
    tracker._state[doc_id] = {
        "document_id": doc_id,
        "status": "pending",
        "stages": {},
    }

    formatted = tracker.get_status(doc_id)
    stages = formatted["stages"]

    assert list(stages.keys()) == PIPELINE_STAGES
    assert len(stages) == 7
    for name in PIPELINE_STAGES:
        assert stages[name]["status"] == "pending"
        assert stages[name]["duration_sec"] is None


# ============================================================================
# 2. INVARIANT CHECKS: DEGRADED FALLBACK EXECUTION TRUTHFULNESS & NO PHANTOMS
# ============================================================================

@pytest.mark.parametrize(
    "analysis_mode,pipeline_status,mapping_status,expected_mapping_status",
    [
        # Normal completed LLM run
        ("llm", "completed", "completed", "completed"),
        # Deterministic fallback mode completed
        ("deterministic_fallback", "completed", "completed", "fallback"),
        # Pipeline status degraded completed
        ("llm", "degraded", "completed", "fallback"),
        # Both analysis_mode=deterministic_fallback and status=degraded
        ("deterministic_fallback", "degraded", "completed", "fallback"),
        # Degraded, but mapping stage failed outright
        ("deterministic_fallback", "degraded", "failed", "failed"),
        # Degraded, but mapping stage is still running
        ("deterministic_fallback", "degraded", "running", "running"),
        # Degraded, but mapping stage is still pending
        ("deterministic_fallback", "degraded", "pending", "pending"),
        # Degraded, but mapping stage was skipped
        ("deterministic_fallback", "degraded", "skipped", "skipped"),
    ],
)
def test_degraded_fallback_mapping_truthfulness(
    analysis_mode, pipeline_status, mapping_status, expected_mapping_status
):
    """Mapping stage reports status truthfully according to execution state without phantom rows."""
    tracker = PipelineStateTracker()
    doc_id = f"doc-fallback-{analysis_mode}-{pipeline_status}-{mapping_status}"

    tracker._state[doc_id] = {
        "document_id": doc_id,
        "status": pipeline_status,
        "analysis_mode": analysis_mode,
        "active_model": "qwen3.5:4b",
        "stages": {
            s: {
                "name": s,
                "status": mapping_status if s == "mapping" else "completed",
                "progress": 100,
                "duration_sec": 0.1,
                "input_size": 10,
                "output_size": 10,
                "quality_metrics": {},
                "errors": [],
            }
            for s in PIPELINE_STAGES
        },
    }

    status = tracker.get_status(doc_id)
    stages = status["stages"]

    assert list(stages.keys()) == PIPELINE_STAGES
    assert len(stages) == 7
    assert "llm_analysis" not in stages

    mapping = stages["mapping"]
    assert mapping["status"] == expected_mapping_status
    assert mapping["analysis_mode"] == analysis_mode
    assert mapping["model"] == "qwen3.5:4b"


# ============================================================================
# 3. INVARIANT CHECKS: THROUGHPUT CALCULATION IN EMBEDDING (DIVISION BY ZERO)
# ============================================================================

@pytest.mark.parametrize(
    "dur,in_sz,q_metrics,expected_throughput",
    [
        # Zero duration
        (0.0, 100, {}, None),
        # None duration
        (None, 100, {}, None),
        # Negative duration (clock skew / edge case)
        (-0.5, 100, {}, None),
        # Zero chunk count with valid duration
        (1.5, 0, {}, 0.0),
        # None input_size and no q_metrics chunk_count
        (1.5, None, {}, None),
        # Standard positive throughput
        (0.5, 100, {}, 200.0),
        # Sub-second duration with fractional throughput
        (0.125, 5, {}, 40.0),
        # Sub-millisecond positive duration (extremely fast) -> no crash/overflow
        (0.0001, 10, {}, 100000.0),
        # Chunk count from quality_metrics when in_sz is None
        (2.0, None, {"chunk_count": 50}, 25.0),
        # Both in_sz and duration 0
        (0.0, 0, {}, None),
        # Both None
        (None, None, {}, None),
    ],
)
def test_embedding_throughput_resilience(dur, in_sz, q_metrics, expected_throughput):
    """Embedding throughput calculation must never raise ZeroDivisionError, TypeError, or overflow."""
    tracker = PipelineStateTracker()
    doc_id = "doc-throughput-test"

    tracker._state[doc_id] = {
        "document_id": doc_id,
        "status": "completed",
        "stages": {
            s: {
                "name": s,
                "status": "completed" if s == "embedding" else "pending",
                "duration_sec": dur if s == "embedding" else None,
                "input_size": in_sz if s == "embedding" else None,
                "output_size": in_sz if s == "embedding" else None,
                "quality_metrics": dict(q_metrics) if s == "embedding" else {},
                "errors": [],
            }
            for s in PIPELINE_STAGES
        },
    }

    status = tracker.get_status(doc_id)
    embedding_stage = status["stages"]["embedding"]

    actual_throughput = embedding_stage["throughput_chunks_sec"]
    assert actual_throughput == expected_throughput
    if expected_throughput is not None:
        assert embedding_stage["quality_metrics"]["throughput_chunks_sec"] == expected_throughput
    else:
        assert "throughput_chunks_sec" not in embedding_stage["quality_metrics"]
    assert "device" in embedding_stage
    if actual_throughput is not None:
        assert isinstance(actual_throughput, (int, float))
        assert not math.isnan(actual_throughput)
        assert not math.isinf(actual_throughput)


# ============================================================================
# 4. INVARIANT CHECKS: TYPE STABILITY OF input_size, output_size, duration_sec
# ============================================================================

def test_type_stability_across_lifecycle():
    """Verify input_size and output_size remain strictly int or None (never strings, floats, NaN)."""
    tracker = PipelineStateTracker()
    doc_id = "doc-type-stability"
    auditor = PipelineAuditor(doc_id, tracker)
    tracker.start_pipeline(doc_id)

    # 1. Pending stages
    status = tracker.get_status(doc_id)
    for s_name in PIPELINE_STAGES:
        st = status["stages"][s_name]
        assert st["input_size"] is None
        assert st["output_size"] is None
        assert st["duration_sec"] is None

    # 2. Running stage
    auditor.start_stage("ingestion", input_size=1024)
    status = tracker.get_status(doc_id)
    ing_stage = status["stages"]["ingestion"]
    assert isinstance(ing_stage["input_size"], int)
    assert ing_stage["input_size"] == 1024
    assert ing_stage["output_size"] is None
    # duration_sec of currently running stage has not ended yet
    assert ing_stage["duration_sec"] is None

    # 3. Completed stage with sub-second duration
    time.sleep(0.005)  # 5ms
    auditor.complete_stage("ingestion", output_size=980, quality_metrics={"word_count": 150})
    status = tracker.get_status(doc_id)
    ing_stage = status["stages"]["ingestion"]
    assert isinstance(ing_stage["input_size"], int)
    assert ing_stage["input_size"] == 1024
    assert isinstance(ing_stage["output_size"], int)
    assert ing_stage["output_size"] == 980
    assert isinstance(ing_stage["duration_sec"], float)
    assert ing_stage["duration_sec"] > 0.0
    # Must preserve sub-second precision, not rounded to 0.0
    assert ing_stage["duration_sec"] < 1.0


def test_type_coercion_adversarial_dirty_inputs():
    """Verify resilience when dirty values (floats, strings, negatives) are supplied."""
    tracker = PipelineStateTracker()
    doc_id = "doc-dirty-inputs"
    auditor = PipelineAuditor(doc_id, tracker)
    tracker.start_pipeline(doc_id)

    # Auditor start_stage with negative size -> clamped to 0
    auditor.start_stage("ingestion", input_size=-50)
    assert auditor.stages["ingestion"]["input_size"] == 0

    # Auditor complete_stage with float size -> converted to int
    auditor.complete_stage("ingestion", output_size=123.7)
    assert auditor.stages["ingestion"]["output_size"] == 123
    assert isinstance(auditor.stages["ingestion"]["output_size"], int)

    # Directly injected dirty strings in tracker._state
    tracker._state[doc_id]["stages"]["chunking"] = {
        "name": "chunking",
        "status": "completed",
        "duration_sec": "0.045",  # string representation of float
        "input_size": 123,
        "output_size": "15",      # string representation of int
        "quality_metrics": {},
        "errors": [],
    }

    formatted = tracker.get_status(doc_id)
    chunking = formatted["stages"]["chunking"]
    # duration_sec coerced to float
    assert isinstance(chunking["duration_sec"], float)
    assert chunking["duration_sec"] == 0.045
    # chunk_count derived safely as int
    assert isinstance(chunking["chunk_count"], int)
    assert chunking["chunk_count"] == 15


# ============================================================================
# 5. CONCURRENCY & THREAD-SAFETY STRESS TEST
# ============================================================================

def test_concurrent_pipeline_updates():
    """Stress-test PipelineStateTracker under heavy multi-threaded concurrent access."""
    tracker = PipelineStateTracker()
    num_docs = 20
    doc_ids = [f"doc-thread-{i}" for i in range(num_docs)]
    errors = []

    def worker_thread(doc_id: str):
        try:
            tracker.start_pipeline(doc_id)
            for stage in PIPELINE_STAGES:
                tracker.start_stage(doc_id, stage, input_size=100)
                time.sleep(0.001)
                tracker.complete_stage(doc_id, stage, output_size=50)
            tracker.complete_pipeline(doc_id, is_degraded=False, analysis_mode="llm", active_model="qwen3.5:4b")
            status = tracker.get_status(doc_id)
            assert status["document_id"] == doc_id
            assert status["status"] == "completed"
            assert len(status["stages"]) == 7
            assert list(status["stages"].keys()) == PIPELINE_STAGES
        except Exception as exc:
            errors.append(f"Thread for {doc_id} failed: {exc}")

    threads = [threading.Thread(target=worker_thread, args=(did,)) for did in doc_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrency errors occurred: {errors}"
    all_statuses = tracker.get_status()["pipelines"]
    assert len(all_statuses) == num_docs
    for st in all_statuses:
        assert len(st["stages"]) == 7
        assert list(st["stages"].keys()) == PIPELINE_STAGES


# ============================================================================
# 6. END-TO-END STATUS CONTRACT & JSON SCHEMA VALIDATION
# ============================================================================

def test_full_e2e_json_serialization_conformance():
    """Verify serialized pipeline status strictly conforms to OpenAPI and JSON schema rules."""
    tracker = PipelineStateTracker()
    doc_id = "doc-e2e-json-contract"
    auditor = PipelineAuditor(doc_id, tracker)
    tracker.start_pipeline(doc_id)

    # Walk through full realistic pipeline
    auditor.start_stage("ingestion", input_size=12500)
    auditor.complete_stage("ingestion", output_size=12000, quality_metrics={"word_count": 2100})

    auditor.start_stage("chunking", input_size=12000)
    auditor.complete_stage("chunking", output_size=18, quality_metrics={"chunk_count": 18, "avg_chunk_words": 116.6})

    auditor.start_stage("embedding", input_size=18)
    auditor.complete_stage("embedding", output_size=18, quality_metrics={"indexed": True, "chunk_count": 18})

    auditor.start_stage("mapping", input_size=18)
    auditor.complete_stage("mapping", output_size=42, quality_metrics={"concept_count": 24, "relation_count": 18})

    auditor.start_stage("graph_building", input_size=42)
    auditor.complete_stage("graph_building", output_size=42, quality_metrics={"node_count": 24, "edge_count": 18})

    auditor.start_stage("evaluation", input_size=18)
    auditor.complete_stage("evaluation", output_size=3, quality_metrics={"overall_score": 0.91, "risk_score": 0.09, "flags_count": 3})

    auditor.start_stage("hierarchy", input_size=42)
    auditor.complete_stage("hierarchy", output_size=4, quality_metrics={"layers_count": 4, "chains_count": 3})

    tracker.complete_pipeline(doc_id, is_degraded=False, analysis_mode="llm", active_model="qwen3.5:4b")

    status = tracker.get_status(doc_id)

    # Test 1: JSON serializability without custom encoders
    json_str = json.dumps(status)
    assert json_str is not None
    parsed = json.loads(json_str)

    # Test 2: Top-level schema properties
    assert parsed["document_id"] == doc_id
    assert parsed["status"] == "completed"
    assert parsed["overall_progress"] == 100
    assert parsed["analysis_mode"] == "llm"
    assert parsed["active_model"] == "qwen3.5:4b"
    assert isinstance(parsed["total_duration_sec"], float)
    assert parsed["total_duration_sec"] > 0

    # Test 3: Stages schema properties
    stages = parsed["stages"]
    assert len(stages) == 7
    assert list(stages.keys()) == PIPELINE_STAGES

    for stage_name in PIPELINE_STAGES:
        st = stages[stage_name]
        assert st["name"] == stage_name
        assert st["status"] == "completed"
        assert isinstance(st["progress"], int)
        assert st["progress"] == 100
        assert isinstance(st["duration_sec"], (float, int))
        assert st["duration_sec"] >= 0.0
        assert isinstance(st["input_size"], int)
        assert isinstance(st["output_size"], int)
        assert isinstance(st["quality_metrics"], dict)
        assert isinstance(st["errors"], list)
        assert st["start_time"] is not None
        assert st["end_time"] is not None

    # Test 4: Semantic Metric Disambiguation
    # Evaluation: input candidate relations, output risk flags
    eval_stage = stages["evaluation"]
    assert eval_stage["input_size"] == 18
    assert eval_stage["output_size"] == 3
    assert eval_stage["quality_metrics"]["flags_count"] == 3
    assert eval_stage["quality_metrics"]["overall_score"] == 0.91

    # Hierarchy: input graph elements, output hierarchy levels
    hier_stage = stages["hierarchy"]
    assert hier_stage["input_size"] == 42
    assert hier_stage["output_size"] == 4
    assert hier_stage["quality_metrics"]["layers_count"] == 4

    # Graph building: node and edge counts
    graph_stage = stages["graph_building"]
    assert graph_stage["node_count"] == 24
    assert graph_stage["edge_count"] == 18


# ============================================================================
# 7. COLD RECOVERY ADVERSARIAL STRESS TEST
# ============================================================================

def test_cold_recovery_with_partial_or_corrupted_report(isolated_settings, monkeypatch):
    """Test cold recovery from database when persisted report is partially corrupted or incomplete."""
    doc_id = "doc-cold-corrupted"
    report_service = AnalysisReportService()

    # Report has empty stages_completed and no audit, but has stage_validations
    corrupted_report = {
        "document_id": doc_id,
        "status": "degraded",
        "analysis_mode": "deterministic_fallback",
        "active_model": "qwen3.5:4b",
        "total_duration_sec": 0.88,
        "stages_completed": ["ingestion", "chunking", "mapping"],
        "stage_validations": [
            {
                "stage": "ingestion",
                "status": "completed",
                "duration_sec": 0.02,
                "input_size": 500,
                "output_size": 480,
            },
            {
                "stage": "chunking",
                "status": "completed",
                "duration_sec": 0.03,
                "input_size": 480,
                "output_size": 2,
            },
            {
                "stage": "mapping",
                "status": "completed",
                "duration_sec": 0.15,
                "input_size": 2,
                "output_size": 5,
            },
        ],
    }

    report_service.save_report(doc_id, corrupted_report)

    monkeypatch.setattr(system, "analysis_report_service", report_service)
    empty_tracker = PipelineStateTracker()
    monkeypatch.setattr(system, "pipeline_tracker", empty_tracker)

    status = asyncio.run(system.get_pipeline_status(document_id=doc_id))

    assert status["document_id"] == doc_id
    assert status["status"] == "degraded"
    assert status["analysis_mode"] == "deterministic_fallback"

    stages = status["stages"]
    # Invariant: exactly 7 stages, no duplicates
    assert len(stages) == 7
    assert list(stages.keys()) == PIPELINE_STAGES
    assert "llm_analysis" not in stages

    # Ingestion recovered
    assert stages["ingestion"]["status"] == "completed"
    assert stages["ingestion"]["duration_sec"] == 0.02
    assert stages["ingestion"]["input_size"] == 500

    # Mapping truthfully reflects fallback because rep_mode is deterministic_fallback
    assert stages["mapping"]["status"] == "fallback"
    assert stages["mapping"]["duration_sec"] == 0.15

    # Unexecuted stages recovered as pending
    for unexecuted in ["embedding", "graph_building", "evaluation", "hierarchy"]:
        assert stages[unexecuted]["status"] == "pending"
        assert stages[unexecuted]["duration_sec"] is None
        assert stages[unexecuted]["input_size"] is None
        assert stages[unexecuted]["output_size"] is None


def test_estimate_size_edge_cases():
    """Verify estimate_size handles pathological payloads without throwing exceptions."""
    assert estimate_size(None) == 0
    assert estimate_size("") == 0
    assert estimate_size("hello") == 5
    assert estimate_size(b"bytes") == 5
    assert estimate_size([]) == 0
    assert estimate_size([1, 2, 3]) == 3
    assert estimate_size({"a": 1}) == 1
    assert estimate_size(42) == 1
    assert estimate_size(3.14) == 1

    class CustomObj:
        word_count = 500
    assert estimate_size(CustomObj()) == 500

    class BrokenObj:
        @property
        def word_count(self):
            raise RuntimeError("boom")
    assert estimate_size(BrokenObj()) == 0
