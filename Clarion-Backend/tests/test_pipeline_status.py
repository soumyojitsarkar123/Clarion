"""
Targeted tests for Authoritative Pipeline Telemetry Schema and Status Serializer.

Verifies:
1. Exactly 7 authoritative pipeline stages present in status output.
2. No synthetic duplicate 'llm_analysis' stage in stages.
3. Unexecuted / pending stages have duration_sec=None (never coerced to 0.0).
4. Genuine metrics (input_size, output_size, duration_sec, quality_metrics) pass through without stripping.
5. Derived metrics (chunk_count, throughput, node_count, edge_count, model) are included.
6. Pipeline report persists complete stage telemetry in stage_validations and audit.
7. Cold recovery restores truthful durations, input/output sizes, and quality metrics from saved reports.
8. Degraded fallback mode truthfully reflects fallback status on mapping without duplicating stages.
"""

import asyncio
import time
import pytest

from services.pipeline_observability import (
    PIPELINE_STAGES,
    PipelineStateTracker,
    PipelineAuditor,
)
from services.analysis_report_service import AnalysisReportService
from services.processing_pipeline import ProcessingPipeline
from routers import system
from utils.config import settings


def test_authoritative_7_stages_and_no_llm_analysis():
    """PipelineStateTracker returns strictly the 7 authoritative stages without llm_analysis."""
    tracker = PipelineStateTracker()
    tracker.start_pipeline("doc-schema-test")

    status = tracker.get_status("doc-schema-test")
    assert status["document_id"] == "doc-schema-test"
    stages = status["stages"]

    # Exactly 7 stages
    assert len(stages) == 7
    assert set(stages.keys()) == set(PIPELINE_STAGES)
    assert "llm_analysis" not in stages

    for stage_name in PIPELINE_STAGES:
        st = stages[stage_name]
        assert st["name"] == stage_name
        assert st["status"] == "pending"
        # Pending stage durations must be None, not coerced 0.0
        assert st["duration_sec"] is None
        assert st["input_size"] is None
        assert st["output_size"] is None
        if stage_name == "mapping":
            assert st["quality_metrics"]["analysis_mode"] == "llm"
            assert st["quality_metrics"]["model"]
        else:
            assert st["quality_metrics"] == {}


def test_unmeasured_derived_metrics_remain_unavailable():
    tracker = PipelineStateTracker()
    tracker.start_pipeline("doc-unmeasured")
    tracker.complete_stage("doc-unmeasured", "chunking")
    tracker.complete_stage("doc-unmeasured", "embedding")
    tracker.complete_stage("doc-unmeasured", "graph_building")
    stages = tracker.get_status("doc-unmeasured")["stages"]

    assert "chunk_count" not in stages["chunking"]
    assert stages["embedding"]["throughput_chunks_sec"] is None
    assert "node_count" not in stages["graph_building"]
    assert "edge_count" not in stages["graph_building"]


def test_estimate_size_handles_raising_attribute_access():
    from services.pipeline_observability import estimate_size

    class RaisingProperty:
        @property
        def number_of_nodes(self):
            raise RuntimeError("unusual property")

    class RaisingGetattribute:
        def __getattribute__(self, name):
            if name == "number_of_nodes":
                raise RuntimeError("unusual __getattribute__")
            return super().__getattribute__(name)

    assert estimate_size(RaisingProperty()) == 1
    assert estimate_size(RaisingGetattribute()) == 1


def test_all_7_stages_truthful_metrics_passthrough():
    """All 7 stages retain genuine input_size, output_size, duration_sec, and quality_metrics."""
    tracker = PipelineStateTracker()
    doc_id = "doc-full-telemetry"
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)
    tracker.start_pipeline(doc_id)

    # 1. Ingestion
    auditor.start_stage("ingestion", input_size=5000, progress=5)
    time.sleep(0.005)
    auditor.complete_stage("ingestion", output_size=4800, quality_metrics={"word_count": 800, "has_text": True}, progress=10)

    # 2. Chunking
    auditor.start_stage("chunking", input_size=4800, progress=15)
    time.sleep(0.005)
    auditor.complete_stage("chunking", output_size=12, quality_metrics={"chunk_count": 12, "avg_chunk_words": 66.7, "sections_detected": 3}, progress=30)

    # 3. Embedding
    auditor.start_stage("embedding", input_size=12, progress=35)
    time.sleep(0.005)
    auditor.complete_stage("embedding", output_size=12, quality_metrics={"indexed": True, "chunk_count": 12}, progress=50)

    # 4. Mapping
    auditor.start_stage("mapping", input_size=12, progress=55)
    time.sleep(0.005)
    auditor.complete_stage("mapping", output_size=25, quality_metrics={"concept_count": 15, "relation_count": 10, "relation_concept_ratio": 0.67}, progress=65)

    # 5. Graph building
    auditor.start_stage("graph_building", input_size=25, progress=70)
    time.sleep(0.005)
    auditor.complete_stage("graph_building", output_size=25, quality_metrics={"node_count": 15, "edge_count": 10, "density": 0.095, "is_weakly_connected": True}, progress=75)

    # 6. Evaluation
    auditor.start_stage("evaluation", input_size=22, progress=80)
    time.sleep(0.005)
    auditor.complete_stage("evaluation", output_size=2, quality_metrics={"overall_score": 0.88, "risk_score": 0.12, "flags_count": 2}, progress=90)

    # 7. Hierarchy
    auditor.start_stage("hierarchy", input_size=25, progress=95)
    time.sleep(0.005)
    auditor.complete_stage("hierarchy", output_size=5, quality_metrics={"layers_count": 3, "chains_count": 2, "has_topic_tree": True}, progress=100)

    tracker.complete_pipeline(doc_id, is_degraded=False, analysis_mode="llm", active_model="qwen3.5:4b")

    status = tracker.get_status(doc_id)
    assert status["status"] == "completed"
    assert status["total_duration_sec"] is not None
    assert status["total_duration_sec"] > 0
    stages = status["stages"]

    assert len(stages) == 7
    assert "llm_analysis" not in stages

    # Ingestion: check base sizes are not stripped
    assert stages["ingestion"]["input_size"] == 5000
    assert stages["ingestion"]["output_size"] == 4800
    assert stages["ingestion"]["duration_sec"] is not None and stages["ingestion"]["duration_sec"] > 0
    assert stages["ingestion"]["quality_metrics"]["word_count"] == 800

    # Chunking: check chunk_count and base sizes
    assert stages["chunking"]["input_size"] == 4800
    assert stages["chunking"]["output_size"] == 12
    assert stages["chunking"]["chunk_count"] == 12
    assert stages["chunking"]["quality_metrics"]["chunk_count"] == 12

    # Embedding: throughput, device, chunk_count
    assert stages["embedding"]["input_size"] == 12
    assert stages["embedding"]["output_size"] == 12
    assert stages["embedding"]["throughput_chunks_sec"] > 0
    assert "device" in stages["embedding"]

    # Mapping: model, analysis_mode, concept_count
    assert stages["mapping"]["input_size"] == 12
    assert stages["mapping"]["output_size"] == 25
    assert stages["mapping"]["model"] == "qwen3.5:4b"
    assert stages["mapping"]["quality_metrics"]["concept_count"] == 15

    # Graph building: node_count, edge_count, base sizes
    assert stages["graph_building"]["input_size"] == 25
    assert stages["graph_building"]["output_size"] == 25
    assert stages["graph_building"]["node_count"] == 15
    assert stages["graph_building"]["edge_count"] == 10

    # Evaluation
    assert stages["evaluation"]["input_size"] == 22
    assert stages["evaluation"]["output_size"] == 2
    assert stages["evaluation"]["quality_metrics"]["overall_score"] == 0.88

    # Hierarchy
    assert stages["hierarchy"]["input_size"] == 25
    assert stages["hierarchy"]["output_size"] == 5
    assert stages["hierarchy"]["quality_metrics"]["layers_count"] == 3


def test_cold_recovery_restores_truthful_durations_and_sizes(isolated_settings, monkeypatch):
    """Cold recovery from persisted SQLite report restores truthful stage metrics."""
    doc_id = "doc-cold-test"
    report_service = AnalysisReportService()

    persisted_report = {
        "document_id": doc_id,
        "status": "completed",
        "analysis_mode": "llm",
        "active_model": "qwen3.5:4b",
        "total_duration_sec": 1.45,
        "stages_completed": PIPELINE_STAGES,
        "metrics": {"words": 800, "chunks": 12, "concepts": 15},
        "stage_validations": [
            {
                "stage": "ingestion",
                "status": "completed",
                "duration_sec": 0.05,
                "input_size": 5000,
                "output_size": 4800,
                "quality_metrics": {"word_count": 800},
            },
            {
                "stage": "chunking",
                "status": "completed",
                "duration_sec": 0.08,
                "input_size": 4800,
                "output_size": 12,
                "quality_metrics": {"chunk_count": 12, "avg_chunk_words": 66.7},
            },
            {
                "stage": "embedding",
                "status": "completed",
                "duration_sec": 0.25,
                "input_size": 12,
                "output_size": 12,
                "quality_metrics": {"indexed": True, "chunk_count": 12},
            },
            {
                "stage": "mapping",
                "status": "completed",
                "duration_sec": 0.60,
                "input_size": 12,
                "output_size": 25,
                "quality_metrics": {"concept_count": 15, "relation_count": 10},
            },
            {
                "stage": "graph_building",
                "status": "completed",
                "duration_sec": 0.12,
                "input_size": 25,
                "output_size": 25,
                "quality_metrics": {"node_count": 15, "edge_count": 10},
            },
            {
                "stage": "evaluation",
                "status": "completed",
                "duration_sec": 0.15,
                "input_size": 22,
                "output_size": 2,
                "quality_metrics": {"overall_score": 0.88, "flags_count": 2},
            },
            {
                "stage": "hierarchy",
                "status": "completed",
                "duration_sec": 0.20,
                "input_size": 25,
                "output_size": 5,
                "quality_metrics": {"layers_count": 3, "chains_count": 2},
            },
        ],
    }

    report_service.save_report(doc_id, persisted_report)

    # Monkeypatch router's analysis_report_service and pipeline_tracker
    monkeypatch.setattr(system, "analysis_report_service", report_service)
    empty_tracker = PipelineStateTracker()
    monkeypatch.setattr(system, "pipeline_tracker", empty_tracker)

    # Call get_pipeline_status
    status = asyncio.run(system.get_pipeline_status(document_id=doc_id))

    assert status["document_id"] == doc_id
    assert status["status"] == "completed"
    assert status["total_duration_sec"] == 1.45
    stages = status["stages"]

    # Authoritative 7 stages, no llm_analysis
    assert len(stages) == 7
    assert "llm_analysis" not in stages

    # Check restored truthful metrics
    assert stages["ingestion"]["duration_sec"] == 0.05
    assert stages["ingestion"]["input_size"] == 5000
    assert stages["ingestion"]["output_size"] == 4800

    assert stages["chunking"]["duration_sec"] == 0.08
    assert stages["chunking"]["chunk_count"] == 12

    assert stages["embedding"]["duration_sec"] == 0.25
    assert stages["embedding"]["throughput_chunks_sec"] == 48.0

    assert stages["mapping"]["duration_sec"] == 0.60
    assert stages["mapping"]["quality_metrics"]["concept_count"] == 15

    assert stages["graph_building"]["duration_sec"] == 0.12
    assert stages["graph_building"]["node_count"] == 15
    assert stages["graph_building"]["edge_count"] == 10

    assert stages["evaluation"]["duration_sec"] == 0.15
    assert stages["evaluation"]["output_size"] == 2

    assert stages["hierarchy"]["duration_sec"] == 0.20
    assert stages["hierarchy"]["quality_metrics"]["layers_count"] == 3


def test_degraded_fallback_truthful_mapping_status():
    """When pipeline degrades to deterministic fallback, mapping status reflects fallback with no synthetic rows."""
    tracker = PipelineStateTracker()
    doc_id = "doc-degraded-truth"
    tracker.start_pipeline(doc_id)
    tracker.start_stage(doc_id, "mapping", input_size=5)
    tracker.complete_stage(doc_id, "mapping", output_size=8)
    tracker.complete_pipeline(
        doc_id,
        is_degraded=True,
        analysis_mode="deterministic_fallback",
        active_model="qwen3.5:4b",
    )

    status = tracker.get_status(doc_id)
    assert status["status"] == "degraded"
    assert status["analysis_mode"] == "deterministic_fallback"

    stages = status["stages"]
    assert len(stages) == 7
    assert "llm_analysis" not in stages

    assert stages["mapping"]["status"] == "fallback"
    assert stages["mapping"]["analysis_mode"] == "deterministic_fallback"
    assert stages["mapping"]["model"] == "qwen3.5:4b"


def test_processing_pipeline_persists_full_stage_telemetry(isolated_settings):
    """ProcessingPipeline._build_analysis_report persists full stage telemetry in stage_validations and audit."""
    pipeline = ProcessingPipeline()
    doc_id = "doc-persist-test"

    audit_payload = {
        "document_id": doc_id,
        "stages": [
            {
                "stage": "ingestion",
                "status": "completed",
                "duration_sec": 0.042,
                "input_size": 2048,
                "output_size": 1900,
                "start_time": "2026-11-11T05:00:00Z",
                "end_time": "2026-11-11T05:00:01Z",
                "quality_metrics": {"word_count": 350},
                "validation": {"passed": True, "issues": []},
            },
            {
                "stage": "chunking",
                "status": "completed",
                "duration_sec": 0.085,
                "input_size": 1900,
                "output_size": 4,
                "start_time": "2026-11-11T05:00:01Z",
                "end_time": "2026-11-11T05:00:02Z",
                "quality_metrics": {"chunk_count": 4},
                "validation": {"passed": True, "issues": []},
            },
        ],
    }

    results = {
        "document_id": doc_id,
        "status": "completed",
        "analysis_mode": "llm",
        "active_model": "qwen3.5:4b",
        "stages_completed": ["ingestion", "chunking"],
        "stages_failed": [],
        "skipped_stages": [],
        "audit": audit_payload,
        "artifacts": {
            "document": {"word_count": 350, "page_count": 1},
            "chunks": {"count": 4},
            "summary": {"section_count": 1},
        },
    }

    report = pipeline._build_analysis_report(
        results=results,
        success=True,
        error=None,
        knowledge_map=None,
        graph=None,
        evaluation={},
        hierarchy={},
        summary=None,
    )

    assert "audit" in report
    assert report["audit"] == audit_payload
    assert "stage_validations" in report
    validations = {v["stage"]: v for v in report["stage_validations"]}

    assert "ingestion" in validations
    ing = validations["ingestion"]
    assert ing["duration_sec"] == 0.042
    assert ing["input_size"] == 2048
    assert ing["output_size"] == 1900
    assert ing["start_time"] == "2026-11-11T05:00:00Z"
    assert ing["end_time"] == "2026-11-11T05:00:01Z"
    assert ing["quality_metrics"]["word_count"] == 350

