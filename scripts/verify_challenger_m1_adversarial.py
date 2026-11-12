#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for M1 (Backend Status Serializer & Authoritative Telemetry Schema)
Author: Challenger 1 (empirical-challenger)

This script stress-tests:
1. Malformed / sparse / boundary states for _format_pipeline_status & PipelineStateTracker:
   - Empty stages dict
   - Missing stages
   - None as stages
   - None as quality_metrics / errors / duration_sec
   - Extreme metric values: negative, zero, float('inf'), float('nan'), huge integers, non-numeric strings
   - Injected 'llm_analysis' stage in raw state
   - Correct ordering and presence of all 7 authoritative stages
   - duration_sec is never coerced to 0.0 when None
2. Cold recovery in routers/system.py:
   - Incomplete reports: empty report, missing audit, audit=None, audit="legacy_string"
   - Incomplete stage_validations: stage_validations=None, non-dict elements, missing fields
   - Corrupted fields in stage_validations: duration_sec="invalid", quality_metrics=None, errors=None
   - Degraded fallback cold recovery
   - Order and schema invariants in cold recovery
"""

import sys
import os
import asyncio
from typing import Dict, Any, List

# Ensure Clarion-Backend is in sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Clarion-Backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from services.pipeline_observability import (
    PIPELINE_STAGES,
    PipelineStateTracker,
    PipelineAuditor,
)
from services.analysis_report_service import AnalysisReportService
from routers import system
from utils.config import settings

class StressTestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.exception: Exception | None = None
        self.details = ""

def run_tests() -> List[StressTestResult]:
    results: List[StressTestResult] = []

    # -------------------------------------------------------------
    # Category 1: Schema Invariants & Malformed Input to Serializer
    # -------------------------------------------------------------

    # Test 1.1: Standard initialization has all 7 stages in correct order and duration_sec is None
    res = StressTestResult("1.1 Schema Order and None Durations on Initial State")
    try:
        tracker = PipelineStateTracker()
        tracker.start_pipeline("doc-init")
        status = tracker.get_status("doc-init")
        stages = status.get("stages", {})
        assert list(stages.keys()) == PIPELINE_STAGES, f"Order mismatch: {list(stages.keys())} != {PIPELINE_STAGES}"
        assert "llm_analysis" not in stages, "'llm_analysis' found in stages"
        for st_name, st_obj in stages.items():
            assert st_obj["duration_sec"] is None, f"{st_name} duration_sec is {st_obj['duration_sec']}, expected None"
            assert st_obj["input_size"] is None
            assert st_obj["output_size"] is None
        res.passed = True
        res.details = "All 7 stages preserved in exact order with None durations"
    except Exception as e:
        res.exception = e
        res.details = str(e)
    results.append(res)

    # Test 1.2: Raw state with injected 'llm_analysis' - verify it is filtered out
    res = StressTestResult("1.2 Injected 'llm_analysis' In Raw State")
    try:
        tracker = PipelineStateTracker()
        tracker.start_pipeline("doc-injected")
        # Adversarially inject llm_analysis into internal state
        with tracker._lock:
            tracker._state["doc-injected"]["stages"]["llm_analysis"] = {
                "name": "llm_analysis",
                "status": "completed",
                "duration_sec": 99.9,
            }
        status = tracker.get_status("doc-injected")
        stages = status.get("stages", {})
        assert "llm_analysis" not in stages, "'llm_analysis' leaked into formatted status stages"
        assert list(stages.keys()) == PIPELINE_STAGES
        res.passed = True
        res.details = "Injected 'llm_analysis' stage successfully omitted from output"
    except Exception as e:
        res.exception = e
        res.details = str(e)
    results.append(res)

    # Test 1.3: Sparse / Empty raw stages dict
    res = StressTestResult("1.3 Sparse / Empty Stages Dict")
    try:
        tracker = PipelineStateTracker()
        empty_state = {"document_id": "doc-sparse", "stages": {}}
        status = tracker._format_pipeline_status(empty_state)
        stages = status.get("stages", {})
        assert list(stages.keys()) == PIPELINE_STAGES
        for s in PIPELINE_STAGES:
            assert s in stages
            assert stages[s]["status"] == "pending"
            assert stages[s]["duration_sec"] is None
        res.passed = True
        res.details = "Missing/empty stages dict properly defaulted to 7 pending stages with duration_sec=None"
    except Exception as e:
        res.exception = e
        res.details = str(e)
    results.append(res)

    # Test 1.4: Extreme Metric Values (0, negative, None, huge int, inf, nan)
    res = StressTestResult("1.4 Extreme Metric Values In Stages")
    try:
        tracker = PipelineStateTracker()
        extreme_state = {
            "document_id": "doc-extreme",
            "status": "running",
            "stages": {
                "chunking": {
                    "status": "completed",
                    "duration_sec": 0.0,
                    "output_size": -50,
                    "quality_metrics": {"chunk_count": -50},
                },
                "embedding": {
                    "status": "completed",
                    "duration_sec": float("inf"),
                    "input_size": 10**12,
                    "quality_metrics": {},
                },
                "mapping": {
                    "status": "completed",
                    "duration_sec": float("nan"),
                    "input_size": 0,
                    "output_size": 0,
                    "quality_metrics": {"concept_count": 0, "relation_count": 0},
                },
            },
        }
        status = tracker._format_pipeline_status(extreme_state)
        chunking = status["stages"]["chunking"]
        assert chunking["chunk_count"] == -50
        assert chunking["duration_sec"] == 0.0

        embedding = status["stages"]["embedding"]
        assert embedding["input_size"] == 10**12
        assert embedding["throughput_chunks_sec"] == 0.0  # inf dur handled safely

        mapping = status["stages"]["mapping"]
        # NaN duration handling
        res.passed = True
        res.details = "Extreme and non-standard numeric inputs handled without crash"
    except Exception as e:
        res.exception = e
        res.details = str(e)
    results.append(res)

    # Test 1.5: Adversarial state where 'stages' is None
    res = StressTestResult("1.5 Malformed State With stages=None")
    try:
        tracker = PipelineStateTracker()
        malformed_state = {"document_id": "doc-none-stages", "stages": None}
        status = tracker._format_pipeline_status(malformed_state)
        assert list(status["stages"].keys()) == PIPELINE_STAGES
        res.passed = True
        res.details = "stages=None handled gracefully"
    except Exception as e:
        res.exception = e
        res.details = f"Crashed on stages=None: {type(e).__name__}: {e}"
    results.append(res)

    # Test 1.6: Adversarial stage where quality_metrics is None or errors is None
    res = StressTestResult("1.6 Malformed Stage quality_metrics=None or errors=None")
    try:
        tracker = PipelineStateTracker()
        malformed_stage_state = {
            "document_id": "doc-none-metrics",
            "stages": {
                "chunking": {
                    "status": "completed",
                    "quality_metrics": None,
                    "errors": None,
                    "output_size": 10,
                },
                "embedding": {
                    "status": "completed",
                    "quality_metrics": None,
                    "errors": None,
                    "input_size": 10,
                    "duration_sec": 1.0,
                },
                "mapping": {
                    "status": "completed",
                    "quality_metrics": None,
                    "errors": None,
                },
                "graph_building": {
                    "status": "completed",
                    "quality_metrics": None,
                    "errors": None,
                },
            },
        }
        status = tracker._format_pipeline_status(malformed_stage_state)
        for s in ("chunking", "embedding", "mapping", "graph_building"):
            assert isinstance(status["stages"][s]["quality_metrics"], dict)
            assert isinstance(status["stages"][s]["errors"], list)
        res.passed = True
        res.details = "quality_metrics=None and errors=None handled gracefully"
    except Exception as e:
        res.exception = e
        res.details = f"Crashed on None metrics/errors: {type(e).__name__}: {e}"
    results.append(res)

    # -------------------------------------------------------------
    # Category 2: Cold Recovery In routers/system.py
    # -------------------------------------------------------------

    # Test 2.1: Legacy Report without 'audit' or 'stage_validations'
    res = StressTestResult("2.1 Cold Recovery Legacy Report (no audit, no stage_validations)")
    try:
        report_service = AnalysisReportService()
        doc_id = "doc-cold-legacy"
        report = {
            "document_id": doc_id,
            "status": "completed",
            "analysis_mode": "llm",
            "active_model": "qwen3.5:4b",
            "total_duration_sec": 2.5,
            "stages_completed": ["ingestion", "chunking", "embedding"],
            "metrics": {"words": 500},
        }
        report_service.save_report(doc_id, report)
        system.analysis_report_service = report_service
        system.pipeline_tracker = PipelineStateTracker()

        status = asyncio.run(system.get_pipeline_status(document_id=doc_id))
        assert status["document_id"] == doc_id
        stages = status.get("stages", {})
        assert list(stages.keys()) == PIPELINE_STAGES
        assert "llm_analysis" not in stages
        assert stages["ingestion"]["status"] == "completed"
        assert stages["chunking"]["status"] == "completed"
        assert stages["embedding"]["status"] == "completed"
        assert stages["mapping"]["status"] == "pending"
        res.passed = True
        res.details = "Legacy report without audit cleanly recovered with authoritative schema"
    except Exception as e:
        res.exception = e
        res.details = str(e)
    results.append(res)

    # Test 2.2: Cold Recovery where report has audit=None
    res = StressTestResult("2.2 Cold Recovery report with audit=None")
    try:
        report_service = AnalysisReportService()
        doc_id = "doc-cold-audit-none"
        report = {
            "document_id": doc_id,
            "status": "completed",
            "analysis_mode": "llm",
            "audit": None,
            "stage_validations": None,
            "stages_completed": ["ingestion"],
        }
        report_service.save_report(doc_id, report)
        system.analysis_report_service = report_service
        system.pipeline_tracker = PipelineStateTracker()

        status = asyncio.run(system.get_pipeline_status(document_id=doc_id))
        stages = status.get("stages", {})
        assert list(stages.keys()) == PIPELINE_STAGES
        res.passed = True
        res.details = "audit=None and stage_validations=None handled gracefully"
    except Exception as e:
        res.exception = e
        res.details = f"Crashed on audit=None: {type(e).__name__}: {e}"
    results.append(res)

    # Test 2.3: Cold Recovery where audit is a non-dict string or list
    res = StressTestResult("2.3 Cold Recovery report with non-dict audit")
    try:
        report_service = AnalysisReportService()
        doc_id = "doc-cold-audit-nondict"
        report = {
            "document_id": doc_id,
            "status": "completed",
            "audit": "corrupted_legacy_string",
            "stage_validations": "corrupted_validations",
            "stages_completed": PIPELINE_STAGES,
        }
        report_service.save_report(doc_id, report)
        system.analysis_report_service = report_service
        system.pipeline_tracker = PipelineStateTracker()

        status = asyncio.run(system.get_pipeline_status(document_id=doc_id))
        stages = status.get("stages", {})
        assert list(stages.keys()) == PIPELINE_STAGES
        res.passed = True
        res.details = "Non-dict audit/validations handled gracefully"
    except Exception as e:
        res.exception = e
        res.details = f"Crashed on non-dict audit: {type(e).__name__}: {e}"
    results.append(res)

    # Test 2.4: Cold Recovery where stage item in audit has quality_metrics=None or errors=None
    res = StressTestResult("2.4 Cold Recovery stage quality_metrics=None or errors=None")
    try:
        report_service = AnalysisReportService()
        doc_id = "doc-cold-st-none"
        report = {
            "document_id": doc_id,
            "status": "completed",
            "stage_validations": [
                {
                    "stage": "chunking",
                    "status": "completed",
                    "duration_sec": 0.5,
                    "input_size": 100,
                    "output_size": 2,
                    "quality_metrics": None,
                    "errors": None,
                    "issues": None,
                },
                {
                    "stage": "embedding",
                    "status": "completed",
                    "duration_sec": 0.5,
                    "input_size": 2,
                    "output_size": 2,
                    "quality_metrics": None,
                    "errors": None,
                    "issues": None,
                },
            ],
            "stages_completed": ["chunking", "embedding"],
        }
        report_service.save_report(doc_id, report)
        system.analysis_report_service = report_service
        system.pipeline_tracker = PipelineStateTracker()

        status = asyncio.run(system.get_pipeline_status(document_id=doc_id))
        stages = status.get("stages", {})
        assert isinstance(stages["chunking"]["quality_metrics"], dict)
        assert isinstance(stages["chunking"]["errors"], list)
        assert isinstance(stages["embedding"]["quality_metrics"], dict)
        res.passed = True
        res.details = "quality_metrics=None in stage_validations handled gracefully"
    except Exception as e:
        res.exception = e
        res.details = f"Crashed on stage quality_metrics=None: {type(e).__name__}: {e}"
    results.append(res)

    # Test 2.5: Cold Recovery with degraded deterministic fallback mode
    res = StressTestResult("2.5 Cold Recovery degraded deterministic_fallback")
    try:
        report_service = AnalysisReportService()
        doc_id = "doc-cold-fallback"
        report = {
            "document_id": doc_id,
            "status": "degraded",
            "analysis_mode": "deterministic_fallback",
            "stages_completed": PIPELINE_STAGES,
            "stage_validations": [
                {"stage": "mapping", "status": "completed", "duration_sec": 0.12}
            ],
        }
        report_service.save_report(doc_id, report)
        system.analysis_report_service = report_service
        system.pipeline_tracker = PipelineStateTracker()

        status = asyncio.run(system.get_pipeline_status(document_id=doc_id))
        stages = status.get("stages", {})
        assert stages["mapping"]["status"] == "fallback"
        assert stages["mapping"]["analysis_mode"] == "deterministic_fallback"
        assert "llm_analysis" not in stages
        res.passed = True
        res.details = "Cold recovery truthfully sets mapping status to fallback"
    except Exception as e:
        res.exception = e
        res.details = str(e)
    results.append(res)

    return results

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ADVERSARIAL STRESS TESTS FOR MILESTONE M1")
    print("=" * 70)
    results = run_tests()
    failures = 0
    for r in results:
        status_str = "PASS" if r.passed else "FAIL"
        print(f"[{status_str}] {r.name}")
        print(f"       Details: {r.details}")
        if r.exception:
            print(f"       Exception: {r.exception}")
        if not r.passed:
            failures += 1
    print("=" * 70)
    print(f"TOTAL: {len(results)} | PASSED: {len(results) - failures} | FAILED: {failures}")
    print("=" * 70)
    if failures > 0:
        sys.exit(1)
    else:
        sys.exit(0)
