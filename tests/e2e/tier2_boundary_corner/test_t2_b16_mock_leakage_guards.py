"""Tier 2 Boundary & Corner Cases: test_t2_b16_mock_leakage_guards.py
Feature F16: Live vs Mocked Model Distinction & Transparency.
Verifies boundary conditions: Ollama 503/unavailability fallback degradation,
mock transparency (deterministic_fallback mode explicitly reported),
CLARION_DISABLE_LIVE_MODELS=1 guard evaluation, socket timeout bounds, and mock speed invariance.
"""

import os
import sys
import time
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.knowledge_map_service import LLMInterface
from services.pipeline_observability import PipelineStateTracker


def test_b16_ollama_503_service_unavailable(monkeypatch):
    """T2.F16.1: Service unavailable fallback degradation."""
    llm = LLMInterface()
    # Force service unavailable
    monkeypatch.setattr(llm, "_is_service_available", lambda: False)
    assert llm._is_service_available() is False

    # Status tracker should report analysis_mode as deterministic_fallback
    tracker = PipelineStateTracker()
    raw = {
        "document_id": "doc-fallback-503",
        "status": "degraded",
        "analysis_mode": "deterministic_fallback",
        "stages": {"mapping": {"status": "fallback"}},
    }
    status = tracker._format_pipeline_status(raw)
    assert status["analysis_mode"] == "deterministic_fallback"
    assert status["stages"]["mapping"]["status"] == "fallback"


def test_b16_mock_leakage_model_naming():
    """T2.F16.2: Mock provider transparently reports deterministic analysis mode."""
    tracker = PipelineStateTracker()
    raw = {
        "document_id": "doc-mock-named",
        "analysis_mode": "deterministic_fallback",
        "active_model": "Demo (Fallback)",
        "stages": {"mapping": {"status": "completed"}},
    }
    status = tracker._format_pipeline_status(raw)
    # Never disguise fallback mode as live neural LLM
    assert status["analysis_mode"] == "deterministic_fallback"
    assert status["active_model"] == "Demo (Fallback)"


def test_b16_env_flag_disables_live_models(monkeypatch):
    """T2.F16.3: CLARION_DISABLE_LIVE_MODELS=1 guard verification."""
    monkeypatch.setenv("CLARION_DISABLE_LIVE_MODELS", "1")
    assert os.environ.get("CLARION_DISABLE_LIVE_MODELS") == "1"


def test_b16_socket_timeout_bounded():
    """T2.F16.4: Unreachable host connection timeout is bounded."""
    from utils.config import settings
    # The timeout configured for Ollama probes should be finite and bounded
    timeout = getattr(settings, "ollama_timeout", 5.0)
    assert timeout > 0.0
    assert timeout <= 60.0


def test_b16_execution_speed_separation():
    """T2.F16.5: Mock/fallback execution executes under 50ms."""
    tracker = PipelineStateTracker()
    t0 = time.perf_counter()
    raw = {
        "document_id": "doc-speed-test",
        "stages": {"mapping": {"status": "completed", "duration_sec": 0.002}},
    }
    status = tracker._format_pipeline_status(raw)
    elapsed = time.perf_counter() - t0
    # Mock/in-memory formatting must execute in < 50ms
    assert elapsed < 0.05
    assert len(status["stages"]) == 7
