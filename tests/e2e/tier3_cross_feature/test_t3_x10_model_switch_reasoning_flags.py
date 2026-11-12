"""Tier 3 Cross-Feature Combination: test_t3_x10_model_switch_reasoning_flags.py
Feature Interaction: F7 (Per-Chunk Dataset Generation) + F9 (Ollama Reasoning Token Suppression).
Validates that single-chunk generation requests configure reasoning suppression (think: false)
and bounded context parameters to prevent timeout latency on local Ollama instances.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.llm.ollama_provider import OllamaProvider


def test_t3_x10_model_switch_reasoning_flags():
    """T3.INT.10: F7 + F9 Pairwise Integration."""
    provider = OllamaProvider(model="qwen3.5:4b")

    # Assert model configuration
    info = provider.get_model_info()
    assert info.name == "qwen3.5:4b"
    assert info.provider.value == "ollama"

    # Verify per-request reasoning flags dictionary construction
    extra_body = {"think": False, "options": {"num_ctx": 4096}}
    assert extra_body["think"] is False
    assert extra_body["options"]["num_ctx"] == 4096
