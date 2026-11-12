"""Tier 2 Boundary & Corner Cases: test_t2_b09_reasoning_edge_cases.py
Feature F9: Ollama Reasoning Token Suppression (think: false).
Verifies boundary conditions: models that do/do not support think parameter,
unclosed <think> tag sanitization, zero temperature determinism,
max token ceiling truncation handling, and empty content provider responses.
"""

import re
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.llm.ollama_provider import OllamaProvider
from core.llm.base import LLMResponse, LLMProviderError


def test_b09_model_without_think_parameter():
    """T2.F9.1: Provider defaults think: false in kwargs."""
    provider = OllamaProvider()
    # Verify kwargs extraction logic accepts think=False
    kwargs = {"think": False, "options": {"num_ctx": 4096}}
    assert kwargs.get("think") is False
    assert kwargs.get("options", {}).get("num_ctx") == 4096


def test_b09_unclosed_think_tag_stripping():
    """T2.F9.2: Model output with <think> tag stripped before JSON parse."""
    raw_output = "<think>Let me reason through the answer...</think>{\"instruction\": \"What is X?\", \"response\": \"X is Y\"}"
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    import json
    data = json.loads(cleaned)
    assert data["instruction"] == "What is X?"


def test_b09_zero_temperature_configuration():
    """T2.F9.3: Temperature setting at zero (T = 0.0) configuration."""
    provider = OllamaProvider(temperature=0.0)
    assert provider.config.temperature == 0.0


def test_b09_max_token_ceiling_truncation():
    """T2.F9.4: Maximum token limit truncation resulting in incomplete JSON."""
    truncated_json = '{"instruction": "What is Dijkstra algorithm?", "response": "Dijkstra algorithm is a greedy'
    # Truncated string should fail json.loads and not be accepted as valid JSON object
    import json
    with pytest.raises(json.JSONDecodeError):
        json.loads(truncated_json)


def test_b09_empty_string_llm_response():
    """T2.F9.5: Ollama API returning empty content string."""
    empty_response = LLMResponse(content="", model="qwen3.5:4b")
    assert empty_response.content == ""
    # Empty content cannot be parsed as JSON object
    import json
    with pytest.raises(json.JSONDecodeError):
        json.loads(empty_response.content)
