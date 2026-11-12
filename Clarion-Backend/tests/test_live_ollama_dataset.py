"""
Integration tests for live Ollama model generation (qwen3.5:4b).

Explicitly checks for a running Ollama daemon at localhost:11434 with qwen3.5:4b.
If unavailable, tests are cleanly skipped.
All tests use the isolated_settings fixture to ensure production SQLite databases
remain completely untouched.
"""

import asyncio
import time
import pytest
import httpx

from core.llm.ollama_provider import OllamaProvider
from services.dataset_factory_service import DatasetFactoryService
from utils.config import settings


def _is_live_ollama_available() -> bool:
    """Check if Ollama is reachable and has the expected model."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            return any("qwen" in m.lower() for m in models)
        return False
    except Exception:
        return False


LIVE_OLLAMA = _is_live_ollama_available()


@pytest.mark.skipif(not LIVE_OLLAMA, reason="Live Ollama daemon with Qwen model unavailable at localhost:11434")
def test_live_ollama_reasoning_suppression_fast_path(isolated_settings):
    """
    Test live Ollama native /api/chat path with think=False.
    Verifies that reasoning tokens are suppressed and output is returned in <45s.
    """
    provider = OllamaProvider(model="qwen3.5:4b", timeout=45.0)

    start = time.time()
    response = asyncio.run(
        provider.generate(
            prompt="Explain the difference between gradient descent and stochastic gradient descent in two sentences.",
            system_message="You are a machine learning instructor. Be concise.",
            think=False,
            max_tokens=150,
        )
    )
    elapsed = time.time() - start

    assert len(response.content) > 20, "Generated content must not be empty"
    assert "<think>" not in response.content, "Thinking tokens must be suppressed"
    assert elapsed < 45.0, f"Native chat must complete in under 45s (took {elapsed:.2f}s)"


@pytest.mark.skipif(not LIVE_OLLAMA, reason="Live Ollama daemon with Qwen model unavailable at localhost:11434")
def test_live_ollama_structured_json_generation(isolated_settings):
    """
    Test live Ollama generate_structured with constrained schema.
    Verifies valid JSON parsing conforming to schema.
    """
    provider = OllamaProvider(model="qwen3.5:4b", timeout=45.0)
    schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["topic", "summary", "key_points"],
    }

    result = asyncio.run(
        provider.generate_structured(
            prompt="Summarize the concept of learning rate in optimization.",
            output_schema=schema,
            timeout=45.0,
            think=False,
        )
    )

    assert isinstance(result, dict)
    assert "topic" in result
    assert "summary" in result
    assert isinstance(result.get("key_points"), list)
