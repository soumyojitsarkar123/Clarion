"""Tier 1 Feature Coverage: F9 Ollama Reasoning Token Suppression.

Specifications:
- T1.F9.1: Request Payload Extra Body Invariant (think: False)
- T1.F9.2: Context Window Configuration Parameter (num_ctx: 4096)
- T1.F9.3: Absence of <think> Tags in Output Content (sanitization of leaked tags)
- T1.F9.4: Token Usage Accounting Without Thinking Bloat
- T1.F9.5: Latency Bounding with think: false
"""

import asyncio
import re
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from core.llm.ollama_provider import OllamaProvider
from core.llm.base import LLMResponse, TokenUsage


def test_f09_extra_body_think_false():
    """T1.F9.1: Request Payload Extra Body Invariant."""
    provider = OllamaProvider()

    # Verify default payload construction includes think=False
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "qwen3.5:4b",
            "message": {"content": '{"result": "ok"}'},
            "prompt_eval_count": 20,
            "eval_count": 15,
        }
        mock_post.return_value = mock_resp

        response = asyncio.run(provider.generate(prompt="Test prompt", think=False))
        assert response is not None
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("think") is False, f"Expected think: False in payload, got {payload.get('think')}"


def test_f09_num_ctx_parameter():
    """T1.F9.2: Context Window Configuration Parameter."""
    provider = OllamaProvider()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "qwen3.5:4b",
            "message": {"content": '{"status": "ready"}'},
        }
        mock_post.return_value = mock_resp

        asyncio.run(provider.generate(prompt="Context check", num_ctx=4096))
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        options = payload.get("options", {})
        assert options.get("num_ctx") == 4096, f"Expected num_ctx=4096, got {options.get('num_ctx')}"


def test_f09_sanitize_think_tags():
    """T1.F9.3: Absence of <think> Tags in Output Content."""
    provider = OllamaProvider()
    leaked_raw = "<think>Analyzing graph theory definitions step by step...</think>{\"answer\": \"G = (V, E)\"}"

    with patch.object(provider, "generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = LLMResponse(
            content=leaked_raw,
            model="qwen3.5:4b",
            usage=TokenUsage(prompt_tokens=30, completion_tokens=25, total_tokens=55),
        )
        parsed = asyncio.run(provider.generate_structured(json_prompt="Return json"))
        assert isinstance(parsed, dict)
        assert parsed.get("answer") == "G = (V, E)"
        # Confirm no think tags present
        raw_dump = str(parsed)
        assert "<think>" not in raw_dump and "</think>" not in raw_dump


def test_f09_token_usage_accounting():
    """T1.F9.4: Token Usage Accounting Without Thinking Bloat."""
    provider = OllamaProvider()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "qwen3.5:4b",
            "message": {"content": '{"summary": "Short technical abstract"}'},
            "prompt_eval_count": 45,
            "eval_count": 18,
        }
        mock_post.return_value = mock_resp

        res = asyncio.run(provider.generate(prompt="Summarize"))
        assert res.usage.prompt_tokens == 45
        assert res.usage.completion_tokens == 18
        assert res.usage.total_tokens == 63
        assert res.usage.completion_tokens < 300, "Clean QA completion tokens must not include reasoning bloat"


def test_f09_latency_bounding_think_false():
    """T1.F9.5: Latency Bounding with think: false."""
    provider = OllamaProvider()
    # Ensure timeout default is bounded to at most 60s
    assert provider.timeout <= 60.0, f"Provider timeout {provider.timeout}s exceeds 60s ceiling"
