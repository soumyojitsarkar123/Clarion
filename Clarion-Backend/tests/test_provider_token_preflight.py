"""Budget enforcement shared by every first-party provider implementation."""

import pytest

from core.llm.base import LLMProviderError, LLMResponse, TokenUsage
from core.llm.ollama_provider import OllamaProvider
from services.token_budget_service import TokenBudgetExceeded


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(f"{message['role']}|{message['content']}" for message in messages)
        return rendered + ("\nassistant|" if add_generation_prompt else "")


def test_provider_preflight_uses_runtime_context_output_reserve_and_exact_tokenizer():
    provider = OllamaProvider(
        model="qwen3.5:4b",
        tokenizer=CharacterTokenizer(),
        token_safety_tokens=1,
    )
    plan = provider.preflight_request(
        "abc", "sys", max_tokens=2, runtime_context_limit=100
    )
    assert plan.context_limit == 100
    assert plan.input_tokens == len("system|sys\nuser|abc\nassistant|")
    assert plan.reserved_output_tokens == 2
    assert plan.remaining_tokens == 100 - plan.input_tokens - 3
    assert plan.exact is True
    assert provider.last_token_metrics["actual_model_token_count"] == plan.input_tokens
    assert provider.last_token_metrics["utf8_byte_length"] == plan.input_tokens


def test_provider_preflight_counts_structured_schema_and_rejects_before_network():
    provider = OllamaProvider(
        model="qwen3.5:4b",
        tokenizer=CharacterTokenizer(),
        token_safety_tokens=0,
    )
    with pytest.raises(TokenBudgetExceeded):
        provider.preflight_request(
            "", max_tokens=1, schema={"required": ["field"]},
            runtime_context_limit=5,
        )


def test_provider_usage_is_kept_and_output_limit_is_not_silent():
    provider = OllamaProvider(model="qwen3.5:4b")
    with pytest.raises(LLMProviderError, match="may be incomplete"):
        provider.record_response_usage(LLMResponse(
            content="partial", model="qwen3.5:4b", finish_reason="length",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=7, total_tokens=19),
        ))
    assert provider.last_token_metrics["actual_prompt_tokens"] == 12
    assert provider.last_token_metrics["actual_output_tokens"] == 7
    assert provider.last_token_metrics["truncated"] is True
