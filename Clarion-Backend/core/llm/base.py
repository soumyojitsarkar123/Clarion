"""
Base LLM Provider - Abstract interface for LLM implementations.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from services.token_budget_service import TokenBudgetService, TokenBudgetExceeded

THINK_TAG_REGEX = re.compile(r"<\s*think\b[^>]*>[\s\S]*?(?:<\s*/\s*think\s*>|$)", re.IGNORECASE)


def sanitize_llm_text(text: Optional[str]) -> str:
    """
    Sanitize model output by stripping reasoning tokens (<think>...</think>)
    and trimming surrounding whitespace.
    """
    if not text:
        return ""
    cleaned = THINK_TAG_REGEX.sub("", str(text))
    return cleaned.strip()



class LLMRuntimeState(str, Enum):
    """Runtime availability state of LLM subsystem."""
    LLM_AVAILABLE = "LLM_AVAILABLE"
    LLM_DEGRADED = "LLM_DEGRADED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    FALLBACK_USED = "FALLBACK_USED"


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    KIMI = "kimi"
    GLM = "glm"


class ModelInfo(BaseModel):
    """Information about an LLM model."""

    provider: ProviderType = Field(..., description="Provider type")
    model_name: str = Field(..., description="Model identifier")
    max_tokens: int = Field(2000, description="Maximum tokens per request")
    context_window: int = Field(8192, description="Context window size")
    supports_functions: bool = Field(False, description="Supports function calling")
    supports_json_mode: bool = Field(False, description="Supports JSON mode")
    cost_per_1k_input: Optional[float] = Field(
        None, description="Cost per 1K input tokens"
    )
    cost_per_1k_output: Optional[float] = Field(
        None, description="Cost per 1K output tokens"
    )
    api_base: Optional[str] = Field(None, description="API base URL")


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(0, description="Tokens in prompt")
    completion_tokens: int = Field(0, description="Tokens in completion")
    total_tokens: int = Field(0, description="Total tokens used")


class LLMResponse(BaseModel):
    """Standardized LLM response."""

    content: str = Field(..., description="Generated text content")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage")
    model: str = Field(..., description="Model used")
    finish_reason: str = Field("stop", description="Reason for completion")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    response_time_ms: Optional[float] = Field(
        None, description="Response time in milliseconds"
    )


class LLMProviderError(Exception):
    """Error in LLM provider operation."""

    pass


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    All LLM implementations must inherit from this class and implement
    the abstract methods. This provides a unified interface for
    switching between different LLM providers.

    Attributes:
        api_key: API key for the provider
        model: Model name/identifier
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
        **kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.additional_config = kwargs
        self._client = None
        self.last_token_metrics = {}
        self._budget_tokenizer = kwargs.get("tokenizer")

    def preflight_request(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        *,
        schema: Optional[Dict[str, Any]] = None,
        runtime_context_limit: Optional[int] = None,
    ):
        """Enforce a model/provider budget before any provider request is sent.

        Exact counts require an explicitly configured tokenizer matching the
        served model. Otherwise UTF-8 bytes are an intentionally conservative
        estimate and the returned metrics identify that limitation.
        """
        import os

        info = self.get_model_info()
        context_limit = int(
            runtime_context_limit
            or self.additional_config.get("context_window")
            or info.context_window
        )
        if info.provider == ProviderType.OLLAMA:
            # Ollama's API default is 4096 unless num_ctx is supplied.
            context_limit = int(runtime_context_limit or self.additional_config.get("num_ctx", 4096))
        output_reserve = int(max_tokens if max_tokens is not None else self.max_tokens)
        safety = int(self.additional_config.get("token_safety_tokens", 256))

        if self._budget_tokenizer is None:
            tokenizer_name = self.additional_config.get("tokenizer_name") or os.getenv("LLM_TOKENIZER_NAME")
            if tokenizer_name:
                try:
                    from transformers import AutoTokenizer
                    self._budget_tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name, use_fast=True, local_files_only=True
                    )
                except Exception:
                    # The estimate fallback remains explicit in metrics.
                    self._budget_tokenizer = None

        service = TokenBudgetService(
            context_limit=context_limit,
            reserved_output_tokens=output_reserve,
            safety_tokens=safety,
            tokenizer=self._budget_tokenizer,
        )
        messages = ([{"role": "system", "content": system_message}] if system_message else [])
        user_content = prompt
        if schema is not None:
            user_content += "\nStructured output schema:\n" + json.dumps(
                schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        messages.append({"role": "user", "content": user_content})
        template_used = False
        material = None
        apply_template = getattr(self._budget_tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                material = apply_template(messages, tokenize=False, add_generation_prompt=True)
                template_used = True
            except Exception:
                material = None
        if material is None:
            # Canonical conservative serialization includes role labels, message
            # boundaries, and structured-output schema in the counted payload.
            material = json.dumps(
                {"messages": messages, "structured_output_schema": schema},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
        plan = service.plan(material)
        exact_for_deployed_request = bool(plan.exact and template_used and schema is None)
        if plan.exact and not template_used:
            from dataclasses import replace
            plan = replace(plan, exact=False, counting_method="configured_tokenizer_serialized_request_estimate")
        special_overhead = None
        if self._budget_tokenizer is not None:
            try:
                with_special = self._budget_tokenizer.encode(material, add_special_tokens=True, truncation=False)
                without_special = self._budget_tokenizer.encode(material, add_special_tokens=False, truncation=False)
                special_overhead = max(0, len(with_special) - len(without_special))
            except Exception:
                pass
        self.last_token_metrics = {
            "input_tokens": plan.input_tokens,
            "actual_model_token_count": plan.input_tokens if exact_for_deployed_request else None,
            "estimated_token_count": None if exact_for_deployed_request else plan.input_tokens,
            "input_tokens_exact": exact_for_deployed_request,
            "counting_method": plan.counting_method,
            "utf8_byte_length": len(material.encode("utf-8")),
            "special_token_overhead": special_overhead,
            "chat_template_applied": template_used,
            "structured_schema_included": schema is not None,
            "context_limit": plan.context_limit,
            "context_utilization": round(plan.input_tokens / plan.context_limit, 6),
            "reserved_output_tokens": plan.reserved_output_tokens,
            "safety_tokens": plan.safety_tokens,
            "remaining_tokens": plan.remaining_tokens,
            "budget_rejected": not plan.fits,
        }
        if not plan.fits:
            service.require_fits(material)
        return plan

    def record_response_usage(self, response: "LLMResponse") -> "LLMResponse":
        """Attach measured provider usage to the preflight budget metrics."""
        usage = response.usage
        finish_reason = str(response.finish_reason or "").casefold()
        truncated = finish_reason in {"length", "max_tokens", "max tokens", "max_tokens_reached"}
        self.last_token_metrics.update({
            "actual_prompt_tokens": getattr(usage, "prompt_tokens", None),
            "actual_output_tokens": getattr(usage, "completion_tokens", None),
            "actual_total_tokens": getattr(usage, "total_tokens", None),
            "provider_finish_reason": response.finish_reason,
            "truncated": truncated,
        })
        response.metadata = dict(response.metadata or {})
        response.metadata["token_budget"] = dict(self.last_token_metrics)
        if truncated:
            raise LLMProviderError("Provider stopped at its output token limit; response may be incomplete")
        return response

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initialize the provider-specific API client.

        Must set self._client to the provider's client instance.
        Raises LLMProviderError if initialization fails.
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt text
            system_message: Optional system/instruction message
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with standardized format

        Raises:
            LLMProviderError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) matching a schema.

        Args:
            prompt: The user prompt text
            output_schema: JSON schema describing expected output structure
            system_message: Optional system/instruction message
            **kwargs: Provider-specific parameters

        Returns:
            Parsed JSON object matching the schema

        Raises:
            LLMProviderError: If generation or parsing fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the configured model.

        Returns:
            ModelInfo with provider, capabilities, and pricing
        """
        pass

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> Optional[float]:
        """
        Estimate the cost of a request in USD.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Estimated cost or None if pricing unavailable
        """
        info = self.get_model_info()

        if info.cost_per_1k_input and info.cost_per_1k_output:
            input_cost = (prompt_tokens / 1000) * info.cost_per_1k_input
            output_cost = (completion_tokens / 1000) * info.cost_per_1k_output
            return round(input_cost + output_cost, 6)

        return None

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate provider configuration.

        Returns:
            (is_valid, error_message)
        """
        if not self.api_key:
            return False, "API key not provided"

        if not self.model:
            return False, "Model not specified"

        return True, None

    def format_prompt_for_logging(self, prompt: str, max_length: int = 200) -> str:
        """
        Format prompt for logging (truncate if too long).

        Args:
            prompt: Original prompt
            max_length: Maximum length to log

        Returns:
            Truncated prompt string
        """
        if len(prompt) <= max_length:
            return prompt
        return prompt[:max_length] + "..."
