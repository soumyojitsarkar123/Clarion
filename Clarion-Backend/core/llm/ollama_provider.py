"""
Ollama Provider - Implementation for local Ollama models.
"""

import logging
import time
import re
import json
from typing import Dict, Any, Optional

from core.llm.base import (
    BaseLLMProvider, LLMResponse, ModelInfo, ProviderType,
    TokenUsage, LLMProviderError, sanitize_llm_text
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider implementation for local LLM models.

    Supports Ollama local deployment with models like Qwen, DeepSeek, Gemma, etc.
    Uses OpenAI-compatible API format for compatibility.

    Example:
        provider = OllamaProvider(
            model="qwen3.5:9b",
            api_base="http://localhost:11434/v1",
            temperature=0.3
        )
    """

    DEFAULT_MODEL = "qwen3.5:4b"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 2000

    # Common Ollama model configurations
    MODEL_CONFIGS = {
        "qwen3.5:9b": {
            "context_window": 32768,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        },
        "qwen3.5:4b": {
            "context_window": 32768,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        },
        "qwen3:latest": {
            "context_window": 32768,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        },
        "deepseek-coder:6.7b": {
            "context_window": 16384,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        },
        "gemma3:4b": {
            "context_window": 8192,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        },
        "llama3:8b": {
            "context_window": 8192,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        },
        "mistral:7b": {
            "context_window": 8192,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        }
    }

    def __init__(self, **kwargs):
        # Ollama doesn't require API key, use placeholder
        if "api_key" not in kwargs or kwargs["api_key"] is None:
            kwargs["api_key"] = "ollama"
        
        # Set default API base if not provided
        if "api_base" not in kwargs:
            kwargs["api_base"] = "http://localhost:11434/v1"
        
        super().__init__(**kwargs)
        self.api_base = kwargs.get("api_base", "http://localhost:11434/v1")
        self._initialize_client()

    def _get_native_base_url(self) -> str:
        """Return Ollama base URL without the OpenAI compatibility suffix (/v1)."""
        base = self.api_base or "http://localhost:11434/v1"
        return base[:-3] if base.endswith("/v1") else base.rstrip("/")

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client for Ollama."""
        try:
            from openai import AsyncOpenAI
            # Ollama provides OpenAI-compatible API
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                max_retries=0,
                timeout=float(self.timeout),
            )
            logger.info(f"Initialized Ollama client at {self.api_base}")
        except ImportError:
            raise LLMProviderError("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise LLMProviderError(f"Failed to connect to Ollama at {self.api_base}: {str(e)}")

    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Ollama chat completion API.

        Attempts native Ollama /api/chat async request path via httpx.AsyncClient first
        with stream: False, think: False, and format constraint if provided, falling back
        to OpenAI client with reasoning extraction if native fails.

        Args:
            prompt: User prompt
            system_message: System instructions
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments (timeout, think, num_ctx, format, etc.)

        Returns:
            LLMResponse with generated content

        Raises:
            LLMProviderError: If API call fails
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        timeout = float(kwargs.get("timeout", self.timeout))
        think = kwargs.get("think", False)
        num_ctx = kwargs.get("num_ctx", self.additional_config.get("num_ctx", 4096))
        budget_metrics = self.preflight_request(
            prompt, system_message, max_tokens, schema=kwargs.get("format"), runtime_context_limit=num_ctx
        )

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        # Fast Path: Native Ollama /api/chat with think: false
        native_url = f"{self._get_native_base_url()}/api/chat"
        payload = {
            "model": self.model or self.DEFAULT_MODEL,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }
        if "format" in kwargs:
            payload["format"] = kwargs["format"]

        try:
            import httpx
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(native_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    if not content and "thinking" in msg:
                        content = msg.get("thinking", "")
                    content = sanitize_llm_text(content)

                    elapsed_ms = (time.time() - start_time) * 1000
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)

                    return self.record_response_usage(LLMResponse(
                        content=content,
                        model=data.get("model", self.model or self.DEFAULT_MODEL),
                        usage=TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                        finish_reason=data.get("done_reason", "stop"),
                        response_time_ms=elapsed_ms,
                        metadata={"provider": "ollama_native", "total_duration_ns": data.get("total_duration"), "token_budget": budget_metrics.__dict__},
                    ))
                else:
                    logger.debug(f"Native Ollama request returned status {resp.status_code}: {resp.text}")
        except Exception as native_err:
            if self.last_token_metrics.get("truncated"):
                raise
            logger.debug(f"Native Ollama request failed, falling back to OpenAI client: {native_err}")

        # Fallback Path: OpenAI compatibility layer
        if not self._client:
            self._initialize_client()

        try:
            response = await self._client.chat.completions.create(
                model=self.model or self.DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"think": think, "options": {"num_ctx": num_ctx}},
                timeout=timeout,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            msg = response.choices[0].message
            content = getattr(msg, "content", None) or getattr(msg, "reasoning", "") or ""
            content = sanitize_llm_text(content)

            return self.record_response_usage(LLMResponse(
                content=content,
                model=response.model if hasattr(response, 'model') else (self.model or self.DEFAULT_MODEL),
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                ),
                finish_reason=response.choices[0].finish_reason,
                response_time_ms=elapsed_ms,
                metadata={"provider": "ollama_openai_fallback", "token_budget": budget_metrics.__dict__},
            ))
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise LLMProviderError(f"Ollama generation failed: {str(e)}")

    async def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) matching a schema.

        Passes format=output_schema for native Ollama grammar constraint.

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
        # Add instruction to output JSON
        schema_str = json.dumps(output_schema, indent=2)
        json_prompt = f"""{prompt}

Please respond with a valid JSON object matching this schema:
{schema_str}

Respond with ONLY the JSON object, no other text."""

        # Add system message to enforce JSON
        if system_message:
            system_message += "\n\nYou MUST respond with valid JSON only."
        else:
            system_message = "You are a JSON generation assistant. Respond with valid JSON only, no other text."

        # Pass format=output_schema for native Ollama grammar constraint
        kwargs.setdefault("format", output_schema)

        response = await self.generate(
            prompt=json_prompt,
            system_message=system_message,
            temperature=0.1,  # Lower temperature for structured output
            **kwargs
        )

        clean_content = sanitize_llm_text(response.content)

        # Stage 1: Strip thinking tags if any leaked through
        clean_content = re.sub(r"<think>.*?</think>", "", clean_content, flags=re.DOTALL).strip()

        # Stage 2: Strip markdown code fences
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
            clean_content = re.sub(r"\s*```$", "", clean_content).strip()

        try:
            # Stage 3: Try to parse the response directly
            result = json.loads(clean_content)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {clean_content}")

            # Stage 4: Try to extract balanced outer JSON from response
            json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError:
                    pass

            raise LLMProviderError(f"Failed to parse structured output: {str(e)}")

    def get_model_info(self) -> ModelInfo:
        """Get information about the configured model."""
        model_name = self.model or self.DEFAULT_MODEL
        
        # Try to find matching config (handle partial matches)
        config = self.MODEL_CONFIGS.get(model_name)
        if not config:
            # Try partial match
            for key in self.MODEL_CONFIGS:
                if model_name.startswith(key.split(':')[0]):
                    config = self.MODEL_CONFIGS[key]
                    break
        
        if not config:
            # Default config
            config = {
                "context_window": 8192,
                "cost_input": 0.0,
                "cost_output": 0.0,
                "supports_json": True
            }

        return ModelInfo(
            provider=ProviderType.OLLAMA,
            model_name=model_name,
            max_tokens=self.max_tokens,
            context_window=config.get("context_window", 8192),
            supports_functions=False,
            supports_json_mode=config.get("supports_json", True),
            cost_per_1k_input=config.get("cost_input"),
            cost_per_1k_output=config.get("cost_output"),
            api_base=self.api_base
        )

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate Ollama configuration.

        Returns:
            (is_valid, error_message)
        """
        if not self.model:
            return False, "Model not specified"
        
        # Try to ping Ollama to verify connection
        try:
            import requests
            response = requests.get(f"{self.api_base}/models", timeout=5)
            if response.status_code != 200:
                return False, f"Ollama API returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to Ollama at {self.api_base}: {str(e)}"
        except ImportError:
            # If requests not available, skip connection test
            pass
        
        return True, None
