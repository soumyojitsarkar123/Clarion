"""
Ollama Provider - Implementation for local Ollama models.
"""

import logging
import time
from typing import Dict, Any, Optional

from core.llm.base import (
    BaseLLMProvider, LLMResponse, ModelInfo, ProviderType,
    TokenUsage, LLMProviderError
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

    DEFAULT_MODEL = "qwen3.5:9b"
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

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client for Ollama."""
        try:
            from openai import AsyncOpenAI
            # Ollama provides OpenAI-compatible API
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base
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

        Args:
            prompt: User prompt
            system_message: System instructions
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            LLMResponse with generated content

        Raises:
            LLMProviderError: If API call fails
        """
        if not self._client:
            self._initialize_client()

        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            start_time = time.time()
            response = await self._client.chat.completions.create(
                model=self.model or self.DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout
            )

            elapsed_ms = (time.time() - start_time) * 1000

            content = response.choices[0].message.content
            if content is None:
                content = ""

            return LLMResponse(
                content=content,
                model=response.model if hasattr(response, 'model') else (self.model or self.DEFAULT_MODEL),
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0
                ),
                finish_reason=response.choices[0].finish_reason,
                response_time_ms=elapsed_ms
            )
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
        import json
        
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

        response = await self.generate(
            prompt=json_prompt,
            system_message=system_message,
            temperature=0.1,  # Lower temperature for structured output
            **kwargs
        )

        try:
            # Try to parse the response
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response.content}")
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
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
