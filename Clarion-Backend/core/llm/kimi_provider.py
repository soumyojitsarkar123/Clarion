"""
Kimi Provider - Implementation for Moonshot AI (Kimi) models.
"""

import json
import logging
import time
from typing import Dict, Any, Optional

import httpx

from core.llm.base import (
    BaseLLMProvider, LLMResponse, ModelInfo, ProviderType,
    TokenUsage, LLMProviderError
)

logger = logging.getLogger(__name__)


class KimiProvider(BaseLLMProvider):
    """
    Moonshot AI (Kimi) provider implementation.
    
    Supports Kimi API for Chinese language understanding and generation.
    
    Example:
        provider = KimiProvider(
            api_key="your_kimi_api_key_here",
            model="moonshot-v1-8k",
            temperature=0.3
        )
    """
    
    DEFAULT_MODEL = "moonshot-v1-8k"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_API_BASE = "https://api.moonshot.cn/v1"
    
    # Model configurations
    MODEL_CONFIGS = {
        "moonshot-v1-8k": {
            "context_window": 8000,
            "cost_input": 0.002,
            "cost_output": 0.006,
            "supports_json": True
        },
        "moonshot-v1-32k": {
            "context_window": 32000,
            "cost_input": 0.006,
            "cost_output": 0.018,
            "supports_json": True
        },
        "moonshot-v1-128k": {
            "context_window": 128000,
            "cost_input": 0.02,
            "cost_output": 0.06,
            "supports_json": True
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            raise LLMProviderError("Kimi API key is required")
        self.api_base = kwargs.get("api_base", self.DEFAULT_API_BASE)
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize httpx async client for Kimi API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers=headers,
            timeout=self.timeout
        )
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Kimi API.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            temperature: Sampling temperature (0-1)
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
            
            payload = {
                "model": self.model or self.DEFAULT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            start_time = time.time()
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            elapsed_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                model=result.get("model", self.model),
                usage=TokenUsage(
                    prompt_tokens=result["usage"]["prompt_tokens"],
                    completion_tokens=result["usage"]["completion_tokens"],
                    total_tokens=result["usage"]["total_tokens"]
                ),
                finish_reason=result["choices"][0].get("finish_reason", "stop"),
                response_time_ms=elapsed_ms
            )
        except httpx.HTTPError as e:
            logger.error(f"Kimi API HTTP error: {e}")
            raise LLMProviderError(f"Kimi API call failed: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Kimi API response parse error: {e}")
            raise LLMProviderError(f"Failed to parse Kimi response: {str(e)}")
        except Exception as e:
            logger.error(f"Kimi generation error: {e}")
            raise LLMProviderError(f"Kimi generation failed: {str(e)}")
    
    def get_model_info(self) -> ModelInfo:
        """Get information about the configured model."""
        model_name = self.model or self.DEFAULT_MODEL
        config = self.MODEL_CONFIGS.get(model_name, {
            "context_window": 8000,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        })
        
        return ModelInfo(
            provider=ProviderType.KIMI,
            model_name=model_name,
            max_tokens=self.max_tokens,
            context_window=config.get("context_window", 8000),
            supports_functions=True,
            supports_json_mode=config.get("supports_json", True),
            cost_per_1k_input=config.get("cost_input"),
            cost_per_1k_output=config.get("cost_output")
        )
