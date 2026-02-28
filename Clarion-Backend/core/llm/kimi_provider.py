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
    
    Kimi is particularly strong at Chinese language understanding
    and supports very long context windows (up to 128K tokens).
    
    Example:
        provider = KimiProvider(
            api_key="your-kimi-api-key",
            model="moonshot-v1-32k",
            temperature=0.3
        )
        response = await provider.generate("Extract concepts from...")
    """
    
    DEFAULT_MODEL = "moonshot-v1-32k"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_API_BASE = "https://api.moonshot.cn/v1"
    
    # Model configurations
    MODEL_CONFIGS = {
        "moonshot-v1-8k": {
            "context_window": 8192,
            "cost_input": None,  # Pricing not publicly available
            "cost_output": None,
            "supports_json": True
        },
        "moonshot-v1-32k": {
            "context_window": 32768,
            "cost_input": None,
            "cost_output": None,
            "supports_json": True
        },
        "moonshot-v1-128k": {
            "context_window": 128000,
            "cost_input": None,
            "cost_output": None,
            "supports_json": True
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_base = kwargs.get("api_base", self.DEFAULT_API_BASE)
    
    def _initialize_client(self) -> None:
        """Initialize httpx async client for Kimi API."""
        if not self.api_key:
            raise LLMProviderError("Kimi API key not provided")
        
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
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
        Generate text using Kimi chat completions API.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
        
        Returns:
            LLMResponse with generated content
        """
        if not self._client:
            self._initialize_client()
        
        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request
        request_data = {
            "model": self.model or self.DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            request_data["top_p"] = kwargs["top_p"]
        
        start_time = time.time()
        
        try:
            response = await self._client.post(
                "/chat/completions",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            
            response_time = (time.time() - start_time) * 1000
            
            # Extract usage (Kimi uses similar format to OpenAI)
            usage_data = data.get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
            
            # Build response
            choice = data["choices"][0]
            return LLMResponse(
                content=choice["message"]["content"],
                usage=usage,
                model=data.get("model", self.model),
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=response_time
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Kimi API error: {e.response.status_code} - {e.response.text}")
            raise LLMProviderError(f"Kimi API error: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Kimi generation failed: {e}")
            raise LLMProviderError(f"Generation failed: {e}") from e
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using Kimi.
        
        Note: Kimi may not have native JSON mode like OpenAI,
        so we rely on prompt engineering.
        
        Args:
            prompt: User prompt
            output_schema: JSON schema for expected output
            system_message: System instructions
            **kwargs: Additional parameters
        
        Returns:
            Parsed JSON object
        """
        # Enhance system message with JSON instructions
        enhanced_system = system_message or ""
        enhanced_system += """\n\nYou are a structured data extraction assistant. 
You must respond with valid JSON only, no markdown formatting, no explanations.
"""
        
        schema_prompt = f"""{prompt}

Return your response as JSON matching this schema:
{json.dumps(output_schema, indent=2)}

Respond with JSON only, no markdown code blocks."""
        
        response = await self.generate(
            schema_prompt,
            system_message=enhanced_system,
            **kwargs
        )
        
        # Parse JSON
        try:
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output: {e}")
            logger.debug(f"Raw content: {response.content[:500]}")
            raise LLMProviderError(f"Failed to parse JSON response: {e}") from e
    
    def get_model_info(self) -> ModelInfo:
        """Get information about configured Kimi model."""
        model = self.model or self.DEFAULT_MODEL
        config = self.MODEL_CONFIGS.get(model, {})
        
        return ModelInfo(
            provider=ProviderType.KIMI,
            model_name=model,
            max_tokens=self.max_tokens,
            context_window=config.get("context_window", 32768),
            supports_functions=False,  # Kimi may not support function calling
            supports_json_mode=config.get("supports_json", True),
            cost_per_1k_input=config.get("cost_input"),
            cost_per_1k_output=config.get("cost_output"),
            api_base=self.api_base
        )
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
