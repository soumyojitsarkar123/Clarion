"""
OpenAI Provider - Implementation for OpenAI GPT models.
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


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT provider implementation.
    
    Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, and other OpenAI models.
    Provides structured output via JSON mode and cost estimation.
    
    Example:
        provider = OpenAIProvider(
            api_key="sk-...",
            model="gpt-4",
            temperature=0.3
        )
        response = await provider.generate("Extract concepts from...")
    """
    
    DEFAULT_MODEL = "gpt-4"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 2000
    API_BASE = "https://api.openai.com/v1"
    
    # Model configurations with context windows and pricing
    MODEL_CONFIGS = {
        "gpt-4": {
            "context_window": 8192,
            "cost_input": 0.03,
            "cost_output": 0.06,
            "supports_json": True
        },
        "gpt-4-turbo": {
            "context_window": 128000,
            "cost_input": 0.01,
            "cost_output": 0.03,
            "supports_json": True
        },
        "gpt-4-turbo-preview": {
            "context_window": 128000,
            "cost_input": 0.01,
            "cost_output": 0.03,
            "supports_json": True
        },
        "gpt-3.5-turbo": {
            "context_window": 16385,
            "cost_input": 0.0005,
            "cost_output": 0.0015,
            "supports_json": True
        },
        "gpt-3.5-turbo-16k": {
            "context_window": 16385,
            "cost_input": 0.001,
            "cost_output": 0.002,
            "supports_json": True
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_base = kwargs.get("api_base", self.API_BASE)
    
    def _initialize_client(self) -> None:
        """Initialize httpx async client for OpenAI API."""
        if not self.api_key:
            raise LLMProviderError("OpenAI API key not provided")
        
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
        Generate text using OpenAI chat completions API.
        
        Args:
            prompt: User prompt
            system_message: System instructions
            temperature: Sampling temperature (0-2)
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
        if "response_format" in kwargs:
            request_data["response_format"] = kwargs["response_format"]
        if "top_p" in kwargs:
            request_data["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            request_data["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            request_data["presence_penalty"] = kwargs["presence_penalty"]
        
        start_time = time.time()
        
        try:
            response = await self._client.post(
                "/chat/completions",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Extract usage
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
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise LLMProviderError(f"OpenAI API error: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise LLMProviderError(f"Generation failed: {e}") from e
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using OpenAI's JSON mode.
        
        Args:
            prompt: User prompt
            output_schema: JSON schema for expected output
            system_message: System instructions
            **kwargs: Additional parameters
        
        Returns:
            Parsed JSON object
        """
        # Enhance prompt with schema information
        schema_prompt = f"""{prompt}

You must respond with valid JSON matching this schema:
{json.dumps(output_schema, indent=2)}

Important:
- Respond with JSON only, no markdown formatting
- Do not include ```json or ``` markers
- Ensure all required fields are present
- Use null for optional fields that don't apply
"""
        
        # Check if model supports JSON mode
        model = self.model or self.DEFAULT_MODEL
        config = self.MODEL_CONFIGS.get(model, {})
        
        if config.get("supports_json", False):
            # Use native JSON mode
            response = await self.generate(
                schema_prompt,
                system_message=system_message,
                response_format={"type": "json_object"},
                **kwargs
            )
        else:
            # Fallback: request JSON in prompt
            response = await self.generate(
                schema_prompt,
                system_message=system_message,
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
        """Get information about configured OpenAI model."""
        model = self.model or self.DEFAULT_MODEL
        
        # Handle model variants (e.g., gpt-4-0613 -> gpt-4)
        base_model = model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model
        if base_model not in self.MODEL_CONFIGS:
            base_model = self.DEFAULT_MODEL
        
        config = self.MODEL_CONFIGS.get(base_model, {})
        
        return ModelInfo(
            provider=ProviderType.OPENAI,
            model_name=model,
            max_tokens=self.max_tokens,
            context_window=config.get("context_window", 8192),
            supports_functions=True,
            supports_json_mode=config.get("supports_json", True),
            cost_per_1k_input=config.get("cost_input"),
            cost_per_1k_output=config.get("cost_output"),
            api_base=self.api_base
        )
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
