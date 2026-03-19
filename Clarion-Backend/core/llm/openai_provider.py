"""
OpenAI Provider - Implementation for OpenAI GPT models.
"""

import logging
import time
from typing import Dict, Any, Optional

from core.llm.base import (
    BaseLLMProvider, LLMResponse, ModelInfo, ProviderType,
    TokenUsage, LLMProviderError
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation for GPT models.
    
    Supports OpenAI API with GPT-3.5, GPT-4, and other models.
    
    Example:
        provider = OpenAIProvider(
            api_key="your_openai_api_key_here",
            model="gpt-4",
            temperature=0.3
        )
    """
    
    DEFAULT_MODEL = "gpt-4"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 2000
    
    # Model configurations
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
        "gpt-3.5-turbo": {
            "context_window": 4096,
            "cost_input": 0.0015,
            "cost_output": 0.002,
            "supports_json": True
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            raise LLMProviderError("OpenAI API key is required")
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMProviderError("OpenAI library not installed. Install with: pip install openai")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using OpenAI chat completion API.
        
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
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                ),
                finish_reason=response.choices[0].finish_reason,
                response_time_ms=elapsed_ms
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMProviderError(f"OpenAI generation failed: {str(e)}")
    
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
        config = self.MODEL_CONFIGS.get(model_name, {
            "context_window": 4096,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "supports_json": True
        })

        return ModelInfo(
            provider=ProviderType.OPENAI,
            model_name=model_name,
            max_tokens=self.max_tokens,
            context_window=config.get("context_window", 4096),
            supports_functions=True,
            supports_json_mode=config.get("supports_json", True),
            cost_per_1k_input=config.get("cost_input"),
            cost_per_1k_output=config.get("cost_output")
        )
