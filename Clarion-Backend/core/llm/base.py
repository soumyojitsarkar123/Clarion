"""
Base LLM Provider - Abstract interface for LLM implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field


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
