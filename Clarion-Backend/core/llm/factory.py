"""
LLM Factory - Factory for creating LLM provider instances.
"""

import logging
from typing import Dict, Any, Type, Optional
import os

from core.llm.base import BaseLLMProvider, ProviderType, LLMProviderError
from core.llm.openai_provider import OpenAIProvider
from core.llm.kimi_provider import KimiProvider
from core.llm.glm_provider import GLMProvider
from core.llm.ollama_provider import OllamaProvider
from utils.config import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LLM provider instances.

    Supports configuration-driven provider selection and allows
    registration of custom providers.

    Example:
        # Create from configuration
        config = {
            "provider": "openai",
            "api_key": "sk-...",
            "model": "gpt-4",
            "temperature": 0.3
        }
        provider = LLMFactory.create_from_config(config)

        # Or create directly
        provider = LLMFactory.create("openai", api_key="...", model="gpt-4")
    """

    _providers: Dict[ProviderType, Type[BaseLLMProvider]] = {
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.KIMI: KimiProvider,
        ProviderType.GLM: GLMProvider,
    }

    @classmethod
    def register_provider(
        cls, provider_type: ProviderType, provider_class: Type[BaseLLMProvider]
    ) -> None:
        """
        Register a new provider type.

        Args:
            provider_type: Provider type identifier
            provider_class: Provider implementation class
        """
        cls._providers[provider_type] = provider_class
        logger.info(f"Registered provider: {provider_type}")

    @classmethod
    def create(cls, provider_type: str, **config) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Provider type ("openai", "kimi", "glm", etc.)
            **config: Provider configuration (api_key, model, etc.)

        Returns:
            Configured provider instance

        Raises:
            LLMProviderError: If provider type unknown or configuration invalid
        """
        # Convert string to enum
        try:
            provider_enum = ProviderType(provider_type.lower())
        except ValueError:
            raise LLMProviderError(f"Unknown provider type: {provider_type}")

        if provider_enum not in cls._providers:
            raise LLMProviderError(f"Provider not registered: {provider_type}")

        provider_class = cls._providers[provider_enum]

        # Create instance
        try:
            provider = provider_class(**config)

            # Validate configuration
            is_valid, error = provider.validate_config()
            if not is_valid:
                raise LLMProviderError(f"Invalid configuration: {error}")

            logger.info(
                f"Created {provider_type} provider with model {config.get('model', 'default')}"
            )
            return provider

        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise
            raise LLMProviderError(f"Failed to create provider: {e}") from e

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseLLMProvider:
        """
        Create provider from configuration dictionary.

        Configuration format:
        {
            "provider": "openai",
            "api_key": "...",
            "model": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 2000,
            "timeout": 60
        }

        Args:
            config: Provider configuration dictionary

        Returns:
            Configured provider instance
        """
        if "provider" not in config:
            raise LLMProviderError("Configuration must include 'provider' field")

        provider_type = config.pop("provider")
        return cls.create(provider_type, **config)

    @classmethod
    def create_default(cls) -> BaseLLMProvider:
        """
        Create default provider from environment variables.

        Returns:
            Default configured provider
        """
        provider_type = str(
            os.getenv("LLM_PROVIDER")
            or os.getenv("DEFAULT_LLM_PROVIDER")
            or settings.llm_provider
            or "ollama"
        ).lower()

        temperature = float(os.getenv("LLM_TEMPERATURE", str(settings.llm_temperature)))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(settings.llm_max_tokens)))
        timeout = int(
            os.getenv(
                "LLM_TIMEOUT",
                str(
                    settings.ollama_request_timeout_seconds
                    if provider_type == "ollama"
                    else 60
                ),
            )
        )

        if provider_type == "ollama":
            config = {
                "api_key": "ollama",
                "model": os.getenv("OLLAMA_MODEL", settings.ollama_model),
                "api_base": os.getenv("OLLAMA_API_BASE", settings.ollama_api_base),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
        elif provider_type == "openai":
            config = {
                "api_key": os.getenv("OPENAI_API_KEY", settings.openai_api_key or ""),
                "model": os.getenv("OPENAI_MODEL", settings.openai_model),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
        elif provider_type == "glm":
            config = {
                "api_key": os.getenv("GLM_API_KEY", ""),
                "model": os.getenv("GLM_MODEL", "chatglm3-6b"),
                "api_base": os.getenv("GLM_API_BASE", GLMProvider.DEFAULT_API_BASE),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
        elif provider_type == "kimi":
            config = {
                "api_key": os.getenv("KIMI_API_KEY", ""),
                "model": os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            api_base = os.getenv("KIMI_API_BASE")
            if api_base:
                config["api_base"] = api_base
        else:
            raise LLMProviderError(f"Unknown default provider: {provider_type}")

        config = {k: v for k, v in config.items() if v is not None}

        return cls.create(provider_type, **config)

    @classmethod
    def list_providers(cls) -> Dict[str, str]:
        """
        List available provider types.

        Returns:
            Dictionary mapping provider type to description
        """
        descriptions = {
            "ollama": "Ollama local models (qwen, deepseek, gemma, etc.)",
            "openai": "OpenAI GPT models (GPT-4, GPT-3.5)",
            "kimi": "Moonshot AI Kimi models",
            "glm": "ChatGLM local/API models",
        }

        return {
            provider.value: descriptions.get(provider.value, "Custom provider")
            for provider in cls._providers.keys()
        }

        return {
            provider.value: descriptions.get(provider.value, "Custom provider")
            for provider in cls._providers.keys()
        }

    @classmethod
    def get_provider_class(cls, provider_type: str) -> Optional[Type[BaseLLMProvider]]:
        """
        Get provider class without instantiation.

        Args:
            provider_type: Provider type string

        Returns:
            Provider class or None
        """
        try:
            provider_enum = ProviderType(provider_type.lower())
            return cls._providers.get(provider_enum)
        except ValueError:
            return None
