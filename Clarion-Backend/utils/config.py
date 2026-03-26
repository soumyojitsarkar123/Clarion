"""
Application configuration using Pydantic settings.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings


def get_best_device() -> str:
    """Auto-detect best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Intelligent Document Knowledge System"
    app_version: str = "1.0.0"
    debug: bool = False

    # Paths
    data_dir: Path = Path("./data")
    vectorstore_dir: Path = Path("./data/vectorstore")
    logs_dir: Path = Path("./logs")
    dataset_dir: Path = Path("./data/datasets")

    # Database
    database_url: str = "sqlite:///./data/clarion.db"

    # Embedding Model
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str = "auto"
    embedding_device_actual: Optional[str] = None

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100

    # LLM
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2000

    # LLM Provider Selection
    # Options: "ollama", "openai", "deepseek", "gemini"
    llm_provider: str = "ollama"

    # DeepSeek API
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com/v1"

    # Google Gemini API
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    # Ollama (local models)
    ollama_model: str = "qwen3.5:4b"
    ollama_api_base: str = "http://localhost:11434/v1"
    ollama_health_timeout_seconds: float = 2.0
    ollama_request_timeout_seconds: int = 30
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000

    # Dataset Generation
    dataset_generation_enabled: bool = True
    dataset_quality_threshold: float = 0.82
    dataset_dedup_threshold: float = 0.85
    dataset_batch_size: int = 10
    dataset_export_format: str = "jsonl"
    dataset_llm_sample_rate: float = 0.3
    dataset_generation_interval_hours: float = 1.0
    dataset_require_llm_for_export: bool = True
    dataset_background_postprocess_enabled: bool = True

    # Retrieval
    default_top_k: int = 5
    similarity_threshold: float = 0.5

    # File Upload
    max_file_size_mb: int = 50
    allowed_extensions: list = [".pdf", ".docx"]

    # API Security
    cors_allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    cors_allow_credentials: bool = False

    # SQLite reliability
    sqlite_timeout_seconds: float = 30.0
    sqlite_busy_timeout_ms: int = 30000

    # Serialization safety
    allow_legacy_pickle_loading: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()


def initialize_compute_settings() -> str:
    """Initialize compute device settings and return actual device used."""
    if settings.embedding_device == "auto":
        settings.embedding_device_actual = get_best_device()
    else:
        settings.embedding_device_actual = settings.embedding_device

    return settings.embedding_device_actual


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for dir_path in [
        settings.data_dir,
        settings.vectorstore_dir,
        settings.logs_dir,
        settings.dataset_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
