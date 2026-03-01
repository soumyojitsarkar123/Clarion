"""
Application configuration using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


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

    # Database
    database_url: str = "sqlite:///./data/clarion.db"

    # Embedding Model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

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
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    # Ollama (local models)
    ollama_model: str = "qwen3:latest"
    ollama_api_base: str = "http://localhost:11434/v1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000

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


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for dir_path in [settings.data_dir, settings.vectorstore_dir, settings.logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
