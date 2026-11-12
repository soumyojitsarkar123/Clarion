"""
Application configuration using Pydantic settings.
"""

import os
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import model_validator

BACKEND_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DATA_DIR = BACKEND_ROOT / "data"
CANONICAL_LOGS_DIR = BACKEND_ROOT / "logs"


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
    app_name: str = "Clarion"
    app_version: str = "1.0.0"
    debug: bool = False

    # Paths
    data_dir: Path = CANONICAL_DATA_DIR
    vectorstore_dir: Path = CANONICAL_DATA_DIR / "vectorstore"
    logs_dir: Path = CANONICAL_LOGS_DIR
    dataset_dir: Path = CANONICAL_DATA_DIR / "datasets"

    # Database
    database_url: str = f"sqlite:///{CANONICAL_DATA_DIR / 'clarion.db'}"

    @model_validator(mode="after")
    def resolve_canonical_paths(self) -> "Settings":
        """Anchor relative paths to BACKEND_ROOT to ensure consistent runtime across launch CWDs."""
        if not self.data_dir.is_absolute():
            self.data_dir = (BACKEND_ROOT / self.data_dir).resolve()
        if not self.vectorstore_dir.is_absolute():
            self.vectorstore_dir = self.data_dir / "vectorstore"
        if not self.logs_dir.is_absolute():
            self.logs_dir = (BACKEND_ROOT / self.logs_dir).resolve()
        if not self.dataset_dir.is_absolute():
            self.dataset_dir = self.data_dir / "datasets"
        if "sqlite:///./data/clarion.db" in self.database_url or self.database_url.startswith("sqlite:///./"):
            self.database_url = f"sqlite:///{self.data_dir / 'clarion.db'}"
        return self

    def get_db_path(self) -> Path:
        """Return the resolved filesystem path to clarion.db."""
        return self.data_dir / "clarion.db"

    # Embedding Model
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int = 64
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
    ollama_request_timeout_seconds: int = 180
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000
    # Runtime request limits, distinct from any advertised model maximum.
    llm_context_window: int = 4096
    llm_reserved_output_tokens: int = 2000
    llm_token_safety_tokens: int = 256
    # Optional HF tokenizer ID known to match the served model; loaded from cache only.
    llm_tokenizer_name: Optional[str] = None

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
    dataset_chunk_timeout_seconds: float = 60.0
    dataset_max_retries_per_chunk: int = 3
    dataset_backoff_base_seconds: float = 1.0
    dataset_backoff_jitter_factor: float = 0.2
    dataset_chunks_per_prompt: int = 1
    # Structured dataset samples are concise; reserve a bounded response budget
    # per chunk instead of the general-purpose 2K generation ceiling.
    dataset_max_output_tokens: int = 1024

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

    if settings.embedding_device_actual == "cpu":
        try:
            threads = min(8, max(1, os.cpu_count() or 4))
            torch.set_num_threads(threads)
        except Exception:
            pass

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
