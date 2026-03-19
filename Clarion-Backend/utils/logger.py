"""
Structured logging configuration and utilities.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.log_stream import get_log_stream_broker


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as structured JSON lines."""

    @staticmethod
    def _record_timestamp(record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def build_payload(record: logging.LogRecord) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "stage": getattr(record, "stage", "system"),
            "event": getattr(record, "event", "log"),
            "timestamp": StructuredJsonFormatter._record_timestamp(record),
            "document_id": getattr(record, "document_id", None),
            "message": record.getMessage(),
            "logger": record.name,
            "level": record.levelname,
        }

        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict) and metadata:
            payload["metadata"] = metadata

        if record.exc_info:
            payload["error"] = logging.Formatter().formatException(record.exc_info)

        return payload

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self.build_payload(record), ensure_ascii=True)


class StreamBrokerHandler(logging.Handler):
    """Publishes structured logs to in-memory broker for websocket streaming."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = StructuredJsonFormatter.build_payload(record)
            get_log_stream_broker().publish(payload)
        except Exception:
            self.handleError(record)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    formatter = StructuredJsonFormatter()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), StreamBrokerHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)


def log_structured(
    logger: logging.Logger,
    level: int,
    *,
    stage: str,
    event: str,
    message: str,
    document_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured log with normalized fields."""
    logger.log(
        level,
        message,
        extra={
            "stage": stage,
            "event": event,
            "document_id": document_id,
            "metadata": metadata or {},
        },
    )
