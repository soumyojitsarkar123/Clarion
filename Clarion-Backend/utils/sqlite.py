"""
SQLite utility helpers for reliable local concurrency.
"""

import sqlite3
from pathlib import Path
from typing import Union

from utils.config import settings


def connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    """
    Create a SQLite connection with reliability-oriented pragmas.
    """
    conn = sqlite3.connect(str(db_path), timeout=settings.sqlite_timeout_seconds)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        try:
            conn.execute("PRAGMA journal_mode=TRUNCATE")
        except sqlite3.OperationalError:
            conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute(f"PRAGMA busy_timeout={settings.sqlite_busy_timeout_ms}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
