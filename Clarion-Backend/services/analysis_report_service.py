"""
Analysis report persistence service.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.config import settings
from utils.sqlite import connect as sqlite_connect


class AnalysisReportService:
    """Stores and retrieves pipeline analysis reports."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.data_dir / "clarion.db")
        self._init_database()

    def _init_database(self) -> None:
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_reports (
                document_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def save_report(self, document_id: str, report: Dict[str, Any]) -> None:
        payload = dict(report)
        payload["document_id"] = document_id
        payload["updated_at"] = datetime.now().isoformat()
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO analysis_reports (document_id, data, created_at)
            VALUES (?, ?, ?)
            """,
            (document_id, json.dumps(payload, ensure_ascii=True), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def get_report(self, document_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite_connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM analysis_reports WHERE document_id = ?",
            (document_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return json.loads(row["data"])


_instance: Optional[AnalysisReportService] = None


def get_analysis_report_service() -> AnalysisReportService:
    global _instance
    if _instance is None:
        _instance = AnalysisReportService()
    return _instance

