"""
RAG audit service for retrieval verification and hallucination risk analysis.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.retrieval import RetrievalResult
from services.retrieval_service import RetrievalService
from utils.config import settings
from utils.logger import log_structured
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class RagAuditService:
    """Audits retrieval quality and grounding confidence for RAG calls."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.data_dir / "clarion.db")
        self.retrieval_service = RetrievalService()
        self._init_database()

    def _init_database(self) -> None:
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_audits (
                audit_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                average_similarity REAL NOT NULL,
                grounding_confidence REAL NOT NULL,
                hallucination_risk TEXT NOT NULL,
                retrieved_chunks TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_audits_document
            ON rag_audits(document_id)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_audits_created
            ON rag_audits(created_at)
            """
        )
        conn.commit()
        conn.close()

    def audit_query(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        response_text: Optional[str] = None,
        retrieval_results: Optional[List[RetrievalResult]] = None,
    ) -> Dict[str, Any]:
        """Audit a RAG retrieval pass and persist results."""
        results = retrieval_results or self.retrieval_service.retrieve(
            document_id=document_id, query=query, top_k=top_k
        )

        similarities = [float(item.score) for item in results]
        avg_similarity = round(sum(similarities) / len(similarities), 4) if similarities else 0.0
        grounding_confidence = self._calculate_grounding_confidence(
            results=results, top_k=top_k, response_text=response_text
        )
        hallucination_risk = self._classify_hallucination_risk(grounding_confidence)

        retrieved_chunks: List[Dict[str, Any]] = []
        for result in results:
            item = {
                "chunk_id": result.chunk_id,
                "similarity_score": round(float(result.score), 4),
                "section_title": result.section_title,
                "content_preview": result.content[:300],
            }
            retrieved_chunks.append(item)
            log_structured(
                logger,
                level=logging.INFO,
                stage="rag",
                event="retrieved_chunk",
                message="RAG retrieved chunk",
                document_id=document_id,
                metadata={
                    "chunk_id": result.chunk_id,
                    "similarity_score": item["similarity_score"],
                    "query": query[:200],
                },
            )

        audit = {
            "audit_id": str(uuid.uuid4()),
            "document_id": document_id,
            "query": query,
            "top_k": top_k,
            "retrieved_chunks": retrieved_chunks,
            "similarity_scores": similarities,
            "average_similarity": avg_similarity,
            "grounding_confidence": grounding_confidence,
            "hallucination_risk": hallucination_risk,
            "flagged": hallucination_risk in {"high", "critical"},
            "timestamp": _utc_now(),
        }

        self._save_audit(audit, response_text=response_text)

        log_structured(
            logger,
            level=logging.INFO,
            stage="rag",
            event="audit_complete",
            message="RAG audit completed",
            document_id=document_id,
            metadata={
                "query": query[:200],
                "top_k": top_k,
                "average_similarity": avg_similarity,
                "grounding_confidence": grounding_confidence,
                "hallucination_risk": hallucination_risk,
                "retrieved_count": len(retrieved_chunks),
            },
        )

        return audit

    def recent_audits(self, document_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent persisted audit results."""
        conn = sqlite_connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if document_id:
            cursor.execute(
                """
                SELECT * FROM rag_audits
                WHERE document_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (document_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM rag_audits
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()

        audits: List[Dict[str, Any]] = []
        for row in rows:
            audits.append(
                {
                    "audit_id": row["audit_id"],
                    "document_id": row["document_id"],
                    "query": row["query_text"],
                    "top_k": row["top_k"],
                    "average_similarity": row["average_similarity"],
                    "grounding_confidence": row["grounding_confidence"],
                    "hallucination_risk": row["hallucination_risk"],
                    "retrieved_chunks": json.loads(row["retrieved_chunks"]),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "timestamp": row["created_at"],
                }
            )
        return audits

    def _save_audit(self, audit: Dict[str, Any], response_text: Optional[str]) -> None:
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rag_audits (
                audit_id,
                document_id,
                query_text,
                top_k,
                average_similarity,
                grounding_confidence,
                hallucination_risk,
                retrieved_chunks,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit["audit_id"],
                audit["document_id"],
                audit["query"],
                audit["top_k"],
                audit["average_similarity"],
                audit["grounding_confidence"],
                audit["hallucination_risk"],
                json.dumps(audit["retrieved_chunks"], ensure_ascii=True),
                json.dumps(
                    {
                        "response_preview": response_text[:500] if response_text else None,
                        "flagged": audit["flagged"],
                    },
                    ensure_ascii=True,
                ),
                audit["timestamp"],
            ),
        )
        conn.commit()
        conn.close()

    def _calculate_grounding_confidence(
        self,
        results: List[RetrievalResult],
        top_k: int,
        response_text: Optional[str],
    ) -> float:
        if not results:
            return 0.0

        similarity = sum(float(item.score) for item in results) / len(results)
        coverage = min(len(results) / max(top_k, 1), 1.0)
        confidence = (0.7 * similarity) + (0.3 * coverage)

        if response_text:
            context_terms = set()
            for item in results:
                context_terms.update(word.lower() for word in item.content.split()[:500])
            response_terms = set(word.lower() for word in response_text.split()[:500])
            overlap = (
                len(context_terms.intersection(response_terms)) / len(response_terms)
                if response_terms
                else 0.0
            )
            confidence = (0.75 * confidence) + (0.25 * overlap)

        return round(max(0.0, min(1.0, confidence)), 4)

    def _classify_hallucination_risk(self, grounding_confidence: float) -> str:
        if grounding_confidence < 0.35:
            return "critical"
        if grounding_confidence < 0.55:
            return "high"
        if grounding_confidence < 0.75:
            return "medium"
        return "low"


_instance: Optional[RagAuditService] = None


def get_rag_audit_service() -> RagAuditService:
    global _instance
    if _instance is None:
        _instance = RagAuditService()
    return _instance

