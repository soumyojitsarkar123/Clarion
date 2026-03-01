"""
System diagnostics for observability and integrity checks.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from networkx.readwrite import json_graph

from services.embedding_service import EmbeddingService
from services.knowledge_map_service import LLMInterface
from services.relation_dataset_service import RelationDatasetService
from utils.config import settings
from utils.graph_store import load_graph_json
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class DiagnosticsService:
    """Runs modular health checks for core Clarion services."""

    def __init__(self):
        self.relation_dataset_service = RelationDatasetService()

    def run_health_check(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        embedding_health = self.check_embedding_service()
        vectorstore_integrity = self.check_vectorstore_integrity(document_id=document_id)
        graph_integrity = self.check_graph_integrity(document_id=document_id)
        dataset_integrity = self.check_dataset_integrity(document_id=document_id)
        llm_connectivity = self.check_llm_connectivity()

        checks = {
            "embedding_service_health": embedding_health,
            "vectorstore_integrity": vectorstore_integrity,
            "graph_integrity": graph_integrity,
            "dataset_integrity": dataset_integrity,
            "llm_connectivity": llm_connectivity,
        }

        statuses = [item.get("status", "unknown") for item in checks.values()]
        if all(status == "healthy" for status in statuses):
            overall = "healthy"
        elif any(status == "unhealthy" for status in statuses):
            overall = "degraded"
        else:
            overall = "warning"

        return {
            "status": overall,
            "timestamp": _utc_now(),
            "document_id": document_id,
            "checks": checks,
        }

    def check_embedding_service(self) -> Dict[str, Any]:
        try:
            _ = EmbeddingService()
            try:
                import sentence_transformers  # noqa: F401
                package_status = "installed"
            except Exception as package_error:
                return {
                    "status": "unhealthy",
                    "detail": "sentence-transformers import failed",
                    "error": str(package_error),
                }

            return {
                "status": "healthy",
                "detail": "Embedding service initialized",
                "model_name": settings.embedding_model_name,
                "device": settings.embedding_device,
                "package_status": package_status,
            }
        except Exception as error:
            return {"status": "unhealthy", "detail": "Embedding service initialization failed", "error": str(error)}

    def check_vectorstore_integrity(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        vector_dir = settings.vectorstore_dir
        if not vector_dir.exists():
            return {
                "status": "unhealthy",
                "detail": "Vectorstore directory is missing",
                "path": str(vector_dir),
            }

        index_files = list(vector_dir.glob("*.index"))
        if document_id:
            index_files = [path for path in index_files if path.stem == document_id]

        valid_files = [path for path in index_files if path.stat().st_size > 0]
        invalid_files = [path.name for path in index_files if path.stat().st_size <= 0]

        status = "healthy" if not invalid_files else "warning"
        return {
            "status": status,
            "detail": "Vectorstore scanned",
            "index_count": len(index_files),
            "valid_index_count": len(valid_files),
            "invalid_indexes": invalid_files,
            "path": str(vector_dir),
        }

    def check_graph_integrity(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        graph_dir = settings.data_dir / "graphs"
        if not graph_dir.exists():
            return {
                "status": "warning",
                "detail": "Graph directory not found",
                "path": str(graph_dir),
            }

        graph_files = list(graph_dir.glob("*.json"))
        graph_files = [path for path in graph_files if not path.name.endswith("_cytoscape.json")]

        if document_id:
            target = graph_dir / f"{document_id}.json"
            graph_files = [target] if target.exists() else []

        valid = 0
        invalid: List[str] = []
        nodes_total = 0
        edges_total = 0

        for path in graph_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                graph = nx.DiGraph(
                    json_graph.node_link_graph(payload, directed=True, multigraph=False)
                )
                valid += 1
                nodes_total += graph.number_of_nodes()
                edges_total += graph.number_of_edges()
            except Exception:
                invalid.append(path.name)

        status = "healthy" if not invalid else "warning"
        if not graph_files:
            status = "warning"
        return {
            "status": status,
            "detail": "Graph files checked",
            "graph_files_checked": len(graph_files),
            "valid_graph_files": valid,
            "invalid_graph_files": invalid,
            "total_nodes": nodes_total,
            "total_edges": edges_total,
        }

    def check_dataset_integrity(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        dataset_db = settings.data_dir / "relation_dataset.db"
        if not dataset_db.exists():
            return {
                "status": "warning",
                "detail": "Relation dataset DB not found",
                "path": str(dataset_db),
            }

        try:
            conn = sqlite_connect(str(dataset_db))
            cursor = conn.cursor()

            if document_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ?",
                    (document_id,),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM relation_dataset")
            total_records = int(cursor.fetchone()[0])

            if document_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ? AND is_valid IS NOT NULL",
                    (document_id,),
                )
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM relation_dataset WHERE is_valid IS NOT NULL"
                )
            labeled_records = int(cursor.fetchone()[0])

            conn.close()

            return {
                "status": "healthy",
                "detail": "Dataset integrity check passed",
                "total_records": total_records,
                "labeled_records": labeled_records,
                "unlabeled_records": max(total_records - labeled_records, 0),
            }
        except Exception as error:
            return {
                "status": "unhealthy",
                "detail": "Dataset query failed",
                "error": str(error),
            }

    def check_llm_connectivity(self) -> Dict[str, Any]:
        try:
            llm = LLMInterface()
            available = llm._is_service_available()
            return {
                "status": "healthy" if available else "warning",
                "detail": "LLM connectivity check complete",
                "provider": llm.provider,
                "model": llm.model,
                "available": bool(available),
            }
        except Exception as error:
            return {
                "status": "unhealthy",
                "detail": "LLM connectivity check failed",
                "error": str(error),
            }


_instance: Optional[DiagnosticsService] = None


def get_diagnostics_service() -> DiagnosticsService:
    global _instance
    if _instance is None:
        _instance = DiagnosticsService()
    return _instance

