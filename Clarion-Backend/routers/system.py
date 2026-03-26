"""
System observability router: pipeline status, diagnostics, and audit endpoints.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from graph.analyzer import GraphAnalyzer
from services.diagnostics_service import get_diagnostics_service
from services.pipeline_observability import get_pipeline_state_tracker
from services.rag_audit_service import get_rag_audit_service
from services.relation_dataset_service import RelationDatasetService
from services.analysis_report_service import get_analysis_report_service
from services.knowledge_map_service import KnowledgeMapService
from services.document_service import DocumentService
from utils.config import settings
from utils.graph_store import graph_pickle_path, load_graph_json
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])

pipeline_tracker = get_pipeline_state_tracker()
rag_audit_service = get_rag_audit_service()
dataset_service = RelationDatasetService()
diagnostics_service = get_diagnostics_service()
analysis_report_service = get_analysis_report_service()
knowledge_map_service = KnowledgeMapService()
document_service = DocumentService()


class RagAuditRequest(BaseModel):
    document_id: str = Field(..., description="Document ID")
    query: str = Field(..., min_length=1, max_length=2000, description="Query to audit")
    top_k: int = Field(default=5, ge=1, le=25)
    response_text: Optional[str] = Field(
        default=None,
        description="Optional model response for grounding overlap scoring",
    )


@router.post("/rag-audit")
async def run_rag_audit(request: RagAuditRequest):
    """Run and persist RAG retrieval audit for a query."""
    try:
        return rag_audit_service.audit_query(
            document_id=request.document_id,
            query=request.query,
            top_k=request.top_k,
            response_text=request.response_text,
        )
    except Exception as error:
        logger.error("RAG audit failed: %s", error)
        raise HTTPException(status_code=500, detail="Failed to run RAG audit")


@router.get("/rag-audit")
async def get_recent_rag_audits(
    document_id: Optional[str] = Query(None, description="Optional document filter"),
    limit: int = Query(20, ge=1, le=200, description="Max number of audit records"),
):
    """Get recently persisted RAG audits."""
    try:
        audits = rag_audit_service.recent_audits(document_id=document_id, limit=limit)
        return {
            "count": len(audits),
            "audits": audits,
        }
    except Exception as error:
        logger.error("RAG audit retrieval failed: %s", error)
        raise HTTPException(status_code=500, detail="Failed to read RAG audits")


@router.get("/pipeline-status")
async def get_pipeline_status(
    document_id: Optional[str] = Query(None, description="Optional document filter")
):
    """Get detailed stage-level pipeline progress for visualization."""
    try:
        status = pipeline_tracker.get_status(document_id=document_id)
        if document_id and not status:
            return {
                "document_id": document_id,
                "status": "not_found",
                "message": "No pipeline state found for document",
            }
        return status
    except Exception as error:
        logger.error("Pipeline status retrieval failed: %s", error)
        raise HTTPException(status_code=500, detail="Failed to read pipeline status")


@router.get("/graph-stats")
async def get_graph_stats(
    document_id: Optional[str] = Query(None, description="Optional document filter")
):
    """Get graph inspector stats including density and central nodes."""
    try:
        if document_id:
            graph = load_graph_json(document_id)
            if graph is None and settings.allow_legacy_pickle_loading:
                legacy_path = graph_pickle_path(document_id)
                if legacy_path.exists():
                    import pickle

                    with open(legacy_path, "rb") as handle:
                        graph = pickle.load(handle)
            if graph is None:
                raise HTTPException(status_code=404, detail="Graph not found")
            return _single_graph_stats(document_id=document_id, graph=graph)

        graph_dir = settings.data_dir / "graphs"
        graph_files = [
            path
            for path in graph_dir.glob("*.json")
            if not path.name.endswith("_cytoscape.json")
        ]
        aggregates = {
            "documents_with_graphs": len(graph_files),
            "concept_count": 0,
            "relation_count": 0,
            "graph_density_avg": 0.0,
            "central_nodes": [],
            "validation_stats": _dataset_validation_stats(),
        }
        if not graph_files:
            return aggregates

        densities = []
        centrality_pool: Dict[str, float] = {}
        for graph_file in graph_files:
            doc_id = graph_file.stem
            graph = load_graph_json(doc_id)
            if graph is None:
                continue
            stats = _single_graph_stats(document_id=doc_id, graph=graph)
            aggregates["concept_count"] += stats["concept_count"]
            aggregates["relation_count"] += stats["relation_count"]
            densities.append(stats["graph_density"])
            for node in stats["central_nodes"]:
                centrality_pool[node["label"]] = max(
                    centrality_pool.get(node["label"], 0.0), node["score"]
                )

        aggregates["graph_density_avg"] = round(
            sum(densities) / len(densities), 4
        ) if densities else 0.0
        aggregates["central_nodes"] = [
            {"label": key, "score": round(value, 4)}
            for key, value in sorted(
                centrality_pool.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ]
        return aggregates
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Graph stats retrieval failed: %s", error)
        raise HTTPException(status_code=500, detail="Failed to read graph stats")


@router.get("/dataset-stats")
async def get_dataset_stats(
    document_id: Optional[str] = Query(None, description="Optional document filter")
):
    """Get dataset inspector stats including validation details."""
    try:
        if not document_id:
            stats = dataset_service.get_dataset_stats()
            validation = _dataset_validation_stats()
            return {
                **stats,
                "validation_stats": validation,
                "storage_paths": _storage_locations(),
                "latest_export_file": _latest_dataset_export_file(document_id=None),
            }

        conn = sqlite_connect(str(settings.data_dir / "relation_dataset.db"))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ?",
            (document_id,),
        )
        total = int(cursor.fetchone()[0])
        cursor.execute(
            "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ? AND is_valid = 1",
            (document_id,),
        )
        valid = int(cursor.fetchone()[0])
        cursor.execute(
            "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ? AND is_valid = 0",
            (document_id,),
        )
        invalid = int(cursor.fetchone()[0])
        cursor.execute(
            "SELECT AVG(llm_confidence) FROM relation_dataset WHERE document_id = ?",
            (document_id,),
        )
        avg_conf = float(cursor.fetchone()[0] or 0.0)
        cursor.execute(
            """
            SELECT relation_type, COUNT(*)
            FROM relation_dataset
            WHERE document_id = ?
            GROUP BY relation_type
            """,
            (document_id,),
        )
        relation_types = {row[0]: int(row[1]) for row in cursor.fetchall()}
        conn.close()

        labeled = valid + invalid
        return {
            "document_id": document_id,
            "total_records": total,
            "labeled_records": labeled,
            "unlabeled_records": max(total - labeled, 0),
            "relation_types": relation_types,
            "average_llm_confidence": round(avg_conf, 4),
            "validation_stats": {
                "valid_relations": valid,
                "invalid_relations": invalid,
                "validation_rate": round(labeled / total, 4) if total else 0.0,
            },
            "storage_paths": _storage_locations(),
            "latest_export_file": _latest_dataset_export_file(document_id=document_id),
        }
    except sqlite3.OperationalError:
        return {
            "total_records": 0,
            "labeled_records": 0,
            "unlabeled_records": 0,
            "relation_types": {},
            "average_llm_confidence": 0.0,
            "validation_stats": {
                "valid_relations": 0,
                "invalid_relations": 0,
                "validation_rate": 0.0,
            },
            "storage_paths": _storage_locations(),
            "latest_export_file": _latest_dataset_export_file(document_id=document_id),
        }
    except Exception as error:
        logger.error("Dataset stats retrieval failed: %s", error)
        raise HTTPException(status_code=500, detail="Failed to read dataset stats")


@router.get("/health-check")
async def system_health_check(
    document_id: Optional[str] = Query(None, description="Optional document for scoped integrity checks")
):
    """Run system diagnostics across embedding, vectorstore, graph, dataset, and LLM checks."""
    try:
        return diagnostics_service.run_health_check(document_id=document_id)
    except Exception as error:
        logger.error("System health check failed: %s", error)
        raise HTTPException(status_code=500, detail="Failed to run health checks")


@router.get("/storage-paths")
async def get_storage_paths():
    """Return key filesystem paths for persisted artifacts and datasets."""
    return _storage_locations()


@router.get("/analysis-report/{document_id}")
async def get_analysis_report(document_id: str):
    """Get the persisted analysis report for a document."""
    try:
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        report = analysis_report_service.get_report(document_id)
        if report:
            return report

        # Fallback snapshot when persisted report is not yet available.
        knowledge_map = knowledge_map_service.get_knowledge_map(document_id)
        graph = load_graph_json(document_id)
        if graph is None and settings.allow_legacy_pickle_loading:
            legacy_path = graph_pickle_path(document_id)
            if legacy_path.exists():
                import pickle

                with open(legacy_path, "rb") as handle:
                    graph = pickle.load(handle)

        dataset_stats = await get_dataset_stats(document_id=document_id)
        graph_stats = (
            _single_graph_stats(document_id=document_id, graph=graph)
            if graph is not None
            else {
                "document_id": document_id,
                "concept_count": len(knowledge_map.concepts) if knowledge_map else 0,
                "relation_count": len(knowledge_map.relations) if knowledge_map else 0,
                "node_count": 0,
                "edge_count": 0,
                "graph_density": 0.0,
                "central_nodes": [],
                "validation_stats": _dataset_validation_stats(document_id=document_id),
            }
        )

        return {
            "document_id": document_id,
            "success": str(document.status).lower() == "analyzed",
            "status": str(document.status).lower(),
            "generated_at": document.metadata.upload_time.isoformat(),
            "summary": {
                "concept_count": graph_stats.get("concept_count", 0),
                "relation_count": graph_stats.get("relation_count", 0),
                "graph_nodes": graph_stats.get("node_count", 0),
                "graph_edges": graph_stats.get("edge_count", 0),
                "graph_density": graph_stats.get("graph_density", 0.0),
                "evaluation_overall_score": None,
                "evaluation_risk_score": None,
                "evaluation_flags": 0,
                "hierarchy_layers": 0,
                "hierarchy_chains": 0,
            },
            "stages_completed": [],
            "stages_failed": [],
            "skipped_stages": [],
            "stage_validations": [],
            "artifacts": {
                "knowledge_map": {
                    "concept_count": graph_stats.get("concept_count", 0),
                    "relation_count": graph_stats.get("relation_count", 0),
                },
                "graph": {
                    "node_count": graph_stats.get("node_count", 0),
                    "edge_count": graph_stats.get("edge_count", 0),
                },
                "dataset": dataset_stats,
            },
            "evaluation": {},
            "hierarchy": {},
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Analysis report retrieval failed: %s", error)
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis report")


def _single_graph_stats(document_id: str, graph: nx.DiGraph) -> Dict[str, Any]:
    concept_count = sum(
        1
        for _, data in graph.nodes(data=True)
        if data.get("node_type") == "concept"
    )
    relation_count = sum(
        1
        for _, _, data in graph.edges(data=True)
        if data.get("relation_type") != "hierarchy"
    )
    analyzer = GraphAnalyzer(graph)
    centrality = analyzer.get_centrality_metrics(top_n=5)
    central_nodes = []
    for node in centrality.get("pagerank", {}).get("top_nodes", []):
        node_id = node["node_id"]
        label = graph.nodes[node_id].get("label", node_id)
        central_nodes.append({"node_id": node_id, "label": label, "score": node["score"]})

    return {
        "document_id": document_id,
        "concept_count": concept_count,
        "relation_count": relation_count,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "graph_density": round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0.0,
        "central_nodes": central_nodes,
        "validation_stats": _dataset_validation_stats(document_id=document_id),
    }


def _dataset_validation_stats(document_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        conn = sqlite_connect(str(settings.data_dir / "relation_dataset.db"))
        cursor = conn.cursor()
        if document_id:
            cursor.execute(
                "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ?",
                (document_id,),
            )
            total = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ? AND is_valid = 1",
                (document_id,),
            )
            valid = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT COUNT(*) FROM relation_dataset WHERE document_id = ? AND is_valid = 0",
                (document_id,),
            )
            invalid = int(cursor.fetchone()[0])
        else:
            cursor.execute("SELECT COUNT(*) FROM relation_dataset")
            total = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM relation_dataset WHERE is_valid = 1")
            valid = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM relation_dataset WHERE is_valid = 0")
            invalid = int(cursor.fetchone()[0])
        conn.close()
        labeled = valid + invalid
        return {
            "valid_relations": valid,
            "invalid_relations": invalid,
            "validation_rate": round(labeled / total, 4) if total else 0.0,
        }
    except Exception:
        return {"valid_relations": 0, "invalid_relations": 0, "validation_rate": 0.0}


def _storage_locations() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_dataset_dir = repo_root / "data" / "datasets"
    return {
        "data_dir": str(settings.data_dir.resolve()),
        "main_db": str((settings.data_dir / "clarion.db").resolve()),
        "relation_dataset_db": str((settings.data_dir / "relation_dataset.db").resolve()),
        "dataset_exports_dir": str(workspace_dataset_dir.resolve()),
        "vectorstore_dir": str(settings.vectorstore_dir.resolve()),
        "graphs_dir": str((settings.data_dir / "graphs").resolve()),
        "logs_dir": str(settings.logs_dir.resolve()),
    }


def _latest_dataset_export_file(document_id: Optional[str]) -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "data" / "datasets"
    if not dataset_dir.exists():
        return None

    pattern = f"*{document_id}*_relations.json" if document_id else "*_relations.json"
    matches = sorted(dataset_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return str(matches[0].resolve())
