"""
System observability router: pipeline status, diagnostics, and audit endpoints.
"""

from __future__ import annotations

import logging
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from graph.analyzer import GraphAnalyzer
from services.diagnostics_service import get_diagnostics_service
from services.pipeline_observability import PIPELINE_STAGES, get_pipeline_state_tracker
from services.rag_audit_service import get_rag_audit_service
from services.relation_dataset_service import RelationDatasetService
from services.analysis_report_service import get_analysis_report_service
from services.knowledge_map_service import KnowledgeMapService
from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.summary_service import SummaryService
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
chunking_service = ChunkingService()
summary_service = SummaryService()


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
        if document_id:
            # Reconcile with canonical persisted analysis report if available
            report = analysis_report_service.get_report(document_id)
            if report:
                rep_status = report.get("status")
                rep_mode = report.get("analysis_mode")
                rep_model = report.get("active_model")
                rep_dur = report.get("total_duration_sec")
                rep_stages = report.get("stages_completed") or []
                rep_metrics = report.get("metrics") or {}

                saved_stages: Dict[str, Dict[str, Any]] = {}
                for entry in (report.get("audit", {}).get("stages") or []):
                    if isinstance(entry, dict) and entry.get("stage"):
                        saved_stages[entry["stage"]] = deepcopy(entry)
                for entry in (report.get("stage_validations") or []):
                    if isinstance(entry, dict) and entry.get("stage"):
                        st_name = entry["stage"]
                        if st_name not in saved_stages:
                            saved_stages[st_name] = deepcopy(entry)
                        else:
                            for k, val in entry.items():
                                if saved_stages[st_name].get(k) is None and val is not None:
                                    saved_stages[st_name][k] = val

                if not status:
                    stage_map: Dict[str, Dict[str, Any]] = {}
                    for s in PIPELINE_STAGES:
                        st_data = saved_stages.get(s)
                        if st_data:
                            st_status = st_data.get("status")
                            if not st_status:
                                st_status = "completed" if s in rep_stages else "pending"
                            elif rep_mode == "deterministic_fallback" and s == "mapping" and st_status == "completed":
                                st_status = "fallback"

                            q_metrics = deepcopy(st_data.get("quality_metrics", {}))
                            dur_raw = st_data.get("duration_sec")
                            dur = float(dur_raw) if dur_raw is not None else None
                            in_sz = st_data.get("input_size")
                            out_sz = st_data.get("output_size")

                            stage_entry: Dict[str, Any] = {
                                "name": s,
                                "status": st_status,
                                "progress": 100 if st_status in ("completed", "fallback", "skipped") else 0,
                                "start_time": st_data.get("start_time"),
                                "end_time": st_data.get("end_time"),
                                "duration_sec": dur,
                                "input_size": in_sz,
                                "output_size": out_sz,
                                "quality_metrics": q_metrics,
                                "errors": list(st_data.get("errors", st_data.get("issues", []))),
                            }
                            if s == "chunking":
                                chunk_count = int(out_sz if out_sz is not None else q_metrics.get("chunk_count", 0))
                                q_metrics["chunk_count"] = chunk_count
                                stage_entry["chunk_count"] = chunk_count
                            elif s == "embedding":
                                chunk_count = int(in_sz if in_sz is not None else q_metrics.get("chunk_count", 0))
                                throughput = round(chunk_count / dur, 2) if (dur and dur > 0 and chunk_count > 0) else 0.0
                                q_metrics["throughput_chunks_sec"] = throughput
                                device = settings.embedding_device_actual or "cpu"
                                q_metrics["device"] = device
                                stage_entry["throughput_chunks_sec"] = throughput
                                stage_entry["device"] = device
                            elif s == "mapping":
                                q_metrics["analysis_mode"] = rep_mode or "llm"
                                q_metrics["model"] = rep_model or settings.ollama_model
                                stage_entry["analysis_mode"] = rep_mode or "llm"
                                stage_entry["model"] = rep_model or settings.ollama_model
                            elif s == "graph_building":
                                stage_entry["node_count"] = int(q_metrics.get("node_count", 0))
                                stage_entry["edge_count"] = int(q_metrics.get("edge_count", 0))

                            stage_entry["quality_metrics"] = q_metrics
                            stage_map[s] = stage_entry
                        elif s in rep_stages:
                            stage_map[s] = {
                                "name": s,
                                "status": "completed",
                                "progress": 100,
                                "start_time": None,
                                "end_time": None,
                                "duration_sec": None,
                                "input_size": None,
                                "output_size": None,
                                "quality_metrics": {},
                                "errors": [],
                            }
                        else:
                            stage_map[s] = {
                                "name": s,
                                "status": "pending",
                                "progress": 0,
                                "start_time": None,
                                "end_time": None,
                                "duration_sec": None,
                                "input_size": None,
                                "output_size": None,
                                "quality_metrics": {},
                                "errors": [],
                            }

                    status = {
                        "document_id": document_id,
                        "status": rep_status or "completed",
                        "analysis_mode": rep_mode or "llm",
                        "active_model": rep_model or settings.ollama_model,
                        "total_duration_sec": rep_dur,
                        "overall_progress": 100,
                        "stages": stage_map,
                        "metrics": rep_metrics,
                    }
                else:
                    is_running = status.get("status") == "running" or (status.get("overall_progress", 0) < 100)
                    if not is_running:
                        if rep_status:
                            status["status"] = rep_status
                        if rep_mode:
                            status["analysis_mode"] = rep_mode
                        if rep_model:
                            status["active_model"] = rep_model
                        if rep_dur is not None:
                            status["total_duration_sec"] = rep_dur
                        if rep_metrics:
                            status["metrics"] = rep_metrics
                        # Enrich in-memory stages if any metrics were missing
                        if "stages" in status and isinstance(status["stages"], dict):
                            for s_name, s_obj in status["stages"].items():
                                saved_st = saved_stages.get(s_name)
                                if saved_st and isinstance(s_obj, dict):
                                    if s_obj.get("duration_sec") is None and saved_st.get("duration_sec") is not None:
                                        s_obj["duration_sec"] = float(saved_st["duration_sec"])
                                    if s_obj.get("input_size") is None and saved_st.get("input_size") is not None:
                                        s_obj["input_size"] = saved_st["input_size"]
                                    if s_obj.get("output_size") is None and saved_st.get("output_size") is not None:
                                        s_obj["output_size"] = saved_st["output_size"]
                                    if not s_obj.get("quality_metrics") and saved_st.get("quality_metrics"):
                                        s_obj["quality_metrics"] = deepcopy(saved_st["quality_metrics"])

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

        doc_chunks = chunking_service.get_document_chunks(document_id)
        doc_summary = summary_service.get_summary(document_id)
        sec_count = len(doc_summary.sections) if doc_summary and doc_summary.sections else 0
        chunk_count = len(doc_chunks) if doc_chunks else 0

        metrics = {
            "words": document.metadata.word_count,
            "pages": document.metadata.page_count,
            "sections": sec_count,
            "chunks": chunk_count,
            "concepts": graph_stats.get("concept_count", 0),
            "candidate_relations": graph_stats.get("relation_count", 0),
            "graph_nodes": graph_stats.get("node_count", 0),
            "graph_edges": graph_stats.get("edge_count", 0),
        }

        return {
            "document_id": document_id,
            "success": str(document.status).lower() == "analyzed",
            "status": str(document.status).lower(),
            "generated_at": document.metadata.upload_time.isoformat(),
            "metrics": metrics,
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
                "document": {
                    "word_count": document.metadata.word_count,
                    "page_count": document.metadata.page_count,
                },
                "chunks": {
                    "count": chunk_count,
                },
                "summary": {
                    "section_count": sec_count,
                },
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
