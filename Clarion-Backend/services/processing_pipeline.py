"""
Processing Pipeline - Orchestrates document processing stages.

Integrates existing services to execute the full pipeline:
ingestion → chunking → embedding → mapping → graph → evaluation → hierarchy
"""

import logging
import re
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from pathlib import Path
from collections import Counter

import networkx as nx

from models.document import Document, DocumentStatus
from models.chunk import Chunk
from models.knowledge_map import KnowledgeMap
from services.background_service import JobStatus
from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.knowledge_map_service import KnowledgeMapService
from services.relation_dataset_service import RelationDatasetService
from graph.builder import GraphBuilder
from graph.exporter import GraphExporter
from graph.hierarchy import HierarchyGenerator
from core.evaluation.confidence_scorer import ConfidenceScorer
from core.evaluation.hallucination_detector import HallucinationDetector
from utils.config import settings
from utils.logger import log_structured
from utils.graph_store import (
    graph_pickle_path,
    load_graph_json,
    save_graph_json,
)
from services.pipeline_observability import (
    PipelineAuditor,
    get_pipeline_state_tracker,
    estimate_size,
)
from services.analysis_report_service import AnalysisReportService
from services.summary_service import SummaryService

logger = logging.getLogger(__name__)


class PipelineStageError(Exception):
    """Error in pipeline stage execution."""
    pass


class ProcessingPipeline:
    """
    Orchestrates the document processing pipeline.
    
    Pipeline Stages:
    1. Ingestion - Document validation and metadata extraction
    2. Chunking - Structure-aware text segmentation
    3. Embedding - Vector generation and FAISS indexing
    4. Mapping - Concept and relation extraction
    5. Graph Building - NetworkX graph construction
    6. Evaluation - Quality assessment and confidence scoring
    7. Hierarchy - Topic trees and prerequisite chains
    
    Each stage preserves partial outputs on failure for recovery.
    """
    
    def __init__(
        self,
        document_service: Optional[DocumentService] = None,
        chunking_service: Optional[ChunkingService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        knowledge_map_service: Optional[KnowledgeMapService] = None,
        relation_dataset_service: Optional[RelationDatasetService] = None
    ):
        """
        Initialize pipeline with service dependencies.
        
        Services are injected for testability and to avoid circular imports.
        """
        self.document_service = document_service or DocumentService()
        self.chunking_service = chunking_service or ChunkingService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.knowledge_map_service = knowledge_map_service or KnowledgeMapService()
        self.relation_dataset_service = relation_dataset_service or RelationDatasetService()
        
        self.graph_builder = GraphBuilder()
        self.graph_exporter = GraphExporter()
        self.confidence_scorer = ConfidenceScorer(self.embedding_service)
        self.hallucination_detector = HallucinationDetector()
        self.pipeline_tracker = get_pipeline_state_tracker()
        self.analysis_report_service = AnalysisReportService()
        self.summary_service = SummaryService()
        
        logger.info("Processing pipeline initialized")
    
    async def execute(
        self,
        document_id: str,
        progress_callback: Optional[Callable] = None,
        skip_existing: bool = True,
        skip_stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute full processing pipeline for a document.
        
        Args:
            document_id: Document to process
            progress_callback: Async function(stage: str, percent: int)
            skip_existing: Skip stages that have already been completed
            skip_stages: List of stage names to skip (for recovery)
        
        Returns:
            Dictionary with pipeline results and metadata
        
        Raises:
            PipelineStageError: If a stage fails irrecoverably
        """
        skip_stages = skip_stages or []

        results = {
            "document_id": document_id,
            "started_at": datetime.now().isoformat(),
            "stages_completed": [],
            "stages_failed": [],
            "artifacts": {},
            "requested_skip_stages": list(skip_stages),
            "skipped_stages": [],
        }

        self.pipeline_tracker.start_pipeline(document_id)
        auditor = PipelineAuditor(document_id=document_id, tracker=self.pipeline_tracker)
        log_structured(
            logger,
            level=logging.INFO,
            stage="pipeline",
            event="pipeline_start",
            message="Pipeline execution started",
            document_id=document_id,
            metadata={"skip_stages": list(skip_stages), "skip_existing": skip_existing},
        )

        async def update_progress(stage: str, percent: int):
            self.pipeline_tracker.set_overall_progress(document_id, stage, percent)
            if progress_callback:
                await progress_callback(stage, percent)

        def should_skip(stage_name: str) -> bool:
            stage_mapping = {
                "ingestion": "ingestion",
                "chunking": "chunking",
                "embedding": "embedding",
                "mapping": "mapping",
                "graph_building": "graph_building",
                "evaluation": "evaluation",
                "hierarchy": "hierarchy",
            }
            normalized = stage_mapping.get(stage_name, stage_name)
            return any(
                normalized
                and (s.lower() in normalized.lower() or normalized.lower() in s.lower())
                for s in skip_stages
            )

        try:
            chunks = None
            knowledge_map = None
            graph = None
            evaluation = {}
            hierarchy = {}
            summary_obj = None

            # Stage 1: Ingestion (0-10%)
            if not should_skip("ingestion"):
                await update_progress(JobStatus.INGESTING.value, 5)
                initial_doc = self.document_service.get_document(document_id)
                auditor.start_stage(
                    "ingestion",
                    input_size=estimate_size(initial_doc.text_content if initial_doc else None),
                    progress=5,
                )
                try:
                    document = await self._stage_ingestion(document_id)
                except Exception as stage_error:
                    results["stages_failed"].append("ingestion")
                    auditor.fail_stage("ingestion", error=str(stage_error), output_size=0)
                    raise
                ingestion_metrics = {
                    "word_count": int(document.metadata.word_count or 0),
                    "has_text": bool(document.text_content),
                }
                auditor.complete_stage(
                    "ingestion",
                    output_size=estimate_size(document.text_content),
                    quality_metrics=ingestion_metrics,
                    progress=10,
                )
                await update_progress(JobStatus.INGESTING.value, 10)
                results["stages_completed"].append("ingestion")
                results["artifacts"]["document"] = {
                    "id": document.id,
                    "word_count": document.metadata.word_count,
                }
            else:
                document = self.document_service.get_document(document_id)
                await update_progress(JobStatus.INGESTING.value, 10)
                auditor.skip_stage("ingestion", progress=10)
                results["skipped_stages"].append("ingestion")

            if not document:
                raise PipelineStageError(f"Document not found: {document_id}")

            # Stage 2: Chunking (10-30%)
            if not should_skip("chunking"):
                await update_progress(JobStatus.CHUNKING.value, 15)
                auditor.start_stage(
                    "chunking",
                    input_size=estimate_size(document.text_content),
                    progress=15,
                )
                try:
                    chunks = await self._stage_chunking(document, use_existing=skip_existing)
                except Exception as stage_error:
                    results["stages_failed"].append("chunking")
                    auditor.fail_stage("chunking", error=str(stage_error), output_size=0)
                    raise
                total_chunk_words = sum(chunk.word_count for chunk in chunks) if chunks else 0
                chunking_metrics = {
                    "chunk_count": len(chunks) if chunks else 0,
                    "avg_chunk_words": round(
                        total_chunk_words / len(chunks), 3
                    )
                    if chunks
                    else 0.0,
                    "sections_detected": len(
                        [chunk for chunk in chunks if chunk.section_title]
                    )
                    if chunks
                    else 0,
                }
                auditor.complete_stage(
                    "chunking",
                    output_size=estimate_size(chunks),
                    quality_metrics=chunking_metrics,
                    progress=30,
                )
                await update_progress(JobStatus.CHUNKING.value, 30)
                results["stages_completed"].append("chunking")
                results["artifacts"]["chunks"] = {"count": len(chunks) if chunks else 0}
            else:
                chunks = self.chunking_service.get_chunks(document.id)
                await update_progress(JobStatus.CHUNKING.value, 30)
                auditor.skip_stage("chunking", progress=30)
                results["skipped_stages"].append("chunking")

            # Stage 3: Embedding (30-50%)
            if not should_skip("embedding"):
                await update_progress(JobStatus.EMBEDDING.value, 35)
                auditor.start_stage(
                    "embedding", input_size=estimate_size(chunks), progress=35
                )
                try:
                    vector_store = await self._stage_embedding(
                        document_id, chunks, use_existing=skip_existing
                    )
                except Exception as stage_error:
                    results["stages_failed"].append("embedding")
                    auditor.fail_stage("embedding", error=str(stage_error), output_size=0)
                    raise
                embedding_metrics = {
                    "indexed": bool(vector_store),
                    "chunk_count": len(chunks) if chunks else 0,
                }
                auditor.complete_stage(
                    "embedding",
                    output_size=len(chunks) if chunks else 0,
                    quality_metrics=embedding_metrics,
                    progress=50,
                )
                await update_progress(JobStatus.EMBEDDING.value, 50)
                results["stages_completed"].append("embedding")
                results["artifacts"]["embeddings"] = {"indexed": True}
            else:
                await update_progress(JobStatus.EMBEDDING.value, 50)
                auditor.skip_stage("embedding", progress=50)
                results["skipped_stages"].append("embedding")

            # Stage 4: Mapping (50-65%)
            if not should_skip("mapping"):
                await update_progress(JobStatus.MAPPING.value, 55)
                auditor.start_stage("mapping", input_size=estimate_size(chunks), progress=55)
                try:
                    knowledge_map = await self._stage_mapping(
                        document_id, chunks, use_existing=skip_existing
                    )
                except Exception as stage_error:
                    results["stages_failed"].append("mapping")
                    auditor.fail_stage("mapping", error=str(stage_error), output_size=0)
                    raise
                mapping_metrics = {
                    "concept_count": len(knowledge_map.concepts),
                    "relation_count": len(knowledge_map.relations),
                    "relation_concept_ratio": round(
                        len(knowledge_map.relations) / max(len(knowledge_map.concepts), 1),
                        3,
                    ),
                }
                auditor.complete_stage(
                    "mapping",
                    output_size=len(knowledge_map.concepts) + len(knowledge_map.relations),
                    quality_metrics=mapping_metrics,
                    progress=65,
                )
                await update_progress(JobStatus.MAPPING.value, 65)
                results["stages_completed"].append("mapping")
                results["artifacts"]["knowledge_map"] = {
                    "concept_count": len(knowledge_map.concepts),
                    "relation_count": len(knowledge_map.relations),
                }

                if knowledge_map.relations:
                    dataset_records = await self.relation_dataset_service.log_relation_batch(
                        document_id=document_id,
                        knowledge_map=knowledge_map,
                        chunks=chunks,
                    )
                    results["artifacts"]["dataset_records"] = dataset_records
                    dataset_export_path = self.relation_dataset_service.export_document_snapshot(
                        document_id=document_id,
                        document_filename=document.metadata.filename if document.metadata else None,
                    )
                    if dataset_export_path:
                        results["artifacts"]["dataset_export"] = str(dataset_export_path)
            else:
                knowledge_map = self.knowledge_map_service.get_knowledge_map(document_id)
                await update_progress(JobStatus.MAPPING.value, 65)
                auditor.skip_stage("mapping", progress=65)
                results["skipped_stages"].append("mapping")

            # Stage 5: Graph Building (65-75%)
            if not should_skip("graph_building"):
                await update_progress(JobStatus.GRAPH_BUILDING.value, 70)
                map_size = (
                    len(knowledge_map.concepts) + len(knowledge_map.relations)
                    if knowledge_map
                    else 0
                )
                auditor.start_stage("graph_building", input_size=map_size, progress=70)
                try:
                    graph, graph_path = await self._stage_graph_building(
                        document_id, knowledge_map, use_existing=skip_existing
                    )
                except Exception as stage_error:
                    results["stages_failed"].append("graph_building")
                    auditor.fail_stage(
                        "graph_building", error=str(stage_error), output_size=0
                    )
                    raise
                graph_metrics = {
                    "node_count": int(graph.number_of_nodes()),
                    "edge_count": int(graph.number_of_edges()),
                    "density": round(nx.density(graph), 4)
                    if graph.number_of_nodes() > 1
                    else 0.0,
                    "is_weakly_connected": nx.is_weakly_connected(graph)
                    if graph.number_of_nodes() > 0
                    else True,
                }
                auditor.complete_stage(
                    "graph_building",
                    output_size=graph.number_of_nodes() + graph.number_of_edges(),
                    quality_metrics=graph_metrics,
                    progress=75,
                )
                await update_progress(JobStatus.GRAPH_BUILDING.value, 75)
                results["stages_completed"].append("graph_building")
                results["artifacts"]["graph"] = {
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                    "path": str(graph_path) if graph_path else None,
                }
            else:
                graph = load_graph_json(document_id)
                if graph is None and settings.allow_legacy_pickle_loading:
                    legacy_path = graph_pickle_path(document_id)
                    if legacy_path.exists():
                        import pickle

                        with open(legacy_path, "rb") as f:
                            graph = pickle.load(f)
                await update_progress(JobStatus.GRAPH_BUILDING.value, 75)
                auditor.skip_stage("graph_building", progress=75)
                results["skipped_stages"].append("graph_building")

            # Stage 6: Evaluation (75-90%)
            if not should_skip("evaluation"):
                await update_progress(JobStatus.EVALUATING.value, 80)
                eval_input_size = (
                    (len(knowledge_map.relations) if knowledge_map else 0)
                    + (len(chunks) if chunks else 0)
                )
                auditor.start_stage("evaluation", input_size=eval_input_size, progress=80)
                evaluation = await self._stage_evaluation(
                    document_id, knowledge_map, chunks, graph
                )
                evaluation_metrics = {
                    "overall_score": evaluation.get("overall_score", 0),
                    "risk_score": evaluation.get("risk_score", 0),
                    "flags_count": len(evaluation.get("flags", [])),
                }
                auditor.complete_stage(
                    "evaluation",
                    output_size=len(evaluation.get("flags", [])),
                    quality_metrics=evaluation_metrics,
                    progress=90,
                )
                await update_progress(JobStatus.EVALUATING.value, 90)
                results["stages_completed"].append("evaluation")
                results["artifacts"]["evaluation"] = {
                    "overall_score": evaluation.get("overall_score"),
                    "flags_count": len(evaluation.get("flags", [])),
                }
            else:
                await update_progress(JobStatus.EVALUATING.value, 90)
                auditor.skip_stage("evaluation", progress=90)
                results["skipped_stages"].append("evaluation")

            # Stage 7: Hierarchy (90-100%)
            if not should_skip("hierarchy"):
                await update_progress(JobStatus.HIERARCHY.value, 95)
                auditor.start_stage(
                    "hierarchy",
                    input_size=graph.number_of_nodes() + graph.number_of_edges()
                    if graph is not None
                    else 0,
                    progress=95,
                )
                hierarchy = await self._stage_hierarchy(document_id, graph)
                hierarchy_metrics = {
                    "layers_count": len(hierarchy.get("layers", [])),
                    "chains_count": len(hierarchy.get("chains", [])),
                    "has_topic_tree": bool(hierarchy.get("topic_tree")),
                }
                auditor.complete_stage(
                    "hierarchy",
                    output_size=len(hierarchy.get("layers", []))
                    + len(hierarchy.get("chains", [])),
                    quality_metrics=hierarchy_metrics,
                    progress=100,
                )
                await update_progress(JobStatus.HIERARCHY.value, 100)
                results["stages_completed"].append("hierarchy")
                results["artifacts"]["hierarchy"] = {
                    "layers_count": len(hierarchy.get("layers", [])),
                    "chains_count": len(hierarchy.get("chains", [])),
                }
            else:
                await update_progress(JobStatus.HIERARCHY.value, 100)
                auditor.skip_stage("hierarchy", progress=100)
                results["skipped_stages"].append("hierarchy")
            
            # Stage 8: Summarization (Internal)
            if chunks and not should_skip("summarization"):
                try:
                    summary_obj = self.summary_service.generate_summary(
                        document_id=document_id,
                        title=document.metadata.filename if document.metadata else "Document",
                        chunks=chunks
                    )
                    results["artifacts"]["summary"] = {
                        "section_count": len(summary_obj.sections),
                        "has_overall": bool(summary_obj.overall_summary)
                    }
                except Exception as sum_err:
                    logger.warning(f"Summarization failed: {sum_err}")

            # Mark document as analyzed
            self.document_service.update_document_status(document_id, DocumentStatus.ANALYZED)

            results["completed_at"] = datetime.now().isoformat()
            results["success"] = True
            results["audit"] = auditor.report()
            self.pipeline_tracker.complete_pipeline(document_id)
            self.analysis_report_service.save_report(
                document_id=document_id,
                report=self._build_analysis_report(
                    results=results,
                    success=True,
                    error=None,
                    knowledge_map=knowledge_map,
                    graph=graph,
                    evaluation=evaluation,
                    hierarchy=hierarchy,
                    summary=summary_obj,
                ),
            )
            log_structured(
                logger,
                level=logging.INFO,
                stage="pipeline",
                event="pipeline_complete",
                message="Pipeline execution completed",
                document_id=document_id,
                metadata={"stages_completed": results["stages_completed"]},
            )

        except Exception as e:
            logger.error(f"Pipeline failed for document {document_id}: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["failed_at"] = datetime.now().isoformat()
            results["audit"] = auditor.report()
            self.pipeline_tracker.fail_pipeline(document_id, str(e))
            self.analysis_report_service.save_report(
                document_id=document_id,
                report=self._build_analysis_report(
                    results=results,
                    success=False,
                    error=str(e),
                    knowledge_map=knowledge_map,
                    graph=graph,
                    evaluation=evaluation,
                    hierarchy=hierarchy,
                    summary=summary_obj,
                ),
            )
            log_structured(
                logger,
                level=logging.ERROR,
                stage="pipeline",
                event="pipeline_error",
                message="Pipeline execution failed",
                document_id=document_id,
                metadata={"error": str(e), "stages_failed": results["stages_failed"]},
            )

            # Update document status to failed
            self.document_service.update_document_status(
                document_id, DocumentStatus.FAILED, error_message=str(e)
            )

            raise PipelineStageError(f"Pipeline failed: {e}") from e

        return results

    def _build_analysis_report(
        self,
        results: Dict[str, Any],
        success: bool,
        error: Optional[str],
        knowledge_map,
        graph,
        evaluation: Dict[str, Any],
        hierarchy: Dict[str, Any],
        summary: Optional[Any] = None,
    ) -> Dict[str, Any]:
        concept_count = len(knowledge_map.concepts) if knowledge_map else 0
        relation_count = len(knowledge_map.relations) if knowledge_map else 0
        node_count = int(graph.number_of_nodes()) if graph is not None else 0
        edge_count = int(graph.number_of_edges()) if graph is not None else 0
        density = round(nx.density(graph), 4) if graph is not None and node_count > 1 else 0.0
        audit = results.get("audit", {})
        stage_validations = []
        for stage in audit.get("stages", []):
            stage_validations.append(
                {
                    "stage": stage.get("stage"),
                    "status": stage.get("status"),
                    "validation_passed": (stage.get("validation") or {}).get("passed"),
                    "issues": (stage.get("validation") or {}).get("issues", []),
                    "quality_metrics": stage.get("quality_metrics", {}),
                }
            )

        summary_metrics = {
            "concept_count": concept_count,
            "relation_count": relation_count,
            "graph_nodes": node_count,
            "graph_edges": edge_count,
            "graph_density": density,
            "evaluation_overall_score": evaluation.get("overall_score"),
            "evaluation_risk_score": evaluation.get("risk_score"),
            "evaluation_flags": len(evaluation.get("flags", [])),
            "hierarchy_layers": len(hierarchy.get("layers", [])),
            "hierarchy_chains": len(hierarchy.get("chains", [])),
        }
        top_concepts = self._build_top_concepts(knowledge_map)
        relation_breakdown = self._build_relation_breakdown(knowledge_map)
        sample_relations = self._build_sample_relations(knowledge_map)
        key_findings = self._build_key_findings(
            summary=summary_metrics,
            top_concepts=top_concepts,
            relation_breakdown=relation_breakdown,
            structured_summary=summary,
        )
        recommendations = self._build_recommendations(
            evaluation=evaluation,
            stage_validations=stage_validations,
            structured_summary=summary,
        )

        return {
            "document_id": results.get("document_id"),
            "success": success,
            "error": error,
            "generated_at": datetime.now().isoformat(),
            "started_at": results.get("started_at"),
            "completed_at": results.get("completed_at") or results.get("failed_at"),
            "stages_completed": results.get("stages_completed", []),
            "stages_failed": results.get("stages_failed", []),
            "skipped_stages": results.get("skipped_stages", []),
            "summary": summary_metrics,
            "insights": {
                "overview": self._build_overview_text(summary=summary_metrics, narrative_summary=summary.overall_summary if summary else None),
                "narrative_summary": summary.overall_summary if summary else None,
                "key_findings": key_findings,
                "top_concepts": top_concepts,
                "relation_breakdown": relation_breakdown,
                "sample_relations": sample_relations,
                "recommendations": recommendations,
            },
            "stage_validations": stage_validations,
            "artifacts": results.get("artifacts", {}),
            "evaluation": evaluation,
            "hierarchy": hierarchy,
        }

    def _build_overview_text(self, summary: Dict[str, Any], narrative_summary: Optional[str] = None) -> str:
        if narrative_summary and "llm unavailable" not in narrative_summary.lower():
            return narrative_summary
        
        concept_count = int(summary.get("concept_count") or 0)
        relation_count = int(summary.get("relation_count") or 0)
        node_count = int(summary.get("graph_nodes") or 0)
        edge_count = int(summary.get("graph_edges") or 0)
        density = summary.get("graph_density")
        return (
            f"Extracted {concept_count} concepts and {relation_count} relations. "
            f"Built a graph with {node_count} nodes and {edge_count} edges "
            f"(density={density})."
        )

    def _build_top_concepts(self, knowledge_map, limit: int = 8) -> List[Dict[str, Any]]:
        if not knowledge_map or not knowledge_map.concepts:
            return []

        ranked = []
        for concept in knowledge_map.concepts:
            chunk_count = len(concept.chunk_ids or [])
            ranked.append(
                {
                    "name": concept.name,
                    "chunk_mentions": chunk_count,
                    "definition": (concept.definition or "").strip()[:180],
                }
            )

        ranked.sort(key=lambda item: (item["chunk_mentions"], item["name"].lower()), reverse=True)
        return ranked[:limit]

    def _build_relation_breakdown(self, knowledge_map) -> List[Dict[str, Any]]:
        if not knowledge_map or not knowledge_map.relations:
            return []

        counter: Counter = Counter()
        for relation in knowledge_map.relations:
            relation_type = getattr(relation.relation_type, "value", str(relation.relation_type))
            counter[str(relation_type)] += 1

        return [
            {"type": relation_type, "count": int(count)}
            for relation_type, count in counter.most_common()
        ]

    def _build_sample_relations(self, knowledge_map, limit: int = 6) -> List[Dict[str, Any]]:
        if not knowledge_map or not knowledge_map.relations:
            return []

        sorted_relations = sorted(
            knowledge_map.relations,
            key=lambda rel: float(rel.confidence or 0.0),
            reverse=True,
        )
        items: List[Dict[str, Any]] = []
        for relation in sorted_relations[:limit]:
            relation_type = getattr(relation.relation_type, "value", str(relation.relation_type))
            items.append(
                {
                    "from": relation.from_concept,
                    "to": relation.to_concept,
                    "type": str(relation_type),
                    "confidence": round(float(relation.confidence or 0.0), 3),
                    "description": (relation.description or "").strip()[:200],
                }
            )
        return items

    def _build_key_findings(
        self,
        summary: Dict[str, Any],
        top_concepts: List[Dict[str, Any]],
        relation_breakdown: List[Dict[str, Any]],
        structured_summary: Optional[Any] = None,
    ) -> List[str]:
        findings: List[str] = []

        if structured_summary:
            concepts: List[str] = []
            seen_concepts = set()
            for section in structured_summary.sections:
                for concept in section.related_concepts[:4]:
                    normalized = self._normalize_finding_concept(concept)
                    if not normalized or normalized in seen_concepts:
                        continue
                    seen_concepts.add(normalized)
                    concepts.append(normalized)
                    if len(concepts) >= 6:
                        break
                if len(concepts) >= 6:
                    break

            if concepts:
                findings.append(
                    f"Main themes: {self._join_readable_phrases(concepts[:4])}."
                )
                if len(concepts) > 4:
                    findings.append(
                        f"Additional topics: {self._join_readable_phrases(concepts[4:6])}."
                    )

            summary_text = str(getattr(structured_summary, "overall_summary", "") or "")
            if "assessment-style questions" in summary_text.lower():
                findings.append(
                    "Format: assessment-style questions covering both conceptual understanding and applied problems."
                )

            for section in structured_summary.sections[:5]:
                for point in section.key_points[:2]:
                    text = str(point).strip()
                    if not self._is_user_facing_finding(text):
                        continue
                    if text.lower().startswith(("covers ", "also touches on ")):
                        continue
                    if not text.endswith((".", "!", "?")):
                        text = f"{text}."
                    findings.append(text)
                    if len(findings) >= 6:
                        return findings[:6]

            if findings:
                return findings[:6]

        if top_concepts:
            concept_names = ", ".join([item["name"] for item in top_concepts[:5]])
            findings.append(f"Top concepts: {concept_names}.")

        if relation_breakdown:
            top_relation = relation_breakdown[0]
            findings.append(
                f"Most common relation type is '{top_relation['type']}' ({top_relation['count']} links)."
            )

        overall = summary.get("evaluation_overall_score")
        if overall is not None:
            findings.append(f"Overall extraction confidence score: {overall}.")

        risk_score = summary.get("evaluation_risk_score")
        if risk_score is not None:
            findings.append(f"Risk score from hallucination checks: {risk_score}.")

        hierarchy_layers = int(summary.get("hierarchy_layers") or 0)
        if hierarchy_layers > 0:
            findings.append(f"Hierarchy generation produced {hierarchy_layers} learning layers.")

        if not findings:
            findings.append("Analysis completed, but no strong structured findings were produced.")

        return findings

    def _build_recommendations(
        self,
        evaluation: Dict[str, Any],
        stage_validations: List[Dict[str, Any]],
        structured_summary: Optional[Any] = None,
    ) -> List[str]:
        recommendations: List[str] = []

        for suggestion in evaluation.get("recommendations", [])[:4]:
            text = str(suggestion).strip()
            if text:
                recommendations.append(text)

        for validation in stage_validations:
            if validation.get("validation_passed") is False:
                issues = validation.get("issues") or []
                if issues:
                    recommendations.append(
                        f"Review {validation.get('stage')} stage: {', '.join([str(item) for item in issues])}."
                    )

        if not recommendations:
            recommendations.append(
                "No major quality issues detected. The summary is ready for review and downstream Q&A."
            )

        if structured_summary and structured_summary.metadata.get("llm_summary_accepted") is False:
            recommendations.append(
                "The summary was generated from extracted document text because the live model response was unavailable or unusable."
            )

        # Preserve order while removing duplicates.
        deduped = []
        seen = set()
        for item in recommendations:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:6]

    def _is_user_facing_finding(self, text: str) -> bool:
        """Filter summary bullets so the report shows readable content findings."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        lowered = cleaned.lower()
        if lowered.startswith("key focus:"):
            return False
        if any(
            phrase in lowered
            for phrase in [
                "subject code",
                "full tmarks",
                "which of the following is the possible number",
                "page ",
            ]
        ):
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned)
        if len(words) < 5:
            return False
        alpha_count = sum(char.isalpha() for char in cleaned)
        digit_count = sum(char.isdigit() for char in cleaned)
        if alpha_count and digit_count > alpha_count * 0.12:
            return False
        return True

    def _normalize_finding_concept(self, text: str) -> Optional[str]:
        """Normalize section concepts into report-safe topic labels."""
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        cleaned = cleaned.replace("-", " ")
        if not cleaned or any(char.isdigit() for char in cleaned):
            return None
        blocked = {
            "bject", "name", "following", "possible", "number", "all", "attempt",
            "question", "questions", "mathematics lll", "mathematics", "rlation",
            "popr", "mean height", "page", "term", "type", "days", "suppose",
        }
        if cleaned in blocked:
            return None
        words = re.findall(r"[a-z][a-z']+", cleaned)
        if not words:
            return None
        alpha_only = "".join(words)
        vowel_count = sum(char in "aeiou" for char in alpha_only)
        if len(alpha_only) >= 6 and vowel_count < max(1, len(alpha_only) * 0.2):
            return None
        return cleaned

    def _join_readable_phrases(self, items: List[str]) -> str:
        """Join topic labels for a short user-facing phrase."""
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"
    
    async def _stage_ingestion(self, document_id: str) -> Document:
        """Stage 1: Validate document and extract metadata."""
        try:
            document = self.document_service.get_document(document_id)
            
            if not document:
                raise PipelineStageError(f"Document not found: {document_id}")
            
            if not document.text_content:
                raise PipelineStageError(f"Document has no text content: {document_id}")
            
            self.document_service.update_document_status(
                document_id, DocumentStatus.PROCESSING
            )
            
            return document
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise PipelineStageError(f"Ingestion failed: {e}") from e
    
    async def _stage_chunking(self, document: Document, use_existing: bool = True):
        """Stage 2: Structure-aware text chunking."""
        try:
            if use_existing:
                existing_chunks = self.chunking_service.get_chunks(document.id)
                if existing_chunks:
                    return existing_chunks
            else:
                self.chunking_service.delete_chunks(document.id)
            
            chunks = self.chunking_service.chunk_document(
                document.id, 
                document.text_content
            )
            
            return chunks
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise PipelineStageError(f"Chunking failed: {e}") from e
    
    async def _stage_embedding(self, document_id: str, chunks, use_existing: bool = True):
        """Stage 3: Generate embeddings and build FAISS index."""
        try:
            if use_existing:
                existing_store = self.embedding_service.get_vector_store(document_id)
                if existing_store:
                    return existing_store
            else:
                self.embedding_service.delete_embeddings(document_id)
            
            embeddings = self.embedding_service.generate_embeddings(chunks)
            vector_store = self.embedding_service.create_vector_index(
                document_id, chunks, embeddings
            )
            
            return vector_store
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise PipelineStageError(f"Embedding failed: {e}") from e
    
    async def _stage_mapping(self, document_id: str, chunks, use_existing: bool = True):
        """Stage 4: Extract concepts and relations."""
        try:
            if use_existing:
                existing_map = self.knowledge_map_service.get_knowledge_map(document_id)
                if existing_map:
                    return existing_map
            else:
                self.knowledge_map_service.delete_knowledge_map(document_id)
            
            knowledge_map = self.knowledge_map_service.build_knowledge_map(
                document_id, chunks
            )
            
            return knowledge_map
        except Exception as e:
            logger.error(f"Mapping failed: {e}")
            raise PipelineStageError(f"Mapping failed: {e}") from e
    
    async def _stage_graph_building(
        self, document_id: str, knowledge_map, use_existing: bool = True
    ):
        """Stage 5: Build NetworkX graph from knowledge map."""
        try:
            if use_existing:
                existing_graph = load_graph_json(document_id)
                if existing_graph is not None:
                    return existing_graph, settings.data_dir / "graphs" / f"{document_id}.json"

                if settings.allow_legacy_pickle_loading:
                    legacy_path = graph_pickle_path(document_id)
                    if legacy_path.exists():
                        import pickle
                        with open(legacy_path, "rb") as f:
                            graph = pickle.load(f)
                        return graph, legacy_path
            
            graph = self.graph_builder.build_from_knowledge_map(knowledge_map)
            
            graph_path = save_graph_json(graph, document_id)
            
            cytoscape_path = settings.data_dir / "graphs" / f"{document_id}_cytoscape.json"
            self.graph_exporter.save_to_file(graph, cytoscape_path, format="cytoscape")
            
            return graph, graph_path
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            raise PipelineStageError(f"Graph building failed: {e}") from e
    
    async def _stage_evaluation(self, document_id, knowledge_map, chunks, graph):
        """Stage 6: Evaluate quality and detect hallucinations."""
        try:
            confidence_scores = {}
            for relation in knowledge_map.relations:
                score = self.confidence_scorer.score_relation(
                    relation, chunks, graph=graph
                )
                confidence_scores[relation.id] = score
                relation.confidence = score
            
            report = self.hallucination_detector.detect(
                knowledge_map, chunks, graph
            )
            
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
            
            return {
                "document_id": document_id,
                "overall_score": round(avg_confidence, 3),
                "confidence_scores": confidence_scores,
                "flags": [
                    {
                        "type": flag.flag_type,
                        "severity": flag.severity,
                        "item_type": flag.item_type,
                        "item_id": flag.item_id,
                        "description": flag.description,
                        "suggestion": flag.suggestion
                    }
                    for flag in report.flags
                ],
                "recommendations": report.recommendations,
                "risk_score": report.overall_risk_score
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "document_id": document_id,
                "overall_score": 0.5,
                "error": str(e),
                "flags": [],
                "recommendations": ["Evaluation failed - manual review recommended"]
            }
    
    async def _stage_hierarchy(self, document_id: str, graph):
        """Stage 7: Generate hierarchies and learning paths."""
        try:
            generator = HierarchyGenerator(graph)
            
            layers = generator.generate_conceptual_layers()
            chains = generator.generate_prerequisite_chains()
            topic_tree = generator.generate_topic_tree()
            
            return {
                "document_id": document_id,
                "layers": [
                    {
                        "level": layer.level,
                        "name": layer.name,
                        "description": layer.description,
                        "concept_count": len(layer.concept_ids)
                    }
                    for layer in layers
                ],
                "chains": [
                    {
                        "chain_id": chain.chain_id,
                        "target": chain.target_concept_id,
                        "length": chain.total_length,
                        "difficulty": chain.estimated_difficulty
                    }
                    for chain in chains
                ],
                "topic_tree": topic_tree.dict() if topic_tree else None
            }
        except Exception as e:
            logger.error(f"Hierarchy generation failed: {e}")
            return {
                "document_id": document_id,
                "layers": [],
                "chains": [],
                "error": str(e)
            }
