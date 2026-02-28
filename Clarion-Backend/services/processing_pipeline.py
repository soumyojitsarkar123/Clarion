"""
Processing Pipeline - Orchestrates document processing stages.

Integrates existing services to execute the full pipeline:
ingestion → chunking → embedding → mapping → graph → evaluation → hierarchy
"""

import logging
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from pathlib import Path

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
from utils.graph_store import (
    graph_pickle_path,
    load_graph_json,
    save_graph_json,
)

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
            "skipped_stages": list(skip_stages)
        }
        
        async def update_progress(stage: str, percent: int):
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
                "hierarchy": "hierarchy"
            }
            normalized = stage_mapping.get(stage_name, stage_name)
            return any(normalized and (s.lower() in normalized.lower() or normalized.lower() in s.lower()) for s in skip_stages)
        
        try:
            chunks = None
            knowledge_map = None
            graph = None
            
            # Stage 1: Ingestion (0-10%)
            if not should_skip("ingestion"):
                await update_progress(JobStatus.INGESTING.value, 5)
                document = await self._stage_ingestion(document_id)
                await update_progress(JobStatus.INGESTING.value, 10)
                results["stages_completed"].append("ingestion")
                results["artifacts"]["document"] = {"id": document.id, "word_count": document.metadata.word_count}
            else:
                document = self.document_service.get_document(document_id)
                await update_progress(JobStatus.INGESTING.value, 10)
                results["skipped_stages"].append("ingestion")
            
            if not document:
                raise PipelineStageError(f"Document not found: {document_id}")
            
            # Stage 2: Chunking (10-30%)
            if not should_skip("chunking"):
                await update_progress(JobStatus.CHUNKING.value, 15)
                chunks = await self._stage_chunking(document)
                await update_progress(JobStatus.CHUNKING.value, 30)
                results["stages_completed"].append("chunking")
                results["artifacts"]["chunks"] = {"count": len(chunks) if chunks else 0}
            else:
                chunks = self.chunking_service.get_chunks(document.id)
                await update_progress(JobStatus.CHUNKING.value, 30)
                results["skipped_stages"].append("chunking")
            
            # Stage 3: Embedding (30-50%)
            if not should_skip("embedding"):
                await update_progress(JobStatus.EMBEDDING.value, 35)
                vector_store = await self._stage_embedding(document_id, chunks)
                await update_progress(JobStatus.EMBEDDING.value, 50)
                results["stages_completed"].append("embedding")
                results["artifacts"]["embeddings"] = {"indexed": True}
            else:
                await update_progress(JobStatus.EMBEDDING.value, 50)
                results["skipped_stages"].append("embedding")
            
            # Stage 4: Mapping (50-65%)
            if not should_skip("mapping"):
                await update_progress(JobStatus.MAPPING.value, 55)
                knowledge_map = await self._stage_mapping(document_id, chunks)
                await update_progress(JobStatus.MAPPING.value, 65)
                results["stages_completed"].append("mapping")
                results["artifacts"]["knowledge_map"] = {
                    "concept_count": len(knowledge_map.concepts),
                    "relation_count": len(knowledge_map.relations)
                }
                
                if knowledge_map.relations:
                    dataset_records = await self.relation_dataset_service.log_relation_batch(
                        document_id=document_id,
                        knowledge_map=knowledge_map,
                        chunks=chunks
                    )
                    results["artifacts"]["dataset_records"] = dataset_records
            else:
                knowledge_map = self.knowledge_map_service.get_knowledge_map(document_id)
                await update_progress(JobStatus.MAPPING.value, 65)
                results["skipped_stages"].append("mapping")
            
            # Stage 5: Graph Building (65-75%)
            if not should_skip("graph_building"):
                await update_progress(JobStatus.GRAPH_BUILDING.value, 70)
                graph, graph_path = await self._stage_graph_building(document_id, knowledge_map)
                await update_progress(JobStatus.GRAPH_BUILDING.value, 75)
                results["stages_completed"].append("graph_building")
                results["artifacts"]["graph"] = {
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                    "path": str(graph_path) if graph_path else None
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
                results["skipped_stages"].append("graph_building")
            
            # Stage 6: Evaluation (75-90%)
            if not should_skip("evaluation"):
                await update_progress(JobStatus.EVALUATING.value, 80)
                evaluation = await self._stage_evaluation(document_id, knowledge_map, chunks, graph)
                await update_progress(JobStatus.EVALUATING.value, 90)
                results["stages_completed"].append("evaluation")
                results["artifacts"]["evaluation"] = {
                    "overall_score": evaluation.get("overall_score"),
                    "flags_count": len(evaluation.get("flags", []))
                }
            else:
                await update_progress(JobStatus.EVALUATING.value, 90)
                results["skipped_stages"].append("evaluation")
            
            # Stage 7: Hierarchy (90-100%)
            if not should_skip("hierarchy"):
                await update_progress(JobStatus.HIERARCHY.value, 95)
                hierarchy = await self._stage_hierarchy(document_id, graph)
                await update_progress(JobStatus.HIERARCHY.value, 100)
                results["stages_completed"].append("hierarchy")
                results["artifacts"]["hierarchy"] = {
                    "layers_count": len(hierarchy.get("layers", [])),
                    "chains_count": len(hierarchy.get("chains", []))
                }
            else:
                await update_progress(JobStatus.HIERARCHY.value, 100)
                results["skipped_stages"].append("hierarchy")
            
            # Mark document as analyzed
            self.document_service.update_document_status(
                document_id, DocumentStatus.ANALYZED
            )
            
            results["completed_at"] = datetime.now().isoformat()
            results["success"] = True
            
            logger.info(f"Pipeline completed for document {document_id}")
            
        except Exception as e:
            logger.error(f"Pipeline failed for document {document_id}: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["failed_at"] = datetime.now().isoformat()
            
            # Update document status to failed
            self.document_service.update_document_status(
                document_id, DocumentStatus.FAILED, error_message=str(e)
            )
            
            raise PipelineStageError(f"Pipeline failed: {e}") from e
        
        return results
    
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
    
    async def _stage_chunking(self, document: Document):
        """Stage 2: Structure-aware text chunking."""
        try:
            existing_chunks = self.chunking_service.get_chunks(document.id)
            if existing_chunks:
                return existing_chunks
            
            chunks = self.chunking_service.chunk_document(
                document.id, 
                document.text_content
            )
            
            return chunks
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise PipelineStageError(f"Chunking failed: {e}") from e
    
    async def _stage_embedding(self, document_id: str, chunks):
        """Stage 3: Generate embeddings and build FAISS index."""
        try:
            existing_store = self.embedding_service.get_vector_store(document_id)
            if existing_store:
                return existing_store
            
            embeddings = self.embedding_service.generate_embeddings(chunks)
            vector_store = self.embedding_service.create_vector_index(
                document_id, chunks, embeddings
            )
            
            return vector_store
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise PipelineStageError(f"Embedding failed: {e}") from e
    
    async def _stage_mapping(self, document_id: str, chunks):
        """Stage 4: Extract concepts and relations."""
        try:
            existing_map = self.knowledge_map_service.get_knowledge_map(document_id)
            if existing_map:
                return existing_map
            
            knowledge_map = self.knowledge_map_service.build_knowledge_map(
                document_id, chunks
            )
            
            return knowledge_map
        except Exception as e:
            logger.error(f"Mapping failed: {e}")
            raise PipelineStageError(f"Mapping failed: {e}") from e
    
    async def _stage_graph_building(self, document_id: str, knowledge_map):
        """Stage 5: Build NetworkX graph from knowledge map."""
        try:
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
