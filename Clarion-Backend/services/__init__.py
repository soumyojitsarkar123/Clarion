"""
Service layer modules for business logic.
"""

from importlib import import_module

__all__ = [
    "DocumentService",
    "ChunkingService",
    "EmbeddingService",
    "RetrievalService",
    "KnowledgeMapService",
    "SummaryService",
    "BackgroundService",
    "JobRepository",
    "ProcessingJob",
    "JobStatus",
    "ProcessingPipeline",
    "PipelineStageError",
]


_MODULE_MAP = {
    "DocumentService": "services.document_service",
    "ChunkingService": "services.chunking_service",
    "EmbeddingService": "services.embedding_service",
    "RetrievalService": "services.retrieval_service",
    "KnowledgeMapService": "services.knowledge_map_service",
    "SummaryService": "services.summary_service",
    "BackgroundService": "services.background_service",
    "JobRepository": "services.background_service",
    "ProcessingJob": "services.background_service",
    "JobStatus": "services.background_service",
    "ProcessingPipeline": "services.processing_pipeline",
    "PipelineStageError": "services.processing_pipeline",
}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(name)
    module = import_module(_MODULE_MAP[name])
    return getattr(module, name)
