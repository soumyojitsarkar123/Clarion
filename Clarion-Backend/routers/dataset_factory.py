"""
Dataset Factory Router - API endpoints for autonomous dataset generation.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from services.dataset_factory_service import (
    continuous_dataset_generation,
    get_dataset_factory,
)
from services.background_service import get_background_service
from utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dataset-factory", tags=["dataset-factory"])


class DatasetFactoryResponse(BaseModel):
    """Response model for dataset factory operations."""

    status: str
    message: str
    statistics: Optional[dict] = None


class DatasetFactoryConfigUpdate(BaseModel):
    """Request model for updating dataset factory configuration."""

    quality_threshold: Optional[float] = None
    dedup_threshold: Optional[float] = None
    llm_sample_rate: Optional[float] = None
    batch_size: Optional[int] = None
    generation_interval_hours: Optional[float] = None
    chunk_timeout_seconds: Optional[float] = None
    max_retries: Optional[int] = None
    require_llm_for_export: Optional[bool] = None


@router.post("/generate", response_model=DatasetFactoryResponse)
async def trigger_dataset_generation(
    background_tasks: BackgroundTasks, document_id: Optional[str] = None
):
    """
    Trigger dataset generation for a specific document or all documents.

    Args:
        background_tasks: FastAPI background tasks
        document_id: Optional specific document to process

    Returns:
        Status of the triggered generation
    """
    try:
        factory = get_dataset_factory()

        if document_id:
            # Process specific document
            background_tasks.add_task(factory.process_document, document_id)
            message = f"Dataset generation triggered for document {document_id}"
        else:
            # Process all documents
            background_tasks.add_task(factory.process_all_documents)
            message = "Dataset generation triggered for all documents"

        logger.info(message)

        return DatasetFactoryResponse(
            status="triggered", message=message, statistics=factory.get_statistics()
        )

    except Exception as e:
        logger.error(f"Error triggering dataset generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/statistics", response_model=DatasetFactoryResponse)
async def get_dataset_factory_statistics():
    """
    Get dataset factory statistics and status.

    Returns:
        Current statistics and configuration
    """
    try:
        factory = get_dataset_factory()
        stats = factory.get_statistics()

        return DatasetFactoryResponse(
            status="success",
            message="Dataset factory statistics retrieved",
            statistics=stats,
        )

    except Exception as e:
        logger.error(f"Error getting dataset factory statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/configure", response_model=DatasetFactoryResponse)
async def update_dataset_factory_config(config_update: DatasetFactoryConfigUpdate):
    """
    Update dataset factory configuration.

    Args:
        config_update: Configuration parameters to update

    Returns:
        Status of the configuration update
    """
    try:
        factory = get_dataset_factory()
        updated_fields = []

        # Update settings if provided
        if config_update.quality_threshold is not None:
            settings.dataset_quality_threshold = config_update.quality_threshold
            factory.quality_threshold = config_update.quality_threshold
            updated_fields.append("quality_threshold")

        if config_update.dedup_threshold is not None:
            settings.dataset_dedup_threshold = config_update.dedup_threshold
            factory.dedup_threshold = config_update.dedup_threshold
            updated_fields.append("dedup_threshold")

        if config_update.llm_sample_rate is not None:
            settings.dataset_llm_sample_rate = config_update.llm_sample_rate
            factory.llm_sample_rate = config_update.llm_sample_rate
            updated_fields.append("llm_sample_rate")

        if config_update.batch_size is not None:
            settings.dataset_batch_size = config_update.batch_size
            factory.batch_size = config_update.batch_size
            updated_fields.append("batch_size")

        if config_update.generation_interval_hours is not None:
            settings.dataset_generation_interval_hours = (
                config_update.generation_interval_hours
            )
            updated_fields.append("generation_interval_hours")

        if config_update.chunk_timeout_seconds is not None:
            settings.dataset_chunk_timeout_seconds = config_update.chunk_timeout_seconds
            updated_fields.append("chunk_timeout_seconds")

        if config_update.max_retries is not None:
            settings.dataset_max_retries_per_chunk = config_update.max_retries
            updated_fields.append("max_retries")

        if config_update.require_llm_for_export is not None:
            settings.dataset_require_llm_for_export = config_update.require_llm_for_export
            factory.require_llm_for_export = config_update.require_llm_for_export
            updated_fields.append("require_llm_for_export")

        message = f"Updated dataset factory configuration: {', '.join(updated_fields)}"
        logger.info(message)

        return DatasetFactoryResponse(
            status="success", message=message, statistics=factory.get_statistics()
        )

    except Exception as e:
        logger.error(f"Error updating dataset factory configuration: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/attempts")
async def get_generation_attempts(
    document_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Retrieve granular attempt telemetry records.
    """
    try:
        factory = get_dataset_factory()
        return factory.get_generation_attempts(
            document_id=document_id,
            generation_id=generation_id,
            chunk_id=chunk_id,
            status=status,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error retrieving generation attempts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/checkpoints/{document_id}")
async def get_document_checkpoints(document_id: str):
    """
    Retrieve chunk-level checkpoint states and progress summary for a document.
    """
    try:
        factory = get_dataset_factory()
        summary = factory.get_checkpoint_summary(document_id)
        if summary.get("total_chunks", 0) == 0:
            # Check if document exists
            doc = factory.document_service.get_document(document_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving checkpoints for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/export/{document_id}")
async def export_document_dataset(document_id: str):
    """
    Export the dataset for a specific document, enforcing strict gold-export research data integrity gates.
    Under require_llm_for_export=True, refuses export (HTTP 409 Conflict) if:
    - Generation is incomplete (partially_completed, failed, or completed < total)
    - Any sample was generated heuristically (deterministic fallback)
    """
    factory = get_dataset_factory()
    return factory.export_document_dataset_strict(document_id)


@router.post("/clear-cache", response_model=DatasetFactoryResponse)
async def clear_embedding_cache():
    """
    Clear the embedding cache to free memory.

    Returns:
        Status of the cache clearing operation
    """
    try:
        factory = get_dataset_factory()
        factory.clear_cache()

        return DatasetFactoryResponse(
            status="success",
            message="Embedding cache cleared successfully",
            statistics=factory.get_statistics(),
        )

    except Exception as e:
        logger.error(f"Error clearing embedding cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def start_continuous_generation_task(background_tasks: BackgroundTasks):
    """
    Start the continuous dataset generation task.

    Args:
        background_tasks: FastAPI background tasks to add the task to
    """
    background_tasks.add_task(continuous_dataset_generation)
    logger.info("Continuous dataset generation task started")
