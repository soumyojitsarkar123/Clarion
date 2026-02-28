"""
Dataset Router - API endpoints for relation dataset export and management.
"""

import logging
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.relation_dataset_service import RelationDatasetService, RelationDatasetRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dataset", tags=["dataset"])

dataset_service = RelationDatasetService()


class ValidationUpdateRequest(BaseModel):
    """Request model for updating validation status."""
    record_id: str
    is_valid: bool


class DatasetStatsResponse(BaseModel):
    """Response model for dataset statistics."""
    total_records: int
    labeled_records: int
    unlabeled_records: int
    relation_types: dict
    average_llm_confidence: float
    average_cooccurrence: float


@router.get("/relations/export")
async def export_relations_dataset(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    format: str = Query("json", description="Export format: json or csv")
):
    """
    Export the relation dataset for training.
    
    Returns all relation records captured during the mapping stage,
    including:
    - concept_a, concept_b: The related concepts
    - relation_type: Type of relationship
    - llm_confidence: LLM-assigned confidence score
    - cooccurrence_score: Chunk proximity-based score
    - semantic_similarity: Embedding similarity (if available)
    - chunk_context: Source text context
    - is_valid: Validation label (nullable for labeling)
    
    Args:
        document_id: Optional filter by specific document
        format: Output format (json or csv)
    
    Returns:
        Dataset in requested format
    """
    try:
        result = dataset_service.export_dataset(
            format=format,
            document_id=document_id
        )
        
        if format == "csv":
            from fastapi.responses import Response
            return Response(
                content=result["data"],
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=relation_dataset.csv"}
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error exporting dataset: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/relations")
async def get_relations(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    relation_type: Optional[str] = Query(None, description="Filter by relation type"),
    is_valid: Optional[bool] = Query(None, description="Filter by validation status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return")
):
    """
    Get relation dataset records.
    
    Args:
        document_id: Filter by document
        relation_type: Filter by relation type
        is_valid: Filter by validation status (True/False for labeled, None for all)
        limit: Maximum records
    
    Returns:
        List of relation records
    """
    try:
        records = dataset_service.get_dataset(
            document_id=document_id,
            relation_type=relation_type,
            is_valid=is_valid,
            limit=limit
        )
        
        return {
            "count": len(records),
            "records": [
                {
                    "record_id": r.record_id,
                    "document_id": r.document_id,
                    "relation_id": r.relation_id,
                    "concept_a": r.concept_a,
                    "concept_b": r.concept_b,
                    "relation_type": r.relation_type,
                    "llm_confidence": r.llm_confidence,
                    "cooccurrence_score": r.cooccurrence_score,
                    "semantic_similarity": r.semantic_similarity,
                    "chunk_context": r.chunk_context[:300],
                    "is_valid": r.is_valid,
                    "created_at": r.created_at.isoformat()
                }
                for r in records
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting relations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/relations/stats")
async def get_dataset_stats():
    """
    Get dataset statistics.
    
    Returns:
        Dataset statistics including counts and averages
    """
    try:
        stats = dataset_service.get_dataset_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/relations/validate")
async def update_validation_status(request: ValidationUpdateRequest):
    """
    Update validation status for a relation record.
    
    Use this to label relations for training/evaluation.
    
    Args:
        record_id: The record ID to update
        is_valid: Validation label (True/False)
    
    Returns:
        Success status
    """
    try:
        success = dataset_service.update_validation(
            record_id=request.record_id,
            is_valid=request.is_valid
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Record not found")
        
        return {"success": True, "record_id": request.record_id, "is_valid": request.is_valid}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating validation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/relations/{document_id}")
async def delete_document_records(document_id: str):
    """
    Delete all dataset records for a document.
    
    Args:
        document_id: Document ID
    
    Returns:
        Number of records deleted
    """
    try:
        deleted = dataset_service.delete_document_records(document_id)
        return {"deleted": deleted, "document_id": document_id}
        
    except Exception as e:
        logger.error(f"Error deleting records: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
