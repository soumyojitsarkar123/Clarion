"""
Analyze router - handles document analysis with background processing.
"""

import logging
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel

from models.response import AnalyzeResponse
from models.document import DocumentStatus
from services.document_service import DocumentService
from services.background_service import BackgroundService, get_background_service
from services.processing_pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

document_service = DocumentService()
background_service = get_background_service()
pipeline = ProcessingPipeline()


class AnalyzeRequest(BaseModel):
    """Request model for document analysis."""

    generate_hierarchy: bool = True
    run_evaluation: bool = True
    force_reanalyze: bool = False
    wait_for_completion: bool = False  # If True, wait for pipeline to finish
    timeout_seconds: int = 300  # Only used if wait_for_completion=True


@router.post("/{document_id}")
async def analyze_document(document_id: str, request: Optional[AnalyzeRequest] = None):
    """
    Start document analysis pipeline (async background processing).

    Pipeline stages:
    1. Ingestion - Document validation
    2. Chunking - Structure-aware segmentation
    3. Embedding - Vector generation and FAISS indexing
    4. Mapping - Concept and relation extraction
    5. Graph Building - NetworkX graph construction
    6. Evaluation - Quality assessment
    7. Hierarchy - Topic trees and prerequisite chains

    - **document_id**: ID of the document to analyze
    - **request**: Analysis options

    Returns immediately with job_id. Use /status/{document_id} to track progress.
    """
    try:
        request = request or AnalyzeRequest()

        document = document_service.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check if already analyzed
        if document.status == DocumentStatus.ANALYZED and not request.force_reanalyze:
            logger.info(f"Document {document_id} already analyzed")
            return {
                "document_id": document_id,
                "status": "already_analyzed",
                "message": "Document has already been fully analyzed",
            }

        if not document.text_content:
            raise HTTPException(
                status_code=400, detail="Document has no text content to analyze"
            )

        skip_stages = []
        if not request.run_evaluation:
            skip_stages.append("evaluation")
        if not request.generate_hierarchy:
            skip_stages.append("hierarchy")

        if request.wait_for_completion:
            try:
                result = await asyncio.wait_for(
                    pipeline.execute(
                        document_id=document_id,
                        skip_existing=not request.force_reanalyze,
                        skip_stages=skip_stages,
                    ),
                    timeout=request.timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail=f"Analysis timed out after {request.timeout_seconds} seconds",
                )

            return {
                "document_id": document_id,
                "status": "completed" if result.get("success") else "failed",
                "message": "Analysis completed synchronously",
                "results": result,
            }

        # Submit job to background service
        job_id = await background_service.submit_job(
            document_id=document_id,
            process_func=pipeline.execute,
            skip_existing=not request.force_reanalyze,
            skip_stages=skip_stages,
        )

        logger.info(f"Submitted analysis job {job_id} for document {document_id}")

        return {
            "document_id": document_id,
            "job_id": job_id,
            "status": "processing",
            "message": "Analysis job submitted. Use /status/{document_id} to track progress.",
            "estimated_duration_seconds": 120,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis submission error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit analysis")
