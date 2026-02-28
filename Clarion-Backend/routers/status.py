"""
Status Router - API endpoints for job status and progress tracking.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field
from datetime import datetime

from services.background_service import BackgroundService, JobRepository, JobStatus
from services.document_service import DocumentService
from services.processing_pipeline import ProcessingPipeline
from services.knowledge_map_service import KnowledgeMapService
from utils.config import settings
from utils.graph_store import graph_json_path, graph_pickle_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["status"])

# Initialize services
background_service = BackgroundService()
document_service = DocumentService()
pipeline = ProcessingPipeline()
knowledge_map_service = KnowledgeMapService()


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    document_id: str
    status: str
    current_stage: str
    progress_percent: int
    
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    retry_count: int
    max_retries: int
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True


class DocumentStatusResponse(BaseModel):
    """Response model for document processing status."""
    document_id: str
    document_status: str
    
    has_active_job: bool
    current_job: Optional[JobStatusResponse] = None
    
    processing_history: list
    artifacts_ready: dict
    
    estimated_completion: Optional[str] = None


class JobListResponse(BaseModel):
    """Response model for listing jobs."""
    document_id: str
    total_jobs: int
    jobs: list


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str = Path(..., description="Document ID")):
    """
    Get current processing status for a document.
    
    Returns:
        - Document status
        - Active job information (if processing)
        - Processing history
        - Which artifacts are ready
        - Estimated completion time
    """
    try:
        # Get document
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get latest job
        latest_job = background_service.get_latest_job(document_id)
        
        # Build processing history
        all_jobs = background_service.get_document_jobs(document_id)
        history = []
        
        for job in all_jobs[:10]:  # Last 10 jobs
            history.append({
                "job_id": job.job_id,
                "status": job.status,
                "stage": job.current_stage,
                "progress": job.progress_percent,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            })
        
        # Determine which artifacts are ready
        artifacts = {
            "document": True,  # Always have document after upload
            "chunks": len(all_jobs) > 0 and any(
                j.status in [JobStatus.COMPLETED, JobStatus.CHUNKING, JobStatus.EMBEDDING, 
                           JobStatus.MAPPING, JobStatus.GRAPH_BUILDING, JobStatus.EVALUATING, 
                           JobStatus.HIERARCHY] 
                for j in all_jobs
            ),
            "embeddings": (settings.vectorstore_dir / f"{document_id}.index").exists(),
            "knowledge_map": knowledge_map_service.get_knowledge_map(document_id) is not None,
            "graph": graph_json_path(document_id).exists() or (
                settings.allow_legacy_pickle_loading and graph_pickle_path(document_id).exists()
            ),
            "evaluation": latest_job and latest_job.status == JobStatus.COMPLETED,
            "hierarchy": latest_job and latest_job.status == JobStatus.COMPLETED
        }
        
        # Build response
        response = DocumentStatusResponse(
            document_id=document_id,
            document_status=document.status,
            has_active_job=latest_job is not None and latest_job.status not in [
                JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED
            ],
            processing_history=history,
            artifacts_ready=artifacts
        )
        
        # Add current job details if active
        if latest_job and latest_job.status not in [
            JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED
        ]:
            response.current_job = JobStatusResponse(
                job_id=latest_job.job_id,
                document_id=latest_job.document_id,
                status=latest_job.status,
                current_stage=latest_job.current_stage,
                progress_percent=latest_job.progress_percent,
                created_at=latest_job.created_at.isoformat(),
                started_at=latest_job.started_at.isoformat() if latest_job.started_at else None,
                completed_at=latest_job.completed_at.isoformat() if latest_job.completed_at else None,
                retry_count=latest_job.retry_count,
                max_retries=latest_job.max_retries,
                error_message=latest_job.error_message
            )
            
            # Estimate completion (very rough)
            if latest_job.started_at and latest_job.progress_percent > 0:
                elapsed = (datetime.now() - latest_job.started_at).total_seconds()
                rate = elapsed / latest_job.progress_percent
                remaining = rate * (100 - latest_job.progress_percent)
                response.estimated_completion = str(datetime.now().timestamp() + remaining)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str = Path(..., description="Job ID")):
    """
    Get detailed status for a specific job.
    
    Returns complete job information including:
    - Current status and stage
    - Progress percentage
    - Retry count
    - Error messages (if failed)
    """
    try:
        job = background_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job.job_id,
            document_id=job.document_id,
            status=job.status,
            current_stage=job.current_stage,
            progress_percent=job.progress_percent,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            retry_count=job.retry_count,
            max_retries=job.max_retries,
            error_message=job.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/document/{document_id}/jobs", response_model=JobListResponse)
async def get_document_jobs(document_id: str = Path(..., description="Document ID")):
    """
    Get all processing jobs for a document.
    
    Returns complete history of all processing attempts.
    """
    try:
        # Verify document exists
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        jobs = background_service.get_document_jobs(document_id)
        
        return JobListResponse(
            document_id=document_id,
            total_jobs=len(jobs),
            jobs=[
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "current_stage": job.current_stage,
                    "progress_percent": job.progress_percent,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_message": job.error_message,
                    "retry_count": job.retry_count
                }
                for job in jobs
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str = Path(..., description="Job ID")):
    """
    Cancel a pending or processing job.
    
    Note: Can only cancel jobs that haven't completed yet.
    """
    try:
        success = await background_service.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="Job not found or already in terminal state"
            )
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs/active")
async def get_active_jobs():
    """
    Get all currently active jobs (system-wide).
    
    Admin endpoint for monitoring processing queue.
    """
    try:
        jobs = background_service.list_active_jobs()
        
        return {
            "count": len(jobs),
            "max_concurrent": background_service.max_concurrent,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "document_id": job.document_id,
                    "status": job.status,
                    "current_stage": job.current_stage,
                    "progress_percent": job.progress_percent,
                    "started_at": job.started_at.isoformat() if job.started_at else None
                }
                for job in jobs
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting active jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/queue/stats")
async def get_queue_stats():
    """
    Get queue statistics.
    
    Returns information about job queue and concurrency limits.
    """
    try:
        return background_service.get_queue_stats()
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/document/{document_id}/recoverable")
async def check_recoverable(document_id: str = Path(..., description="Document ID")):
    """
    Check if a failed document processing can be recovered.
    
    Returns information about available artifacts and recoverable stages.
    """
    try:
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        recovery_info = background_service.check_recoverable(document_id)
        
        return {
            "document_id": document_id,
            "recoverable": recovery_info["recoverable"],
            "reason": recovery_info.get("reason"),
            "current_stage": recovery_info.get("current_stage"),
            "available_artifacts": recovery_info.get("artifacts", {}),
            "completed_stages": recovery_info.get("available_stages", []),
            "job_status": recovery_info.get("job_status"),
            "error_message": recovery_info.get("error_message")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking recoverability: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/document/{document_id}/recover")
async def recover_document(
    document_id: str = Path(..., description="Document ID"),
    skip_completed: bool = True
):
    """
    Attempt to recover a failed document processing job.
    
    Uses existing artifacts to skip completed stages.
    """
    try:
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        recovery_info = background_service.check_recoverable(document_id)
        
        if not recovery_info["recoverable"]:
            return {
                "document_id": document_id,
                "recovered": False,
                "message": recovery_info.get("reason", "Not recoverable")
            }
        
        job_id = await background_service.recover_job(
            document_id=document_id,
            process_func=pipeline.execute,
            skip_completed_stages=skip_completed
        )
        
        if job_id:
            return {
                "document_id": document_id,
                "recovered": True,
                "job_id": job_id,
                "message": "Recovery job started",
                "available_stages": recovery_info.get("available_stages", [])
            }
        else:
            return {
                "document_id": document_id,
                "recovered": False,
                "message": "Failed to start recovery"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recovering document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
