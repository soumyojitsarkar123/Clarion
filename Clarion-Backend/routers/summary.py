"""
Summary router - handles structured summary retrieval.
"""

import logging
from fastapi import APIRouter, HTTPException

from services.summary_service import SummaryService
from services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary", tags=["summary"])

summary_service = SummaryService()
document_service = DocumentService()


@router.get("/{document_id}")
async def get_summary(document_id: str):
    """
    Get the structured summary for a document.
    
    - **document_id**: ID of the document
    """
    try:
        document = document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        summary = summary_service.get_summary(document_id)
        
        if not summary:
            raise HTTPException(
                status_code=404, 
                detail="Summary not found. Please analyze the document first."
            )
        
        return {
            "document_id": summary.document_id,
            "title": summary.title,
            "overall_summary": summary.overall_summary,
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "level": s.level,
                    "summary": s.summary,
                    "key_points": s.key_points,
                    "related_concepts": s.related_concepts,
                    "child_sections": s.child_sections
                }
                for s in summary.sections
            ],
            "metadata": summary.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
