"""
Upload router - handles document upload.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from models.response import UploadResponse, ErrorResponse
from services.document_service import DocumentService
from utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

document_service = DocumentService()


@router.post("", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or DOCX) for processing.
    
    - **file**: PDF or DOCX file to upload
    """
    try:
        allowed_extensions = settings.allowed_extensions
        filename = file.filename or "unknown.txt"
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        if f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Supported: {', '.join(allowed_extensions)}"
            )
        
        max_size = settings.max_file_size_mb * 1024 * 1024
        contents = await file.read()
        
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        document = document_service.upload_document(contents, filename)
        
        logger.info(f"Document uploaded: {document.id}")
        
        return UploadResponse(
            document_id=document.id,
            filename=document.metadata.filename,
            status="uploaded",
            message="Document uploaded and processing started"
        )
        
    except ValueError as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/list")
async def list_documents():
    """List all uploaded documents."""
    try:
        documents = document_service.list_documents()
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.metadata.filename,
                    "filename": doc.metadata.filename,
                    "size": doc.metadata.file_size,
                    "status": doc.status,
                    "uploaded_at": doc.metadata.upload_time.isoformat(),
                    "word_count": doc.metadata.word_count
                }
                for doc in documents
            ]
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its associated data."""
    try:
        deleted = document_service.delete_document(document_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
