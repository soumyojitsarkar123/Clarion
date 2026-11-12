"""
Upload router - handles document upload.
"""

import logging
import zipfile
import PyPDF2.errors
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pathlib import Path
import hashlib

from models.response import UploadResponse, ErrorResponse
from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from utils.sqlite import connect as sqlite_connect
from utils.config import settings

logger = logging.getLogger(__name__)
UPLOAD_READ_CHUNK_BYTES = 64 * 1024

router = APIRouter(prefix="/upload", tags=["upload"])

document_service = DocumentService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()


async def _read_upload_limited(file: UploadFile, max_size: int) -> bytes:
    """Read at most max_size + 1 bytes so oversized uploads are rejected early."""
    contents = bytearray()
    while len(contents) <= max_size:
        remaining = max_size + 1 - len(contents)
        chunk = await file.read(min(UPLOAD_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > max_size:
            break
    return bytes(contents)


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
        contents = await _read_upload_limited(file, max_size)
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty (0 bytes). Please upload a valid document."
            )
        
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
            message="Document uploaded successfully. Analysis has not started yet."
        )
        
    except (PyPDF2.errors.EmptyFileError, PyPDF2.errors.PdfReadError, zipfile.BadZipFile, KeyError) as e:
        logger.warning(f"Corrupt or empty document upload error: {str(e)}")
        raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")
    except ValueError as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
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
                    "word_count": doc.metadata.word_count,
                    "page_count": getattr(doc.metadata, "page_count", None),
                    "raw_word_count": doc.metadata.raw_word_count,
                    "extracted_page_count": doc.metadata.extracted_page_count,
                    "text_bearing_page_count": doc.metadata.text_bearing_page_count,
                    "source_file_sha256": doc.metadata.source_file_sha256,
                }
                for doc in documents
            ]
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{document_id}/metrics")
async def get_document_metrics(document_id: str):
    """Return stored document, chunk, embedding-map, and source-page counts."""
    document = document_service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = chunking_service.get_chunks(document_id)
    raw_chunk_ids = {
        raw_id for chunk in chunks
        for raw_id in (chunk.raw_chunk_ids or [chunk.chunk_id])
    }
    indexed_pages = sorted({
        int(page)
        for chunk in chunks
        for page in (chunk.source_provenance or {}).get("page_references", [])
        if str(page).isdigit()
    })

    conn = sqlite_connect(str(settings.data_dir / "clarion.db"))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    segments = None
    mapped_vectors = None
    if "embedding_segments_v2" in tables:
        segments = conn.execute(
            "SELECT COUNT(*) FROM embedding_segments_v2 WHERE document_id=?", (document_id,)
        ).fetchone()[0]
    elif "embedding_segments" in tables:
        segments = conn.execute(
            "SELECT COUNT(*) FROM embedding_segments WHERE document_id=?", (document_id,)
        ).fetchone()[0]
    if "embedding_vector_map" in tables:
        mapped_vectors = conn.execute(
            "SELECT COUNT(*) FROM embedding_vector_map WHERE document_id=?", (document_id,)
        ).fetchone()[0]
    elif "embeddings" in tables:
        mapped_vectors = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE document_id=?", (document_id,)
        ).fetchone()[0]
    conn.close()

    vector_store = embedding_service.get_vector_store(document_id, load_model=False)
    faiss_vectors = int(vector_store.index.ntotal) if vector_store and vector_store.index is not None else None
    successful_pages = document.metadata.extracted_page_count
    text_pages = document.metadata.text_bearing_page_count
    return {
        "document_id": document_id,
        "status": str(document.status),
        "counts": {
            "physical_pages": document.metadata.page_count,
            "extracted_pages": successful_pages,
            "text_bearing_pages": text_pages,
            "empty_text_pages": max(0, successful_pages - text_pages) if successful_pages is not None and text_pages is not None else None,
            "indexed_pages": len(indexed_pages) if document.source_pages else None,
            "indexed_page_numbers": indexed_pages if document.source_pages else None,
            "raw_words": document.metadata.raw_word_count,
            "normalized_words": document.metadata.word_count,
            "candidate_raw_chunks": len(raw_chunk_ids) if chunks else None,
            "retained_chunks": len(chunks) if chunks else None,
            "rejected_chunks": None,
            "embedding_segments": segments,
            "mapped_vectors": mapped_vectors,
            "faiss_vectors": faiss_vectors,
        },
        "word_count_definition": "Whitespace-delimited units after extraction normalization; attached punctuation, contractions, hyphenated terms, numbers, and Unicode text each remain one unit. This is not a tokenizer count.",
        "source_file_sha256": document.metadata.source_file_sha256,
    }


@router.get("/{document_id}/source")
async def get_source_document(document_id: str):
    """Serve the exact retained original upload for citation navigation."""
    document = document_service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document.source_file_path or not document.metadata.source_file_sha256:
        raise HTTPException(status_code=404, detail="Original source file is not available for this legacy document")

    source_root = (settings.data_dir / "source_documents").resolve()
    source_path = Path(document.source_file_path).resolve()
    try:
        source_path.relative_to(source_root)
    except ValueError:
        raise HTTPException(status_code=409, detail="Stored source path is outside the managed source directory")
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Original source file is missing")

    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != document.metadata.source_file_sha256:
        raise HTTPException(status_code=409, detail="Original source file checksum does not match its upload record")
    return FileResponse(source_path, filename=document.metadata.filename)


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
