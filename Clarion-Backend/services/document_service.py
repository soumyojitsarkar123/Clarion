"""
Document Service - handles document ingestion and storage.
"""

import uuid
import sqlite3
import json
import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

from models.document import Document, DocumentMetadata, DocumentStatus
from utils.file_handler import extract_text, get_file_type
from utils.config import settings, ensure_directories
from utils.sqlite import connect as sqlite_connect
from services.provenance_migrations import apply_provenance_schema
from utils.text_metrics import count_words

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for managing document lifecycle.
    Handles upload, storage, and retrieval of documents.
    """
    
    def __init__(self):
        ensure_directories()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for document metadata."""
        db_path = settings.data_dir / "clarion.db"
        self.db_path = str(db_path)
        
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                page_count INTEGER,
                word_count INTEGER,
                status TEXT NOT NULL,
                text_content TEXT,
                error_message TEXT
            )
        """)
        conn.commit()
        apply_provenance_schema(self.db_path)
        
        conn.commit()
        conn.close()
        logger.info("Document database initialized")
    
    def _save_to_db(self, document: Document) -> None:
        """Save document to SQLite database."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (id, filename, file_size, file_type, upload_time, page_count, 
             word_count, status, text_content, error_message, source_provenance_json,
             source_file_path, source_file_sha256, extracted_page_count,
             text_bearing_page_count, raw_word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document.id,
            document.metadata.filename,
            document.metadata.file_size,
            document.metadata.file_type,
            document.metadata.upload_time.isoformat(),
            document.metadata.page_count,
            document.metadata.word_count,
            document.status,
            document.text_content,
            document.error_message,
            json.dumps(document.source_provenance, ensure_ascii=False),
            document.source_file_path,
            document.metadata.source_file_sha256,
            document.metadata.extracted_page_count,
            document.metadata.text_bearing_page_count,
            document.metadata.raw_word_count,
        ))

        cursor.execute("DELETE FROM document_source_pages WHERE document_id = ?", (document.id,))
        cursor.execute("DELETE FROM document_source_blocks WHERE document_id = ?", (document.id,))
        for page in document.source_pages:
            cursor.execute(
                """INSERT INTO document_source_pages
                   (document_id, page_number, raw_text, source_char_start, source_char_end,
                    extraction_method, extraction_version, warnings_json, location_available)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document.id, page.get("page_number"), page.get("raw_text"),
                 page.get("source_char_start"), page.get("source_char_end"),
                 page.get("extraction_method", "unknown"), page.get("extraction_version", "unknown"),
                 json.dumps(page.get("warnings", []), ensure_ascii=False),
                 int(bool(page.get("location_available", False)))),
            )
        for block in document.source_blocks:
            cursor.execute(
                """INSERT INTO document_source_blocks
                   (document_id, block_id, block_type, block_index, raw_text,
                    source_char_start, source_char_end, extraction_method,
                    extraction_version, warnings_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document.id, block.get("block_id"), block.get("block_type", "unknown"),
                 block.get("block_index", 0), block.get("raw_text"),
                 block.get("source_char_start"), block.get("source_char_end"),
                 document.source_provenance.get("extraction_method", "unknown"),
                 document.source_provenance.get("extraction_version", "unknown"),
                 json.dumps(block.get("warnings", []), ensure_ascii=False)),
            )
        
        conn.commit()
        conn.close()
    
    def _load_from_db(self, document_id: str) -> Optional[Document]:
        """Load document from SQLite database."""
        conn = sqlite_connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        source_provenance = {}
        try:
            source_provenance = json.loads(row["source_provenance_json"] or "{}")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            pass
        page_rows = conn.execute(
            "SELECT * FROM document_source_pages WHERE document_id=? ORDER BY page_number",
            (document_id,),
        ).fetchall()
        block_rows = conn.execute(
            "SELECT * FROM document_source_blocks WHERE document_id=? ORDER BY block_index",
            (document_id,),
        ).fetchall()
        conn.close()

        return Document(
            id=row["id"],
            metadata=DocumentMetadata(
                filename=row["filename"],
                file_size=row["file_size"],
                file_type=row["file_type"],
                upload_time=datetime.fromisoformat(row["upload_time"]),
                page_count=row["page_count"],
                word_count=row["word_count"],
                extracted_page_count=row["extracted_page_count"] if "extracted_page_count" in row.keys() else None,
                text_bearing_page_count=row["text_bearing_page_count"] if "text_bearing_page_count" in row.keys() else None,
                raw_word_count=row["raw_word_count"] if "raw_word_count" in row.keys() else None,
                source_file_sha256=row["source_file_sha256"] if "source_file_sha256" in row.keys() else None,
            ),
            status=DocumentStatus(row["status"]),
            text_content=row["text_content"],
            error_message=row["error_message"],
            source_provenance=source_provenance,
            source_pages=[{
                **dict(item),
                "warnings": json.loads(item["warnings_json"] or "[]"),
                "location_available": bool(item["location_available"]),
            } for item in page_rows],
            source_blocks=[{
                **dict(item), "warnings": json.loads(item["warnings_json"] or "[]")
            } for item in block_rows],
            source_file_path=row["source_file_path"] if "source_file_path" in row.keys() else None,
        )
    
    def upload_document(
        self,
        file_content: bytes,
        filename: str
    ) -> Document:
        """
        Upload and process a document.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
        
        Returns:
            Created Document object
        """
        doc_id = str(uuid.uuid4())
        file_size = len(file_content)
        file_type = get_file_type(filename)
        
        if file_type == "unknown":
            raise ValueError(f"Unsupported file type: {filename}")
        
        logger.info(f"Processing document: {filename} ({file_size} bytes)")
        
        text_content = None
        page_count = None
        word_count = None
        
        try:
            text_content, extracted_meta = extract_text(file_content, filename)
            page_count = extracted_meta.get("page_count")
            word_count = extracted_meta.get("normalized_word_count", count_words(text_content))
            
            if file_type == "pdf" and not page_count:
                page_count = extracted_meta.get("page_count")

            if word_count < 20:
                raise ValueError(
                    "No extractable text found in the document. "
                    "Please upload a text-based PDF/DOCX or run OCR first."
                )
                
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
        
        source_file_sha256 = hashlib.sha256(file_content).hexdigest()
        source_dir = settings.data_dir / "source_documents"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_suffix = ".docx" if file_type in ("docx", "doc") else ".pdf"
        source_path = source_dir / f"{doc_id}{source_suffix}"
        fd, temp_path = tempfile.mkstemp(prefix=f".{doc_id}.", suffix=".upload", dir=source_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(file_content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, source_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

        document = Document(
            id=doc_id,
            metadata=DocumentMetadata(
                filename=filename,
                file_size=file_size,
                file_type=file_type,
                upload_time=datetime.now(),
                page_count=page_count,
                word_count=word_count,
                extracted_page_count=extracted_meta.get("extracted_page_count"),
                text_bearing_page_count=extracted_meta.get("text_bearing_page_count"),
                raw_word_count=extracted_meta.get("raw_word_count", word_count),
                source_file_sha256=source_file_sha256,
            ),
            status=DocumentStatus.UPLOADED,
            text_content=text_content,
            source_provenance=extracted_meta.get("source_provenance", {}),
            source_pages=extracted_meta.get("source_pages", []),
            source_blocks=extracted_meta.get("source_blocks", []),
            source_file_path=str(source_path),
        )
        try:
            self._save_to_db(document)
        except Exception:
            source_path.unlink(missing_ok=True)
            raise
        logger.info(f"Document uploaded successfully: {doc_id}")
        
        return document
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Document object if found, None otherwise
        """
        return self._load_from_db(document_id)
    
    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> Document:
        """
        Update document status.
        
        Args:
            document_id: Document identifier
            status: New status
            error_message: Optional error message
        
        Returns:
            Updated Document object
        """
        document = self._load_from_db(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        document.status = status
        if error_message:
            document.error_message = error_message
        
        self._save_to_db(document)
        return document
    
    def update_document_content(
        self,
        document_id: str,
        text_content: str
    ) -> Document:
        """
        Update document text content.
        
        Args:
            document_id: Document identifier
            text_content: New text content
        
        Returns:
            Updated Document object
        """
        document = self._load_from_db(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        document.text_content = text_content
        document.source_provenance = {}
        document.source_pages = []
        document.source_blocks = []
        self._save_to_db(document)
        return document
    
    def list_documents(self) -> List[Document]:
        """
        List all documents.
        
        Returns:
            List of Document objects
        """
        conn = sqlite_connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents ORDER BY upload_time DESC")
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            documents.append(Document(
                id=row["id"],
                metadata=DocumentMetadata(
                    filename=row["filename"],
                    file_size=row["file_size"],
                    file_type=row["file_type"],
                    upload_time=datetime.fromisoformat(row["upload_time"]),
                    page_count=row["page_count"],
                    word_count=row["word_count"],
                    extracted_page_count=row["extracted_page_count"] if "extracted_page_count" in row.keys() else None,
                    text_bearing_page_count=row["text_bearing_page_count"] if "text_bearing_page_count" in row.keys() else None,
                    raw_word_count=row["raw_word_count"] if "raw_word_count" in row.keys() else None,
                    source_file_sha256=row["source_file_sha256"] if "source_file_sha256" in row.keys() else None,
                ),
                status=DocumentStatus(row["status"]),
                text_content=row["text_content"],
                error_message=row["error_message"]
                ,source_file_path=row["source_file_path"] if "source_file_path" in row.keys() else None
            ))
        
        return documents
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT source_file_path FROM documents WHERE id = ?", (document_id,))
        source_row = cursor.fetchone()
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()

        if deleted and source_row and source_row[0]:
            source_root = (settings.data_dir / "source_documents").resolve()
            source_path = Path(source_row[0]).resolve()
            try:
                source_path.relative_to(source_root)
                source_path.unlink(missing_ok=True)
            except (ValueError, OSError):
                logger.warning("Retained source file was not removed because it is outside the managed source directory")
        
        if deleted:
            logger.info(f"Document deleted: {document_id}")
        
        return deleted
