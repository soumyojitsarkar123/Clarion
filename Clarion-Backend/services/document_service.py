"""
Document Service - handles document ingestion and storage.
"""

import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

from models.document import Document, DocumentMetadata, DocumentStatus
from utils.file_handler import extract_text, get_file_type
from utils.config import settings, ensure_directories
from utils.sqlite import connect as sqlite_connect

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
        conn.close()
        logger.info("Document database initialized")
    
    def _save_to_db(self, document: Document) -> None:
        """Save document to SQLite database."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (id, filename, file_size, file_type, upload_time, page_count, 
             word_count, status, text_content, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            document.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def _load_from_db(self, document_id: str) -> Optional[Document]:
        """Load document from SQLite database."""
        conn = sqlite_connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Document(
            id=row["id"],
            metadata=DocumentMetadata(
                filename=row["filename"],
                file_size=row["file_size"],
                file_type=row["file_type"],
                upload_time=datetime.fromisoformat(row["upload_time"]),
                page_count=row["page_count"],
                word_count=row["word_count"]
            ),
            status=DocumentStatus(row["status"]),
            text_content=row["text_content"],
            error_message=row["error_message"]
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
            word_count = len(text_content.split()) if text_content else 0
            
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
        
        document = Document(
            id=doc_id,
            metadata=DocumentMetadata(
                filename=filename,
                file_size=file_size,
                file_type=file_type,
                upload_time=datetime.now(),
                page_count=page_count,
                word_count=word_count
            ),
            status=DocumentStatus.UPLOADED,
            text_content=text_content
        )
        
        self._save_to_db(document)
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
                    word_count=row["word_count"]
                ),
                status=DocumentStatus(row["status"]),
                text_content=row["text_content"],
                error_message=row["error_message"]
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
        
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        if deleted:
            logger.info(f"Document deleted: {document_id}")
        
        return deleted
