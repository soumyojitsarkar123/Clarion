"""
Document models for storing document metadata and content.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status enum."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    ANALYZED = "analyzed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="MIME type of the file")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    page_count: Optional[int] = Field(None, description="Number of pages (for PDF)")
    word_count: Optional[int] = Field(None, description="Total word count")
    extracted_page_count: Optional[int] = Field(None, description="Physical pages for which text extraction completed")
    text_bearing_page_count: Optional[int] = Field(None, description="Pages with non-whitespace extracted text")
    raw_word_count: Optional[int] = Field(None, description="Whitespace-delimited units in raw extracted text")
    source_file_sha256: Optional[str] = Field(None, description="SHA-256 of the retained original upload")


class Document(BaseModel):
    """Main document model."""
    id: str = Field(..., description="Unique document identifier")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="Processing status")
    text_content: Optional[str] = Field(None, description="Extracted raw text content")
    source_provenance: Dict[str, Any] = Field(default_factory=dict, description="Versioned extracted-to-normalized source map")
    source_pages: List[Dict[str, Any]] = Field(default_factory=list, description="PDF page text and source intervals, when available")
    source_blocks: List[Dict[str, Any]] = Field(default_factory=list, description="Format-specific paragraph or table locations")
    source_file_path: Optional[str] = Field(None, description="Internal path to retained original upload")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    class Config:
        use_enum_values = True
