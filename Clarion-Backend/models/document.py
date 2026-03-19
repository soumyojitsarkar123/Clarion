"""
Document models for storing document metadata and content.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
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


class Document(BaseModel):
    """Main document model."""
    id: str = Field(..., description="Unique document identifier")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="Processing status")
    text_content: Optional[str] = Field(None, description="Extracted raw text content")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    class Config:
        use_enum_values = True
