"""
Chunk models for document segmentation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkCreate(BaseModel):
    """Model for creating a new chunk."""
    document_id: str = Field(..., description="Parent document ID")
    section_title: Optional[str] = Field(None, description="Heading/section title if detected")
    content: str = Field(..., description="Chunk text content")
    position_index: int = Field(..., description="Position in original document")


class Chunk(BaseModel):
    """Complete chunk model with ID."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    section_title: Optional[str] = Field(None, description="Heading/section title if detected")
    content: str = Field(..., description="Chunk text content")
    position_index: int = Field(..., description="Position in original document")
    word_count: int = Field(..., description="Number of words in chunk")
    source_provenance: Dict[str, Any] = Field(default_factory=dict, description="Original source intervals and page references")
    raw_chunk_ids: List[str] = Field(default_factory=list, description="Raw chunks merged into this retained chunk")
