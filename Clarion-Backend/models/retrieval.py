"""
Retrieval and query models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A single retrieval result."""
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    section_title: Optional[str] = Field(None, description="Section title")
    position_index: int = Field(..., description="Position in document")
    document_id: Optional[str] = Field(None, description="Source document ID")
    document_name: Optional[str] = Field(None, description="Source document name")
    page_references: List[int] = Field(default_factory=list, description="Verified one-based source pages")
    source_intervals: List[Dict[str, Any]] = Field(default_factory=list, description="Original extracted-text intervals where verifiable")
    segment_id: Optional[str] = Field(None, description="Embedding segment that matched FAISS")
    raw_chunk_ids: List[str] = Field(default_factory=list, description="Raw source chunks represented by this passage")
    embedding_model_id: Optional[str] = Field(None, description="Embedding model/index compatibility identifier")


class QueryRequest(BaseModel):
    """Request for knowledge querying the base."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    top_k: int = Field(default=5, ge=1, le=25, description="Number of results to return")
    include_knowledge_map: bool = Field(default=False, description="Include related knowledge map")
