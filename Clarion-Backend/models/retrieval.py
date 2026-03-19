"""
Retrieval and query models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A single retrieval result."""
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    section_title: Optional[str] = Field(None, description="Section title")
    position_index: int = Field(..., description="Position in document")


class QueryRequest(BaseModel):
    """Request for knowledge querying the base."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    top_k: int = Field(default=5, ge=1, le=25, description="Number of results to return")
    include_knowledge_map: bool = Field(default=False, description="Include related knowledge map")
