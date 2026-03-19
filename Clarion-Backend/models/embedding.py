"""
Embedding record models.
"""

from typing import List
from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    """Record linking chunk to embedding vector."""
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    vector_index: int = Field(..., description="Index in FAISS index")
    embedding: List[float] = Field(..., description="Embedding vector")
