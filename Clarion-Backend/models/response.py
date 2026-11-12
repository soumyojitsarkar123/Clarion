"""
API response models.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")


class UploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str = Field(..., description="Generated document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Response message")


class AnalyzeResponse(BaseModel):
    """Response for document analysis."""
    document_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Analysis status")
    chunk_count: int = Field(..., description="Number of chunks created")
    concept_count: int = Field(..., description="Number of concepts extracted")
    message: str = Field(..., description="Response message")


class KnowledgeMapResponse(BaseModel):
    """Response for knowledge map retrieval."""
    document_id: str = Field(..., description="Document ID")
    extraction_method: str = Field("unknown", description="Concept extraction method")
    analysis_mode: str = Field("unknown", description="Analysis mode")
    main_topics: List[Dict[str, Any]] = Field(..., description="Main topics")
    subtopics: List[Dict[str, Any]] = Field(..., description="Subtopics")
    dependencies: List[Dict[str, str]] = Field(..., description="Concept dependencies")
    concepts: List[Dict[str, Any]] = Field(..., description="Extracted concepts")
    relations: List[Dict[str, Any]] = Field(..., description="Concept relations")


class QueryResponse(BaseModel):
    """Response for knowledge query."""
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Retrieval results")
    response: str = Field(..., description="Generated response")
    knowledge_map: Optional[Dict[str, Any]] = Field(None, description="Related knowledge map")
    fallback_used: bool = Field(False, description="Whether deterministic fallback analysis was used")
    omitted_evidence: List[Dict[str, str]] = Field(default_factory=list, description="Retrieved evidence not sent to the generator and the reason")
    token_metrics: Optional[Dict[str, Any]] = Field(None, description="Token budget and provider usage; estimates are labeled")
