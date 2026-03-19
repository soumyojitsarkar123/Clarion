"""
Structured summary models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SummarySection(BaseModel):
    """A section in the structured summary."""
    id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    level: int = Field(..., description="Hierarchy level (1=main, 2+=sub)")
    summary: str = Field(..., description="Section summary")
    key_points: List[str] = Field(default_factory=list, description="Key points")
    related_concepts: List[str] = Field(default_factory=list, description="Related concept names")
    child_sections: List[str] = Field(default_factory=list, description="Child section IDs")


class StructuredSummary(BaseModel):
    """Structured hierarchical summary of a document."""
    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    overall_summary: str = Field(..., description="Overall document summary")
    sections: List[SummarySection] = Field(default_factory=list, description="Hierarchical sections")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
