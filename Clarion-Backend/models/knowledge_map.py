"""
Knowledge map models for concept hierarchy and relationships.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Types of semantic relationships between concepts."""
    PREREQUISITE = "prerequisite"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    CAUSE_EFFECT = "cause-effect"
    EXAMPLE_OF = "example-of"
    SIMILAR_TO = "similar-to"
    PART_OF = "part-of"
    DERIVES_FROM = "derives-from"


class Concept(BaseModel):
    """A key concept extracted from document."""
    id: str = Field(..., description="Unique concept identifier")
    name: str = Field(..., description="Concept name")
    definition: Optional[str] = Field(None, description="Concept definition if found")
    context: Optional[str] = Field(None, description="Surrounding context")
    chunk_ids: List[str] = Field(default_factory=list, description="Source chunk IDs")


class Relation(BaseModel):
    """A relationship between two concepts."""
    id: str = Field(..., description="Unique relation identifier")
    from_concept: str = Field(..., description="Source concept name")
    to_concept: str = Field(..., description="Target concept name")
    relation_type: RelationType = Field(..., description="Type of relationship")
    description: Optional[str] = Field(None, description="Relation description")
    confidence: float = Field(default=1.0, description="Confidence score 0-1")


class MainTopic(BaseModel):
    """A main topic in the document hierarchy."""
    id: str = Field(..., description="Topic identifier")
    title: str = Field(..., description="Topic title")
    description: Optional[str] = Field(None, description="Topic summary")
    concept_ids: List[str] = Field(default_factory=list, description="Associated concept IDs")
    subtopic_ids: List[str] = Field(default_factory=list, description="Subtopic IDs")


class Subtopic(BaseModel):
    """A subtopic under a main topic."""
    id: str = Field(..., description="Subtopic identifier")
    title: str = Field(..., description="Subtopic title")
    description: Optional[str] = Field(None, description="Subtopic summary")
    parent_topic_id: str = Field(..., description="Parent main topic ID")
    concept_ids: List[str] = Field(default_factory=list, description="Associated concept IDs")


class KnowledgeMap(BaseModel):
    """Complete knowledge map for a document."""
    document_id: str = Field(..., description="Document identifier")
    main_topics: List[MainTopic] = Field(default_factory=list, description="Main topics")
    subtopics: List[Subtopic] = Field(default_factory=list, description="Subtopics")
    concepts: List[Concept] = Field(default_factory=list, description="Extracted concepts")
    relations: List[Relation] = Field(default_factory=list, description="Concept relationships")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def get_dependencies(self) -> List[Dict[str, str]]:
        """Get dependencies in simplified format."""
        return [
            {
                "from": r.from_concept,
                "to": r.to_concept,
                "relation": r.relation_type.value
            }
            for r in self.relations
        ]
