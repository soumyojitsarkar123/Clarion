"""
Pydantic data models for Ground Truth relation validation records.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ValidityLabel(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    AMBIGUOUS = "AMBIGUOUS"


class ErrorCategory(str, Enum):
    NONE = "NONE"
    UNSUPPORTED_ASSERTION = "UNSUPPORTED_ASSERTION"
    DIRECTION_INVERSION = "DIRECTION_INVERSION"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    SPURIOUS_COOCCURRENCE = "SPURIOUS_COOCCURRENCE"
    NOISE_ENTITY = "NOISE_ENTITY"


class GroundTruthRecord(BaseModel):
    """
    A single ground-truth annotated relation record under closed-domain protocol.
    """
    record_id: str = Field(..., description="Unique record identifier")
    document_id: str = Field(..., description="Source document identifier")
    relation_id: str = Field(..., description="Original candidate relation identifier")
    concept_a: str = Field(..., min_length=1, description="Head concept name")
    concept_b: str = Field(..., min_length=1, description="Tail concept name")
    relation_type: str = Field(..., description="Semantic relation type (e.g. prerequisite, explanation)")
    label: ValidityLabel = Field(..., description="Ground-truth validity assessment")
    error_category: Optional[ErrorCategory] = Field(
        default=ErrorCategory.NONE,
        description="Specific error category if label is INVALID",
    )
    evidence_text: str = Field(..., description="Exact textual excerpt supporting the annotation")
    source_chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs containing the evidence")
    annotator_id: str = Field(default="adjudicator", description="Identifier of annotator or consensus committee")
    confidence_rating: int = Field(default=5, ge=1, le=5, description="Annotator confidence rating (1-5)")
    notes: Optional[str] = Field(default=None, description="Optional annotation notes")
    annotator_a_label: Optional[str] = Field(default=None, description="Initial label from Annotator A")
    annotator_b_label: Optional[str] = Field(default=None, description="Initial label from Annotator B if in overlap")
    adjudication_status: Optional[str] = Field(default=None, description="Status: SINGLE_ANNOTATED, DUAL_AGREED, or ADJUDICATED")
    raw_candidate: Optional[dict] = Field(default=None, description="Original candidate extraction metadata")

    @property
    def is_valid(self) -> bool:
        """Binary representation: True if VALID, False if INVALID or AMBIGUOUS."""
        return self.label == ValidityLabel.VALID

    @field_validator("error_category", mode="before")
    @classmethod
    def validate_error_category(cls, v, info):
        return v or ErrorCategory.NONE


class GroundTruthDataset(BaseModel):
    """
    A collection of ground-truth annotated relation records with metadata.
    """
    dataset_version: str = Field(default="1.0.0", description="Semantic dataset version")
    domain: str = Field(default="closed-domain-technical", description="Target domain description")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    description: str = Field(..., description="Dataset provenance and description")
    records: List[GroundTruthRecord] = Field(default_factory=list)
