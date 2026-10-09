"""
Tests for Ground Truth Dataset Validator and Protocol conformance.
"""

from pathlib import Path
import pytest

from core.benchmarking.dataset_validator import GroundTruthValidator, DatasetValidationError
from models.ground_truth import ValidityLabel, ErrorCategory


@pytest.fixture
def validator():
    return GroundTruthValidator()


@pytest.fixture
def fixture_path():
    return Path(__file__).resolve().parents[1] / "data" / "ground_truth" / "synthetic_benchmark_fixture.json"


def test_validate_synthetic_fixture(validator, fixture_path):
    """Verify that the synthetic benchmark fixture cleanly validates against the protocol."""
    dataset, stats = validator.validate_file(fixture_path)

    assert dataset.dataset_version == "1.0.0"
    assert stats["total_records"] == 20
    assert stats["distinct_documents"] == 5
    assert set(stats["document_ids"]) == {
        "doc_statistics", "doc_calculus", "doc_neural_nets", "doc_databases", "doc_algorithms"
    }

    # Verify presence of all classes
    assert stats["label_distribution"][ValidityLabel.VALID.value] >= 10
    assert stats["label_distribution"][ValidityLabel.INVALID.value] >= 5
    assert stats["label_distribution"][ValidityLabel.AMBIGUOUS.value] >= 1

    # Verify coverage of every single error taxonomy category
    errors = stats["error_distribution"]
    assert ErrorCategory.UNSUPPORTED_ASSERTION.value in errors
    assert ErrorCategory.DIRECTION_INVERSION.value in errors
    assert ErrorCategory.TYPE_MISMATCH.value in errors
    assert ErrorCategory.SPURIOUS_COOCCURRENCE.value in errors
    assert ErrorCategory.NOISE_ENTITY.value in errors


def test_validator_rejects_duplicate_ids(validator):
    """Verify that duplicate record IDs are rejected."""
    records = [
        {
            "record_id": "dup-1",
            "document_id": "doc-a",
            "relation_id": "r-1",
            "concept_a": "A",
            "concept_b": "B",
            "relation_type": "prerequisite",
            "label": "VALID",
            "evidence_text": "Evidence A to B.",
        },
        {
            "record_id": "dup-1",
            "document_id": "doc-b",
            "relation_id": "r-2",
            "concept_a": "C",
            "concept_b": "D",
            "relation_type": "prerequisite",
            "label": "VALID",
            "evidence_text": "Evidence C to D.",
        },
    ]
    with pytest.raises(DatasetValidationError, match="Duplicate record_id"):
        validator.validate_records(records)


def test_validator_rejects_invalid_without_error_category(validator):
    """Verify that INVALID records missing an error category violate protocol."""
    records = [
        {
            "record_id": "rec-1",
            "document_id": "doc-a",
            "relation_id": "r-1",
            "concept_a": "A",
            "concept_b": "B",
            "relation_type": "prerequisite",
            "label": "INVALID",
            "error_category": "NONE",
            "evidence_text": "Some text.",
        }
    ]
    with pytest.raises(DatasetValidationError, match="marked INVALID but has no specific error_category"):
        validator.validate_records(records)


def test_validator_rejects_empty_evidence(validator):
    """Verify that records with missing or whitespace-only evidence are rejected."""
    records = [
        {
            "record_id": "rec-1",
            "document_id": "doc-a",
            "relation_id": "r-1",
            "concept_a": "A",
            "concept_b": "B",
            "relation_type": "prerequisite",
            "label": "VALID",
            "evidence_text": "   ",
        }
    ]
    with pytest.raises(DatasetValidationError, match="empty evidence_text"):
        validator.validate_records(records)
