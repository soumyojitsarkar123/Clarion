"""
Tests for gold dataset contract eligibility validation (Item 10).

Verifies:
- Records with all required fields pass validation
- Records missing any required field fail with an appropriate violation
- INVALID label without error_category is rejected
- AMBIGUOUS records are eligible but counted as non-trainable
- is_promotable is False iff any violations exist
"""

import pytest
from core.benchmarking.gold_dataset_contract import (
    GoldDatasetValidator,
    STATE_ADJUDICATED_GOLD,
    STATE_RAW_ANNOTATIONS,
)
from models.ground_truth import ErrorCategory, GroundTruthRecord, ValidityLabel


def _make_record(**overrides) -> GroundTruthRecord:
    """Build a minimal valid GroundTruthRecord with all required fields."""
    defaults = dict(
        record_id="rec-001",
        document_id="doc-A",
        relation_id="rel-1",
        concept_a="Alpha",
        concept_b="Beta",
        relation_type="explanation",
        label=ValidityLabel.VALID,
        error_category=ErrorCategory.NONE,
        evidence_text="Alpha is a component that explains Beta in the context of the system.",
        source_chunk_ids=["chunk-1"],
        annotator_id="AM1",
        confidence_rating=4,
        notes=None,
    )
    defaults.update(overrides)
    return GroundTruthRecord(**defaults)


class TestGoldDatasetValidatorEligible:
    """All required fields present → record is eligible."""

    def test_valid_record_passes_contract(self):
        validator = GoldDatasetValidator(annotation_version="1.0.0")
        records = [_make_record()]
        result = validator.validate_records(records)
        assert result.total_records == 1
        assert result.eligible_records == 1
        assert result.ineligible_records == 0
        assert len(result.violations) == 0
        assert result.is_promotable is True
        assert result.dataset_state == STATE_ADJUDICATED_GOLD

    def test_invalid_label_with_error_category_passes(self):
        validator = GoldDatasetValidator()
        records = [_make_record(
            label=ValidityLabel.INVALID,
            error_category=ErrorCategory.UNSUPPORTED_ASSERTION,
        )]
        result = validator.validate_records(records)
        assert result.eligible_records == 1
        assert len(result.violations) == 0

    def test_ambiguous_record_is_eligible_but_non_trainable(self):
        validator = GoldDatasetValidator()
        records = [_make_record(label=ValidityLabel.AMBIGUOUS)]
        result = validator.validate_records(records)
        assert result.eligible_records == 1
        assert result.non_trainable_records == 1
        assert result.is_promotable is True

    def test_multiple_valid_records(self):
        validator = GoldDatasetValidator()
        records = [_make_record(record_id=f"rec-{i:03}") for i in range(5)]
        result = validator.validate_records(records)
        assert result.eligible_records == 5
        assert result.ineligible_records == 0
        assert result.is_promotable is True


class TestGoldDatasetValidatorViolations:
    """Missing or invalid fields → record is ineligible.

    NOTE: GroundTruthRecord is a Pydantic model that rejects genuinely null required
    fields at construction time. To test that GoldDatasetValidator catches downstream
    violations (e.g. records constructed from raw dicts in validate_json_file, or
    records created with model_construct bypassing validation), we use
    GroundTruthRecord.model_construct() which bypasses Pydantic validation.
    """

    def test_null_label_fails(self):
        """Records with null label (from raw JSON paths) are caught by the contract."""
        validator = GoldDatasetValidator()
        # Use model_construct to simulate a record that bypassed Pydantic (e.g. from raw JSON)
        rec = GroundTruthRecord.model_construct(
            record_id="rec-bad-label",
            document_id="doc-A",
            relation_id="rel-1",
            concept_a="Alpha",
            concept_b="Beta",
            relation_type="explanation",
            label=None,
            error_category=ErrorCategory.NONE,
            evidence_text="Some evidence text here.",
            source_chunk_ids=["chunk-1"],
            annotator_id="AM1",
            confidence_rating=4,
        )
        result = validator.validate_records([rec])
        assert result.ineligible_records == 1
        assert any("label" in v.field for v in result.violations)

    def test_null_annotator_fails(self):
        """Records with null annotator_id are caught by the contract."""
        validator = GoldDatasetValidator()
        rec = GroundTruthRecord.model_construct(
            record_id="rec-bad-ann",
            document_id="doc-A",
            relation_id="rel-1",
            concept_a="Alpha",
            concept_b="Beta",
            relation_type="explanation",
            label=ValidityLabel.VALID,
            error_category=ErrorCategory.NONE,
            evidence_text="Some evidence text here.",
            source_chunk_ids=["chunk-1"],
            annotator_id=None,
            confidence_rating=4,
        )
        result = validator.validate_records([rec])
        assert result.ineligible_records == 1
        assert any("annotator_id" in v.field for v in result.violations)

    def test_placeholder_annotator_fails(self):
        """Records with placeholder annotator IDs are rejected by the contract."""
        validator = GoldDatasetValidator()
        for placeholder in ["unknown", "none", "placeholder", "null"]:
            rec = _make_record(annotator_id=placeholder)
            result = validator.validate_records([rec])
            assert result.ineligible_records == 1, f"Placeholder '{placeholder}' should fail"
            assert any("annotator_id" in v.field for v in result.violations)

    def test_empty_evidence_text_fails(self):
        """Records with empty evidence_text are rejected — this passes Pydantic but fails contract."""
        validator = GoldDatasetValidator()
        # Pydantic allows empty string for evidence_text; the contract requires non-empty
        rec = GroundTruthRecord.model_construct(
            record_id="rec-empty-ev",
            document_id="doc-A",
            relation_id="rel-1",
            concept_a="Alpha",
            concept_b="Beta",
            relation_type="explanation",
            label=ValidityLabel.VALID,
            error_category=ErrorCategory.NONE,
            evidence_text="",
            source_chunk_ids=["chunk-1"],
            annotator_id="AM1",
            confidence_rating=4,
        )
        result = validator.validate_records([rec])
        assert result.ineligible_records == 1
        assert any("evidence_text" in v.field for v in result.violations)

    def test_invalid_without_error_category_fails(self):
        """INVALID label with NONE error_category fails the contract — Pydantic allows it."""
        validator = GoldDatasetValidator()
        records = [_make_record(
            label=ValidityLabel.INVALID,
            error_category=ErrorCategory.NONE,
        )]
        result = validator.validate_records(records)
        assert result.ineligible_records == 1
        assert any("error_category" in v.field for v in result.violations)

    def test_confidence_rating_out_of_range_fails(self):
        """Confidence ratings outside [1,5] are caught by the contract via model_construct."""
        validator = GoldDatasetValidator()
        for bad_rating in [0, 6, -1, 100]:
            rec = GroundTruthRecord.model_construct(
                record_id=f"rec-conf-{bad_rating}",
                document_id="doc-A",
                relation_id="rel-1",
                concept_a="Alpha",
                concept_b="Beta",
                relation_type="explanation",
                label=ValidityLabel.VALID,
                error_category=ErrorCategory.NONE,
                evidence_text="Some evidence text.",
                source_chunk_ids=["chunk-1"],
                annotator_id="AM1",
                confidence_rating=bad_rating,
            )
            result = validator.validate_records([rec])
            assert result.ineligible_records == 1, (
                f"confidence_rating={bad_rating} should fail"
            )
            assert any("confidence_rating" in v.field for v in result.violations)

    def test_null_record_id_fails(self):
        """Records with null record_id are caught by the contract."""
        validator = GoldDatasetValidator()
        rec = GroundTruthRecord.model_construct(
            record_id=None,
            document_id="doc-A",
            relation_id="rel-1",
            concept_a="Alpha",
            concept_b="Beta",
            relation_type="explanation",
            label=ValidityLabel.VALID,
            error_category=ErrorCategory.NONE,
            evidence_text="Some evidence text.",
            source_chunk_ids=["chunk-1"],
            annotator_id="AM1",
            confidence_rating=4,
        )
        result = validator.validate_records([rec])
        assert result.ineligible_records == 1

    def test_empty_records_list(self):
        validator = GoldDatasetValidator()
        result = validator.validate_records([])
        assert result.is_promotable is False
        assert result.total_records == 0

    def test_mixed_eligible_ineligible(self):
        """Mix of valid and invalid records produces correct eligible/ineligible counts."""
        validator = GoldDatasetValidator()
        ok_rec = _make_record(record_id="ok-1")
        bad_label = GroundTruthRecord.model_construct(
            record_id="bad-1",
            document_id="doc-A",
            relation_id="rel-1",
            concept_a="Alpha",
            concept_b="Beta",
            relation_type="explanation",
            label=None,
            error_category=ErrorCategory.NONE,
            evidence_text="Some text.",
            source_chunk_ids=["chunk-1"],
            annotator_id="AM1",
            confidence_rating=4,
        )
        bad_evidence = GroundTruthRecord.model_construct(
            record_id="bad-2",
            document_id="doc-A",
            relation_id="rel-2",
            concept_a="Alpha",
            concept_b="Gamma",
            relation_type="explanation",
            label=ValidityLabel.VALID,
            error_category=ErrorCategory.NONE,
            evidence_text="",
            source_chunk_ids=["chunk-2"],
            annotator_id="AM1",
            confidence_rating=4,
        )
        records = [ok_rec, bad_label, bad_evidence]
        result = validator.validate_records(records)
        assert result.total_records == 3
        assert result.eligible_records == 1
        assert result.ineligible_records == 2
        assert result.is_promotable is False


class TestGoldContractResult:
    """Test GoldContractResult helper methods."""

    def test_raise_if_not_promotable_raises_on_violations(self):
        validator = GoldDatasetValidator()
        # Use model_construct to bypass Pydantic for a null-label record
        rec = GroundTruthRecord.model_construct(
            record_id="rec-bad",
            document_id="doc-A",
            relation_id="rel-1",
            concept_a="Alpha",
            concept_b="Beta",
            relation_type="explanation",
            label=None,  # invalid — contract catches this
            error_category=ErrorCategory.NONE,
            evidence_text="Some evidence.",
            source_chunk_ids=["chunk-1"],
            annotator_id="AM1",
            confidence_rating=4,
        )
        result = validator.validate_records([rec])
        with pytest.raises(ValueError, match="Gold dataset contract FAILED"):
            result.raise_if_not_promotable()

    def test_raise_if_not_promotable_passes_on_clean_dataset(self):
        validator = GoldDatasetValidator()
        records = [_make_record()]
        result = validator.validate_records(records)
        result.raise_if_not_promotable()  # should not raise

    def test_to_dict_contains_expected_keys(self):
        validator = GoldDatasetValidator()
        records = [_make_record()]
        result = validator.validate_records(records)
        d = result.to_dict()
        assert "dataset_state" in d
        assert "eligible_records" in d
        assert "ineligible_records" in d
        assert "is_promotable" in d
        assert "sha256" in d
        assert "violation_count" in d
