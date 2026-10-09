"""
Unit tests for Inter-Annotator Agreement Engine and Annotation Tooling.
"""

import json
from pathlib import Path
import pytest

from core.benchmarking.agreement import (
    compute_cohens_kappa,
    compute_fleiss_kappa,
    extract_disagreements_for_adjudication,
    interpret_kappa,
)
from core.benchmarking.annotation_tool import (
    export_annotation_batch,
    validate_annotation_batch,
    import_annotation_batch_to_ground_truth,
)
from models.ground_truth import ValidityLabel, ErrorCategory


def test_cohens_kappa_perfect_agreement():
    """Verify kappa is 1.0 when raters agree on every item."""
    r1 = ["VALID", "INVALID", "VALID", "AMBIGUOUS"]
    r2 = ["VALID", "INVALID", "VALID", "AMBIGUOUS"]
    ids = ["rec-1", "rec-2", "rec-3", "rec-4"]

    res = compute_cohens_kappa(r1, r2, item_ids=ids)
    assert res.observed_agreement == 1.0
    assert res.kappa == 1.0
    assert "Almost perfect" in res.interpretation
    assert len(res.disagreement_indices) == 0


def test_cohens_kappa_disagreement():
    """Verify kappa correctly drops and isolates disagreements."""
    r1 = ["VALID", "VALID", "INVALID", "INVALID"]
    r2 = ["VALID", "INVALID", "VALID", "INVALID"]
    ids = ["rec-1", "rec-2", "rec-3", "rec-4"]

    res = compute_cohens_kappa(r1, r2, item_ids=ids)
    assert res.observed_agreement == 0.50
    assert res.kappa == 0.0  # By chance agreement when marginals are symmetric 0.5/0.5
    assert set(res.disagreement_indices) == {"rec-2", "rec-3"}


def test_fleiss_kappa_multiple_raters():
    """Verify Fleiss' Kappa for 3 raters across items."""
    # 4 items, 3 raters
    matrix = [
        ["VALID", "VALID", "VALID"],      # perfect agreement
        ["INVALID", "INVALID", "INVALID"],# perfect agreement
        ["VALID", "VALID", "INVALID"],    # 2 vs 1
        ["AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS"], # perfect
    ]
    ids = ["item-1", "item-2", "item-3", "item-4"]
    res = compute_fleiss_kappa(matrix, item_ids=ids)

    assert res.num_raters == 3
    assert res.num_items == 4
    assert res.observed_agreement > 0.70
    assert res.kappa > 0.60
    assert res.disagreement_indices == ["item-3"]


def test_extract_disagreements_for_adjudication():
    """Verify divergent annotations are identified for committee review."""
    batch_a = {
        "records": [
            {"record_id": "r-1", "concept_a": "A", "concept_b": "B", "relation_type": "rel1", "label": "VALID"},
            {"record_id": "r-2", "concept_a": "C", "concept_b": "D", "relation_type": "rel2", "label": "VALID"},
        ]
    }
    batch_b = {
        "records": [
            {"record_id": "r-1", "concept_a": "A", "concept_b": "B", "relation_type": "rel1", "label": "VALID"},
            {"record_id": "r-2", "concept_a": "C", "concept_b": "D", "relation_type": "rel2", "label": "INVALID"},
        ]
    }

    disagreements = extract_disagreements_for_adjudication({"rater_1": batch_a, "rater_2": batch_b})
    assert len(disagreements) == 1
    assert disagreements[0]["record_id"] == "r-2"
    assert disagreements[0]["assigned_labels"] == {"rater_1": "VALID", "rater_2": "INVALID"}


def test_annotation_export_and_validation(tmp_path):
    """Verify export from relation_dataset.db and validation checks."""
    db_path = "Clarion-Backend/data/relation_dataset.db"
    if not Path(db_path).exists():
        pytest.skip("relation_dataset.db not found")

    out_file = tmp_path / "test_annotation_queue.json"
    summary = export_annotation_batch(
        db_path=db_path,
        output_path=str(out_file),
        sample_size=10,
        random_seed=42,
    )

    assert summary["total_records"] == 10
    assert out_file.exists()

    # Initial export should have 0 completed annotations
    val_initial = validate_annotation_batch(str(out_file))
    assert val_initial["is_complete"] is False
    assert val_initial["completed_records"] == 0

    # Fill in annotations for testing
    with open(out_file, "r") as f:
        data = json.load(f)

    for i, r in enumerate(data["records"]):
        if i % 2 == 0:
            r["label"] = "VALID"
        else:
            r["label"] = "INVALID"
            r["error_category"] = "UNSUPPORTED_ASSERTION"
        r["annotator_id"] = "test_annotator"
        r["confidence_rating"] = 5

    with open(out_file, "w") as f:
        json.dump(data, f)

    val_completed = validate_annotation_batch(str(out_file))
    assert val_completed["is_valid"] is True
    assert val_completed["is_complete"] is True
    assert val_completed["completed_records"] == 10

    # Test conversion to GroundTruthDataset
    gold_out = tmp_path / "test_gold.json"
    ds = import_annotation_batch_to_ground_truth(
        input_path=str(out_file),
        output_gold_path=str(gold_out),
        db_path=None,  # don't mutate DB in unit test
    )
    assert len(ds.records) == 10
    assert ds.records[0].label in (ValidityLabel.VALID, ValidityLabel.INVALID)


def test_export_overlap_batch(tmp_path):
    """Verify export of independent overlap batch for Annotator B."""
    primary_file = tmp_path / "primary_batch.json"
    records = [
        {
            "record_id": f"rec-{i}",
            "document_id": f"doc-{i % 3}",
            "relation_id": f"rel-{i}",
            "concept_a": f"ConceptA_{i}",
            "concept_b": f"ConceptB_{i}",
            "relation_type": "explanation",
            "chunk_context": f"Evidence text for {i}",
            "source_chunk_ids": [f"c-{i}"],
        }
        for i in range(12)
    ]
    with open(primary_file, "w") as f:
        json.dump({"manifest_version": "1.0.0", "records": records}, f)

    from core.benchmarking.annotation_tool import export_overlap_batch
    overlap_file = tmp_path / "overlap_b.json"
    summary = export_overlap_batch(
        primary_batch_path=str(primary_file),
        output_path=str(overlap_file),
        overlap_size=6,
        random_seed=42,
        annotator_b_id="AnnB",
    )

    assert summary["total_records"] == 6
    assert overlap_file.exists()

    with open(overlap_file, "r") as f:
        ovl_data = json.load(f)

    assert len(ovl_data["records"]) == 6
    # Verify records are fresh / unannotated for independent judgment
    for r in ovl_data["records"]:
        assert r["label"] is None
        assert r["annotator_id"] == "AnnB"
        assert r["confidence_rating"] is None


def test_compute_agreement_pending_when_unannotated(tmp_path):
    """Verify system reports HUMAN ANNOTATION PENDING when labels are null."""
    file_a = tmp_path / "file_a.json"
    file_b = tmp_path / "file_b.json"
    records = [
        {"record_id": f"rec-{i}", "label": None}
        for i in range(5)
    ]
    with open(file_a, "w") as f:
        json.dump({"records": records}, f)
    with open(file_b, "w") as f:
        json.dump({"records": records}, f)

    from core.benchmarking.annotation_tool import compute_agreement_between_files
    report = compute_agreement_between_files(str(file_a), str(file_b))
    assert report["status"] == "HUMAN ANNOTATION PENDING — AGREEMENT NOT COMPUTED"
    assert report["completed_in_a"] == 0
    assert report["completed_in_b"] == 0


def test_compute_agreement_and_adjudication_flow(tmp_path):
    """Verify agreement calculation, disagreement queue creation, and gold assembly."""
    from core.benchmarking.annotation_tool import (
        compute_agreement_between_files,
        create_adjudication_queue,
        assemble_gold_dataset_v1,
    )

    file_a = tmp_path / "annotator_a.json"
    file_b = tmp_path / "annotator_b.json"

    # Annotator A: 4 items (3 VALID, 1 INVALID)
    records_a = [
        {
            "record_id": "r-1", "document_id": "d-1", "relation_id": "rel-1",
            "concept_a": "A", "concept_b": "B", "relation_type": "explanation",
            "chunk_context": "Evidence 1 text.", "source_chunk_ids": ["c1"],
            "label": "VALID", "error_category": "NONE", "annotator_id": "AnnA",
            "confidence_rating": 5, "adjudicated_evidence_text": "Evidence 1 text.",
        },
        {
            "record_id": "r-2", "document_id": "d-1", "relation_id": "rel-2",
            "concept_a": "C", "concept_b": "D", "relation_type": "explanation",
            "chunk_context": "Evidence 2 text.", "source_chunk_ids": ["c2"],
            "label": "VALID", "error_category": "NONE", "annotator_id": "AnnA",
            "confidence_rating": 4, "adjudicated_evidence_text": "Evidence 2 text.",
        },
        {
            "record_id": "r-3", "document_id": "d-2", "relation_id": "rel-3",
            "concept_a": "E", "concept_b": "F", "relation_type": "explanation",
            "chunk_context": "Evidence 3 text.", "source_chunk_ids": ["c3"],
            "label": "INVALID", "error_category": "UNSUPPORTED_ASSERTION", "annotator_id": "AnnA",
            "confidence_rating": 5, "adjudicated_evidence_text": "Evidence 3 text.",
        },
        {
            "record_id": "r-4", "document_id": "d-2", "relation_id": "rel-4",
            "concept_a": "G", "concept_b": "H", "relation_type": "explanation",
            "chunk_context": "Evidence 4 text.", "source_chunk_ids": ["c4"],
            "label": "VALID", "error_category": "NONE", "annotator_id": "AnnA",
            "confidence_rating": 5, "adjudicated_evidence_text": "Evidence 4 text.",
        },
    ]

    # Annotator B: overlap on r-1, r-2, r-3. Agrees on r-1 and r-3, disagrees on r-2.
    records_b = [
        {
            "record_id": "r-1", "document_id": "d-1", "relation_id": "rel-1",
            "concept_a": "A", "concept_b": "B", "relation_type": "explanation",
            "chunk_context": "Evidence 1 text.", "source_chunk_ids": ["c1"],
            "label": "VALID", "error_category": "NONE", "annotator_id": "AnnB",
            "confidence_rating": 5, "adjudicated_evidence_text": "Evidence 1 text.",
        },
        {
            "record_id": "r-2", "document_id": "d-1", "relation_id": "rel-2",
            "concept_a": "C", "concept_b": "D", "relation_type": "explanation",
            "chunk_context": "Evidence 2 text.", "source_chunk_ids": ["c2"],
            "label": "INVALID", "error_category": "DIRECTION_INVERSION", "annotator_id": "AnnB",
            "confidence_rating": 4, "adjudicated_evidence_text": "Evidence 2 text.",
        },
        {
            "record_id": "r-3", "document_id": "d-2", "relation_id": "rel-3",
            "concept_a": "E", "concept_b": "F", "relation_type": "explanation",
            "chunk_context": "Evidence 3 text.", "source_chunk_ids": ["c3"],
            "label": "INVALID", "error_category": "UNSUPPORTED_ASSERTION", "annotator_id": "AnnB",
            "confidence_rating": 4, "adjudicated_evidence_text": "Evidence 3 text.",
        },
    ]

    with open(file_a, "w") as f:
        json.dump({"manifest_version": "1.0.0", "records": records_a}, f)
    with open(file_b, "w") as f:
        json.dump({"manifest_version": "1.0.0", "records": records_b}, f)

    # 1. Compute agreement
    report = compute_agreement_between_files(str(file_a), str(file_b))
    assert report["status"] == "AGREEMENT_COMPUTED"
    assert report["overlapping_items"] == 3
    assert report["observed_agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["disagreement_count"] == 1
    assert report["disagreement_record_ids"] == ["r-2"]

    # 2. Extract adjudication queue
    adj_queue_file = tmp_path / "adjudication_queue.json"
    adj_summary = create_adjudication_queue(str(file_a), str(file_b), str(adj_queue_file))
    assert adj_summary["total_disagreements"] == 1
    assert adj_queue_file.exists()

    with open(adj_queue_file, "r") as f:
        adj_data = json.load(f)
    assert len(adj_data["disagreements"]) == 1
    disag = adj_data["disagreements"][0]
    assert disag["record_id"] == "r-2"
    assert disag["annotator_a"]["label"] == "VALID"
    assert disag["annotator_b"]["label"] == "INVALID"
    # Adjudication fields start as None
    assert disag["adjudicated_label"] is None

    # Simulate human adjudicator resolving the disagreement
    disag["adjudicated_label"] = "INVALID"
    disag["adjudicated_error_category"] = "DIRECTION_INVERSION"
    disag["adjudicator_id"] = "SeniorAdjudicator"
    disag["adjudicated_evidence_text"] = "Evidence 2 text proving inversion."
    disag["adjudication_rationale"] = "Annotator B correctly identified direction inversion."
    with open(adj_queue_file, "w") as f:
        json.dump(adj_data, f)

    # 3. Assemble Gold Dataset V1
    gold_file = tmp_path / "human_gold_v1.json"
    gold_ds = assemble_gold_dataset_v1(
        primary_batch_path=str(file_a),
        adjudication_path=str(adj_queue_file),
        secondary_batch_path=str(file_b),
        output_gold_path=str(gold_file),
    )

    assert len(gold_ds.records) == 4
    # r-1: agreed
    rec1 = next(r for r in gold_ds.records if r.record_id == "r-1")
    assert rec1.label == ValidityLabel.VALID
    assert rec1.adjudication_status == "DUAL_AGREED"
    assert rec1.annotator_a_label == "VALID"
    assert rec1.annotator_b_label == "VALID"

    # r-2: adjudicated
    rec2 = next(r for r in gold_ds.records if r.record_id == "r-2")
    assert rec2.label == ValidityLabel.INVALID
    assert rec2.error_category == ErrorCategory.DIRECTION_INVERSION
    assert rec2.annotator_id == "SeniorAdjudicator"
    assert rec2.adjudication_status == "ADJUDICATED"
    assert rec2.annotator_a_label == "VALID"
    assert rec2.annotator_b_label == "INVALID"

    # r-4: single annotated (not in overlap)
    rec4 = next(r for r in gold_ds.records if r.record_id == "r-4")
    assert rec4.label == ValidityLabel.VALID
    assert rec4.adjudication_status == "SINGLE_ANNOTATED"
    assert rec4.annotator_b_label is None
