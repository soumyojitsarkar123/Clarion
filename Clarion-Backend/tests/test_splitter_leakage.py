"""
Tests for document-level data splitting and leak-free cross-validation.

Ensures that models evaluated on document knowledge graphs are never tested
on intra-document vocabulary or relation contexts seen during training.
"""

import pytest

from core.benchmarking.splitter import (
    SplitConfig,
    SplitResult,
    split_by_document,
    split_train_val_test,
    split_document_kfold,
)


@pytest.fixture
def multi_doc_records():
    """Create synthetic records spanning 6 distinct documents with multiple relations per document."""
    records = []
    for doc_idx in range(1, 7):
        doc_id = f"doc_{doc_idx}"
        for rel_idx in range(1, 6):
            records.append({
                "record_id": f"rec_{doc_idx}_{rel_idx}",
                "document_id": doc_id,
                "concept_a": f"Concept_{doc_idx}_A{rel_idx}",
                "concept_b": f"Concept_{doc_idx}_B{rel_idx}",
                "relation_type": "prerequisite" if rel_idx % 2 == 0 else "explanation",
                "llm_confidence": 0.8,
                "is_valid": True,
            })
    return records


def test_split_by_document_disjointness(multi_doc_records):
    """Verify that train, validation, and test document sets are strictly disjoint."""
    config = SplitConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        split_strategy="document",
        seed=42,
    )
    result = split_by_document(multi_doc_records, config)

    assert result.document_isolated is True
    # Mathematical assertion: zero document overlap
    result.assert_no_document_leakage()

    train_docs = result.train_doc_ids
    val_docs = result.val_doc_ids
    test_docs = result.test_doc_ids

    assert len(train_docs) >= 1
    assert len(val_docs) >= 1
    assert len(test_docs) >= 1

    assert train_docs.isdisjoint(val_docs)
    assert train_docs.isdisjoint(test_docs)
    assert val_docs.isdisjoint(test_docs)


def test_split_by_document_preserves_intra_document_relations(multi_doc_records):
    """Verify all records belonging to a document are placed in the same split."""
    config = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    result = split_by_document(multi_doc_records, config)

    # Check total record conservation
    total_records = len(result.train) + len(result.validation) + len(result.test)
    assert total_records == len(multi_doc_records)

    # Every document in train must have ALL its relations in train
    for doc_id in result.train_doc_ids:
        doc_records = [r for r in multi_doc_records if r["document_id"] == doc_id]
        train_doc_records = [r for r in result.train if r["document_id"] == doc_id]
        assert len(doc_records) == len(train_doc_records)


def test_assertion_detects_injected_leakage(multi_doc_records):
    """Verify that assert_no_document_leakage detects any document overlap."""
    config = SplitConfig(seed=42)
    result = split_by_document(multi_doc_records, config)

    # Artificially inject a record from test into train
    leaked_record = dict(result.test[0])
    result.train.append(leaked_record)

    with pytest.raises(AssertionError, match="Document-level data leakage detected"):
        result.assert_no_document_leakage()


def test_split_document_kfold(multi_doc_records):
    """Verify K-Fold cross-validation grouped by document guarantees fold isolation."""
    folds = split_document_kfold(multi_doc_records, n_splits=3, seed=42)
    assert len(folds) == 3

    for fold_idx, (train_recs, test_recs) in enumerate(folds):
        train_docs = {r["document_id"] for r in train_recs}
        test_docs = {r["document_id"] for r in test_recs}

        assert train_docs.isdisjoint(test_docs), (
            f"Fold {fold_idx} has overlapping documents: {train_docs & test_docs}"
        )
        assert len(train_recs) + len(test_recs) == len(multi_doc_records)


def test_split_train_val_test_fallback_on_insufficient_documents():
    """Verify graceful fallback to record-level split when distinct documents < min_docs."""
    # Only 1 document with 6 records
    single_doc_records = [
        {"record_id": f"r_{i}", "document_id": "single_doc", "relation_type": "type_a"}
        for i in range(6)
    ]
    config = SplitConfig(split_strategy="document", min_docs_for_grouping=3, seed=42)
    result = split_train_val_test(single_doc_records, config)

    assert result.document_isolated is False
    assert len(result.train) + len(result.validation) + len(result.test) == 6
