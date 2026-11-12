"""Unit tests for relation dataset capture: provenance, quality flags, export."""

import asyncio
import json

import pytest

from models.knowledge_map import Concept, KnowledgeMap, Relation, RelationType
from services.relation_dataset_service import RelationDatasetService


def make_concept(name, chunk_ids):
    return Concept(id=f"c-{name}", name=name, definition=f"Definition of {name}", chunk_ids=chunk_ids)


def make_relation(from_name, to_name, description=None):
    return Relation(
        id=f"r-{from_name}-{to_name}",
        from_concept=from_name,
        to_concept=to_name,
        relation_type=RelationType.EXPLANATION,
        description=description or "Inferred from repeated co-occurrence within the same chunks.",
        confidence=0.6,
    )


def make_map(document_id, chunks):
    concepts = [
        make_concept("Hypothesis Testing", [chunks[0].chunk_id]),
        make_concept("Null Hypothesis", [chunks[0].chunk_id]),
        make_concept("Website", [chunks[1].chunk_id]),
    ]
    relations = [
        make_relation("Hypothesis Testing", "Null Hypothesis"),
        make_relation("Website", "Null Hypothesis"),
    ]
    return KnowledgeMap(document_id=document_id, concepts=concepts, relations=relations)


@pytest.fixture
def dataset_chunks(isolated_settings):
    from models.chunk import Chunk

    return [
        Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Hypothesis testing evaluates the null hypothesis using a sample.",
            position_index=0,
            word_count=10,
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="The website frontend was assigned to the team with roll no details.",
            position_index=10,
            word_count=13,
        ),
    ]


def test_log_relation_batch_records_provenance(isolated_settings, dataset_chunks):
    service = RelationDatasetService()
    knowledge_map = make_map("doc-1", dataset_chunks)
    count = asyncio.run(
        service.log_relation_batch(
            document_id="doc-1",
            knowledge_map=knowledge_map,
            chunks=dataset_chunks,
        )
    )
    assert count == 2

    records = service.get_dataset(document_id="doc-1")
    assert len(records) == 2
    for record in records:
        assert record.record_id
        assert record.is_valid is None
        assert record.metadata.get("confidence_source") == "heuristic_cooccurrence"
        assert record.source_chunk_ids


def test_quality_flags_mark_generic_concepts(isolated_settings, dataset_chunks):
    service = RelationDatasetService()
    knowledge_map = make_map("doc-1", dataset_chunks)
    asyncio.run(
        service.log_relation_batch(
            document_id="doc-1",
            knowledge_map=knowledge_map,
            chunks=dataset_chunks,
        )
    )
    records = {record.concept_a: record for record in service.get_dataset(document_id="doc-1")}
    website_record = records["Website"]
    assert "generic_concept_a" in website_record.metadata.get("quality_flags", [])


def test_validation_update_and_stats(isolated_settings, dataset_chunks):
    service = RelationDatasetService()
    knowledge_map = make_map("doc-1", dataset_chunks)
    asyncio.run(
        service.log_relation_batch(
            document_id="doc-1",
            knowledge_map=knowledge_map,
            chunks=dataset_chunks,
        )
    )
    records = service.get_dataset(document_id="doc-1")
    assert service.update_validation(records[0].record_id, True)
    assert service.update_validation(records[1].record_id, False)

    stats = service.get_dataset_stats()
    assert stats["total_records"] == 2
    assert stats["labeled_records"] == 2
    assert stats["unlabeled_records"] == 0


def test_export_json_and_csv(isolated_settings, dataset_chunks):
    service = RelationDatasetService()
    knowledge_map = make_map("doc-1", dataset_chunks)
    asyncio.run(
        service.log_relation_batch(
            document_id="doc-1",
            knowledge_map=knowledge_map,
            chunks=dataset_chunks,
        )
    )

    json_export = service.export_dataset(format="json", document_id="doc-1")
    assert json_export["format"] == "json"
    assert json_export["count"] == 2
    assert "quality_summary" in json_export
    for record in json_export["records"]:
        assert record["is_valid"] is None
        assert "llm_confidence" in record

    csv_export = service.export_dataset(format="csv", document_id="doc-1")
    assert csv_export["format"] == "csv"
    assert "record_id" in csv_export["data"]


def test_export_document_snapshot(isolated_settings, dataset_chunks):
    service = RelationDatasetService()
    knowledge_map = make_map("doc-1", dataset_chunks)
    asyncio.run(
        service.log_relation_batch(
            document_id="doc-1",
            knowledge_map=knowledge_map,
            chunks=dataset_chunks,
        )
    )
    path = service.export_document_snapshot(
        document_id="doc-1", document_filename="Stats Paper.pdf"
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["document_id"] == "doc-1"
    assert payload["document_filename"] == "Stats Paper.pdf"
    assert payload["count"] == 2


def test_safe_export_stem(isolated_settings):
    service = RelationDatasetService()
    assert service._safe_export_stem("My: Paper?.pdf") == "My_Paper"
    assert service._safe_export_stem("") == "dataset"


def test_workspace_dataset_mirror_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CLARION_DISABLE_WORKSPACE_DATASET_MIRROR", "1")
    assert RelationDatasetService()._workspace_dataset_dir() is None
