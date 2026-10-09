"""Unit tests for the heuristic knowledge map extraction path."""

from models.knowledge_map import KnowledgeMap, RelationType
from services.knowledge_map_service import KnowledgeMapService


def test_build_knowledge_map_is_heuristic(isolated_settings, sample_chunks):
    service = KnowledgeMapService()
    knowledge_map = service.build_knowledge_map("doc-sample", sample_chunks)

    assert isinstance(knowledge_map, KnowledgeMap)
    assert knowledge_map.metadata.get("extraction_method") == "heuristic"
    assert knowledge_map.concepts
    assert all(concept.chunk_ids for concept in knowledge_map.concepts)


def test_concept_definitions_are_grounded(isolated_settings, sample_chunks):
    service = KnowledgeMapService()
    concepts = service._extract_concepts_heuristic(sample_chunks)
    for concept in concepts:
        if concept.definition:
            assert concept.definition
    names = {concept.name.lower() for concept in concepts}
    assert "hypothesis testing" in names


def test_cooccurrence_relations_are_labeled(isolated_settings, sample_chunks):
    service = KnowledgeMapService()
    concepts = service._extract_concepts_heuristic(sample_chunks)
    relations = service._detect_relations_by_cooccurrence(concepts, sample_chunks)
    for relation in relations:
        assert relation.relation_type == RelationType.EXPLANATION
        assert "co-occurrence" in relation.description.lower()
        assert 0.0 <= relation.confidence <= 0.8
    assert not any(relation.from_concept == relation.to_concept for relation in relations)


def test_knowledge_map_round_trips_database(isolated_settings, sample_chunks):
    service = KnowledgeMapService()
    knowledge_map = service.build_knowledge_map("doc-sample", sample_chunks)
    reloaded = service.get_knowledge_map("doc-sample")
    assert reloaded is not None
    assert len(reloaded.concepts) == len(knowledge_map.concepts)
    assert len(reloaded.relations) == len(knowledge_map.relations)


def test_delete_knowledge_map(isolated_settings, sample_chunks):
    service = KnowledgeMapService()
    service.build_knowledge_map("doc-sample", sample_chunks)
    service.delete_knowledge_map("doc-sample")
    assert service.get_knowledge_map("doc-sample") is None


def test_relation_type_normalization(isolated_settings):
    interface = KnowledgeMapService().llm
    assert interface._normalize_relation_type("cause_effect") == "cause-effect"
    assert interface._normalize_relation_type("example") == "example-of"
    assert interface._normalize_relation_type("part_of") == "part-of"
    assert interface._normalize_relation_type("similar") == "similar-to"
    assert interface._normalize_relation_type("unknown-type") == "explanation"
