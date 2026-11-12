"""Tier 1 Feature Coverage: F1 Genuine Stage Metrics Passthrough.

Specifications:
- T1.F1.1: Ingestion Stage Metrics Passthrough (input_size, output_size > 0, completed)
- T1.F1.2: Chunking Stage Metrics Passthrough (chunk_count, output_size, input_size > 0)
- T1.F1.3: Embedding Stage Metrics & Throughput Passthrough (duration_sec > 0, throughput > 0, device)
- T1.F1.4: Graph Building Stage Metrics Passthrough (node_count, edge_count, completed)
- T1.F1.5: Evaluation & Hierarchy Stages Metrics Passthrough (candidate relations, hierarchy levels)
"""

import time
import pytest
from services.pipeline_observability import (
    get_pipeline_state_tracker,
    PipelineAuditor,
)


def test_f01_ingestion_metrics_passthrough(e2e_client):
    """T1.F1.1: Ingestion Stage Metrics Passthrough."""
    doc_id = "doc-t1-f01-ingest"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)

    auditor.start_stage("ingestion", input_size=1500, progress=5)
    time.sleep(0.005)
    auditor.complete_stage(
        "ingestion",
        output_size=1450,
        quality_metrics={"word_count": 220, "has_text": True},
        progress=10,
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200, f"Status API returned {response.status_code}"
    payload = response.json()

    assert "stages" in payload, "Payload missing stages dictionary"
    ingestion = payload["stages"].get("ingestion", {})
    assert ingestion.get("status") == "completed", f"Expected completed, got {ingestion.get('status')}"
    assert ingestion.get("input_size") == 1500, f"Expected 1500, got {ingestion.get('input_size')}"
    assert ingestion.get("output_size") == 1450, f"Expected 1450, got {ingestion.get('output_size')}"
    assert ingestion.get("duration_sec") is not None and ingestion.get("duration_sec") >= 0.0


def test_f01_chunking_metrics_passthrough(e2e_client):
    """T1.F1.2: Chunking Stage Metrics Passthrough."""
    doc_id = "doc-t1-f01-chunk"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)

    auditor.start_stage("chunking", input_size=5000, progress=15)
    time.sleep(0.005)
    auditor.complete_stage(
        "chunking",
        output_size=8,
        quality_metrics={"chunk_count": 8, "avg_chunk_words": 62.5},
        progress=30,
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    chunking = payload["stages"].get("chunking", {})
    assert chunking.get("status") == "completed"
    assert chunking.get("chunk_count") == 8
    assert chunking.get("output_size") == 8
    assert chunking.get("input_size") == 5000


def test_f01_embedding_metrics_passthrough(e2e_client):
    """T1.F1.3: Embedding Stage Metrics & Throughput Passthrough."""
    doc_id = "doc-t1-f01-embed"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)

    auditor.start_stage("embedding", input_size=8, progress=35)
    time.sleep(0.005)
    auditor.complete_stage(
        "embedding",
        output_size=8,
        quality_metrics={"indexed": True, "chunk_count": 8},
        progress=50,
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    embedding = payload["stages"].get("embedding", {})
    assert embedding.get("status") == "completed"
    assert embedding.get("duration_sec") is not None
    assert embedding.get("duration_sec") > 0.0
    assert "throughput_chunks_sec" in embedding
    assert embedding.get("throughput_chunks_sec") > 0.0
    assert embedding.get("device") in ["cpu", "cuda"]


def test_f01_graph_building_metrics_passthrough(e2e_client):
    """T1.F1.4: Graph Building Stage Metrics Passthrough."""
    doc_id = "doc-t1-f01-graph"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)

    auditor.start_stage("graph_building", input_size=27, progress=70)
    time.sleep(0.005)
    auditor.complete_stage(
        "graph_building",
        output_size=27,
        quality_metrics={
            "node_count": 12,
            "edge_count": 15,
            "density": 0.113,
            "is_weakly_connected": True,
        },
        progress=80,
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    graph_stage = payload["stages"].get("graph_building", {})
    assert graph_stage.get("status") == "completed"
    assert graph_stage.get("node_count") == 12
    assert graph_stage.get("edge_count") == 15
    assert graph_stage.get("output_size") == 27


def test_f01_evaluation_hierarchy_metrics_passthrough(e2e_client):
    """T1.F1.5: Evaluation & Hierarchy Stages Metrics Passthrough."""
    doc_id = "doc-t1-f01-eval-hier"
    tracker = get_pipeline_state_tracker()
    tracker.start_pipeline(doc_id)
    auditor = PipelineAuditor(document_id=doc_id, tracker=tracker)

    # Evaluation
    auditor.start_stage("evaluation", input_size=15, progress=85)
    time.sleep(0.005)
    auditor.complete_stage(
        "evaluation",
        output_size=2,
        quality_metrics={"overall_score": 0.91, "risk_score": 0.09, "flags_count": 2},
        progress=92,
    )

    # Hierarchy
    auditor.start_stage("hierarchy", input_size=27, progress=95)
    time.sleep(0.005)
    auditor.complete_stage(
        "hierarchy",
        output_size=3,
        quality_metrics={"layers_count": 3, "chains_count": 2},
        progress=100,
    )

    response = e2e_client.get(f"/system/pipeline-status?document_id={doc_id}")
    assert response.status_code == 200
    payload = response.json()

    eval_stage = payload["stages"].get("evaluation", {})
    hier_stage = payload["stages"].get("hierarchy", {})

    assert eval_stage.get("status") == "completed"
    assert eval_stage.get("input_size") == 15
    assert eval_stage.get("output_size") == 2

    assert hier_stage.get("status") == "completed"
    assert hier_stage.get("input_size") == 27
    assert hier_stage.get("output_size") == 3
