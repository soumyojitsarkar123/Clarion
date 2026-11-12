"""Tier 4 Real-World Application Scenario: test_t4_s06_concurrent_uploads.py
Scenario 6: Concurrent Multi-Document Ingestion.
Simulates parallel ingestion and status tracking for multiple concurrent documents,
asserting zero state bleeding between pipelines and absolute production database immutability.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
PROD_DB = BACKEND_DIR / "data" / "clarion.db"
PROD_DB_EXPECTED_SIZE = 34533376

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pipeline_observability import PipelineStateTracker


def _simulate_doc_pipeline(doc_id: str, tracker: PipelineStateTracker, size: int) -> dict:
    tracker.init_pipeline(doc_id, total_steps=7)
    tracker.start_stage(doc_id, "ingestion")
    tracker.complete_stage(doc_id, "ingestion", input_size=size, output_size=size)
    tracker.start_stage(doc_id, "chunking")
    tracker.complete_stage(doc_id, "chunking", input_size=size, output_size=max(1, size // 500))
    return tracker.get_status(doc_id)


def test_t4_s06_concurrent_uploads():
    """T4.SCEN.6: Concurrent document ingestion isolation."""
    tracker = PipelineStateTracker()
    doc_specs = [
        (f"doc-concurrent-{i}", 1000 * (i + 1))
        for i in range(10)
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_simulate_doc_pipeline, doc_id, tracker, size)
            for doc_id, size in doc_specs
        ]
        results = [f.result() for f in futures]

    assert len(results) == 10
    # Assert each pipeline retained its exact input size without race conditions
    for i, res in enumerate(results):
        expected_size = 1000 * (i + 1)
        assert res["stages"]["ingestion"]["input_size"] == expected_size
        assert len(res["stages"]) == 7

    # Assert production DB remained unmodified
    assert PROD_DB.stat().st_size == PROD_DB_EXPECTED_SIZE
