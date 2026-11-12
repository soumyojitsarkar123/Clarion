"""Tier 3 Cross-Feature Combination: test_t3_x16_multi_tenant_isolation_sessions.py
Feature Interaction: F3 (Authoritative Telemetry Schema) + F15 (Production Database Isolation & Safety).
Validates that concurrent document pipelines maintain strict session isolation in state tracking,
preventing cross-document metric bleeding while guaranteeing production database immutability.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
PROD_DB = BACKEND_DIR / "data" / "clarion.db"
PROD_DB_EXPECTED_SIZE = 34533376

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pipeline_observability import PipelineStateTracker


def test_t3_x16_multi_tenant_isolation_sessions():
    """T3.INT.16: F3 + F15 Pairwise Integration."""
    tracker = PipelineStateTracker()
    doc_a = "doc-tenant-alpha"
    doc_b = "doc-tenant-beta"

    # Initialize two distinct pipelines
    tracker.init_pipeline(doc_a, total_steps=7)
    tracker.init_pipeline(doc_b, total_steps=7)

    # Document A completes ingestion with 1,500 bytes
    tracker.start_stage(doc_a, "ingestion")
    tracker.complete_stage(doc_a, "ingestion", input_size=1500, output_size=1500)

    # Document B completes ingestion with 9,000 bytes
    tracker.start_stage(doc_b, "ingestion")
    tracker.complete_stage(doc_b, "ingestion", input_size=9000, output_size=9000)

    status_a = tracker.get_status(doc_a)
    status_b = tracker.get_status(doc_b)

    # Invariant: Strict multi-tenant metric isolation
    assert status_a["stages"]["ingestion"]["input_size"] == 1500
    assert status_b["stages"]["ingestion"]["input_size"] == 9000

    # Production DB safety check
    assert PROD_DB.stat().st_size == PROD_DB_EXPECTED_SIZE
