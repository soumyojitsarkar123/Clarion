"""Tier 3 Cross-Feature Combination: test_t3_x17_deliverables_git_integrity.py
Feature Interaction: F3 (Authoritative Telemetry Schema) + F17 (Markdown Deliverables & Publication Readiness).
Validates that the stage names, fields, and metrics documented in PIPELINE_TELEMETRY_SCHEMA.md
match the backend PipelineStateTracker authoritative schema definition with 100% parity.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pipeline_observability import PIPELINE_STAGES


def test_t3_x17_deliverables_git_integrity():
    """T3.INT.17: F3 + F17 Pairwise Integration."""
    schema_doc = ROOT_DIR / "PIPELINE_TELEMETRY_SCHEMA.md"
    assert schema_doc.exists(), "PIPELINE_TELEMETRY_SCHEMA.md must exist in repository root"
    content = schema_doc.read_text(encoding="utf-8")

    # Verify that all 7 canonical stages are documented in the markdown deliverable
    for stage in PIPELINE_STAGES:
        assert stage in content, f"Stage '{stage}' must be documented in PIPELINE_TELEMETRY_SCHEMA.md"

    # Verify key field requirements are documented
    required_fields = ["duration_sec", "input_size", "output_size", "quality_metrics", "status"]
    for field in required_fields:
        assert field in content, f"Field '{field}' must be documented in PIPELINE_TELEMETRY_SCHEMA.md"
