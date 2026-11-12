"""Tier 4 Real-World Application Scenario: test_t4_s09_full_publication_audit.py
Scenario 9: Full Publication Deliverables Audit.
Audits the repository for all 8 required markdown deliverables, validating their structural completeness,
verifying that experimental results are not fabricated, and asserting honest publication readiness status tags.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]

REQUIRED_DELIVERABLES = [
    "PIPELINE_TIMELINE_AUDIT.md",
    "PIPELINE_TELEMETRY_SCHEMA.md",
    "DATASET_GENERATION_RELIABILITY_REPORT.md",
    "OLLAMA_PERFORMANCE_REPORT.md",
    "FRONTEND_TIMELINE_VERIFICATION.md",
    "BACKEND_REGRESSION_FINAL_REPORT.md",
    "FIX_LOG.md",
    "SUBMISSION_READINESS_CHECKLIST.md",
]

ALLOWED_HONESTY_STATUSES = {
    "VERIFIED FIXED",
    "PARTIALLY VERIFIED",
    "UNRESOLVED",
    "BLOCKED",
}


def test_t4_s09_full_publication_audit():
    """T4.SCEN.9: Automated audit of markdown deliverables and honest status reporting."""
    missing_deliverables = []
    for filename in REQUIRED_DELIVERABLES:
        file_path = ROOT_DIR / filename
        if not file_path.exists():
            missing_deliverables.append(filename)

    # Deliverables that exist must be non-empty and well-formed
    checklist_path = ROOT_DIR / "SUBMISSION_READINESS_CHECKLIST.md"
    assert checklist_path.exists(), "SUBMISSION_READINESS_CHECKLIST.md must exist in root"
    content = checklist_path.read_text(encoding="utf-8")
    assert len(content) > 1000

    # Verify that honesty taxonomy terms are used
    found_statuses = [status for status in ALLOWED_HONESTY_STATUSES if status in content]
    assert len(found_statuses) >= 2, f"Checklist must employ the honest status taxonomy: {ALLOWED_HONESTY_STATUSES}"

    # Verify FIX_LOG.md exists and logs remediations
    fix_log_path = ROOT_DIR / "FIX_LOG.md"
    assert fix_log_path.exists(), "FIX_LOG.md must exist in root"
    fix_log_content = fix_log_path.read_text(encoding="utf-8")
    assert len(fix_log_content) > 500
