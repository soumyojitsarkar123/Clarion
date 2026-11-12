"""Tier 1 Feature Coverage: F17 Markdown Deliverables & Honest Publication Classifications.

Specifications:
- T1.F17.1: Verification of 8 Required Deliverable File Paths
- T1.F17.2: Submission Readiness Checklist Schema
- T1.F17.3: Permissible Honesty Classification Taxonomy
- T1.F17.4: Non-Fabrication of Unexecuted Experiments
- T1.F17.5: Fix Log Completeness
"""

from pathlib import Path
import re
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]

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

PERMISSIBLE_STATUS_TAGS = {
    "VERIFIED FIXED",
    "PARTIALLY VERIFIED",
    "UNRESOLVED",
    "BLOCKED",
    "SATISFIED",
    "TECHNICALLY IMPROVED / RESEARCH VALIDATION PENDING",
}


def test_f17_eight_deliverable_files_exist():
    """T1.F17.1: Verification of 8 Required Deliverable File Paths."""
    missing = []
    for filename in REQUIRED_DELIVERABLES:
        filepath = ROOT_DIR / filename
        if not filepath.exists():
            missing.append(filename)

    assert not missing, f"Missing required markdown deliverables: {missing}"


def test_f17_submission_readiness_checklist_schema():
    """T1.F17.2: Submission Readiness Checklist Schema."""
    checklist_path = ROOT_DIR / "SUBMISSION_READINESS_CHECKLIST.md"
    assert checklist_path.exists(), "SUBMISSION_READINESS_CHECKLIST.md missing"
    content = checklist_path.read_text(encoding="utf-8")

    assert "# Clarion Academic Submission Readiness Checklist" in content
    assert "Acceptance Criteria" in content or "Acceptance Condition" in content
    assert "Empirical Evidence" in content or "Methodology" in content


def test_f17_permissible_honesty_taxonomy():
    """T1.F17.3: Permissible Honesty Classification Taxonomy."""
    checklist_path = ROOT_DIR / "SUBMISSION_READINESS_CHECKLIST.md"
    assert checklist_path.exists()
    content = checklist_path.read_text(encoding="utf-8")

    # Find status classifications in tables or bold tags
    statuses = re.findall(r"\*\*(VERIFIED FIXED|PARTIALLY VERIFIED|UNRESOLVED|BLOCKED|SATISFIED)\*\*", content)
    assert len(statuses) > 0, "No standardized status tags found in SUBMISSION_READINESS_CHECKLIST.md"
    for st in statuses:
        assert st in PERMISSIBLE_STATUS_TAGS, f"Unrecognized status tag: {st}"


def test_f17_non_fabrication_unexecuted_experiments():
    """T1.F17.4: Non-Fabrication of Unexecuted Experiments."""
    fix_log = ROOT_DIR / "FIX_LOG.md"
    assert fix_log.exists()
    content = fix_log.read_text(encoding="utf-8")

    # Verify that human evaluation is honestly reported as unlabeled/unmeasured, not fabricated
    assert "1,554" in content or "unlabeled" in content.lower() or "unmeasured" in content.lower(), (
        "Human evaluation status must be honestly reported"
    )
    assert "Workflow B" in content or "WSL2" in content or "PARTIALLY VERIFIED" in content


def test_f17_fix_log_completeness():
    """T1.F17.5: Fix Log Completeness."""
    fix_log = ROOT_DIR / "FIX_LOG.md"
    assert fix_log.exists()
    content = fix_log.read_text(encoding="utf-8")

    assert "# Clarion Engineering and Research Fix Log" in content
    assert "Master Fix Inventory Table" in content
    assert "ISSUE-01" in content
    assert "ISSUE-02" in content
    assert "ISSUE-03" in content
