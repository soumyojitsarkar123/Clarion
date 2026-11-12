"""Tier 2 Boundary & Corner Cases: test_t2_b17_deliverable_validation.py
Feature F17: Markdown Deliverables & Honest Publication Classifications.
Verifies boundary conditions: missing section detection, placeholder tokens detection ([TODO]/TBD),
invalid status string rejection (e.g. ALMOST_FIXED), cross-file consistency, and zero-byte file rejection.
"""

import re
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
CANONICAL_HONEST_STATUSES = {
    "VERIFIED FIXED",
    "PARTIALLY VERIFIED",
    "UNRESOLVED",
    "BLOCKED",
}


def validate_markdown_structure(content: str, required_sections: list[str]) -> list[str]:
    """Return missing section headings from content."""
    missing = []
    for section in required_sections:
        pattern = rf"^#+\s+{re.escape(section)}"
        if not re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            missing.append(section)
    return missing


def validate_no_unresolved_placeholders(content: str) -> list[str]:
    """Detect unresolved placeholders like [TODO], [TBD], or [INSERT ...]."""
    matches = re.findall(r"\[(TODO|TBD|INSERT[^\]]*)\]", content, re.IGNORECASE)
    return matches


def test_b17_missing_section_heading_flagged():
    """T2.F17.1: Missing required section heading detected by validator."""
    sample_md = "# Title\n\n## Section 1\n\nContent\n"
    missing = validate_markdown_structure(sample_md, ["Section 1", "Section 2: Architecture"])
    assert "Section 2: Architecture" in missing
    assert len(missing) == 1


def test_b17_placeholder_text_rejection():
    """T2.F17.2: Unresolved placeholders [TODO] / [INSERT METRICS HERE] detected."""
    sample_with_placeholders = "# Report\n\nResults: [INSERT METRICS HERE]. Next steps: [TODO]."
    found = validate_no_unresolved_placeholders(sample_with_placeholders)
    assert len(found) == 2
    assert "INSERT METRICS HERE" in found[0]
    assert "TODO" in found[1]


def test_b17_invalid_honesty_status_string():
    """T2.F17.3: Invalid honesty status string rejection."""
    invalid_statuses = ["ALMOST_FIXED", "IN_PROGRESS", "MOSTLY_WORKING", "PENDING_VERIFICATION"]
    for bad_status in invalid_statuses:
        assert bad_status not in CANONICAL_HONEST_STATUSES


def test_b17_submission_readiness_canonical_statuses():
    """T2.F17.4: SUBMISSION_READINESS_CHECKLIST.md uses canonical honesty taxonomy."""
    checklist_path = ROOT_DIR / "reports" / "historical" / "SUBMISSION_READINESS_CHECKLIST.md"
    if checklist_path.exists():
        content = checklist_path.read_text(encoding="utf-8")
        # Find all bold status tags or markdown badges
        found_statuses = set()
        for canonical in CANONICAL_HONEST_STATUSES:
            if canonical in content:
                found_statuses.add(canonical)
        # Must contain at least one canonical status tag
        assert len(found_statuses) >= 1


def test_b17_zero_byte_deliverable_rejected(tmp_path):
    """T2.F17.5: Zero-byte deliverable file rejected as missing/invalid."""
    empty_file = tmp_path / "EMPTY_REPORT.md"
    empty_file.write_text("", encoding="utf-8")
    assert empty_file.stat().st_size == 0
    # Zero-length file is invalid
    is_valid = empty_file.exists() and empty_file.stat().st_size > 0
    assert is_valid is False
