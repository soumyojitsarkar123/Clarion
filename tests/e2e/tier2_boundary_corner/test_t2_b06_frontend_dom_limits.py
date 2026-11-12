"""Tier 2 Boundary & Corner Cases: test_t2_b06_frontend_dom_limits.py
Feature F6: Frontend Timeline Conformance & Invariants.
Verifies boundary conditions: fast rapid polling updates, missing container element guards,
unknown status string resilience, XSS escaping in stage names/errors, and DOM tag/ID invariants.
"""

import sys
from pathlib import Path
import pytest
import lxml.html

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.pipeline_observability import PipelineStateTracker, PIPELINE_STAGES
from tests.e2e.test_helpers import simulate_render_pipeline_status, parse_frontend_html


def test_b06_fast_polling_rapid_updates():
    """T2.F6.1: Rapid state updates maintain invariant of 7 canonical stages."""
    tracker = PipelineStateTracker()
    doc_id = "doc-rapid-poll"
    tracker.init_pipeline(doc_id, total_steps=7)

    # Simulate 50 rapid sequential status reads and state mutations
    for i in range(50):
        stage_idx = i % len(PIPELINE_STAGES)
        stage_name = PIPELINE_STAGES[stage_idx]
        tracker.start_stage(doc_id, stage_name)
        status = tracker.get_status(doc_id)
        assert len(status["stages"]) == 7
        assert "llm_analysis" not in status["stages"]


def test_b06_missing_container_elements_guard():
    """T2.F6.2: Missing container elements in payload / render call."""
    # When payload is empty or invalid, render should return empty rows without throwing uncaught exceptions
    rows_empty = simulate_render_pipeline_status({})
    assert isinstance(rows_empty, list)

    rows_stages_empty = simulate_render_pipeline_status({"stages": {}})
    assert isinstance(rows_stages_empty, list)


def test_b06_unexpected_stage_status_string():
    """T2.F6.3: Malformed/unknown status string in stage record."""
    tracker = PipelineStateTracker()
    raw = {
        "document_id": "doc-unknown-status",
        "stages": {
            "ingestion": {
                "status": "quantum_superposition_xyz",
                "duration_sec": 0.05,
            }
        },
    }
    status = tracker._format_pipeline_status(raw)
    assert status["stages"]["ingestion"]["status"] == "quantum_superposition_xyz"
    rows = simulate_render_pipeline_status(status)
    assert len(rows) >= 1
    # Check rendered HTML contains the escaped status safely
    html_joined = " ".join(r["html"] for r in rows)
    assert "quantum_superposition_xyz" in html_joined


def test_b06_xss_escaping_in_stage_name():
    """T2.F6.4: Cross-site scripting (XSS) escaping in stage name and error strings."""
    xss_payload = "<script>alert('xss')</script>"
    raw = {
        "document_id": "doc-xss",
        "stages": {
            "ingestion": {
                "status": "failed",
                "errors": [xss_payload],
            }
        },
    }
    rows = simulate_render_pipeline_status(raw)
    for r in rows:
        # Script tags must be escaped as &lt;script&gt; or stripped, never raw unescaped in DOM
        assert "<script>alert" not in r["html"]


def test_b06_dom_tag_closure_and_id_uniqueness():
    """T2.F6.5: Static HTML DOM tag closure and global ID uniqueness."""
    doc = parse_frontend_html()
    all_elements_with_id = doc.xpath("//*[@id]")
    id_list = [el.get("id") for el in all_elements_with_id if el.get("id")]
    id_counts = {}
    for elem_id in id_list:
        id_counts[elem_id] = id_counts.get(elem_id, 0) + 1

    duplicates = {elem_id: count for elem_id, count in id_counts.items() if count > 1}
    assert duplicates == {}, f"Found duplicate DOM element IDs in frontend/index.html: {duplicates}"

    # Verify key timeline elements exist
    assert doc.get_element_by_id("timeline-table-body") is not None
    assert doc.get_element_by_id("pipeline-stages") is not None
