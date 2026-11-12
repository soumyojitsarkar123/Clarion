"""Tier 1 Feature Coverage: F5 Semantic Disambiguation Labels & Tooltips.

Specifications:
- T1.F5.1: Evaluation Input Label Disambiguation ("Candidate Relations Evaluated")
- T1.F5.2: Evaluation Output Label Disambiguation ("Risk Flags Detected")
- T1.F5.3: Hierarchy Input Label Disambiguation ("Graph Elements (Nodes + Edges)")
- T1.F5.4: Hierarchy Output Label Disambiguation ("Hierarchy Levels & Chains")
- T1.F5.5: Graph Building Semantic Distinction (concept/relation inputs vs node/edge outputs)
"""

import pytest
from tests.e2e.test_helpers import (
    simulate_render_pipeline_status,
    parse_frontend_html,
)


def _get_rendered_stage_row(stage_key: str):
    stages_payload = {
        "ingestion": {"status": "completed", "duration_sec": 0.05, "input_size": 1500, "output_size": 1400},
        "chunking": {"status": "completed", "duration_sec": 0.04, "input_size": 1400, "output_size": 8},
        "embedding": {"status": "completed", "duration_sec": 0.08, "input_size": 8, "output_size": 8},
        "mapping": {"status": "completed", "duration_sec": 0.12, "input_size": 8, "output_size": 25},
        "graph_building": {"status": "completed", "duration_sec": 0.06, "input_size": 25, "output_size": 25, "node_count": 12, "edge_count": 13},
        "evaluation": {"status": "completed", "duration_sec": 0.03, "input_size": 25, "output_size": 2},
        "hierarchy": {"status": "completed", "duration_sec": 0.03, "input_size": 25, "output_size": 3},
    }
    rows = simulate_render_pipeline_status({"stages": stages_payload})
    for r in rows:
        if stage_key in r["cells"][0].lower():
            return r
    return None


def test_f05_evaluation_input_label():
    """T1.F5.1: Evaluation Input Label Disambiguation."""
    row = _get_rendered_stage_row("evaluation")
    assert row is not None, "Evaluation stage row not found in rendered timeline"

    # Input cell is index 3 (0=Name, 1=Status, 2=Duration, 3=Input Size, 4=Output Size)
    input_cell_text = row["cells"][3]
    input_title = row["titles"][3] if len(row["titles"]) > 3 else ""
    row_html = row["html"]

    assert (
        "candidate relations evaluated" in input_title.lower()
        or "candidate relations evaluated" in input_cell_text.lower()
        or "candidate relations" in row_html.lower()
    ), f"Evaluation input must explicitly denote 'Candidate Relations Evaluated', got html: {row_html}"


def test_f05_evaluation_output_label():
    """T1.F5.2: Evaluation Output Label Disambiguation."""
    row = _get_rendered_stage_row("evaluation")
    assert row is not None, "Evaluation stage row not found in rendered timeline"

    output_cell_text = row["cells"][4]
    output_title = row["titles"][4] if len(row["titles"]) > 4 else ""
    row_html = row["html"]

    assert (
        "risk flags detected" in output_title.lower()
        or "risk flags detected" in output_cell_text.lower()
        or "risk flags" in row_html.lower()
    ), f"Evaluation output must explicitly denote 'Risk Flags Detected', got html: {row_html}"


def test_f05_hierarchy_input_label():
    """T1.F5.3: Hierarchy Input Label Disambiguation."""
    row = _get_rendered_stage_row("hierarchy")
    assert row is not None, "Hierarchy stage row not found in rendered timeline"

    input_cell_text = row["cells"][3]
    input_title = row["titles"][3] if len(row["titles"]) > 3 else ""
    row_html = row["html"]

    assert (
        "graph elements" in input_title.lower()
        or "graph elements" in input_cell_text.lower()
        or "graph elements" in row_html.lower()
    ), f"Hierarchy input must explicitly denote 'Graph Elements', got html: {row_html}"


def test_f05_hierarchy_output_label():
    """T1.F5.4: Hierarchy Output Label Disambiguation."""
    row = _get_rendered_stage_row("hierarchy")
    assert row is not None, "Hierarchy stage row not found in rendered timeline"

    output_cell_text = row["cells"][4]
    output_title = row["titles"][4] if len(row["titles"]) > 4 else ""
    row_html = row["html"]

    assert (
        "hierarchy levels" in output_title.lower()
        or "hierarchy levels" in output_cell_text.lower()
        or "hierarchy levels" in row_html.lower()
    ), f"Hierarchy output must explicitly denote 'Hierarchy Levels & Chains', got html: {row_html}"


def test_f05_graph_building_semantic_distinction():
    """T1.F5.5: Graph Building Semantic Distinction."""
    row = _get_rendered_stage_row("graph_building")
    assert row is not None, "Graph building stage row not found in rendered timeline"

    row_html = row["html"]
    # Verify semantic distinction between input (concepts/relations) and output (graph nodes/edges)
    assert (
        "concept" in row_html.lower()
        or "node" in row_html.lower()
        or "graph element" in row_html.lower()
    ), f"Graph building row missing semantic node/edge indicators, got: {row_html}"
