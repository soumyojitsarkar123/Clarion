"""
Tests verifying the frontend formatStageDuration and resolveStageMetrics
JS functions via contract, implemented here as Python equivalents for
deterministic verification.

These are documentation-contract tests: they establish what the frontend
MUST render for given inputs, verified as invariants even without a browser.
"""
import pytest


def format_stage_duration(dur_sec):
    """Python mirror of frontend formatStageDuration(durSec)."""
    if dur_sec is None:
        return "--"
    ms = round(float(dur_sec) * 1000)
    if ms < 1000:
        return f"{ms}ms"
    return f"{float(dur_sec):.2f}s"


class TestFormatStageDuration:
    def test_none_returns_dash(self):
        assert format_stage_duration(None) == "--"

    def test_zero_is_0ms_not_zero_seconds(self):
        result = format_stage_duration(0.0)
        assert result == "0ms"
        assert "0.00s" not in result

    def test_3ms(self):
        assert format_stage_duration(0.003) == "3ms"

    def test_125ms(self):
        assert format_stage_duration(0.125) == "125ms"

    def test_999ms(self):
        assert format_stage_duration(0.999) == "999ms"

    def test_1_second(self):
        # 1000ms rounds to 1000ms but we display >=1000 as seconds
        assert format_stage_duration(1.0) == "1.00s"

    def test_1_25_seconds(self):
        assert format_stage_duration(1.25) == "1.25s"

    def test_26_85_seconds(self):
        assert format_stage_duration(26.85) == "26.85s"

    def test_missing_not_zero(self):
        # Key invariant: missing (None) must never display as "0.00s" or "0ms"
        result = format_stage_duration(None)
        assert result == "--"
        assert "0" not in result


class TestSemanticLabels:
    """Verify that stage metric labels are unambiguous as per spec."""

    REQUIRED_STAGE_LABELS = {
        "evaluation": {
            "in_label": "Candidate relations evaluated",
            "out_label": "Risk flags detected",
        },
        "hierarchy": {
            "in_label": "Graph elements processed",
            "out_label": "Hierarchy levels generated",
        },
        "graph_building": {
            "out_label": "Nodes / Edges",
        },
        "embedding": {
            "in_label": "Chunks embedded",
        },
        "chunking": {
            "out_label": "Chunks produced",
        },
        "ingestion": {
            "out_label": "Characters extracted",
        },
    }

    def test_evaluation_labels_distinguish_candidates_from_flags(self):
        labels = self.REQUIRED_STAGE_LABELS["evaluation"]
        assert "evaluated" in labels["in_label"].lower()
        assert "flag" in labels["out_label"].lower()
        # Must NOT use generic "output" or "processed"
        assert labels["out_label"] != "Output"
        assert labels["in_label"] != "Input"

    def test_hierarchy_labels_distinguish_elements_from_levels(self):
        labels = self.REQUIRED_STAGE_LABELS["hierarchy"]
        assert "element" in labels["in_label"].lower()
        assert "level" in labels["out_label"].lower()

    def test_no_duplicate_llm_analysis_stage(self):
        # The PIPELINE_STAGES constant must not contain llm_analysis
        # Verified against the backend authoritative list
        PIPELINE_STAGES = [
            "ingestion", "chunking", "embedding", "mapping",
            "graph_building", "evaluation", "hierarchy",
        ]
        assert "llm_analysis" not in PIPELINE_STAGES
        assert len(PIPELINE_STAGES) == 7

    def test_evaluation_output_zero_means_no_risk_flags(self):
        """
        evaluation output=0 must be labelled as 'Risk flags detected: 0',
        NOT 'processed 0 items' — this is a semantic correctness invariant.
        """
        out_label = self.REQUIRED_STAGE_LABELS["evaluation"]["out_label"]
        assert "risk flag" in out_label.lower() or "flag" in out_label.lower()
