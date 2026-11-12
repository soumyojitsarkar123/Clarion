"""Tier 3 Cross-Feature Combination: test_t3_x08_format_duration_pipeline_poll.py
Feature Interaction: F1 (Genuine Metrics Passthrough) + F4 (Sub-second Duration Formatting).
Validates that sub-second durations recorded during stage polling (e.g., 4ms, 125ms)
are rendered with explicit millisecond precision without converting to 0.00s.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.pipeline_observability import PipelineStateTracker
from tests.e2e.test_helpers import evaluate_frontend_format_duration


def test_t3_x08_format_duration_pipeline_poll():
    """T3.INT.8: F1 + F4 Pairwise Integration."""
    tracker = PipelineStateTracker()
    doc_id = "doc-x08-fast-stages"

    raw = {
        "document_id": doc_id,
        "stages": {
            "ingestion": {"status": "completed", "duration_sec": 0.004},  # 4ms
            "chunking": {"status": "completed", "duration_sec": 0.125},   # 125ms
            "embedding": {"status": "completed", "duration_sec": 1.450},  # 1.45s
        },
    }
    status = tracker._format_pipeline_status(raw)
    stages = status["stages"]

    dur_ing = stages["ingestion"]["duration_sec"]
    dur_chk = stages["chunking"]["duration_sec"]
    dur_emb = stages["embedding"]["duration_sec"]

    # F1 passthrough precision
    assert dur_ing == pytest.approx(0.004, abs=1e-5)
    assert dur_chk == pytest.approx(0.125, abs=1e-5)
    assert dur_emb == pytest.approx(1.450, abs=1e-5)

    # F4 formatting verification
    fmt_ing = evaluate_frontend_format_duration(dur_ing)
    fmt_chk = evaluate_frontend_format_duration(dur_chk)
    fmt_emb = evaluate_frontend_format_duration(dur_emb)

    assert fmt_ing in ("4ms", "<10ms")
    assert fmt_chk == "125ms"
    assert fmt_emb == "1.45s"
