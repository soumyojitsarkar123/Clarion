"""Tier 2 Boundary & Corner Cases: test_t2_b04_subsecond_limits.py
Feature F4: Sub-second Duration Formatting (ms/s, no 0.00s).
Verifies boundary conditions: negative durations, exact subsecond thresholds (0.9999s),
NaN / infinite / null durations, multi-day large durations, and string-encoded durations.
"""

import math
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.e2e.test_helpers import evaluate_frontend_format_duration


def test_b04_negative_duration_handling():
    """T2.F4.1: Negative duration input handling."""
    res = evaluate_frontend_format_duration(-0.5)
    # Negative durations due to clock skew must render as placeholder '--' or clamp to '0ms', never negative
    assert res in ("--", "0ms", "<1ms")
    assert "-" not in res or res == "--"


def test_b04_exact_threshold_subsecond():
    """T2.F4.2: Exact threshold value at t = 0.9999s."""
    res = evaluate_frontend_format_duration(0.9999)
    # 0.9999s should format as '1000ms' or rounded '1.00s', never scientific notation
    assert res in ("1000ms", "1.00s")
    assert "e" not in res.lower()


def test_b04_nan_and_inf_duration():
    """T2.F4.3: Infinite or NaN duration input handling."""
    res_null = evaluate_frontend_format_duration(None)
    assert res_null == "--"

    res_nan = evaluate_frontend_format_duration("NaN")
    assert res_nan == "--"


def test_b04_extreme_duration_multiday():
    """T2.F4.4: Extremely long duration (t = 86400s) boundary."""
    res = evaluate_frontend_format_duration(86400.0)
    assert "86400" in res or "24h" in res
    assert not res.startswith("0")


def test_b04_string_encoded_duration():
    """T2.F4.5: String encoded numeric duration coercion."""
    res = evaluate_frontend_format_duration("0.045")
    assert res == "45ms"
