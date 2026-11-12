"""Tier 1 Feature Coverage: F4 Sub-second Duration Formatting.

Specifications:
- T1.F4.1: Millisecond Unit Formatting for t < 1.0s (0.125 -> "125ms")
- T1.F4.2: Small Millisecond Unit Formatting (0.003 -> "3ms")
- T1.F4.3: Sub-Millisecond Unit Formatting (0.0004 -> "<1ms" or "0ms", never "0.00s")
- T1.F4.4: Second Unit Formatting for t >= 1.0s (1.25 -> "1.25s")
- T1.F4.5: Preservation of Missing Durations as "--" (None -> "--", never "0.00s")
"""

import pytest
from tests.e2e.test_helpers import evaluate_frontend_format_duration


def test_f04_millisecond_formatting():
    """T1.F4.1: Millisecond Unit Formatting for t < 1.0s."""
    formatted = evaluate_frontend_format_duration(0.125)
    assert formatted == "125ms", f"Expected '125ms', got '{formatted}'"


def test_f04_small_millisecond_formatting():
    """T1.F4.2: Small Millisecond Unit Formatting."""
    formatted = evaluate_frontend_format_duration(0.003)
    assert formatted == "3ms", f"Expected '3ms', got '{formatted}'"


def test_f04_sub_millisecond_formatting():
    """T1.F4.3: Sub-Millisecond Unit Formatting."""
    formatted = evaluate_frontend_format_duration(0.0004)
    assert formatted in ["<1ms", "0ms"], f"Expected '<1ms' or '0ms', got '{formatted}'"
    assert formatted != "0.00s", "Must never format sub-millisecond durations as '0.00s'"


def test_f04_second_formatting():
    """T1.F4.4: Second Unit Formatting for t >= 1.0s."""
    formatted = evaluate_frontend_format_duration(1.25)
    assert formatted == "1.25s", f"Expected '1.25s', got '{formatted}'"


def test_f04_missing_duration_formatting():
    """T1.F4.5: Preservation of Missing Durations as '--'."""
    formatted = evaluate_frontend_format_duration(None)
    assert formatted == "--", f"Expected '--' for missing/null duration, got '{formatted}'"
