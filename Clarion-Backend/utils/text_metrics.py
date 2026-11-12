"""Shared text counting rules used by ingestion, chunking, and API metrics."""

from __future__ import annotations


def count_words(text: str | None) -> int:
    """Count whitespace-delimited units in normalized extracted text.

    Attached punctuation, hyphenated terms, contractions, numbers, and Unicode
    text count as one unit. A punctuation-only unit also counts because the
    application uses whitespace units, not language-model tokens.
    """
    return len((text or "").split())
