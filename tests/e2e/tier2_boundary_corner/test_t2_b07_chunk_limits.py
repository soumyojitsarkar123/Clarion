"""Tier 2 Boundary & Corner Cases: test_t2_b07_chunk_limits.py
Feature F7: Per-Chunk / Small-Batch Dataset Generation.
Verifies boundary conditions: chunks with excessive whitespace/newlines, markdown code/LaTeX formulas,
multilingual Unicode/CJK/Cyrillic content, sub-threshold word counts, and zero trainable chunks in document.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_b07_chunk_whitespace_and_newlines(tmp_path):
    """T2.F7.1: Chunk text with extreme whitespace and newlines."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_ws.db")
    raw_text = "\n\n   \t\t\n" * 100 + "A" + "\n\n   " * 100
    normalized = factory._normalize_training_text(raw_text)
    # Consecutive whitespace must collapse to a single space
    assert "   " not in normalized
    assert "\t" not in normalized
    assert normalized == "A"


def test_b07_chunk_markdown_latex_formulas(tmp_path):
    """T2.F7.2: Chunk text with LaTeX formulas and markdown formatting."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_latex.db")
    latex_text = (
        "Gaussian Integration and Probability Distributions.\n\n"
        "The standard normal density satisfies the identity "
        r"$\int_{-\infty}^{\infty} e^{-x^2 / 2} dx = \sqrt{2\pi}$."
        " Therefore the cumulative distribution function is strictly monotonic."
    )
    normalized = factory._normalize_training_text(latex_text)
    # LaTeX backslashes and formulas must be preserved
    assert r"\int" in normalized or r"\sqrt" in normalized
    assert "e^{-x^2 / 2}" in normalized


def test_b07_chunk_unicode_cjk_cyrillic(tmp_path):
    """T2.F7.3: Chunk containing non-English Unicode, CJK, and Cyrillic characters."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_unicode.db")
    multilingual = "Алгоритм Дейкстры для поиска кратчайшего пути. 迪杰斯特拉最短路径算法。"
    normalized = factory._normalize_training_text(multilingual)
    assert "Алгоритм" in normalized
    assert "迪杰斯特拉" in normalized


def test_b07_sub_threshold_word_count(tmp_path):
    """T2.F7.4: Chunk with sub-threshold word count (<180 chars or <20 tokens)."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_short.db")
    short_chunk = "This is a very short chunk with ten words only."
    is_trainable = factory._is_trainable_chunk(short_chunk)
    # Below 180 characters and 20 alpha tokens must be rejected as non-trainable
    assert is_trainable is False


def test_b07_zero_trainable_chunks_in_document(tmp_path):
    """T2.F7.5: Zero trainable chunks in entire document (all administrative boilerplate)."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_admin.db")
    admin_text = (
        "Roll No: 101. Enrollment Number: EN2026. Submitted by: Student Name.\n"
        "Work assigned: None. Group member attendance: Present. Marks obtained: 0.\n"
        "Department of Computer Science and Engineering. Prof. Smith."
    )
    is_trainable = factory._is_trainable_chunk(admin_text)
    assert is_trainable is False
