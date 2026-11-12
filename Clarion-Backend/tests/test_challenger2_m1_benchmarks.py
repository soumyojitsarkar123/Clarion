"""
Challenger 2 Empirical Verification, Fuzzing, Throughput Benchmarking,
and Real Document Regression Suite for Milestone 1.

Targets:
- normalize_extracted_text
- _segment_concatenated_token
- extract_text_from_pdf
"""

import time
import random
import string
import sqlite3
import re
import pytest
from pathlib import Path
import sys

# Ensure Clarion-Backend is in python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.file_handler import (
    normalize_extracted_text,
    _segment_concatenated_token,
    KNOWN_VALID_WORDS,
    SEGMENTATION_VOCAB,
    extract_text_from_pdf,
)


class TestDPComplexityAndFuzzing:
    """Empirical tests for dynamic programming segmentation complexity and fuzzing."""

    def test_dp_pathological_repeated_single_char(self):
        """Test DP segmenter against long strings of 'a' (the only valid 1-char word)."""
        lengths = [10, 50, 100, 500, 1000]
        times = []
        for l in lengths:
            token = 'a' * l
            t0 = time.perf_counter()
            result = _segment_concatenated_token(token)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            expected = ' '.join(['a'] * l)
            assert result == expected
            assert elapsed < 1.0, f"Length {l} took too long: {elapsed:.4f}s"
        assert times[-1] < 0.5, f"1000 repeated 'a' took {times[-1]:.4f}s"

    def test_dp_pathological_repeated_two_char_words(self):
        """Test DP segmenter against repeated 2-char dictionary words ('in', 'to', 'is')."""
        for word in ['in', 'to', 'is', 'of']:
            for count in [10, 50, 100, 250]:
                token = word * count
                t0 = time.perf_counter()
                result = _segment_concatenated_token(token)
                elapsed = time.perf_counter() - t0
                assert elapsed < 0.5, f"{token[:20]}... count={count} took {elapsed:.4f}s"
                assert result == ' '.join([word] * count)

    def test_dp_unmatchable_long_tokens(self):
        """Test DP segmenter on long random lowercase strings that have no dictionary matches."""
        for length in [50, 100, 500, 1000, 5000]:
            token = ''.join(random.choices('bcdfghjklmnpqrstvwxyz', k=length))
            t0 = time.perf_counter()
            result = _segment_concatenated_token(token)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.5, f"Length {length} took {elapsed:.4f}s"
            assert isinstance(result, str)

    def test_dp_alternating_branching_pressure(self):
        """Test DP segmenter against alternating high-branching vocabulary combinations."""
        token = "whenspeedisimportant" * 20  # 400 chars
        t0 = time.perf_counter()
        result = _segment_concatenated_token(token)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.2, f"Branching token took {elapsed:.4f}s"
        assert "speed" in result
        assert "important" in result

    def test_regex_catastrophic_backtracking_fuzz(self):
        """Test normalize_extracted_text against adversarial regex backtracking inputs."""
        fuzz_cases = [
            "(" * 500 + "nested" + ")" * 500,
            "word" + "?!:,;" * 1000 + "next",
            "A" * 5000,
            "COA" * 1000 + "Final",
            "MIPS" * 1000 + "and",
            "9" * 5000 + "questions",
            "A." * 2000,
            "\x00\x01\x02\x03" * 500,
            "test\u200b\u200bzero\ufeffwidth\u00a0nonbreaking",
            "计算机体系结构" * 100 + "RAWcanoccur" + "🚀🔥💡" * 50,
            "word" + " \t \t   \t" * 1000 + "end",
        ]
        for idx, case in enumerate(fuzz_cases):
            t0 = time.perf_counter()
            normalized = normalize_extracted_text(case)
            elapsed = time.perf_counter() - t0
            assert elapsed < 1.0, f"Fuzz case {idx} took {elapsed:.4f}s (potential catastrophic backtracking)"
            assert isinstance(normalized, str)


class TestThroughputAndScaling:
    """Benchmark throughput (words/sec) across various text sizes."""

    @staticmethod
    def generate_synthetic_text(word_count: int) -> str:
        words_pool = [
            "Ina", "WAWcanoccur", "RAWcanoccur", "Carry", "Look", "Ahead", "Adderis",
            "used", "whenspeedisimportant", "generally", "parallel", "separate",
            "significance", "terms", "hazards", "pipeline", "allows", "computer",
            "architecture", "Section1", "10questions", "analysis", "synthesis",
            "The", "processor", "computes", "results", "witha", "high", "frequency."
        ]
        selected = random.choices(words_pool, k=word_count)
        return " ".join(selected)

    def test_throughput_scaling_1k_to_50k_words(self):
        """Benchmark normalization throughput on 1k, 5k, 10k, and 50k words."""
        sizes = [1000, 5000, 10000, 50000]
        results = {}
        for size in sizes:
            text = self.generate_synthetic_text(size)
            char_count = len(text)
            
            # Warm up
            _ = normalize_extracted_text(text[:200])
            
            t0 = time.perf_counter()
            out = normalize_extracted_text(text)
            elapsed = time.perf_counter() - t0
            
            words_per_sec = size / elapsed if elapsed > 0 else float('inf')
            chars_per_sec = char_count / elapsed if elapsed > 0 else float('inf')
            results[size] = {
                "elapsed_sec": elapsed,
                "words_per_sec": words_per_sec,
                "chars_per_sec": chars_per_sec,
            }
            # Acceptance threshold: at least 5,000 words/second throughput
            assert words_per_sec > 5000, f"Throughput too low for {size} words: {words_per_sec:.0f} words/sec"
            assert "In a" in out
            assert "RAW can occur" in out
            assert "when speed is important" in out

        print("\nThroughput Benchmark Results:")
        for size, r in results.items():
            print(f"  {size:6d} words: {r['elapsed_sec']:.4f}s ({r['words_per_sec']:,.0f} words/sec, {r['chars_per_sec']:,.0f} chars/sec)")


class TestRealDocumentRegressions:
    """Regression checks on real document extracts."""

    def test_coa_exam_prediction_sheet_regression(self):
        """Verify normalization on COA_Final_Exam_Prediction_Sheet.pdf extract."""
        db_path = Path(BACKEND_DIR) / "data" / "clarion.db"
        if not db_path.exists():
            db_path = Path("/mnt/d/Clarion/data/clarion.db")
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        row = cur.execute(
            "SELECT text_content FROM documents WHERE id = 'cf1f04f1-fe0a-45d5-b151-f7e107126e2a'"
        ).fetchone()
        assert row is not None, "COA document cf1f04f1-fe0a-45d5-b151-f7e107126e2a not found in clarion.db"
        raw_text = row[0]
        assert raw_text and len(raw_text) > 500

        t0 = time.perf_counter()
        normalized = normalize_extracted_text(raw_text)
        elapsed = time.perf_counter() - t0
        words = len(raw_text.split())
        throughput = words / elapsed if elapsed > 0 else float('inf')
        print(f"\nCOA Normalized {words} words in {elapsed:.4f}s ({throughput:,.0f} words/sec)")

        # Target Acceptance Criteria for COA:
        # 1. Glued phrases split
        assert "Usethisasa" not in normalized
        assert "Use this as a" in normalized
        assert "step by step" in normalized
        assert "or derivation" in normalized
        assert "in a simple" in normalized.lower()
        assert "raw can occur" in normalized.lower()

        # Target prompt assertions from worker handoff:
        target_checks = [
            ("Ina simple", "In a simple"),
            ("WAWcanoccur", "WAW can occur"),
            ("RAWcanoccur", "RAW can occur"),
            ("Carry Look Ahead Adderis used whenspeedisimportant", "Carry Look Ahead Adder is used when speed is important"),
            ("generally", "generally"),
            ("parallel", "parallel"),
            ("significance", "significance")
        ]
        for inp, exp in target_checks:
            assert normalize_extracted_text(inp) == exp

    def test_daa_final_regression(self):
        """Verify normalization on DAA FINAL.pdf extract."""
        pdf_path = Path(BACKEND_DIR) / "data" / "DAA FINAL.pdf"
        assert pdf_path.exists(), f"DAA FINAL.pdf not found at {pdf_path}"
        
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        t0 = time.perf_counter()
        text, metadata = extract_text_from_pdf(pdf_bytes)
        elapsed = time.perf_counter() - t0
        word_count = len(text.split())
        throughput = word_count / elapsed if elapsed > 0 else float('inf')
        print(f"\nDAA FINAL.pdf: {metadata['page_count']} pages, {word_count} words extracted & normalized in {elapsed:.4f}s ({throughput:,.0f} words/sec)")

        # Verify integrity of DAA algorithms and terms
        assert "DAA" in text
        assert "Dijkstra" in text
        assert "Knapsack" in text
        assert re.search(r"Dynamic\s+Programming", text, re.I) is not None
        assert metadata["page_count"] > 0
        assert word_count > 1000

    def test_ai_book_sample_regression(self):
        """Verify normalization on AI_Book.pdf extract."""
        pdf_path = Path(BACKEND_DIR) / "data" / "AI_Book.pdf"
        assert pdf_path.exists(), f"AI_Book.pdf not found at {pdf_path}"

        import io
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
            num_pages = len(reader.pages)
            pages_to_test = min(15, num_pages)
            sample_text = "\n\n".join([reader.pages[i].extract_text() or "" for i in range(pages_to_test)])

        word_count = len(sample_text.split())
        t0 = time.perf_counter()
        normalized = normalize_extracted_text(sample_text)
        elapsed = time.perf_counter() - t0
        throughput = word_count / elapsed if elapsed > 0 else float('inf')
        print(f"\nAI_Book sample ({pages_to_test} pages, {word_count} words) normalized in {elapsed:.4f}s ({throughput:,.0f} words/sec)")

        assert word_count > 1000
        assert throughput > 5000, f"AI_Book throughput too low: {throughput:.0f} words/sec"
        assert len(normalized) > 0


class TestAdversarialFindingsAndDefectReproduction:
    """
    Adversarial verification tests validating resolved defects:
    1. TITLE_VERB_MERGER_PATTERN preserves common English TitleCase words.
    2. Vocabulary expansions allow DP segmenter to resolve glued phrases.
    3. Sentence-starting glued words with initial uppercase are properly segmented.
    """

    def test_reproduce_title_verb_over_segmentation_defect(self):
        """
        DEFECT RESOLUTION: TITLE_VERB_MERGER_PATTERN preserves valid English words
        such as 'Compare', 'American', 'Prepare', 'Declare', 'Square', etc.
        """
        valid_words = [
            "Compare",
            "American",
            "Prepare",
            "Declare",
            "Square",
            "Beware",
            "Welfare",
            "Metropolis",
            "Tennis",
            "Debris",
        ]
        for word in valid_words:
            result = normalize_extracted_text(word)
            assert result == word, (
                f"Word {word} should be preserved intact, but got {result}"
            )
        assert normalize_extracted_text("Howtousethis") == "How to use this"

    def test_reproduce_capitalized_glued_phrases_bypassing_dp(self):
        """
        DEFECT RESOLUTION: Long glued tokens starting with uppercase are
        properly segmented by DP segmenter while preserving leading capitalization.
        """
        title_glued = "Thethreecommonmethods"
        result = normalize_extracted_text(title_glued)
        assert result == "The three common methods"

    def test_reproduce_unrepaired_glued_tokens_due_to_vocab_gaps(self):
        """
        DEFECT RESOLUTION: DP segmenter resolves glued tokens with expanded vocabulary.
        """
        repaired = {
            "orexecutesinonesecond": "or executes in one second",
            "becauseitischeap": "because it is cheap",
            "dividesinstructionexecution": "divides instruction execution",
        }
        for token, expected in repaired.items():
            result = normalize_extracted_text(token)
            assert result == expected, f"Expected {token} -> {expected}, but got {result}"
