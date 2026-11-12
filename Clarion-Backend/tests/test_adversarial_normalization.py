"""
Adversarial Stress Test Suite for Milestone 1: Text Extraction & Normalization.
Authored by Challenger 1 (Milestone 1).

Tests:
1. Corner cases: empty strings, uppercase strings, single characters, punctuation, mixed alphanumeric codes.
2. Known word protection: ensure long technical words are never damaged.
3. Glued phrase separation: function words, nouns, and verbs.
4. Chunking boundary tests: headings and section splitting on normalized text.
5. Regressions: TitleCase false positives, plural acronym mangling, and alphanumeric code preservation.
"""

import pytest
from utils.file_handler import normalize_extracted_text
from services.chunking_service import ChunkingService


# ============================================================================
# 1. Corner Cases
# ============================================================================

def test_corner_cases_empty_and_whitespace():
    """Empty and whitespace strings must normalize cleanly to empty strings."""
    assert normalize_extracted_text("") == ""
    assert normalize_extracted_text("   ") == ""
    assert normalize_extracted_text("\t\n\r  \n") == ""
    assert normalize_extracted_text(None) == ""


def test_corner_cases_single_characters():
    """Single characters of various types must not be altered or crash."""
    assert normalize_extracted_text("a") == "a"
    assert normalize_extracted_text("A") == "A"
    assert normalize_extracted_text("1") == "1"
    assert normalize_extracted_text(".") == "."
    assert normalize_extracted_text("?") == "?"
    assert normalize_extracted_text("-") == "-"


def test_corner_cases_purely_uppercase():
    """Purely uppercase phrases must not be split into spurious fragments."""
    assert normalize_extracted_text("MICROPROCESSOR") == "MICROPROCESSOR"
    assert normalize_extracted_text("ADVANCED COMPUTER ARCHITECTURE") == "ADVANCED COMPUTER ARCHITECTURE"
    assert normalize_extracted_text("SECTION 1: INTRODUCTION") == "SECTION 1: INTRODUCTION"


def test_corner_cases_punctuation_heavy():
    """Punctuation sequences should be handled safely."""
    assert normalize_extracted_text("... (parentheses) [brackets] {braces} -- -") == "... (parentheses) [brackets] {braces} -- -"
    # Heading colon separation
    assert normalize_extracted_text("Section 1:Introduction") == "Section 1: Introduction"
    assert normalize_extracted_text("Chapter 2:Memory") == "Chapter 2: Memory"


def test_corner_cases_mixed_alphanumeric_codes():
    """Hardware and protocol codes must remain intact."""
    assert normalize_extracted_text("x86_64") == "x86_64"
    assert normalize_extracted_text("IEEE-754") == "IEEE-754"
    assert normalize_extracted_text("L1/L2") == "L1/L2"
    assert normalize_extracted_text("MIPS32") == "MIPS32"
    assert normalize_extracted_text("MIPS-32") == "MIPS-32"
    assert normalize_extracted_text("x86-64") == "x86-64"
    assert normalize_extracted_text("RISC-V") == "RISC-V"
    assert normalize_extracted_text("UTF-8") == "UTF-8"
    assert normalize_extracted_text("RS-232") == "RS-232"
    assert normalize_extracted_text("0xdeadbeef") == "0xdeadbeef"
    assert normalize_extracted_text("0xffff") == "0xffff"
    assert normalize_extracted_text("0xabc") == "0xabc"
    assert normalize_extracted_text("The risc isa is simple") == "The risc isa is simple"


# ============================================================================
# 2. Known Word Protection
# ============================================================================

def test_known_word_protection_long_technical_words():
    """Long technical words specified in dispatch must never be decomposed."""
    long_words = [
        "multiprocessor",
        "synchronization",
        "microarchitecture",
        "programmable",
        "interconnection",
        "pipelining",
        "multiplicand",
        "superscalar",
        "hyperthreading",
        "virtualization",
        "demultiplexer",
        "microprocessor",
    ]
    for word in long_words:
        assert normalize_extracted_text(word) == word, f"Word '{word}' was corrupted to '{normalize_extracted_text(word)}'"


def test_known_word_protection_greek_latin_is_words():
    """English words ending in -is or -sis must never be split by verb merger regex."""
    is_words = [
        "Analysis",
        "Synthesis",
        "Basis",
        "Thesis",
        "Crisis",
        "Emphasis",
        "Hypothesis",
        "Diagnosis",
        "Genesis",
    ]
    for word in is_words:
        assert normalize_extracted_text(word) == word, f"Word '{word}' was corrupted to '{normalize_extracted_text(word)}'"


# ============================================================================
# 3. Glued Phrase Splitting
# ============================================================================

def test_glued_phrases_mandated_targets():
    """Target phrases from ORIGINAL_REQUEST § R3 must normalize cleanly."""
    assert normalize_extracted_text("Ina simple") == "In a simple"
    assert normalize_extracted_text("WAWcanoccur") == "WAW can occur"
    assert normalize_extracted_text("RAWcanoccur") == "RAW can occur"
    assert normalize_extracted_text("WARcanoccur") == "WAR can occur"
    assert normalize_extracted_text("Carry Look Ahead Adderis used whenspeedisimportant") == "Carry Look Ahead Adder is used when speed is important"
    assert normalize_extracted_text("Whatis the answer") == "What is the answer"
    assert normalize_extracted_text("Itis clear that") == "It is clear that"
    assert normalize_extracted_text("Thisis important") == "This is important"


def test_glued_phrases_clitics():
    """Clitic phrases like 'ofa', 'toa', 'fora', 'witha' must be split."""
    assert normalize_extracted_text("part ofa system") == "part of a system"
    assert normalize_extracted_text("send toa node") == "send to a node"
    assert normalize_extracted_text("used fora cache") == "used for a cache"
    assert normalize_extracted_text("built witha bus") == "built with a bus"
    assert normalize_extracted_text("data froma port") == "data from a port"
    assert normalize_extracted_text("both anda signal") == "both and a signal"


# ============================================================================
# 4. Chunking Boundary Tests
# ============================================================================

def test_chunking_with_normalized_glued_headings(isolated_settings):
    """ChunkingService must detect section headings that were originally glued in PDF text."""
    service = ChunkingService()
    raw_pdf_text = (
        "Section1:ProcessorArchitecture\n"
        "Ina simple pipelined CPU, WAWcanoccur and RAWcanoccur.\n"
        "Carry Look Ahead Adderis used whenspeedisimportant.\n\n"
        "Section2:MemoryOrganization\n"
        "SRAM and DRAM provide different latency and capacity trade-offs."
    )
    chunks = service.chunk_document("doc-adversarial-1", raw_pdf_text)
    assert len(chunks) >= 1
    # Check that Section 1 heading was detected
    assert any("Processor Architecture" in (c.section_title or "") for c in chunks)
    # Check that normalized text in chunks has no glued words
    for c in chunks:
        assert "Ina" not in c.content
        assert "WAWcanoccur" not in c.content
        assert "RAWcanoccur" not in c.content
        assert "Adderis" not in c.content
        assert "whenspeedisimportant" not in c.content


def test_chunking_boundary_clean_word_counts(isolated_settings):
    """Chunk word counts and slices must remain valid after normalization."""
    isolated_settings.chunk_size = 30
    isolated_settings.chunk_overlap = 5
    isolated_settings.min_chunk_size = 10
    service = ChunkingService()
    
    text = "# Section 1\n" + " ".join(["word"] * 50) + "\n# Section 2\n" + " ".join(["term"] * 40)
    chunks = service.chunk_document("doc-boundary-1", text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.word_count >= 10
        assert chunk.document_id == "doc-boundary-1"


# ============================================================================
# 5. Adversarial Flaw Verifications (Now Resolved & Passing)
# ============================================================================

def test_flaw_titlecase_verb_merger_false_positives():
    """
    TITLE_VERB_MERGER_PATTERN aggressively splits standard English words ending in 'are', 'can', 'uses', 'is':
    - 'Compare' -> 'Comp are'
    - 'Declare' -> 'Decl are'
    - 'Prepare' -> 'Prep are'
    - 'Square' -> 'Squ are'
    - 'American' (e.g. ASCII) -> 'Ameri can'
    - 'Clauses' -> 'Cla uses'
    - 'Refuses' -> 'Ref uses'
    - 'Confuses' -> 'Conf uses'
    """
    assert normalize_extracted_text("Compare") == "Compare"
    assert normalize_extracted_text("Declare") == "Declare"
    assert normalize_extracted_text("Prepare") == "Prepare"
    assert normalize_extracted_text("Square") == "Square"
    assert normalize_extracted_text("American") == "American"
    assert normalize_extracted_text("Clauses") == "Clauses"
    assert normalize_extracted_text("Refuses") == "Refuses"
    assert normalize_extracted_text("Confuses") == "Confuses"


def test_flaw_plural_acronym_and_arch_code_mangling():
    """
    Regex `([A-Z]+)(?=[A-Z][a-z])` and `([A-Z]{2,})(?=[a-z])` splits plural acronyms and codes:
    - 'GPUs' -> 'GP Us'
    - 'PDFs' -> 'PD Fs'
    - 'APIs' -> 'AP Is'
    - 'URLs' -> 'UR Ls'
    - 'CPUs' -> 'CPU s'
    - 'FPGAs' -> 'FPG As'
    - 'ARMv8' -> 'AR Mv8'
    - 'IPv4' -> 'I Pv4'
    - 'IPv6' -> 'I Pv6'
    """
    assert normalize_extracted_text("GPUs") == "GPUs"
    assert normalize_extracted_text("PDFs") == "PDFs"
    assert normalize_extracted_text("APIs") == "APIs"
    assert normalize_extracted_text("URLs") == "URLs"
    assert normalize_extracted_text("CPUs") == "CPUs"
    assert normalize_extracted_text("FPGAs") == "FPGAs"
    assert normalize_extracted_text("ARMv8") == "ARMv8"
    assert normalize_extracted_text("IPv4") == "IPv4"
    assert normalize_extracted_text("IPv6") == "IPv6"


def test_flaw_common_prepositions_with_the():
    """
    Common glued prepositions with 'the' fail because length < 7 and no explicit rules:
    - 'inthe' -> 'in the'
    - 'ofthe' -> 'of the'
    - 'tothe' -> 'to the'
    - 'forthe' -> 'for the'
    - 'onthe' -> 'on the'
    """
    assert normalize_extracted_text("inthe") == "in the"
    assert normalize_extracted_text("ofthe") == "of the"
    assert normalize_extracted_text("tothe") == "to the"
    assert normalize_extracted_text("forthe") == "for the"
    assert normalize_extracted_text("onthe") == "on the"
