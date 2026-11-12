"""
File handling utilities for document processing.
"""

import io
import re
import unicodedata
import zipfile
from pathlib import PurePosixPath
from pathlib import Path
from typing import Tuple, Optional
import logging

from services.source_provenance import (
    EXTRACTION_VERSION,
    NORMALIZATION_MAP_VERSION,
    build_normalization_map,
)
from utils.text_metrics import count_words

logger = logging.getLogger(__name__)


def get_file_type(filename: str) -> str:
    """
    Determine the file type from filename extension.
    
    Args:
        filename: Name of the file
    
    Returns:
        File type string ('pdf', 'docx', or 'unknown')
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext == ".docx":
        return "docx"
    elif ext == ".doc":
        return "doc"
    return "unknown"


def validate_file_signature(file_content: bytes, filename: str) -> None:
    """Validate file signatures and bound DOCX archive expansion before parsing."""
    file_type = get_file_type(filename)
    if file_type == "pdf":
        if b"%PDF-" not in file_content[:1024]:
            raise ValueError("File content does not have a valid PDF signature")
        return

    if file_type != "docx":
        raise ValueError(f"Unsupported file type: {filename}")

    try:
        with zipfile.ZipFile(io.BytesIO(file_content)) as archive:
            entries = archive.infolist()
            if len(entries) > 10_000:
                raise ValueError("DOCX archive contains too many entries")

            uncompressed_total = 0
            for entry in entries:
                path = PurePosixPath(entry.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("DOCX archive contains an unsafe path")
                if entry.flag_bits & 0x1:
                    raise ValueError("Encrypted DOCX archives are not supported")
                if entry.file_size > 64 * 1024 * 1024:
                    raise ValueError("DOCX archive entry exceeds the 64 MB limit")
                uncompressed_total += entry.file_size
                if uncompressed_total > 256 * 1024 * 1024:
                    raise ValueError("DOCX archive exceeds the 256 MB uncompressed limit")
                if entry.file_size and entry.file_size / max(entry.compress_size, 1) > 1000:
                    raise ValueError("DOCX archive has an unsafe compression ratio")

            names = {entry.filename for entry in entries}
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise ValueError("Invalid DOCX file signature")
    except zipfile.BadZipFile as error:
        raise ValueError("Invalid DOCX file signature") from error


# ============================================================================
# Text Normalization & Acronym Repair Dictionaries
# ============================================================================

KNOWN_VALID_WORDS = {
    # Acronyms & technical abbreviations in lowercase
    'cpi', 'cpu', 'alu', 'ram', 'rom', 'risc', 'cisc', 'mips', 'amat', 'tlb', 'dma', 'bus',
    'sram', 'dram', 'raw', 'waw', 'war', 'isa',
    # Historically mangled words to never split
    'generally', 'usually', 'parallel', 'separate', 'generate', 'significance', 'significant',
    'terms', 'hazards', 'signals', 'small', 'pipeline', 'allows', 'they', 'relate', 'each',
    'other', 'derivation',
    # English/Greek/Latin words ending in -is or -sis
    'analysis', 'synthesis', 'basis', 'thesis', 'crisis', 'emphasis', 'hypothesis',
    'diagnosis', 'prognosis', 'genesis', 'osmosis', 'paralysis', 'dialysis', 'nemesis',
    'iris', 'paris', 'axis', 'ellipsis', 'parenthesis', 'metropolis', 'synopsis',
    # Common English words ending in -are, -can, -uses, -is (protect from TitleCase verb merger)
    'compare', 'declare', 'prepare', 'square', 'beware', 'welfare', 'middleware', 'malware',
    'healthcare', 'hardware', 'software',
    'american', 'african', 'republican', 'pelican', 'toucan',
    'clause', 'clauses', 'refuse', 'refuses', 'confuse', 'confuses', 'focus', 'focuses',
    'diffuse', 'diffuses', 'campus', 'campuses', 'virus', 'viruses', 'bonus', 'bonuses',
    'status', 'statuses', 'tennis', 'chassis', 'debris',
    # Common English compound words (protect from DP over-segmentation)
    'sometimes', 'somewhere', 'somehow', 'timeline', 'downtime', 'meantime', 'underway', 'runtime',
    # Additional technical vocabulary
    'cache', 'mapping', 'locality', 'reference', 'throughput', 'bandwidth', 'common', 'method',
    'methods', 'divides', 'executes', 'improves', 'short', 'logic', 'table',
    # Domain & technical vocabulary
    'computer', 'structure', 'function', 'register', 'registers',
    'speedup', 'virtual', 'buffer', 'interrupt', 'vectored', 'between', 'complex', 'program',
    'cycles', 'frequency', 'inverse', 'higher', 'variable', 'policy', 'direct', 'memory',
    'arithmetic', 'output', 'control', 'stores', 'accepts', 'results', 'formula', 'points',
    'concept', 'access', 'fields', 'single', 'hazard', 'branch', 'simple', 'fixed', 'total',
    'sheet', 'patterns', 'previous', 'papers', 'semester', 'syllabus', 'explain', 'algorithm',
    'remember', 'system', 'process', 'number', 'address', 'design', 'level', 'value', 'order',
    'blocks', 'input', 'state', 'unit', 'data', 'from', 'with', 'then', 'each', 'what',
    'mean', 'have', 'more', 'some', 'than', 'into', 'most', 'very', 'only', 'same', 'also',
    'over', 'both', 'even', 'used', 'such', 'make', 'part', 'line', 'last', 'exam', 'show',
    'rate', 'bits', 'sign', 'bias', 'rule', 'term', 'year', 'read', 'word', 'high', 'need',
    'well', 'much', 'load', 'must', 'seen', 'find', 'down', 'cost', 'take', 'come', 'made',
    'good', 'form', 'long', 'type', 'give', 'help', 'keep', 'hold', 'away', 'left', 'full',
    'late', 'fast', 'slow', 'main', 'five', 'time', 'hits', 'miss',
    'organization', 'architecture', 'functional', 'functionalblocks', 'instructions',
    'calculations', 'coordinates', 'multiplicand', 'pipelining', 'associative',
    'replacement', 'arbitration', 'asynchronous', 'synchronous', 'translation',
    'lookaside', 'structural', 'prediction', 'forwarding', 'definition',
    'conclusion', 'percentage', 'precision', 'operations', 'performed',
    'instruction', 'processor', 'microprocessor', 'controller', 'datapath', 'interconnect',
    'interconnection', 'semiconductor', 'transistor', 'propagation', 'circuit', 'digital',
    'analog', 'binary', 'hexadecimal', 'floating', 'integer', 'decode', 'fetch', 'store',
    'prefetch', 'subroutine', 'compilation', 'compiler', 'benchmark', 'multiprocessor',
    'coherence', 'consistency', 'synchronization', 'deadlock', 'starvation', 'scheduling',
    'multithreading', 'hyperthreading', 'superscalar', 'speculation', 'branching',
    'segmentation', 'paging', 'virtualization', 'representation', 'implementation',
    'configuration', 'specification', 'application', 'classification', 'performance',
    'capacity', 'efficiency', 'scalability', 'reliability',
    'availability', 'multiplier', 'multiplication', 'divider', 'division', 'adder', 'adders',
    'comparator', 'multiplexer', 'demultiplexer', 'encoder', 'decoder', 'flipflop', 'latch',
    'operating', 'thread', 'threads', 'execution', 'operand', 'operands', 'addressing',
    'opcode', 'firmware', 'important', 'different', 'following', 'necessary', 'particular',
    'possible', 'standard', 'available', 'appropriate', 'consistent', 'additional',
    'understanding', 'information', 'environment', 'development', 'management',
    'relationship', 'communication', 'description', 'consideration', 'distribution',
    'investigation', 'characteristic', 'transformation', 'recommendation', 'suitable',
    'simpler', 'faster', 'carried', 'carries', 'carrying', 'computes', 'computed',
    'computing', 'generated', 'generates', 'generating', 'propagated', 'propagates',
    'propagating', 'extension', 'secondary', 'storage', 'loading', 'required',
    'perform', 'performs', 'performed', 'performing',
    'show', 'shows', 'showed', 'showing',
    'give', 'gives', 'gave', 'given', 'giving',
    'load', 'loads', 'loaded', 'loading',
    'program', 'programs', 'programmed', 'programming',
    'larger', 'largest', 'large',
    'run', 'runs', 'running',
    'update', 'updates', 'updated', 'updating',
    'first', 'second', 'third',
    'late', 'later', 'latest',
    'replace', 'replaces', 'replaced', 'replacing',
    'spatial', 'temporal', 'locality',
    'parallel', 'generate', 'generates', 'generated',
    'propagate', 'propagates', 'propagated',
    'suitable', 'faster', 'cheaper', 'simpler',
    'tends', 'tended', 'tending', 'tend',
    'repeatedly', 'nearby', 'recent',
    'logical', 'physical', 'virtual', 'address', 'space'
}

SEGMENTATION_VOCAB = set(KNOWN_VALID_WORDS) | {
    'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'so', 'yet', 'if', 'then', 'else',
    'because', 'as', 'than', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
    'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'of', 'about', 'into', 'through',
    'after', 'before', 'above', 'below', 'under', 'between', 'without', 'within',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'done', 'doing', 'can', 'could', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must', 'use', 'uses', 'used', 'using', 'occur', 'occurs', 'occurred',
    'it', 'its', 'they', 'them', 'their', 'there', 'this', 'that',
    'execute', 'executes', 'executed', 'cheap', 'cheaper',
    'write', 'writes', 'writing', 'read', 'reads', 'reading',
    'reference', 'general', 'moves', 'moving', 'improves', 'throughput', 'divides',
    'done', 'short', 'concept', 'logic', 'carryout', 'direct', 'table', 'memory',
    'system', 'method', 'methods', 'common', 'cache', 'mapping', 'locality',
    'speed', 'carry', 'look', 'ahead', 'adder', 'signals', 'signal', 'terms', 'term',
    'hazards', 'hazard', 'pipeline', 'pipe', 'line', 'lines', 'step', 'steps',
    'faster', 'simpler', 'suitable', 'well', 'all', 'any', 'one', 'two', 'three', 'four',
    'five', 'first', 'second', 'third', 'last', 'new', 'old', 'good', 'bad', 'great',
    'small', 'large', 'high', 'low', 'fast', 'slow', 'main', 'secondary', 'storage',
    'pages', 'page', 'demand', 'operate', 'operates', 'load', 'loads', 'loading',
    'extension', 'required', 'requires', 'require', 'way', 'ways', 'time', 'times'
}

ACRONYM_REPAIRS = [
    (re.compile(r'\bRA\s+W(?![A-Z])'), 'RAW'),
    (re.compile(r'\bWA\s+W(?![A-Z])'), 'WAW'),
    (re.compile(r'\bWA\s+R(?![A-Z])'), 'WAR'),
    (re.compile(r'\bCP\s+I(?![A-Z])'), 'CPI'),
    (re.compile(r'\bCP\s+U(?![A-Z])'), 'CPU'),
    (re.compile(r'\bSRA\s+M(?![A-Z])'), 'SRAM'),
    (re.compile(r'\bDRA\s+M(?![A-Z])'), 'DRAM'),
    (re.compile(r'\bMIP\s+S(?![A-Z])'), 'MIPS'),
    (re.compile(r'\bAM\s+AT(?![A-Z])'), 'AMAT'),
    (re.compile(r'\bAMA\s+T(?![A-Z])'), 'AMAT'),
]

WORD_REPAIRS = [
    (re.compile(r'\bgener\s+all\s+y\b', re.IGNORECASE), 'generally'),
    (re.compile(r'\busu\s+all\s+y\b', re.IGNORECASE), 'usually'),
    (re.compile(r'\bpar\s+all\s+el\b', re.IGNORECASE), 'parallel'),
    (re.compile(r'\bsepa\s+rate\b', re.IGNORECASE), 'separate'),
    (re.compile(r'\bgene\s+rate\b', re.IGNORECASE), 'generate'),
    (re.compile(r'\bsign\s+ifi\s+can\s+ce\b', re.IGNORECASE), 'significance'),
    (re.compile(r'\bsign\s+ifi\s+can\s+ce(?=[a-z])', re.IGNORECASE), 'significance '),
    (re.compile(r'\bsign\s+ifi\s+can\s+t\b', re.IGNORECASE), 'significant'),
    (re.compile(r'\bterm\s+s\b', re.IGNORECASE), 'terms'),
    (re.compile(r'\bhazard\s+s\b', re.IGNORECASE), 'hazards'),
    (re.compile(r'\bsign\s+als\b', re.IGNORECASE), 'signals'),
    (re.compile(r'\bsm\s+all\b', re.IGNORECASE), 'small'),
    (re.compile(r'\bpipe\s+line\b', re.IGNORECASE), 'pipeline'),
    (re.compile(r'\ball\s+owsa\b', re.IGNORECASE), 'allows a'),
    (re.compile(r'\ball\s+ows\b', re.IGNORECASE), 'allows'),
    (re.compile(r'\bthe\s+yre\s+late\b', re.IGNORECASE), 'they relate'),
    (re.compile(r'\beach\s+o\s+the\s+r\b', re.IGNORECASE), 'each other'),
    (re.compile(r'\border\s+ivation\b', re.IGNORECASE), 'or derivation'),
    (re.compile(r'\bper\s+form\s+s\b', re.IGNORECASE), 'performs'),
    (re.compile(r'\bper\s+form\s+ed\b', re.IGNORECASE), 'performed'),
    (re.compile(r'\bper\s+form\b', re.IGNORECASE), 'perform'),
    (re.compile(r'\bshow\s+s\b', re.IGNORECASE), 'shows'),
    (re.compile(r'\bload\s+s\b', re.IGNORECASE), 'loads'),
    (re.compile(r'\bgive\s+s\b', re.IGNORECASE), 'gives'),
    (re.compile(r'\bprogram\s+s\b', re.IGNORECASE), 'programs'),
    (re.compile(r'\bto\s*run\b', re.IGNORECASE), 'to run'),
    (re.compile(r'\bspati\s+all\s+ocality\b', re.IGNORECASE), 'spatial locality'),
    (re.compile(r'\bspati\s+all\b', re.IGNORECASE), 'spatial'),
    (re.compile(r'\bmean\s+s\b', re.IGNORECASE), 'means'),
    (re.compile(r'\bdesign\s+s\b', re.IGNORECASE), 'designs'),
    (re.compile(r'\bexam\s+s\b', re.IGNORECASE), 'exams'),
    (re.compile(r'\bfast\s+er\s+but\b', re.IGNORECASE), 'faster but'),
    (re.compile(r'\bfast\s+er\b', re.IGNORECASE), 'faster'),
    (re.compile(r'\blate\s+r\s+when\b', re.IGNORECASE), 'later when'),
    (re.compile(r'\blate\s+r\b', re.IGNORECASE), 'later'),
]

def _make_word_replacer(rep: str):
    def replacer(m: re.Match) -> str:
        orig = m.group(0)
        if orig[0].isupper():
            return rep[0].upper() + rep[1:]
        return rep
    return replacer

KNOWN_ACRONYMS_PATTERN = re.compile(
    r'\b(RAW|WAW|WAR|CPI|SRAM|DRAM|MIPS|AMAT|TLB|DMA|ALU|CPU|ROM|RAM|RISC|CISC|IEEE|COA|ISA|FIFO|LRU)(?!(?:s\b|v\d))(?=[a-z])'
)

TITLE_VERB_MERGER_PATTERN = re.compile(
    r'\b([A-Z][a-z]{2,})(is|are|uses|computes|can|carries|shows|gives|means|contains)\b'
)

def _split_title_verb(match: re.Match) -> str:
    token = match.group(0)
    if token.lower() in KNOWN_VALID_WORDS:
        return token
    prefix = match.group(1).lower()
    if prefix in KNOWN_VALID_WORDS:
        return f"{match.group(1)} {match.group(2)}"
    return token

GLUED_RULES = [
    (re.compile(r'\bIna\b'), 'In a'),
    (re.compile(r'\bina\b'), 'in a'),
    (re.compile(r'\bWhatis\b'), 'What is'),
    (re.compile(r'\bwhatis\b'), 'what is'),
    (re.compile(r'\bHowis\b'), 'How is'),
    (re.compile(r'\bhowis\b'), 'how is'),
    (re.compile(r'\bWhatdo\b'), 'What do'),
    (re.compile(r'\bwhatdo\b'), 'what do'),
    (re.compile(r'\bHowdo\b'), 'How do'),
    (re.compile(r'\bhowdo\b'), 'how do'),
    (re.compile(r'\bItis\b'), 'It is'),
    (re.compile(r'\bitis\b'), 'it is'),
    (re.compile(r'\bThisis\b'), 'This is'),
    (re.compile(r'\bthisis\b'), 'this is'),
    (re.compile(r'\bThatis\b'), 'That is'),
    (re.compile(r'\bthatis\b'), 'that is'),
    (re.compile(r'\bThereis\b'), 'There is'),
    (re.compile(r'\bthereis\b'), 'there is'),
    (re.compile(r'\bHowto\b'), 'How to'),
    (re.compile(r'\bhowto\b'), 'how to'),
    (re.compile(r'\bHowmany\b'), 'How many'),
    (re.compile(r'\bhowmany\b'), 'how many'),
    (re.compile(r'\bNamethem\b'), 'Name them'),
    (re.compile(r'\bnamethem\b'), 'name them'),
    (re.compile(r'\bUsethisasa\b'), 'Use this as a'),
    (re.compile(r'\bcanoccur\b'), 'can occur'),
    (re.compile(r'\bsoitis\b'), 'so it is'),
    (re.compile(r'\bfasterbut\b'), 'faster but'),
    (re.compile(r'\bsimplerbut\b'), 'simpler but'),
    (re.compile(r'\bissuitable\b'), 'is suitable'),
    (re.compile(r'\bstepbystep\b'), 'step by step'),
    (re.compile(r'\baswell\b'), 'as well'),
    (re.compile(r'\bofa\b'), 'of a'),
    (re.compile(r'\btoa\b'), 'to a'),
    (re.compile(r'\bfora\b'), 'for a'),
    (re.compile(r'\bwitha\b'), 'with a'),
    (re.compile(r'\bfroma\b'), 'from a'),
    (re.compile(r'\banda\b'), 'and a'),
    (re.compile(r'\bInthe\b'), 'In the'),
    (re.compile(r'\binthe\b'), 'in the'),
    (re.compile(r'\bOfthe\b'), 'Of the'),
    (re.compile(r'\bofthe\b'), 'of the'),
    (re.compile(r'\bTothe\b'), 'To the'),
    (re.compile(r'\btothe\b'), 'to the'),
    (re.compile(r'\bForthe\b'), 'For the'),
    (re.compile(r'\bforthe\b'), 'for the'),
    (re.compile(r'\bOnthe\b'), 'On the'),
    (re.compile(r'\bonthe\b'), 'on the'),
]

def _segment_concatenated_token(token: str) -> str:
    """Dynamic programming word segmentation for long lowercase concatenated strings."""
    if token in KNOWN_VALID_WORDS:
        return token
    n = len(token)
    if n < 7:
        return token
    dp = [None] * (n + 1)
    dp[0] = (0, [])
    max_word_len = 24
    for i in range(1, n + 1):
        best_score = -1
        best_words = None
        min_j = max(0, i - max_word_len)
        for j in range(min_j, i):
            if dp[j] is not None:
                sub = token[j:i]
                if len(sub) == 1 and sub != 'a':
                    continue
                if sub in SEGMENTATION_VOCAB:
                    sub_score = len(sub) ** 2
                    prev_score, prev_words = dp[j]
                    total_score = prev_score + sub_score
                    if total_score > best_score:
                        best_score = total_score
                        best_words = prev_words + [sub]
        if best_words is not None:
            dp[i] = (best_score, best_words)
    if dp[n] is not None and len(dp[n][1]) > 1:
        return ' '.join(dp[n][1])
    return token

def _segment_long_words(match: re.Match) -> str:
    token = match.group(0)
    if token.isupper() or token.lower() in KNOWN_VALID_WORDS:
        return token
    is_title = token[0].isupper() and token[1:].islower()
    if is_title:
        segmented = _segment_concatenated_token(token.lower())
        if segmented != token.lower():
            return segmented[0].upper() + segmented[1:]
        return token
    if token.islower():
        return _segment_concatenated_token(token)
    return token


def normalize_extracted_text(text: str) -> str:
    """Normalize whitespace and separate words erroneously merged without spaces by PDF extractors."""
    if not text:
        return ""
    # Join a line-end hyphenation while leaving ordinary in-line hyphens intact.
    # The normalization offset map records the resulting source span as a
    # replacement, so citations remain anchored to the original extracted text.
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    text = unicodedata.normalize("NFC", text)
    # 1. Pre-repair historically mangled acronyms and words
    for pat, rep in ACRONYM_REPAIRS:
        text = pat.sub(rep, text)
    for pat, rep in WORD_REPAIRS:
        text = pat.sub(_make_word_replacer(rep), text)
        
    # 2. Separate known acronyms followed by lowercase words (e.g. RAWcanoccur -> RAW canoccur, MIPSand -> MIPS and)
    text = KNOWN_ACRONYMS_PATTERN.sub(r'\1 ', text)
    
    # 3. Separate Acronym block followed by Capital+lower (e.g. 'COAFinal' -> 'COA Final')
    # Protect plural acronyms (GPUs, CPUs) and version codes (ARMv8, IPv4) from being split
    text = re.sub(r'([A-Z]+)(?=[A-Z](?!s\b)(?!v\d)[a-z])', r'\1 ', text)
    
    # 4. Separate Acronym block of 2+ uppercase letters followed by lowercase (generic fallback)
    text = re.sub(r'([A-Z]{2,})(?!(?:s\b|v\d))(?=[a-z])', r'\1 ', text)
    
    # 5. CamelCase: lower or digit followed by Upper (AdderIs -> Adder Is)
    text = re.sub(r'([a-z0-9])(?=[A-Z])', r'\1 ', text)
    
    # 6. Separate section labels and digits, and numbers followed by words
    # Preserving hex constants (0xdeadbeef, 0xffff, 0xabc) and identifiers like w0, token0, r0
    text = re.sub(r'\b(Section|Chapter|Figure|Table|Page|Question|Part|Module)(\d+)\b', r'\1 \2', text)
    text = re.sub(r'(\d+)(?!(?:st|nd|rd|th)\b)(?![xX][0-9a-fA-F])([a-zA-Z]{3,})', r'\1 \2', text)
    
    # 7. Separate punctuation glued to text
    text = re.sub(r'([a-zA-Z0-9])([:;,\?!])([a-zA-Z0-9])', r'\1\2 \3', text)
    text = re.sub(r'([a-zA-Z])\(', r'\1 (', text)
    text = re.sub(r'\)([a-zA-Z])', r') \1', text)
    text = re.sub(r'(\d)\.([A-Za-z])', r'\1. \2', text)
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # 8. TitleCase + verb mergers (Adderis -> Adder is, Addercomputes -> Adder computes)
    text = TITLE_VERB_MERGER_PATTERN.sub(_split_title_verb, text)
    
    # 9. Glued function words & clitics
    for pat, rep in GLUED_RULES:
        text = pat.sub(rep, text)
        
    # 10. Dynamic programming word segmentation on long tokens (handles lower and TitleCase)
    text = re.sub(r'\b[A-Za-z]{7,}\b', _segment_long_words, text)
    
    # 11. Post-repair safety pass
    for pat, rep in ACRONYM_REPAIRS:
        text = pat.sub(rep, text)
    for pat, rep in WORD_REPAIRS:
        text = pat.sub(_make_word_replacer(rep), text)
        
    # Collapse consecutive spaces/tabs while preserving newlines
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_content: bytes) -> Tuple[str, dict]:
    """
    Extract text from PDF file content.
    
    Args:
        file_content: Raw PDF file bytes
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    try:
        import PyPDF2

        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)

        text_parts: list[str] = []
        pages: list[dict] = []
        cursor = 0
        metadata = {
            "page_count": len(reader.pages),
            "has_images": False,
            "extraction_method": "PyPDF2.PageObject.extract_text",
            "extraction_version": getattr(PyPDF2, "__version__", "unknown"),
        }

        for page_num, page in enumerate(reader.pages):
            warnings = [
                "layout_coordinates_unavailable",
                "reading_order_not_verified",
                "table_structure_not_detected",
                "multicolumn_order_not_verified",
                "header_footer_removal_not_applied",
            ]
            try:
                page_text = page.extract_text()
            except Exception as page_error:
                page_text = None
                warnings.append(f"page_text_extraction_failed:{type(page_error).__name__}")
                logger.warning("PDF page %d extraction failed: %s", page_num + 1, page_error)
            if page_num:
                text_parts.append("\n\n")
                cursor += 2
            start = cursor if page_text is not None else None
            page_text = page_text or ""
            text_parts.append(page_text)
            cursor += len(page_text)
            pages.append({
                "page_number": page_num + 1,
                "raw_text": page_text if page_text else None,
                "source_char_start": start if page_text else None,
                "source_char_end": cursor if page_text else None,
                "extraction_method": "PyPDF2.PageObject.extract_text",
                "extraction_version": getattr(PyPDF2, "__version__", "unknown"),
                "warnings": warnings if page_text else [*warnings, "page_text_unavailable"],
                "location_available": bool(page_text),
            })

        raw_text = "".join(text_parts)
        normalized_text = normalize_extracted_text(raw_text).replace("\x00", "")
        metadata["source_pages"] = pages
        metadata["extracted_page_count"] = sum(
            1 for page in pages
            if not any(w.startswith("page_text_extraction_failed:") for w in page["warnings"])
        )
        metadata["text_bearing_page_count"] = sum(1 for page in pages if (page["raw_text"] or "").strip())
        metadata["raw_word_count"] = count_words(raw_text)
        metadata["normalized_word_count"] = count_words(normalized_text)
        metadata["source_provenance"] = {
            "schema_version": 1,
            "extraction_version": EXTRACTION_VERSION,
            "extraction_method": "PyPDF2.PageObject.extract_text",
            "normalization_version": NORMALIZATION_MAP_VERSION,
            "location_kind": "pdf_extracted_page_text",
            "offset_map": build_normalization_map(raw_text, normalized_text),
        }
        
        if reader.metadata:
            metadata["title"] = reader.metadata.get("/Title", "")
            metadata["author"] = reader.metadata.get("/Author", "")
        
        logger.info("Extracted text from %d PDF pages", len(reader.pages))
        return normalized_text, metadata
        
    except ImportError:
        logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        raise ImportError("PyPDF2 is required for PDF processing")
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise


def extract_text_from_docx(file_content: bytes) -> Tuple[str, dict]:
    """
    Extract text from DOCX file content.
    
    Args:
        file_content: Raw DOCX file bytes
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    try:
        from docx import Document
        
        doc_file = io.BytesIO(file_content)
        try:
            doc = Document(doc_file)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid DOCX document: {str(e)}")
        
        text_parts: list[str] = []
        blocks: list[dict] = []
        cursor = 0
        metadata = {
            "paragraph_count": len(doc.paragraphs),
            "has_tables": len(doc.tables) > 0
        }
        
        def append_block(block_type: str, block_index: int, value: str) -> None:
            nonlocal cursor
            if not value.strip():
                return
            if text_parts:
                text_parts.append("\n\n")
                cursor += 2
            start = cursor
            text_parts.append(value)
            cursor += len(value)
            blocks.append({
                "block_id": f"{block_type}:{block_index}",
                "block_type": block_type,
                "block_index": block_index,
                "raw_text": value,
                "source_char_start": start,
                "source_char_end": cursor,
                "location_available": True,
                "warnings": [],
            })

        for index, para in enumerate(doc.paragraphs):
            append_block("paragraph", index, para.text)
        
        for table_index, table in enumerate(doc.tables):
            for row_index, row in enumerate(table.rows):
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                append_block("table_row", table_index * 100000 + row_index, row_text)
        
        raw_text = "".join(text_parts)
        full_text = normalize_extracted_text(raw_text).replace("\x00", "")
        metadata["raw_word_count"] = count_words(raw_text)
        metadata["normalized_word_count"] = count_words(full_text)
        metadata["source_blocks"] = blocks
        metadata["extraction_method"] = "python-docx.paragraphs-and-table-rows"
        metadata["extraction_version"] = "python-docx"
        metadata["source_provenance"] = {
            "schema_version": 1,
            "extraction_version": EXTRACTION_VERSION,
            "extraction_method": metadata["extraction_method"],
            "normalization_version": NORMALIZATION_MAP_VERSION,
            "location_kind": "docx_paragraph_table_block",
            "offset_map": build_normalization_map(raw_text, full_text),
        }
        
        core_props = doc.core_properties
        metadata["title"] = core_props.title or ""
        metadata["author"] = core_props.author or ""
        
        logger.info(f"Extracted text from DOCX with {len(doc.paragraphs)} paragraphs")
        return full_text, metadata
        
    except ImportError:
        logger.error("python-docx not installed. Install with: pip install python-docx")
        raise ImportError("python-docx is required for DOCX processing")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {str(e)}")
        raise ValueError(f"Corrupted or invalid DOCX document: {str(e)}")


def extract_text(file_content: bytes, filename: str) -> Tuple[str, dict]:
    """
    Extract text from file based on file type.
    
    Args:
        file_content: Raw file bytes
        filename: Name of the file
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    
    Raises:
        ValueError: If file type is not supported
    """
    file_type = get_file_type(filename)
    validate_file_signature(file_content, filename)
    
    if file_type == "pdf":
        text, metadata = extract_text_from_pdf(file_content)
    elif file_type in ("docx", "doc"):
        text, metadata = extract_text_from_docx(file_content)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Keep the public extraction boundary safe even when an extractor plugin
    # or adapter returns text that bypassed the built-in format-specific
    # sanitizers.
    text = text.replace("\x00", "")
    
    return text, metadata
