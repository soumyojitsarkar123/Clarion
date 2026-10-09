"""
Evidence Verifier for Closed-Domain Document Knowledge Graphs.

Extracts sentence-level evidence for candidate relations from document chunks
and assesses lexical/syntactic evidence support and sentence grounding.

NOTE ON SCIENTIFIC TERMINOLOGY:
This module implements rule-based lexical evidence support and sentence-level
grounding. It does NOT use a trained Natural Language Inference (NLI) model.
All metrics derived from this module reflect lexical and pattern-based evidence
support, NOT deep semantic entailment.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Lexical predicate indicators for closed-domain relation types
RELATION_INDICATOR_PATTERNS = {
    "prerequisite": [
        r"\brequir(e|es|ed|ing)\b",
        r"\bprerequisite\b",
        r"\bdepend(s|ed|ing)? on\b",
        r"\bneed(s|ed)? (first|before)\b",
        r"\bbased on\b",
        r"\bfoundation for\b",
        r"\bprior to\b",
        r"\bbefore\b",
    ],
    "definition": [
        r"\bdefined as\b",
        r"\brefers to\b",
        r"\bis a\b",
        r"\bmeans\b",
        r"\bformally defined\b",
        r"\bdenotes\b",
    ],
    "explanation": [
        r"\bexplain(s|ed|ing)?\b",
        r"\bdescrib(e|es|ed|ing)\b",
        r"\bprovid(e|es|ed)? (a formula|an explanation|details)\b",
        r"\billustrat(e|es|ed)\b",
        r"\bshow(s|ed)? that\b",
    ],
    "cause-effect": [
        r"\bcause(s|ed|ing)?\b",
        r"\bresult(s|ed|ing)? in\b",
        r"\blead(s|ing)? to\b",
        r"\bproduc(e|es|ed|ing)\b",
        r"\bgive(s)? rise to\b",
        r"\bconsequently\b",
    ],
    "part-of": [
        r"\bpart of\b",
        r"\bcomponent of\b",
        r"\bconsist(s|ed)? of\b",
        r"\bcompris(e|es|ed)\b",
        r"\binclud(e|es|ed|ing)\b",
        r"\bcontain(s|ed|ing)?\b",
    ],
    "derives-from": [
        r"\bderiv(e|es|ed|ing)? from\b",
        r"\bcalculat(ed)? from\b",
        r"\bcomput(ed)? from\b",
        r"\bfollow(s)? from\b",
        r"\bobtain(ed)? from\b",
    ],
    "example-of": [
        r"\bexample of\b",
        r"\binstance of\b",
        r"\bsuch as\b",
        r"\bfor example\b",
        r"\be\.g\.\b",
    ],
    "similar-to": [
        r"\bsimilar to\b",
        r"\banalogous to\b",
        r"\blike\b",
        r"\bresembl(e|es|ed)\b",
        r"\bequivalent to\b",
    ],
}

# Negation and contradiction patterns
NEGATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bnot\b",
        r"\bnever\b",
        r"\bno longer\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdoes not\b",
        r"\bdoesn't\b",
        r"\bdid not\b",
        r"\bdidn't\b",
        r"\bis not\b",
        r"\bisn't\b",
        r"\bare not\b",
        r"\baren't\b",
        r"\bwas not\b",
        r"\bwasn't\b",
        r"\bfails? to\b",
        r"\bno evidence\b",
        r"\bcontradict(s|ed)?\b",
        r"\bdisprov(e|es|ed)\b",
        r"\bunrelated\b",
    ]
]

# Contrastive / independent clause boundary markers
CONTRASTIVE_MARKERS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhereas\b",
        r"\bwhile\b",
        r"\bhowever\b",
        r"\balthough\b",
        r"\bthough\b",
        r"\bin contrast\b",
        r"\bconversely\b",
        r"\bon the other hand\b",
        r"\bmeanwhile\b",
        r"\bnevertheless\b",
    ]
]


@dataclass
class EvidenceExtractionResult:
    """Detailed result of evidence sentence extraction."""
    concept_a: str
    concept_b: str
    relation_type: str
    best_sentence: str
    all_matching_sentences: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    cooccurrence_found: bool = False
    lexical_support_score: float = 0.0
    pattern_matches: List[str] = field(default_factory=list)
    method: str = "lexical_grounding"

    @property
    def evidence_support_score(self) -> float:
        """Scientifically accurate alias for lexical_support_score."""
        return self.lexical_support_score


class EvidenceVerifier:
    """
    Extracts and scores textual evidence for relation validation within closed-domain documents.
    Operates on rule-based lexical matching and syntactic heuristics.
    """

    def __init__(self):
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for rel_type, patterns in RELATION_INDICATOR_PATTERNS.items():
            self._compiled_patterns[rel_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly using standard punctuation boundaries."""
        if not text:
            return []
        cleaned = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    @staticmethod
    def _extract_chunk_id(chunk: Any) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("id") or chunk.get("chunk_id") or "")
        return str(getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None) or "")

    @staticmethod
    def _extract_chunk_text(chunk: Any) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("text") or chunk.get("content") or "")
        return str(getattr(chunk, "content", None) or getattr(chunk, "text", None) or "")

    def extract_evidence(
        self,
        concept_a: str,
        concept_b: str,
        relation_type: str,
        chunks: Sequence[Any],
    ) -> EvidenceExtractionResult:
        """
        Extract the best evidence sentence and matching chunk IDs from source chunks.
        """
        ca_lower = concept_a.strip().lower()
        cb_lower = concept_b.strip().lower()
        patterns = self._compiled_patterns.get(relation_type.lower(), [])

        best_sentence = ""
        best_score = -1.0
        matching_chunk_ids: List[str] = []
        all_matches: List[str] = []
        cooccurrence_found = False
        matched_patterns: List[str] = []

        # 1. Search chunk by chunk, sentence by sentence
        for chunk in chunks:
            chunk_id = self._extract_chunk_id(chunk)
            chunk_text = self._extract_chunk_text(chunk)
            if not chunk_text:
                continue

            sentences = self.split_sentences(chunk_text)
            for sent in sentences:
                sent_lower = sent.lower()
                has_a = ca_lower in sent_lower
                has_b = cb_lower in sent_lower

                if not (has_a or has_b):
                    continue

                if has_a and has_b:
                    cooccurrence_found = True
                    all_matches.append(sent)
                    if chunk_id and chunk_id not in matching_chunk_ids:
                        matching_chunk_ids.append(str(chunk_id))

                    score = self.verify_support(concept_a, concept_b, relation_type, sent)
                    curr_matched_patterns = [
                        pat.pattern for pat in patterns if pat.search(sent)
                    ]

                    if score > best_score:
                        best_score = score
                        best_sentence = sent
                        matched_patterns = curr_matched_patterns

        # 2. Chunk-level fallback if concepts span adjacent sentences
        if not cooccurrence_found:
            for chunk in chunks:
                chunk_id = self._extract_chunk_id(chunk)
                chunk_text = self._extract_chunk_text(chunk)
                if not chunk_text:
                    continue
                ct_lower = chunk_text.lower()
                if ca_lower in ct_lower and cb_lower in ct_lower:
                    cooccurrence_found = True
                    if chunk_id and str(chunk_id) not in matching_chunk_ids:
                        matching_chunk_ids.append(str(chunk_id))
                    sentences = self.split_sentences(chunk_text)
                    relevant_sents = [s for s in sentences if (ca_lower in s.lower() or cb_lower in s.lower())]
                    if relevant_sents:
                        all_matches.extend(relevant_sents)
                        candidate_text = " ".join(relevant_sents[:2])
                        cand_score = 0.35
                        cand_patterns = [pat.pattern for pat in patterns if pat.search(candidate_text)]
                        if cand_patterns:
                            cand_score += 0.35
                        if cand_score > best_score:
                            best_score = cand_score
                            best_sentence = candidate_text
                            matched_patterns = cand_patterns

        normalized_score = round(max(0.0, min(1.0, best_score if best_score > 0 else 0.0)), 4)

        return EvidenceExtractionResult(
            concept_a=concept_a,
            concept_b=concept_b,
            relation_type=relation_type,
            best_sentence=best_sentence,
            all_matching_sentences=all_matches,
            source_chunk_ids=matching_chunk_ids,
            cooccurrence_found=cooccurrence_found,
            lexical_support_score=normalized_score,
            pattern_matches=matched_patterns,
            method="lexical_grounding",
        )

    def verify_support(
        self,
        concept_a: str,
        concept_b: str,
        relation_type: str,
        evidence_text: str,
    ) -> float:
        """
        Compute a rule-based lexical evidence support score for an evidence excerpt.

        Evaluates:
        1. Strict sentence co-occurrence (both concepts must be present).
        2. Negation/contradiction detection (penalizes negated assertions).
        3. Contrastive/unrelated clause boundary detection.
        4. Relational indicator pattern matching for the specific relation type.
        5. Directional alignment for asymmetric relations.

        Returns:
            Float support score in [0.0, 1.0]
        """
        if not evidence_text or not evidence_text.strip():
            return 0.0

        ev_lower = evidence_text.lower()
        ca_lower = concept_a.strip().lower()
        cb_lower = concept_b.strip().lower()
        rel_lower = relation_type.strip().lower()

        # Hard Rule 4: Both concepts must be mentioned in evidence
        has_a = ca_lower in ev_lower
        has_b = cb_lower in ev_lower
        if not (has_a and has_b):
            return 0.0

        # Locating concept positions
        pos_a = ev_lower.find(ca_lower)
        pos_b = ev_lower.find(cb_lower)
        first_pos = min(pos_a, pos_b)
        first_len = len(ca_lower) if pos_a < pos_b else len(cb_lower)
        second_pos = max(pos_a, pos_b)
        between_span = ev_lower[first_pos + first_len : second_pos]

        # Hard Rule 1 & 6: Negation and contradiction detection
        for neg_pat in NEGATION_PATTERNS:
            if neg_pat.search(between_span):
                # Negation between concepts directly negates the relationship
                return 0.0
            if neg_pat.search(ev_lower):
                # Check if negation precedes the first concept or immediately modifies the sentence
                neg_match = neg_pat.search(ev_lower)
                if neg_match and neg_match.start() < first_pos and (first_pos - neg_match.end() < 30):
                    return 0.0

        # Hard Rule 3: Unrelated contrasting clauses
        for con_pat in CONTRASTIVE_MARKERS:
            if con_pat.search(between_span):
                return 0.05
            if con_pat.search(ev_lower[:first_pos]) and ("," in between_span or ";" in between_span):
                return 0.05

        # Base co-occurrence score
        base_score = 0.35

        # Hard Rule 7: Relation-type specific pattern match
        patterns = self._compiled_patterns.get(rel_lower, [])
        matched_target_pattern = False
        for pat in patterns:
            if pat.search(evidence_text):
                base_score += 0.35
                matched_target_pattern = True
                break

        # Check if text matches an incompatible rival relation type
        if not matched_target_pattern:
            for other_rel, other_patterns in self._compiled_patterns.items():
                if other_rel == rel_lower:
                    continue
                for pat in other_patterns:
                    if pat.search(between_span):
                        # Strong indicator for a different relation type in the between span
                        base_score -= 0.15
                        break

        # Hard Rule 2: Directional alignment for asymmetric relations
        # For candidate relation (A, rel, B):
        if rel_lower == "prerequisite":
            # A is prerequisite for B (A comes first).
            # Patterns like "B requires A" or "B depends on A" mean A is before B.
            # If text says "A requires B", then B is prerequisite for A (inversion!).
            if pos_a < pos_b:
                # Text order: A ... B
                # If text says "A requires B", direction is inverted!
                if re.search(r"\b(requir(es|ed|ing)?|depend(s|ed)? on|need(s|ed)?)\b", between_span):
                    return 0.10  # Inversion penalty
                elif re.search(r"\b(is a prerequisite for|foundation for|prior to|before)\b", between_span):
                    base_score += 0.30
                else:
                    base_score += 0.10
            else:
                # Text order: B ... A
                # If text says "B requires A" or "B depends on A", this supports candidate A -> B!
                if re.search(r"\b(requir(es|ed|ing)?|depend(s|ed)? on|need(s|ed)?|based on)\b", between_span):
                    base_score += 0.30
                else:
                    base_score -= 0.10
        elif rel_lower in {"cause-effect", "derives-from"}:
            if pos_a < pos_b:
                if re.search(r"\b(cause(s|ed)?|result(s|ed)? in|lead(s)? to)\b", between_span):
                    base_score += 0.30
            else:
                if re.search(r"\b(deriv(es|ed)? from|calculat(ed)? from|result(s)? of)\b", between_span):
                    base_score += 0.30
        else:
            base_score += 0.10

        return round(max(0.0, min(1.0, base_score)), 4)
