"""
Summary Service - generates structured summaries from documents.
"""

import uuid
import sqlite3
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence
import logging

from models.chunk import Chunk
from models.summary import StructuredSummary, SummarySection
from .knowledge_map_service import LLMInterface
from utils.config import settings

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "carries",
    "does", "each", "for", "from", "given", "has", "have", "how", "if", "in", "into",
    "is", "it", "its", "more", "most", "not", "of", "on", "only", "or", "that", "the",
    "their", "them", "these", "this", "those", "through", "to", "using", "was", "what",
    "when", "where", "which", "with", "would", "your", "you", "question", "questions",
    "attempt", "marks", "mark", "page", "part", "section", "document", "text", "summary",
    "response", "json", "analysis", "chapter", "topic", "topics", "degree", "stream",
    "subject", "code", "duration", "semester", "examination", "november", "january",
    "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "december", "kolkata", "institute", "engineering", "management",
    "town", "full", "hour", "hours", "term", "paper", "btech", "cse", "2024", "2025",
    "2026", "ll", "iii", "ii", "iv", "distinguish", "explain", "define", "find",
    "construct", "support", "set", "drawn", "taken", "shows", "show", "using",
    "acceptable", "obtained", "randomly", "would", "given", "question", "questions",
    "suppose",
}

_SUMMARY_GENERIC_TOPICS = {
    "document",
    "section",
    "question",
    "questions",
    "attempt",
    "marks",
    "page",
    "part",
    "topic",
    "topics",
    "text",
    "summary",
    "paper",
}


class SummaryService:
    """
    Service for generating structured hierarchical summaries.
    Creates summaries organized by conceptual hierarchy rather than paragraphs.
    """
    
    def __init__(self):
        self.llm = LLMInterface()
        self._prefer_deterministic = False
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for summaries."""
        db_path = settings.data_dir / "clarion.db"
        self.db_path = str(db_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                document_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_summary(
        self,
        document_id: str,
        title: str,
        chunks: List[Chunk]
    ) -> StructuredSummary:
        """
        Generate structured summary for a document.
        
        Args:
            document_id: Document identifier
            title: Document title
            chunks: Document chunks
        
        Returns:
            StructuredSummary object
        """
        logger.info(f"Generating structured summary for document {document_id}")
        self._prefer_deterministic = False
        
        sections = self._generate_sections(chunks)
        llm_summary_result = self._generate_overall_summary(title=title, chunks=chunks, sections=sections)
        overall_summary = llm_summary_result["text"]
        llm_summary_accepted = bool(llm_summary_result["llm_used"])
        
        summary = StructuredSummary(
            document_id=document_id,
            title=title,
            overall_summary=overall_summary,
            sections=sections,
            metadata={
                "chunk_count": len(chunks),
                "section_count": len(sections),
                "llm_summary_accepted": llm_summary_accepted,
                "llm_last_error": self.llm.last_error,
            }
        )
        
        self._save_summary(summary)
        
        logger.info(f"Summary generated with {len(sections)} sections")
        return summary
    
    def _generate_sections(self, chunks: List[Chunk]) -> List[SummarySection]:
        """Generate hierarchical sections from chunks."""
        sections: List[SummarySection] = []
        grouped_chunks = self._group_chunks_by_section(chunks)

        for index, group in enumerate(grouped_chunks):
            title = self._resolve_section_title(group, index)
            section_text = "\n".join(
                [self._prepare_text_for_analysis(chunk.content) for chunk in group if chunk.content]
            ).strip()
            if not section_text:
                continue

            section_id = str(uuid.uuid4())
            sections.append(
                SummarySection(
                    id=section_id,
                    title=title,
                    level=1,
                    summary=self._summarize_section(section_text, title=title),
                    key_points=self._extract_key_points(section_text, title=title),
                    related_concepts=self._extract_related_concepts(section_text),
                    child_sections=[],
                )
            )
        
        return sections

    def _generate_overall_summary(
        self,
        title: str,
        chunks: List[Chunk],
        sections: Sequence[SummarySection],
    ) -> Dict[str, Any]:
        """Generate a concise overall summary, falling back to deterministic extraction."""
        return {
            "text": self._build_fallback_overall_summary(title=title, chunks=chunks, sections=sections),
            "llm_used": False,
        }

    def _summarize_section(self, text: str, title: Optional[str] = None) -> str:
        """Generate a brief summary of section text."""
        return self._build_fallback_section_summary(text, title=title)
    
    def _extract_key_points(self, text: str, title: Optional[str] = None) -> List[str]:
        """Extract key points from text."""
        return self._build_fallback_key_points(text)

    def _group_chunks_by_section(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """Group consecutive chunks that appear to belong to the same section."""
        groups: List[List[Chunk]] = []
        current_group: List[Chunk] = []
        current_key = None

        for index, chunk in enumerate(chunks):
            title = self._clean_title(chunk.section_title or "")
            chunk_text = self._prepare_text_for_analysis(chunk.content)
            derived_title = title or self._derive_topic_label(chunk_text, fallback=f"Section {index + 1}")
            grouping_key = derived_title.lower()

            if current_group and grouping_key != current_key:
                groups.append(current_group)
                current_group = []

            current_group.append(chunk)
            current_key = grouping_key

        if current_group:
            groups.append(current_group)

        return groups

    def _resolve_section_title(self, group: Sequence[Chunk], index: int) -> str:
        """Choose a stable human-readable title for a section group."""
        candidates = [
            self._clean_title(chunk.section_title or "")
            for chunk in group
            if self._clean_title(chunk.section_title or "")
        ]
        if candidates:
            return candidates[0]

        combined_text = "\n".join([chunk.content for chunk in group if chunk.content])
        return self._derive_topic_label(combined_text, fallback=f"Section {index + 1}")

    def _build_fallback_overall_summary(
        self,
        title: str,
        chunks: List[Chunk],
        sections: Sequence[SummarySection],
    ) -> str:
        """Build a deterministic overall summary from the cleanest extracted document text."""
        combined_text = "\n".join([chunk.content for chunk in chunks if chunk.content])
        profile = self._build_document_profile(title=title, chunks=chunks, sections=sections)
        topics = profile["topics"]
        sample_sentences = profile["sample_sentences"]
        document_kind = profile["document_kind"]

        summary_parts: List[str] = []
        if document_kind == "assessment":
            if topics:
                summary_parts.append(
                    f"This document is an assessment-style document focused on {self._join_phrases(topics[:4]).lower()}."
                )
            else:
                summary_parts.append(
                    "This document is an assessment-style document built around the uploaded subject matter."
                )
        elif topics:
            summary_parts.append(
                f"This document focuses on {self._join_phrases(topics[:4]).lower()}."
            )
        else:
            cleaned_title = self._clean_title(title) or "the uploaded material"
            summary_parts.append(
                f"This document presents material related to {cleaned_title.lower()}."
            )

        if document_kind == "assessment":
            action_topics = self._join_phrases(topics[:3]).lower() if topics else "the main concepts"
            summary_parts.append(
                f"It presents question-driven material that asks the reader to interpret, compare, and apply {action_topics}."
            )
        elif topics:
            summary_parts.append(
                f"The main themes include {self._join_phrases(topics[:5]).lower()}."
            )

        if sample_sentences:
            supporting = self._rewrite_sentence_for_summary(sample_sentences[0], topics)
            if supporting:
                summary_parts.append(supporting)

        if not summary_parts:
            summary_parts.append(
                "This document contains extracted text, but a high-quality summary could not be formed from the current content."
            )

        summary_text = " ".join(
            [self._clean_sentence(part) for part in summary_parts if part and part.strip()]
        )
        return self._clean_sentence(summary_text)

    def _build_fallback_section_summary(self, text: str, title: Optional[str] = None) -> str:
        """Produce an extractive section summary when the LLM output is missing or invalid."""
        sentences = self._extract_candidate_sentences(text)
        chosen = self._select_informative_sentences(sentences, limit=2)
        if chosen:
            return self._clean_sentence(" ".join(chosen))

        title_hint = self._clean_title(title or "")
        if title_hint:
            return f"This section focuses on {title_hint.lower()}."
        return "This section contains supporting information from the uploaded document."

    def _build_fallback_key_points(self, text: str) -> List[str]:
        """Extract deterministic bullet points from the section text."""
        points: List[str] = []
        keywords = [self._display_keyword(keyword).lower() for keyword in self._extract_keywords(text, limit=5)]
        if keywords:
            points.append(f"Covers {self._join_phrases(keywords[:3]).lower()}.")
            if len(keywords) > 3:
                points.append(f"Also touches on {self._join_phrases(keywords[3:5]).lower()}.")

        sentences = self._extract_candidate_sentences(text, prefer_low_digits=True)
        selected = self._select_informative_sentences(sentences, limit=2)
        for sentence in selected:
            if sentence.endswith("?"):
                continue
            if sentence not in points:
                points.append(sentence)

        return points[:4]

    def _extract_related_concepts(self, text: str) -> List[str]:
        """Extract a small set of meaningful related concepts for the UI."""
        keywords = self._extract_keywords(text, limit=5)
        return [keyword.title() for keyword in keywords]

    def _build_document_profile(
        self,
        title: str,
        chunks: List[Chunk],
        sections: Sequence[SummarySection],
    ) -> Dict[str, Any]:
        """Collect clean document-level signals for deterministic summaries."""
        clean_chunk_texts: List[str] = []
        for chunk in chunks:
            prepared = self._prepare_text_for_analysis(chunk.content)
            if prepared:
                clean_chunk_texts.append(prepared)

        combined_text = "\n".join(clean_chunk_texts)
        section_titles = [
            section.title
            for section in sections
            if section.title
            and not re.fullmatch(r"Section \d+", section.title, re.IGNORECASE)
            and self._clean_topic_phrase(section.title)
        ]
        keywords = [
            self._display_keyword(keyword)
            for keyword in self._extract_keywords(combined_text, limit=8)
            if self._clean_topic_phrase(self._display_keyword(keyword))
        ]
        sampled_sentences = self._select_informative_sentences(
            self._extract_candidate_sentences(combined_text, prefer_low_digits=True),
            limit=3,
        )

        seen_topics = set()
        topics: List[str] = []
        for candidate in [*keywords, *section_titles]:
            normalized = self._clean_topic_phrase(candidate)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_topics:
                continue
            seen_topics.add(key)
            topics.append(normalized)
            if len(topics) >= 6:
                break

        lowered = combined_text.lower()
        document_kind = "assessment" if self._looks_like_assessment(lowered) else "reference"
        title_hint = self._clean_title(title)

        return {
            "document_kind": document_kind,
            "topics": topics,
            "sample_sentences": sampled_sentences,
            "title_hint": title_hint,
        }

    def _normalize_points(self, points: Sequence[Any]) -> List[str]:
        """Validate and normalize LLM-provided key points."""
        normalized: List[str] = []
        seen = set()
        for item in points:
            candidate = self._clean_sentence(str(item))
            if not self._is_valid_point(candidate):
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
        return normalized[:4]

    def _prepare_text_for_analysis(self, text: str) -> str:
        """Normalize raw chunk text for extractive summarization."""
        lines = []
        for raw_line in (text or "").splitlines():
            line = self._clean_sentence(raw_line)
            if self._should_skip_line(line):
                continue
            lines.append(line)
        return "\n".join(lines)

    def _extract_candidate_sentences(self, text: str, prefer_low_digits: bool = False) -> List[str]:
        """Split text into meaningful summary candidates."""
        normalized = self._prepare_text_for_analysis(text)
        if not normalized:
            return []

        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        sentences: List[str] = []
        for raw_part in raw_parts:
            sentence = self._clean_sentence(raw_part)
            if not self._is_valid_sentence(sentence):
                continue
            if prefer_low_digits:
                digit_count = sum(char.isdigit() for char in sentence)
                alpha_count = sum(char.isalpha() for char in sentence)
                if alpha_count and digit_count > alpha_count * 0.15:
                    continue
            sentences.append(sentence)
        return sentences

    def _select_informative_sentences(self, sentences: Sequence[str], limit: int) -> List[str]:
        """Select diverse, high-signal sentences while avoiding noisy duplicates."""
        if not sentences:
            return []

        scored: List[tuple[float, str]] = []
        for sentence in sentences:
            words = re.findall(r"[A-Za-z][A-Za-z'-]+", sentence.lower())
            unique_words = {word for word in words if word not in _STOPWORDS}
            if not unique_words:
                continue
            digit_penalty = sum(char.isdigit() for char in sentence) * 0.25
            question_penalty = 1.25 if sentence.endswith("?") else 0.0
            instruction_penalty = 1.5 if self._looks_like_instruction(sentence.lower()) else 0.0
            score = len(unique_words) + min(len(words) / 6, 3) - digit_penalty - question_penalty - instruction_penalty
            scored.append((score, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: List[str] = []
        seen_word_sets: List[set[str]] = []
        for _, sentence in scored:
            word_set = {
                word
                for word in re.findall(r"[A-Za-z][A-Za-z'-]+", sentence.lower())
                if word not in _STOPWORDS
            }
            if not word_set:
                continue
            if any(self._jaccard_similarity(word_set, existing) > 0.7 for existing in seen_word_sets):
                continue
            selected.append(sentence)
            seen_word_sets.append(word_set)
            if len(selected) >= limit:
                break

        return selected

    def _extract_keywords(self, text: str, limit: int = 5) -> List[str]:
        """Extract stable keywords from text while filtering numbers and OCR noise."""
        counter: Counter[str] = Counter()
        for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", self._prepare_text_for_analysis(text).lower()):
            canonical = self._canonical_keyword(token)
            if canonical in _STOPWORDS:
                continue
            if len(canonical) <= 2 or len(canonical) > 24:
                continue
            if self._looks_like_noise(canonical):
                continue
            if canonical in _SUMMARY_GENERIC_TOPICS:
                continue
            counter[canonical] += 1

        keywords: List[str] = []
        for token, _ in counter.most_common():
            keywords.append(token)
            if len(keywords) >= limit:
                break
        return keywords

    def _derive_topic_label(self, text: str, fallback: str) -> str:
        """Build a readable topic label from keywords."""
        keywords = self._extract_keywords(text, limit=3)
        if not keywords:
            cleaned_fallback = self._clean_title(fallback)
            return cleaned_fallback or "Section"
        return " and ".join([self._display_keyword(keyword) for keyword in keywords[:2]])

    def _clean_title(self, title: str) -> str:
        """Normalize section titles and reject generic placeholders."""
        cleaned = self._clean_sentence(title)
        if not cleaned:
            return ""
        if re.fullmatch(r"(section|part|page)\s*[\divxlc]+", cleaned, re.IGNORECASE):
            return ""
        if cleaned.lower() in {"untitled", "document", "summary"}:
            return ""
        if self._looks_like_noise(cleaned):
            return ""
        return cleaned[:80]

    def _clean_topic_phrase(self, phrase: str) -> str:
        """Normalize topic phrases and reject generic or noisy labels."""
        cleaned = self._clean_title(phrase)
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered in _SUMMARY_GENERIC_TOPICS:
            return ""
        if self._looks_like_instruction(lowered):
            return ""
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned)
        if len(words) > 5:
            return ""
        if len(words) == 1 and self._canonical_keyword(words[0]) in _SUMMARY_GENERIC_TOPICS:
            return ""
        return cleaned

    def _clean_sentence(self, text: str) -> str:
        """Collapse whitespace and remove obvious prompt/fallback boilerplate."""
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -:\t\r\n")
        cleaned = re.sub(r"^[A-Za-z]?\d+[\.\)]\s*", "", cleaned)
        cleaned = re.sub(r"^(or|and)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace(" ,", ",").replace(" .", ".")
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    def _rewrite_sentence_for_summary(
        self,
        sentence: str,
        topics: Sequence[str],
    ) -> str:
        """Convert a strong source sentence into a safer summary sentence."""
        cleaned = self._clean_sentence(sentence)
        if not cleaned or self._looks_like_instruction(cleaned.lower()):
            return ""
        if cleaned.endswith("?"):
            return ""
        if len(cleaned) > 220:
            cleaned = cleaned[:217].rsplit(" ", 1)[0].strip() + "."
        if topics and sum(topic.lower() in cleaned.lower() for topic in topics[:4]) == 0:
            action_topics = self._join_phrases(topics[:3]).lower()
            return f"It also discusses applications and examples related to {action_topics}."
        if not cleaned.endswith((".", "!", "?")):
            cleaned = f"{cleaned}."
        return cleaned

    def _should_skip_line(self, line: str) -> bool:
        """Skip lines that are mostly formatting, OCR noise, or scoring metadata."""
        if not line:
            return True
        lowered = line.lower()
        if lowered in {"--end--", "thank you"}:
            return False
        if "llm unavailable" in lowered or "deterministic fallback" in lowered:
            return True
        if any(
            marker in lowered
            for marker in [
                "subject code",
                "full tmarks",
                "stream:",
                "third semester examination",
                "part -",
            ]
        ):
            return True
        if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", lowered):
            return True
        if re.fullmatch(r"part\s*[-:]?\s*[a-z0-9]+", lowered):
            return True
        if self._looks_like_instruction(lowered):
            return True
        alpha_count = sum(char.isalpha() for char in line)
        digit_count = sum(char.isdigit() for char in line)
        weird_count = sum(not (char.isalnum() or char.isspace() or char in ".,;:!?()%/-'\"") for char in line)
        if alpha_count < 3 and digit_count > 0:
            return True
        if weird_count > max(2, len(line) * 0.05):
            return True
        return False

    def _is_valid_sentence(self, sentence: str) -> bool:
        """Check whether a sentence is informative enough for summaries."""
        if not sentence:
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", sentence)
        if len(words) < 5:
            return False
        alpha_count = sum(char.isalpha() for char in sentence)
        digit_count = sum(char.isdigit() for char in sentence)
        if alpha_count == 0:
            return False
        if digit_count > alpha_count * 0.45:
            return False
        lowered = sentence.lower()
        if lowered.startswith(("summary:", "response:", "text:", "document content:")):
            return False
        if "llm unavailable" in lowered or "deterministic fallback" in lowered:
            return False
        if self._looks_like_instruction(lowered):
            return False
        useful_words = [word for word in words if self._is_useful_keyword(self._canonical_keyword(word))]
        if useful_words and len(useful_words) / max(len(words), 1) < 0.35:
            return False
        return True

    def _is_valid_summary_text(self, text: str) -> bool:
        """Validate LLM summary output before showing it to users."""
        cleaned = self._clean_sentence(text)
        if not cleaned:
            return False
        lowered = cleaned.lower()
        blocked_phrases = {
            "llm unavailable",
            "deterministic fallback analysis",
            "response (json only)",
            "key points (json array)",
            "summary:",
        }
        if any(phrase in lowered for phrase in blocked_phrases):
            return False
        if len(re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned)) < 8:
            return False
        return True

    def _is_valid_point(self, text: str) -> bool:
        """Validate a single bullet point."""
        if not self._is_valid_sentence(text):
            return False
        if len(text) > 220:
            return False
        return True

    def _looks_like_assessment(self, lowered_text: str) -> bool:
        """Identify question-paper style documents from extracted text markers."""
        markers = [
            "attempt",
            "each question",
            "answer any",
            "question carries",
            "marks",
            "null hypothesis",
            "choose the correct",
            "short answer",
        ]
        question_mark_count = lowered_text.count("?")
        return question_mark_count >= 3 or any(marker in lowered_text for marker in markers)

    def _looks_like_instruction(self, lowered_text: str) -> bool:
        """Reject exam instructions and layout boilerplate from summaries."""
        instruction_markers = [
            "attempt ",
            "answer any",
            "answer all",
            "each question",
            "question carries",
            "full marks",
            "subject code",
            "page ",
            "part ",
            "write short notes",
        ]
        return any(marker in lowered_text for marker in instruction_markers)

    def _join_phrases(self, items: Sequence[str]) -> str:
        """Join phrases into a short natural-language list."""
        cleaned = [self._clean_title(item) for item in items if self._clean_title(item)]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    def _looks_like_noise(self, text: str) -> bool:
        """Detect OCR-heavy fragments that should not appear as labels."""
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return True
        alpha_count = sum(char.isalpha() for char in compact)
        if alpha_count < 3:
            return True
        weird_count = sum(not char.isalpha() for char in compact if not char.isdigit())
        vowel_count = sum(char.lower() in "aeiou" for char in compact if char.isalpha())
        if len(compact) >= 6 and vowel_count < max(1, alpha_count * 0.2):
            return True
        return weird_count > alpha_count * 0.35

    def _canonical_keyword(self, token: str) -> str:
        """Normalize simple keyword variants for cleaner summaries."""
        normalized = token.lower().strip("-'")
        aliases = {
            "samples": "sampling",
            "sample": "sampling",
            "statistic": "statistics",
            "statistics": "statistics",
            "hypothesis": "hypothesis testing",
            "hypotheses": "hypothesis testing",
            "testing": "hypothesis testing",
            "probabilities": "probability",
            "means": "mean",
            "intervals": "confidence intervals",
            "interval": "confidence intervals",
            "population": "population",
            "distribution": "distribution",
            "distributions": "distribution",
            "confidence": "confidence intervals",
        }
        return aliases.get(normalized, normalized)

    def _display_keyword(self, token: str) -> str:
        """Convert a normalized keyword into a user-facing label."""
        display_map = {
            "hypothesis testing": "hypothesis testing",
            "confidence intervals": "confidence intervals",
        }
        return display_map.get(token, token.replace("-", " ").title())

    def _is_useful_keyword(self, token: str) -> bool:
        """Check whether a canonical token is useful for user-facing summaries."""
        if token in _STOPWORDS:
            return False
        if self._looks_like_noise(token):
            return False
        alpha_only = "".join([char for char in token if char.isalpha()])
        if len(alpha_only) < 3:
            return False
        vowel_count = sum(char in "aeiou" for char in alpha_only.lower())
        return vowel_count >= 1

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        """Compute a small similarity signal for duplicate suppression."""
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
    
    def _save_summary(self, summary: StructuredSummary) -> None:
        """Save summary to database."""
        from datetime import datetime
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = summary.model_dump_json()
        
        cursor.execute("""
            INSERT OR REPLACE INTO summaries (document_id, data, created_at)
            VALUES (?, ?, ?)
        """, (summary.document_id, data, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_summary(self, document_id: str) -> Optional[StructuredSummary]:
        """
        Retrieve summary for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            StructuredSummary object if found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT data FROM summaries WHERE document_id = ?",
            (document_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return StructuredSummary.model_validate_json(row[0])
    
    def delete_summary(self, document_id: str) -> None:
        """Delete summary for a document."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM summaries WHERE document_id = ?", (document_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted summary for document {document_id}")
