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
from core.llm.base import sanitize_llm_text
from utils.config import settings

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "but", "can", "carries",
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
    # Additional generic discourse markers, stopwords, and front-matter fillers
    "then", "there", "such", "other", "many", "most", "also", "well", "very", "much",
    "proceedings", "conference", "international", "journal", "figure", "figures",
    "table", "tables", "problem", "problems", "approach", "first", "second", "third",
    "one", "two", "three", "four", "new", "overall", "since", "here", "thus", "hence",
    "therefore", "however", "instead", "rather", "whether", "whereas", "meanwhile",
    "nevertheless", "though", "although", "even", "just", "either", "neither", "both",
    "all", "any", "every",
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
    "then",
    "there",
    "such",
    "more",
    "other",
    "many",
    "most",
    "proceedings",
    "conference",
    "figure",
    "figures",
    "table",
    "tables",
    "problem",
    "problems",
    "approach",
    "item",
    "items",
    "thing",
    "things",
    "way",
    "ways",
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
        
        document_type = self.classify_document(chunks=chunks, title=title)
        sections = self._generate_sections(chunks)
        llm_summary_result = self._generate_overall_summary(
            title=title, chunks=chunks, sections=sections, document_type=document_type
        )
        overall_summary = llm_summary_result["text"]
        llm_summary_accepted = bool(llm_summary_result["llm_used"])
        
        summary = StructuredSummary(
            document_id=document_id,
            title=title,
            document_type=document_type,
            overall_summary=overall_summary,
            sections=sections,
            metadata={
                "chunk_count": len(chunks),
                "section_count": len(sections),
                "document_type": document_type,
                "fallback_used": not llm_summary_accepted,
                "analysis_mode": "llm" if llm_summary_accepted else "deterministic_fallback",
                "llm_summary_accepted": llm_summary_accepted,
                "llm_last_error": self.llm.last_error,
            }
        )
        
        self._save_summary(summary)
        
        logger.info(f"Summary generated with {len(sections)} sections, classified as {document_type}")
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
        document_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Generate a concise overall summary, calling live LLM when available, falling back to deterministic extraction."""
        if not self._prefer_deterministic and self.llm._is_service_available():
            try:
                excerpts = []
                for c in chunks[:5]:
                    content_clean = self._clean_sentence(c.content[:400])
                    if content_clean:
                        excerpts.append(content_clean)
                context_str = "\n".join(excerpts)
                prompt = (
                    f"Provide a concise, factual executive summary (2-3 sentences) for the document titled '{title}'.\n"
                    f"Rely strictly on this document text:\n{context_str}\n\n"
                    f"Write only the 2-3 sentence executive summary with no preamble or labels."
                )
                response = self.llm.generate(prompt)
                clean_text = sanitize_llm_text(response).strip()
                if clean_text and len(clean_text) >= 40 and not clean_text.lower().startswith("llm unavailable"):
                    return {
                        "text": clean_text,
                        "llm_used": True,
                    }
            except Exception as e:
                logger.warning(f"LLM overall summary generation failed, falling back: {e}")

        return {
            "text": self._build_fallback_overall_summary(
                title=title, chunks=chunks, sections=sections, document_type=document_type
            ),
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

    def _is_front_matter_chunk(self, chunk: Chunk) -> bool:
        """Detect whether a chunk is part of front matter (copyright, preface, contributor roster)."""
        title = (chunk.section_title or "").lower()
        content = (chunk.content or "").lower()

        front_matter_markers = (
            "copyright", "all rights reserved", "isbn", "printed in", "published by",
            "library of congress", "contributing authors", "editorial board", "advisory board",
            "reviewers", "acknowledgments", "acknowledgements", "table of contents",
            "preface", "foreword"
        )
        if any(marker in title for marker in front_matter_markers):
            return True

        if re.search(r"\b(?:copyright|isbn\s*[\d\-]+|all rights reserved|published by)\b", content):
            return True

        for sentence in chunk.content.splitlines():
            sentence = sentence.strip()
            if sentence.count(",") >= 4:
                words = sentence.split()
                if words:
                    cap_tokens = sum(1 for w in words if w and w[0].isupper())
                    if cap_tokens >= 6 and (cap_tokens / len(words)) > 0.5:
                        return True

        return False

    def _sample_representative_chunks(
        self, chunks: List[Chunk], max_samples: int = 25
    ) -> List[Chunk]:
        """Sample representative body chunks across the document."""
        if not chunks:
            return []
        body_chunks = [c for c in chunks if not self._is_front_matter_chunk(c)]
        if not body_chunks:
            body_chunks = chunks
        if len(body_chunks) <= max_samples:
            return body_chunks
        step = (len(body_chunks) - 1) / (max_samples - 1) if max_samples > 1 else 1.0
        sampled_indices = sorted(list({int(round(i * step)) for i in range(max_samples)}))
        return [body_chunks[idx] for idx in sampled_indices]

    def _clean_document_title(self, title: str) -> str:
        """Extract a readable, clean document title without file extensions or noise."""
        if not title:
            return ""
        cleaned = re.sub(r"\.(pdf|docx|txt|md|html)$", "", str(title), flags=re.IGNORECASE)
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned or cleaned.lower() in {"untitled", "document", "summary", "unknown", "sample"}:
            return ""
        return cleaned[:100].strip()

    def _is_question_or_exam_stem(self, sentence: str) -> bool:
        """Identify questions, interrogative stems, and exam prompts that should not appear in declarative summaries."""
        if not sentence:
            return True
        cleaned = sentence.strip()
        if "?" in cleaned:
            return True

        # Strip leading numbering/labels like "1.", "Q1.", "Q 1:", "(a)", "Part A:"
        stripped = re.sub(
            r"^(?:q(?:uestion)?\s*\.?\s*\d+|[a-z\d]+[\.\)]|\([a-z\d]+\)|part\s+[a-z\d]+[:.-]?)\s*[:.-]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        lowered = stripped.lower()

        question_stems = (
            "what is", "what are", "what do", "what does", "what will", "what would",
            "how is", "how are", "how do", "how does", "how can", "how to",
            "why is", "why are", "why do", "why does", "why should",
            "which of", "which is", "which are",
            "when is", "when does", "where is", "where does",
            "state the", "state whether", "state and", "state if",
            "explain the", "explain how", "explain why", "explain what", "explain in",
            "describe the", "describe how", "describe why",
            "discuss the", "discuss how",
            "define the", "define ",
            "differentiate between", "distinguish between", "compare and contrast", "compare the",
            "write short note", "write short notes", "write a short note",
            "calculate the", "find the", "determine the", "compute the",
            "show that", "prove that", "derive the", "derive an",
            "give the", "give reasons", "give an example", "give two",
            "illustrate with", "draw a", "draw the", "sketch the",
            "list the", "name the", "mention the", "enumerate ",
        )
        return any(lowered.startswith(stem) for stem in question_stems)

    def _is_bare_formula(self, sentence: str) -> bool:
        """Identify bare formulas lacking declarative explanatory text."""
        cleaned = sentence.strip()
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned)
        if len(words) < 5 and any(op in cleaned for op in ["=", "×", "÷", "+", "->", "→"]):
            return True
        if "=" in cleaned:
            explanatory_indicators = [
                "which", "where", "means", "indicates", "measures", "defines",
                "calculated as", "expressed as", "given by", "represents", "denotes"
            ]
            lowered = cleaned.lower()
            if not any(ind in lowered for ind in explanatory_indicators) and len(words) < 8:
                return True
        return False

    def _formulate_lead_sentence(
        self,
        title: str,
        topics: Sequence[str],
        document_kind: str = "unknown",
    ) -> str:
        """Formulate a natural context lead sentence from cleaned title and prominent domain concepts."""
        cleaned_title = self._clean_document_title(title)
        clean_topics = [t for t in topics if self._clean_topic_phrase(t)]
        
        topic_clause = self._join_phrases(clean_topics[:3]).lower() if clean_topics else ""
        
        if cleaned_title and topic_clause:
            return f"This document examines {cleaned_title}, focusing on core principles across {topic_clause}."
        elif cleaned_title:
            return f"This document provides a technical overview of {cleaned_title} and its primary subject areas."
        elif topic_clause:
            return f"This document examines core principles and architectural mechanisms across {topic_clause}."
        else:
            return "This document presents foundational concepts and technical analyses across the subject matter."

    def _build_fallback_overall_summary(
        self,
        title: str,
        chunks: List[Chunk],
        sections: Sequence[SummarySection],
        document_type: str = "unknown",
    ) -> str:
        """Build an authentic executive summary grounded in the document text without template boilerplate."""
        profile = self._build_document_profile(title=title, chunks=chunks, sections=sections)
        topics = profile["topics"]
        doc_kind = document_type if document_type != "unknown" else profile["document_kind"]

        lead_sentence = self._formulate_lead_sentence(
            title=title, topics=topics, document_kind=doc_kind
        )

        representative_chunks = self._sample_representative_chunks(chunks, max_samples=25)
        clean_chunk_texts: List[str] = []
        for chunk in representative_chunks:
            cleaned = self._prepare_text_for_analysis(chunk.content)
            if cleaned:
                clean_chunk_texts.append(cleaned)
        combined_text = "\n".join(clean_chunk_texts)

        raw_candidates = self._extract_candidate_sentences(combined_text, prefer_low_digits=True)

        scored_candidates: List[tuple[float, str]] = []
        lead_word_set = {
            w.lower()
            for w in re.findall(r"[A-Za-z][A-Za-z'-]+", lead_sentence)
            if w.lower() not in _STOPWORDS
        }
        topic_lower_set = {t.lower() for t in topics}

        for raw_s in raw_candidates:
            s = self._clean_sentence(raw_s)
            if not s or len(s) < 20 or len(s) > 240:
                continue
            if self._is_question_or_exam_stem(s):
                continue
            if self._looks_like_instruction(s.lower()):
                continue
            if self._is_bare_formula(s):
                continue
            if self._is_roster_sentence(s) or self._looks_like_acknowledgment(s.lower()):
                continue
            if "deterministic fallback" in s.lower() or "llm unavailable" in s.lower():
                continue

            words = re.findall(r"[A-Za-z][A-Za-z'-]+", s)
            if len(words) < 7 or len(words) > 40:
                continue

            if not (s[0].isupper() or s[0].isdigit()):
                continue

            if not s.endswith((".", "!", ";")):
                s = f"{s}."

            content_words = [w.lower() for w in words if w.lower() not in _STOPWORDS]
            unique_content = set(content_words)
            if len(unique_content) < 5:
                continue

            concept_density = sum(1 for w in unique_content if w in topic_lower_set)
            length_bonus = 2.0 if 12 <= len(words) <= 28 else 0.5
            digit_penalty = sum(c.isdigit() for c in s) * 0.1
            score = len(unique_content) + (concept_density * 2.5) + length_bonus - digit_penalty

            scored_candidates.append((score, s))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        selected_sentences: List[str] = []
        seen_word_sets: List[set[str]] = [lead_word_set] if lead_word_set else []

        for _, s in scored_candidates:
            words = {
                w.lower()
                for w in re.findall(r"[A-Za-z][A-Za-z'-]+", s)
                if w.lower() not in _STOPWORDS
            }
            if any(self._jaccard_similarity(words, existing) > 0.45 for existing in seen_word_sets):
                continue
            selected_sentences.append(s)
            seen_word_sets.append(words)
            if len(selected_sentences) >= 3:
                break

        summary_parts = [lead_sentence] + selected_sentences
        summary_text = " ".join([p.strip() for p in summary_parts if p.strip()])
        return self._clean_sentence(summary_text)

    def _build_fallback_section_summary(self, text: str, title: Optional[str] = None) -> str:
        """Produce an extractive section summary when the LLM output is missing or invalid."""
        sentences = self._extract_candidate_sentences(text)
        clean_candidates: List[str] = [
            s for s in sentences
            if not self._is_question_or_exam_stem(s)
            and not self._looks_like_instruction(s.lower())
            and not self._is_bare_formula(s)
            and not self._is_roster_sentence(s)
            and not self._looks_like_acknowledgment(s.lower())
        ]
        chosen = self._select_informative_sentences(clean_candidates or sentences, limit=2)
        if chosen:
            return self._clean_sentence(" ".join(chosen))

        title_hint = self._clean_title(title or "")
        if title_hint:
            return f"This section discusses {title_hint.lower()} and its associated principles."
        return "This section contains supporting technical details from the document."

    def _build_fallback_key_points(self, text: str) -> List[str]:
        """Extract authentic, informative bullet points directly from the section text."""
        points: List[str] = []
        sentences = self._extract_candidate_sentences(text, prefer_low_digits=True)
        clean_candidates: List[str] = [
            s for s in sentences
            if not self._is_question_or_exam_stem(s)
            and not self._looks_like_instruction(s.lower())
            and not self._is_bare_formula(s)
            and not self._is_roster_sentence(s)
            and not self._looks_like_acknowledgment(s.lower())
            and 6 <= len(s.split()) <= 40
            and len(s) <= 220
        ]
        selected = self._select_informative_sentences(clean_candidates or sentences, limit=4)
        for sentence in selected:
            if sentence.endswith("?") or len(sentence.split()) < 4:
                continue
            if not sentence.endswith((".", "!")):
                sentence = f"{sentence}."
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
        """Collect clean document-level signals for deterministic summaries using sampled body chunks."""
        sampled_chunks = self._sample_representative_chunks(chunks, max_samples=25)
        clean_chunk_texts: List[str] = []
        for chunk in sampled_chunks:
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
        document_kind = self.classify_document(text=lowered, chunks=chunks, title=title)
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
        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
        for raw_line in raw_parts:
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

    def _is_roster_sentence(self, sentence: str) -> bool:
        """Detect sentences that are editorial contributor rosters rather than technical content."""
        words = sentence.strip().split()
        if len(words) < 3:
            return False
        commas = sentence.count(",")
        if commas >= 3:
            cap_count = sum(1 for w in words if w and w[0].isupper())
            if cap_count / len(words) >= 0.45:
                return True
        return False

    def _looks_like_acknowledgment(self, lowered: str) -> bool:
        """Detect acknowledgments or editorial boilerplate sentences."""
        markers = (
            "thank the following", "grateful to", "acknowledges the", "contributions of",
            "editorial board", "reviewers:", "credits:", "printed in", "all rights reserved",
            "isbn", "published by", "preface to the", "edition was published",
            "bibliography", "abbreviations are used", "abbreviations used", "frequently cited",
            "proceedings of the", "conference on", "in press", "dissertation", "thesis"
        )
        return any(m in lowered for m in markers)

    def _select_informative_sentences(self, sentences: Sequence[str], limit: int) -> List[str]:
        """Select diverse, high-signal sentences while avoiding noisy duplicates and contributor rosters."""
        if not sentences:
            return []

        scored: List[tuple[float, str]] = []
        for sentence in sentences:
            if self._is_roster_sentence(sentence) or self._looks_like_acknowledgment(sentence.lower()):
                continue
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
        """Convert a strong source sentence into a clean declarative summary sentence."""
        cleaned = self._clean_sentence(sentence)
        if not cleaned or self._looks_like_instruction(cleaned.lower()):
            return ""
        if self._is_question_or_exam_stem(cleaned):
            return ""
        if len(cleaned) > 220:
            cleaned = cleaned[:217].rsplit(" ", 1)[0].strip() + "."
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
        if sum(c in "┌─│└├┤┬┴┼═║╔╗╚╝━" for c in line) > 1:
            return True
        alpha_count = sum(char.isalpha() for char in line)
        digit_count = sum(char.isdigit() for char in line)
        allowed_math = ".,;:!?()%/-+=<>[]{}|*&^~_$#@≥≤≠≈→←·×°±√∞∑∏∫∂'\""
        weird_count = sum(not (char.isalnum() or char.isspace() or char in allowed_math or ord(char) > 0x370) for char in line)
        if alpha_count < 3 and digit_count > 0:
            return True
        if weird_count > max(2, len(line) * 0.08):
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

    def classify_document(
        self,
        text: Optional[str] = None,
        chunks: Optional[List[Chunk]] = None,
        title: Optional[str] = None,
    ) -> str:
        """
        Classify document using multiple structural, lexical, and density signals.
        Returns one of: 'textbook', 'academic_paper', 'report', 'assessment',
        'technical_documentation', 'notes', or 'unknown'.
        """
        # Aggregate text from chunks or text argument
        combined = ""
        if text:
            combined = text
        elif chunks:
            # Sample up to 30 chunks to avoid excessive memory for large books
            sample_chunks = chunks[:15] + chunks[-15:] if len(chunks) > 30 else chunks
            combined = "\n".join(c.content for c in sample_chunks)

        lowered = combined.lower()
        title_lower = (title or "").lower()
        total_words = len(lowered.split())
        num_chunks = len(chunks) if chunks else 0

        # Title-based fast indicators
        if any(marker in title_lower for marker in ["exam", "question paper", "quiz", "test paper", "assessment"]):
            return "assessment"
        if any(marker in title_lower for marker in ["lecture notes", "study notes", "meeting notes", "notes on"]):
            return "notes"
        if any(marker in title_lower for marker in ["api reference", "documentation", "user guide", "developer guide"]):
            return "technical_documentation"

        scores: Dict[str, float] = {
            "assessment": 0.0,
            "academic_paper": 0.0,
            "technical_documentation": 0.0,
            "report": 0.0,
            "textbook": 0.0,
            "notes": 0.0,
        }

        # 1. Assessment signals
        exam_instruction_markers = [
            "instructions to candidates",
            "time allowed:",
            "time allowed :",
            "maximum marks",
            "full marks",
            "answer any",
            "attempt any",
            "question carries",
            "roll no",
            "total marks",
            "write short notes on any",
            "choose the correct answer",
            "marks for each question",
        ]
        exam_matches = sum(1 for m in exam_instruction_markers if m in lowered)
        question_count = lowered.count("?")
        q_density_per_k = (question_count / (total_words / 1000.0)) if total_words > 50 else 0.0

        if exam_matches >= 2:
            scores["assessment"] += 8.0 + (exam_matches * 2.0)
        elif exam_matches == 1:
            scores["assessment"] += 4.0

        if q_density_per_k >= 6.0:
            scores["assessment"] += 5.0
        elif q_density_per_k >= 3.0:
            scores["assessment"] += 2.5

        # If document is short/medium and has multiple questions and exam keywords
        if total_words < 5000 and question_count >= 5 and exam_matches >= 1:
            scores["assessment"] += 4.0

        # 2. Textbook signals
        chapter_matches = len(re.findall(r"\bchapter\s+\d+\b", lowered))
        textbook_markers = [
            "table of contents",
            "preface",
            "exercises",
            "problems",
            "glossary",
            "appendix",
            "index",
            "edition",
            "textbook",
            "instructor's manual",
        ]
        tb_marker_count = sum(1 for m in textbook_markers if m in lowered)
        if chapter_matches >= 2:
            scores["textbook"] += 4.0 + min(chapter_matches, 6) * 1.5
        if tb_marker_count >= 2:
            scores["textbook"] += tb_marker_count * 1.5
        if num_chunks > 20 or total_words > 15000:
            scores["textbook"] += 3.0
            # A large textbook with exercises should not be classified as assessment
            scores["assessment"] -= 4.0

        # 3. Academic paper signals
        paper_sections = ["abstract", "introduction", "methodology", "methods", "experiments", "results", "related work", "conclusion", "references"]
        paper_section_count = sum(1 for sec in paper_sections if re.search(rf"\b{sec}\b", lowered))
        paper_academic_markers = ["et al.", "proceedings of", "ieee", "acm", "arxiv", "doi:", "university", "department of", "conference on"]
        paper_marker_count = sum(1 for m in paper_academic_markers if m in lowered)
        has_citations = bool(re.search(r"\[\d{1,3}\]", lowered) or re.search(r"\([A-Z][a-z]+,?\s+\d{4}\)", combined))

        if "abstract" in lowered and paper_section_count >= 3:
            scores["academic_paper"] += 5.0 + paper_section_count
        if paper_marker_count >= 2:
            scores["academic_paper"] += paper_marker_count * 1.5
        if has_citations:
            scores["academic_paper"] += 3.0

        # 4. Technical documentation signals
        tech_markers = [
            "installation", "getting started", "quickstart", "api reference",
            "configuration", "parameters:", "returns:", "raises:", "endpoint",
            "cli", "sdk", "prerequisites", "usage", "pip install", "npm install",
            "curl -x", "docker run", "import ", "def ", "class "
        ]
        tech_count = sum(1 for m in tech_markers if m in lowered)
        if tech_count >= 3:
            scores["technical_documentation"] += tech_count * 1.5

        # 5. Report signals
        report_markers = [
            "executive summary", "key findings", "recommendations", "annual report",
            "status report", "progress report", "prepared by", "deliverable",
            "milestone", "audit report", "overview of findings"
        ]
        report_count = sum(1 for m in report_markers if m in lowered)
        if "executive summary" in lowered:
            scores["report"] += 4.0
        if report_count >= 2:
            scores["report"] += report_count * 1.5

        # 6. Notes signals
        notes_markers = ["lecture notes", "study notes", "meeting notes", "notes:", "key takeaways", "lecture 1", "lecture 2"]
        notes_count = sum(1 for m in notes_markers if m in lowered)
        if notes_count >= 2:
            scores["notes"] += notes_count * 2.0

        # Evaluate best score
        best_type, best_score = max(scores.items(), key=lambda item: item[1])
        # Threshold: require at least 4.0 points of evidence to avoid false positives
        if best_score < 4.0:
            return "unknown"
        return best_type

    def _looks_like_assessment(self, lowered_text: str) -> bool:
        """Identify question-paper style documents using multi-signal classifier."""
        return self.classify_document(text=lowered_text) == "assessment"

    def _looks_like_instruction(self, lowered_text: str) -> bool:
        """Reject exam instructions and layout boilerplate from summaries."""
        instruction_markers = [
            "attempt ",
            "attempt any",
            "answer any",
            "answer all",
            "each question",
            "question carries",
            "full marks",
            "maximum marks",
            "time allowed",
            "subject code",
            "page ",
            "part ",
            "part -",
            "write short notes",
            "instructions to candidates",
            "roll no",
            "total marks",
            "marks for each",
            "-mark questions",
            "-mark question",
            "mark question",
            "mark questions",
            "choose the correct",
            "all questions carry",
            "assume suitable data",
            "figures to the right indicate",
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
