"""
Dataset Factory Service - Autonomous high-quality training dataset generation.

Generates unlimited training datasets from processed documents using:
- LLM-assisted quality scoring
- Embedding-based deduplication
- Multi-format export (JSONL, Parquet, CSV)
- Background continuous processing
"""

import asyncio
import json
import uuid
import sqlite3
import hashlib
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from utils.config import settings
from services.knowledge_map_service import KnowledgeMapService
from services.document_service import DocumentService
from core.llm.factory import LLMFactory
from core.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training sample for LLM fine-tuning."""

    id: str
    document_id: str
    chunk_id: str
    sample_type: str  # "qa", "concept", "relation", "summary"
    input_text: str
    output_text: str
    context: str
    quality_score: float
    metadata: Dict[str, Any]
    created_at: str


@dataclass
class DatasetBatch:
    """Batch of training samples ready for export."""

    id: str
    samples: List[TrainingSample]
    document_ids: List[str]
    sample_count: int
    avg_quality: float
    created_at: str
    exported_files: List[str]
    selected_chunk_count: int
    generation_mode: str


class DatasetFactoryService:
    """
    Autonomous dataset generation factory.

    Continuously processes documents and generates high-quality training data.
    Uses LLM for quality control and embedding for deduplication.
    """

    def __init__(self):
        self.db_path = settings.data_dir / "clarion.db"
        self.dataset_dir = settings.dataset_dir
        self.quality_threshold = settings.dataset_quality_threshold
        self.dedup_threshold = settings.dataset_dedup_threshold
        self.llm_sample_rate = settings.dataset_llm_sample_rate
        self.batch_size = settings.dataset_batch_size
        self.export_format = settings.dataset_export_format
        self.require_llm_for_export = settings.dataset_require_llm_for_export

        # Lazy-loaded services
        self._provider: Optional[BaseLLMProvider] = None
        self._provider_error: Optional[str] = None
        self._embedding_model = None
        self._embedding_cache = {}  # In-memory cache for deduplication
        self.document_service = DocumentService()
        self.knowledge_map_service = KnowledgeMapService()
        self._process_lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_samples": 0,
            "total_exported": 0,
            "quality_filtered": 0,
            "dedup_filtered": 0,
            "last_run": None,
        }

        self._init_database()
        logger.info("DatasetFactoryService initialized")

    @property
    def provider(self) -> Optional[BaseLLMProvider]:
        """Lazy-load the default configured LLM provider."""
        if self._provider is None and self._provider_error is None:
            try:
                self._provider = LLMFactory.create_default()
            except Exception as error:
                self._provider_error = str(error)
                logger.warning("Dataset factory provider unavailable: %s", error)
        return self._provider

    def _init_database(self) -> None:
        """Initialize dataset tracking database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Dataset generation tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_samples (
                sample_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                sample_type TEXT NOT NULL,
                quality_score REAL NOT NULL,
                is_exported INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        # Deduplication index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dataset_quality 
            ON dataset_samples(quality_score)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dataset_exported 
            ON dataset_samples(is_exported)
        """)

        # Generation log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_generation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                samples_generated INTEGER,
                samples_exported INTEGER,
                avg_quality REAL,
                duration_seconds REAL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_exports (
                export_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                export_path TEXT NOT NULL,
                jsonl_path TEXT,
                sample_count INTEGER NOT NULL,
                selected_chunk_count INTEGER NOT NULL,
                generation_mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dataset_exports_doc
            ON dataset_exports(document_id, created_at DESC)
        """)

        conn.commit()
        conn.close()

    def generate_qa_pairs(
        self, chunk_text: str, context: str = ""
    ) -> List[Dict[str, str]]:
        """
        Generate fast deterministic question-answer pairs from chunk text.
        """
        heuristic_pairs = self._generate_qa_pairs_fast(chunk_text)
        return heuristic_pairs

    def extract_concept_definitions(self, chunk_text: str) -> List[Dict[str, str]]:
        """Extract fast deterministic concept-definition pairs from text."""
        heuristic_concepts = self._extract_concepts_fast(chunk_text)
        return heuristic_concepts

    def _generate_qa_pairs_fast(self, chunk_text: str) -> List[Dict[str, str]]:
        """Generate lightweight deterministic Q&A pairs for demo responsiveness."""
        sentences = self._extract_sentences(chunk_text)
        if not sentences:
            return []

        subject = self._infer_subject(chunk_text)
        answer = " ".join(sentences[:2])[:600]
        if len(answer.split()) < 12:
            return []

        return [
            {
                "question": f"What does this passage explain about {subject}?",
                "answer": answer,
                "difficulty": "easy",
            }
        ]

    def _extract_concepts_fast(self, chunk_text: str) -> List[Dict[str, str]]:
        """Extract lightweight concept-definition pairs without blocking on the LLM."""
        sentences = self._extract_sentences(chunk_text)
        if not sentences:
            return []

        definition = sentences[0][:320]
        concept_terms = self._candidate_terms(chunk_text)
        concepts = []
        for term in concept_terms[:2]:
            concepts.append({"concept": term, "definition": definition})
        return concepts

    def _extract_sentences(self, text: str) -> List[str]:
        """Return clean sentences suitable for quick summarization."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        sentences = []
        for sentence in raw_sentences:
            cleaned = " ".join(sentence.split())
            if len(cleaned) >= 40:
                sentences.append(cleaned)
        return sentences

    def _infer_subject(self, text: str) -> str:
        """Infer a short subject phrase from chunk text."""
        candidates = self._candidate_terms(text)
        if candidates:
            return candidates[0]
        return "this topic"

    def _candidate_terms(self, text: str) -> List[str]:
        """Extract a few stable concept candidates from the text."""
        stopwords = {
            "that", "with", "from", "this", "have", "will", "your", "their",
            "about", "there", "which", "using", "used", "into", "them",
        }
        counts: Dict[str, int] = {}
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{3,}\b", text or ""):
            normalized = token.strip()
            lower = normalized.lower()
            if lower in stopwords:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1

        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0].lower()),
        )
        return [term for term, _ in ranked[:5]]

    def score_sample_quality(self, sample_text: str, sample_type: str = "qa") -> float:
        """
        Score sample quality using LLM + heuristics.

        Returns: Quality score 0.0-1.0
        """
        # Heuristic scores (fast, no LLM call)
        length_score = min(1.0, len(sample_text) / 200)  # Prefer longer samples
        word_count = len(sample_text.split())
        completeness_score = min(1.0, word_count / 30)  # At least 30 words

        # Check for common quality issues
        has_question = "?" in sample_text or "what" in sample_text.lower()
        has_answer = len(sample_text) > 50
        no_placeholder = "..." not in sample_text and "[...]" not in sample_text

        heuristic_score = (
            0.3 * length_score
            + 0.3 * completeness_score
            + 0.2 * float(has_question or has_answer)
            + 0.2 * float(no_placeholder)
        )

        llm_score = None
        if self.provider is not None:
            llm_score = self._llm_quality_validation(sample_text, sample_type)

        if llm_score is None:
            return round(heuristic_score, 3)

        return round((0.45 * heuristic_score) + (0.55 * llm_score), 3)

    def _llm_quality_validation(self, text: str, sample_type: str) -> Optional[float]:
        """LLM-based quality validation."""
        if self.provider is None:
            return None

        prompt = f"""Evaluate the quality of this training sample for fine-tuning an AI assistant.

Sample Type: {sample_type}
Sample Text:
{text[:1500]}

Rate the quality from 0.0 to 1.0 based on:
- Clarity: Is the text clear and well-written?
- Accuracy: Is the information accurate and coherent?
- Usefulness: Would this help train a helpful assistant?
- Completeness: Is the sample complete (not cut off)?

Respond with ONLY a number between 0.0 and 1.0 (e.g., 0.85)"""

        try:
            response = asyncio.run(
                self.provider.generate(
                    prompt=prompt,
                    system_message="You are a strict dataset quality rater. Return only a number between 0.0 and 1.0.",
                    temperature=0.0,
                    max_tokens=16,
                )
            )
            # Extract number from response
            import re

            numbers = re.findall(r"\d+\.?\d*", response.content)
            if numbers:
                score = float(numbers[0])
                return min(1.0, max(0.0, score))
        except Exception as e:
            logger.debug(f"LLM quality validation failed: {e}")

        return None

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute a lightweight deterministic embedding for deduplication.

        This keeps dataset generation responsive during demos without loading a
        second large embedding model in parallel with the main pipeline.
        """
        dimension = 256
        vector = np.zeros(dimension, dtype=np.float32)
        tokens = re.findall(r"\b\w+\b", (text or "").lower())

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def is_duplicate(self, text: str, threshold: Optional[float] = None) -> bool:
        """
        Check if text is duplicate of existing samples.

        Uses cosine similarity on embeddings.
        """
        if threshold is None:
            threshold = self.dedup_threshold

        # Compute embedding
        embedding = self.compute_embedding(text)
        embedding_norm = embedding / np.linalg.norm(embedding)

        # Check against cache
        for cached_text, cached_embedding in self._embedding_cache.items():
            similarity = np.dot(embedding_norm, cached_embedding)
            if similarity > threshold:
                logger.debug(f"Duplicate detected (similarity: {similarity:.3f})")
                return True

        # Add to cache (limit cache size)
        if len(self._embedding_cache) < 10000:
            self._embedding_cache[text[:500]] = embedding_norm

        return False

    def _normalize_training_text(self, text: str) -> str:
        """Normalize chunk text before trainability checks."""
        normalized = " ".join((text or "").split())
        normalized = re.sub(r"\b([A-Za-z])(?:\s+[A-Za-z]){3,}\b", lambda m: m.group(0).replace(" ", ""), normalized)
        return normalized.strip()

    def _looks_administrative_or_noisy(self, text: str) -> bool:
        """Detect roster-like, metadata-heavy, or otherwise poor training content."""
        normalized = self._normalize_training_text(text).lower()
        if not normalized:
            return True

        blocked_markers = [
            "roll no",
            "enrollment number",
            "work assigned",
            "name sec",
            "attendance",
            "marks obtained",
            "group members",
            "group member",
            "submitted by",
            "department of",
            "prof.",
            "professor",
        ]
        if any(marker in normalized for marker in blocked_markers):
            return True

        if len(re.findall(r"\d", normalized)) > max(12, len(normalized) // 8):
            return True

        alpha_tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", normalized)
        if len(alpha_tokens) < 20:
            return True

        unique_ratio = len(set(token.lower() for token in alpha_tokens)) / max(len(alpha_tokens), 1)
        if unique_ratio < 0.35:
            return True

        return False

    def _is_trainable_chunk(self, text: str) -> bool:
        """Keep only chunks that are self-contained enough for supervised training."""
        normalized = self._normalize_training_text(text)
        if len(normalized) < 180:
            return False
        if self._looks_administrative_or_noisy(normalized):
            return False

        sentence_like_units = re.split(r"(?<=[.!?])\s+", normalized)
        substantial_units = [unit for unit in sentence_like_units if len(unit.split()) >= 8]
        return len(substantial_units) >= 2

    def _is_trainable_sample(
        self,
        input_text: str,
        output_text: str,
        context: str,
    ) -> bool:
        """Reject samples that still look administrative, fragmentary, or weak."""
        input_clean = self._normalize_training_text(input_text)
        output_clean = self._normalize_training_text(output_text)
        context_clean = self._normalize_training_text(context)
        combined = " ".join([input_clean, output_clean, context_clean]).lower()
        if self._looks_administrative_or_noisy(combined):
            return False

        output_words = re.findall(r"[A-Za-z][A-Za-z'-]+", output_clean)
        if len(output_words) < 18:
            return False
        if len(re.findall(r"[A-Za-z][A-Za-z'-]+", input_clean)) < 4:
            return False
        if len(set(word.lower() for word in output_words)) < 12:
            return False
        if not any(punct in output_clean for punct in [".", ":", ";"]):
            return False
        return True

    def create_training_sample(
        self,
        document_id: str,
        chunk_id: str,
        chunk_text: str,
        sample_type: str,
        input_text: str,
        output_text: str,
        context: str = "",
    ) -> Optional[TrainingSample]:
        """Create a single training sample with quality control."""

        # Combine for quality scoring
        sample_text = f"{input_text} {output_text}"

        if self.require_llm_for_export and self.provider is None:
            return None

        if not self._is_trainable_sample(input_text, output_text, context):
            self.stats["quality_filtered"] += 1
            logger.debug("Sample filtered as non-trainable")
            return None

        # Quality check
        quality_score = self.score_sample_quality(sample_text, sample_type)
        if quality_score < self.quality_threshold:
            self.stats["quality_filtered"] += 1
            logger.debug(
                f"Sample filtered by quality: {quality_score:.3f} < {self.quality_threshold}"
            )
            return None

        # Deduplication check
        if self.is_duplicate(sample_text):
            self.stats["dedup_filtered"] += 1
            logger.debug("Sample filtered as duplicate")
            return None

        # Create sample
        provider_model = None
        if self.provider is not None:
            try:
                provider_model = self.provider.get_model_info().model_name
            except Exception:
                provider_model = None
        sample = TrainingSample(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_id=chunk_id,
            sample_type=sample_type,
            input_text=input_text.strip(),
            output_text=output_text.strip(),
            context=context.strip()[:2000],
            quality_score=quality_score,
            metadata={
                "word_count": len(sample_text.split()),
                "char_count": len(sample_text),
                "model_used": provider_model or settings.ollama_model,
                "quality_threshold": self.quality_threshold,
                "trainable": True,
            },
            created_at=datetime.now().isoformat(),
        )

        return sample

    def process_document(self, document_id: str) -> DatasetBatch:
        """
        Process a single document and generate training samples.

        Builds a per-document training dataset grounded in selected chunks.
        """
        with self._process_lock:
            logger.info(f"Processing document {document_id} for dataset generation")
            start_time = datetime.now()

            samples = []
            total_generated = 0
            quality_scores = []
            exported_files: List[str] = []
            generation_mode = "llm_required"
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            document = self.document_service.get_document(document_id)

            # Get all chunks for this document
            cursor.execute(
                """
                SELECT chunk_id, content, position_index 
                FROM chunks 
                WHERE document_id = ?
                ORDER BY position_index
            """,
                (document_id,),
            )

            chunks = cursor.fetchall()
            logger.info(f"Found {len(chunks)} chunks in document {document_id}")

            concept_titles = self._concept_chunk_titles(document_id)
            chunks_to_process = self._select_chunks_for_dataset(chunks, concept_titles)
            logger.info(
                "Processing %d selected chunks for dataset generation (%d total chunks)",
                len(chunks_to_process),
                len(chunks),
            )

            selected_chunks = []
            for chunk in chunks_to_process:
                chunk_text = self._normalize_training_text(chunk["content"])
                if not self._is_trainable_chunk(chunk_text):
                    continue

                selected_chunks.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "position_index": int(chunk["position_index"]),
                        "section_title": concept_titles.get(chunk["chunk_id"])
                        or self._clean_section_title(
                            chunk["content"], chunk["chunk_id"], chunk["position_index"]
                        ),
                        "content": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "context": self._build_context(cursor, document_id, chunk["position_index"]),
                    }
                )

            generation_records, generation_mode = self._generate_dataset_records(
                document=document,
                selected_chunks=selected_chunks,
            )

            for record in generation_records:
                chunk_id = record.get("chunk_id", "")
                chunk_text = record.get("source_text", "")
                sample = self.create_training_sample(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    sample_type=record.get("sample_type", "qa"),
                    input_text=record.get("instruction", ""),
                    output_text=record.get("response", ""),
                    context=record.get("context", ""),
                )
                if not sample:
                    continue

                sample.metadata.update(
                    {
                        "concepts": record.get("concepts", []),
                        "grounding_excerpt": record.get("grounding_excerpt", ""),
                        "section_title": record.get("section_title", ""),
                        "generation_mode": generation_mode,
                    }
                )
                samples.append(sample)
                total_generated += 1
                quality_scores.append(sample.quality_score)
                self._save_sample(cursor, sample)

            conn.commit()
            conn.close()

            if samples:
                exported_files = self._export_document_dataset(
                    batch_id=str(uuid.uuid4()),
                    document=document,
                    selected_chunks=selected_chunks,
                    samples=samples,
                    generation_mode=generation_mode,
                )

            duration = (datetime.now() - start_time).total_seconds()

            # Create batch record
            batch = DatasetBatch(
                id=str(uuid.uuid4()),
                samples=samples,
                document_ids=[document_id],
                sample_count=total_generated,
                avg_quality=float(np.mean(quality_scores))
                if quality_scores
                else 0.0,
                created_at=datetime.now().isoformat(),
                exported_files=exported_files,
                selected_chunk_count=len(selected_chunks),
                generation_mode=generation_mode,
            )

            # Log generation
            self._log_generation(
                document_id, total_generated, len(samples), batch.avg_quality, duration
            )

            self.stats["total_processed"] += 1
            self.stats["total_samples"] += total_generated
            self.stats["last_run"] = datetime.now().isoformat()

            logger.info(
                f"Generated {total_generated} samples from {document_id} in {duration:.1f}s"
            )

            return batch

    def _select_chunks_for_dataset(
        self,
        chunks: List[sqlite3.Row],
        concept_titles: Optional[Dict[str, str]] = None,
    ) -> List[sqlite3.Row]:
        """Prefer substantial, title-bearing chunks over raw uniform sampling."""
        if not chunks:
            return []
        concept_titles = concept_titles or {}
        candidate_chunks = [
            row for row in chunks if self._is_trainable_chunk(row["content"] or "")
        ]
        if not candidate_chunks:
            return []

        def sort_key(row: sqlite3.Row) -> Tuple[int, int, int]:
            title_bonus = 2 if row["chunk_id"] in concept_titles else 1 if self._clean_section_title(row["content"], row["chunk_id"], row["position_index"]) else 0
            word_count = len((row["content"] or "").split())
            return (title_bonus, min(word_count, 280), -int(row["position_index"]))

        ranked = sorted(candidate_chunks, key=sort_key, reverse=True)
        selected: List[sqlite3.Row] = []
        seen_titles = set()

        if concept_titles:
            concept_rank = {chunk_id: index for index, chunk_id in enumerate(concept_titles.keys())}
            concept_rows = sorted(
                [row for row in candidate_chunks if row["chunk_id"] in concept_titles],
                key=lambda row: concept_rank.get(row["chunk_id"], 9999),
            )
            for row in concept_rows:
                title = concept_titles.get(row["chunk_id"], "").lower()
                if title and title in seen_titles:
                    continue
                selected.append(row)
                if title:
                    seen_titles.add(title)
                if len(selected) >= 5:
                    break
            if len(selected) >= 5:
                return sorted(selected, key=lambda row: int(row["position_index"]))

        for row in ranked:
            title = (
                concept_titles.get(row["chunk_id"])
                or self._clean_section_title(row["content"], row["chunk_id"], row["position_index"])
            ).lower()
            if title and title in seen_titles:
                continue
            selected.append(row)
            if title:
                seen_titles.add(title)
            if len(selected) >= 5:
                break

        if len(selected) < min(4, len(candidate_chunks)):
            fallback = sorted(candidate_chunks, key=lambda row: int(row["position_index"]))
            used_ids = {row["chunk_id"] for row in selected}
            for row in fallback:
                if row["chunk_id"] in used_ids:
                    continue
                selected.append(row)
                if len(selected) >= min(5, len(candidate_chunks)):
                    break

        return sorted(selected, key=lambda row: int(row["position_index"]))

    def _concept_chunk_titles(self, document_id: str) -> Dict[str, str]:
        """Map representative chunk IDs to concept names from the knowledge map."""
        knowledge_map = self.knowledge_map_service.get_knowledge_map(document_id)
        if not knowledge_map:
            return {}

        titles: Dict[str, str] = {}
        for concept in knowledge_map.concepts:
            for chunk_id in concept.chunk_ids[:1]:
                titles.setdefault(chunk_id, concept.name)
        return titles

    def _clean_section_title(
        self, content: str, chunk_id: str, position_index: int
    ) -> str:
        """Infer a stable section title for dataset grounding."""
        banned_titles = {
            "formula",
            "formula:",
            "terms explained",
            "terms explained:",
            "document",
            "function",
            "increase",
            "direction",
            "respect",
            "fixed",
            "vector",
        }
        known_phrases = [
            "Partial Derivatives",
            "Gradient Descent",
            "Chain Rule",
            "Cross-Entropy",
            "Outer Product",
            "Jacobian",
            "Softmax",
            "Gradient",
        ]
        lowered_text = (content or "").lower()
        for phrase in known_phrases:
            if phrase.lower() in lowered_text:
                return phrase

        lines = [line.strip(" :-") for line in (content or "").splitlines() if line.strip()]
        if lines:
            first_line = " ".join(lines[0].split())
            if 2 <= len(first_line.split()) <= 6 and len(first_line) <= 60:
                if (
                    first_line.lower() not in banned_titles
                    and not any(symbol in first_line for symbol in ["=", "∂", "∇", "→", "⊗"])
                ):
                    return first_line

        match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", content or "")
        if match:
            candidate = " ".join(match.group(1).split()).strip(":- ")
            if candidate.lower() not in banned_titles:
                return candidate

        candidates = self._candidate_terms(content or "")
        if candidates:
            for candidate in candidates:
                if candidate.lower() not in banned_titles:
                    return candidate

        return f"Chunk {position_index + 1}"

    def _generate_dataset_records(
        self,
        document,
        selected_chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Generate chunk-grounded dataset records using the available LLM provider."""
        if not selected_chunks:
            return [], "deterministic"

        provider = self.provider
        if provider is not None:
            records = self._generate_with_provider(
                provider=provider,
                document_title=document.metadata.filename if document else "Document",
                selected_chunks=selected_chunks,
            )
            if records:
                provider_name = provider.get_model_info().provider.value
                return records, f"llm_{provider_name}"

        if self.require_llm_for_export:
            return [], "llm_unavailable"

        return self._generate_deterministic_records(selected_chunks), "deterministic"

    def _generate_with_provider(
        self,
        provider: BaseLLMProvider,
        document_title: str,
        selected_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Refine selected chunks into a clean JSON dataset using the configured provider."""
        payload = [
            {
                "chunk_id": chunk["chunk_id"],
                "section_title": chunk["section_title"],
                "text": chunk["content"][:1200],
            }
            for chunk in selected_chunks
        ]
        prompt = f"""
You are preparing a research-quality training dataset from selected document chunks.
Use only the provided chunk text. Do not invent facts. Keep outputs concise, technically clear, and grounded.
When mathematics appears, write notation in LaTeX.

Return a JSON object with this shape:
{{
  "samples": [
    {{
      "chunk_id": "string",
      "sample_type": "qa" | "concept" | "summary",
      "instruction": "string",
      "response": "string",
      "concepts": ["string"],
      "grounding_excerpt": "string"
    }}
  ]
}}

Rules:
- Produce 1 or 2 samples per chunk.
- Prefer concept explanation and concise question-answer supervision.
- Each response should be 1 to 4 sentences.
- grounding_excerpt must be copied from the source chunk and remain under 220 characters.
- concepts must contain the 1 to 4 most relevant terms from that chunk.
- Reject administrative rosters, marksheets, attendance-style text, author lists, team-member tables, and metadata-heavy chunks.
- Skip any chunk that is not self-contained enough for direct supervised fine-tuning.
- Do not emit generic pairs about websites, backend roles, departments, enrollment data, or personnel assignments.
- Return only samples that are directly usable for supervised fine-tuning.

Document title: {document_title}
Selected chunks:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

        try:
            schema = {
                "type": "object",
                "properties": {
                    "samples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chunk_id": {"type": "string"},
                                "sample_type": {"type": "string"},
                                "instruction": {"type": "string"},
                                "response": {"type": "string"},
                                "concepts": {"type": "array", "items": {"type": "string"}},
                                "grounding_excerpt": {"type": "string"},
                            },
                            "required": ["chunk_id", "sample_type", "instruction", "response"],
                        },
                    }
                },
                "required": ["samples"],
            }
            parsed = asyncio.run(
                provider.generate_structured(
                    prompt=prompt,
                    output_schema=schema,
                    system_message=(
                        "You create only high-quality supervised fine-tuning samples. "
                        "Use only the source text. Exclude noisy or administrative content."
                    ),
                )
            ) or {}
            return self._normalize_cloud_samples(parsed.get("samples", []), selected_chunks)
        except Exception as error:
            provider_name = "unknown"
            try:
                provider_name = provider.get_model_info().provider.value
            except Exception:
                pass
            logger.warning("LLM dataset refinement failed (%s): %s", provider_name, error)
            return []

    def _extract_json_object(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from model output."""
        text = (response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None
        return None

    def _normalize_cloud_samples(
        self,
        raw_samples: List[Dict[str, Any]],
        selected_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate and normalize cloud-produced samples."""
        chunk_map = {chunk["chunk_id"]: chunk for chunk in selected_chunks}
        normalized = []
        for item in raw_samples or []:
            chunk_id = str(item.get("chunk_id", "")).strip()
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue

            instruction = " ".join(str(item.get("instruction", "")).split()).strip()
            response = " ".join(str(item.get("response", "")).split()).strip()
            if len(instruction) < 12 or len(response) < 30:
                continue

            normalized.append(
                {
                    "chunk_id": chunk_id,
                    "section_title": chunk["section_title"],
                    "sample_type": item.get("sample_type", "qa"),
                    "instruction": instruction,
                    "response": response,
                    "concepts": [
                        " ".join(str(term).split()).strip()
                        for term in item.get("concepts", [])[:4]
                        if str(term).strip()
                    ],
                    "grounding_excerpt": " ".join(
                        str(item.get("grounding_excerpt", "")).split()
                    )[:220],
                    "context": chunk["context"],
                    "source_text": chunk["content"],
                }
            )
        return normalized

    def _generate_deterministic_records(
        self, selected_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create clean JSON-ready records directly from selected chunks."""
        records: List[Dict[str, Any]] = []
        for chunk in selected_chunks:
            subject = chunk["section_title"] or self._infer_subject(chunk["content"])
            summary = self._fallback_summary(chunk["content"])
            concepts = self._concepts_from_chunk(chunk["content"], chunk["section_title"])
            excerpt = self._grounding_excerpt(chunk["content"])

            if summary:
                records.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "section_title": chunk["section_title"],
                        "sample_type": "qa",
                        "instruction": f"Explain the core idea behind {subject}.",
                        "response": summary,
                        "concepts": concepts,
                        "grounding_excerpt": excerpt,
                        "context": chunk["context"],
                        "source_text": chunk["content"],
                    }
                )

            if concepts:
                concept_name = concepts[0]
                records.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "section_title": chunk["section_title"],
                        "sample_type": "concept",
                        "instruction": f"Define {concept_name} in the context of this document.",
                        "response": summary,
                        "concepts": concepts,
                        "grounding_excerpt": excerpt,
                        "context": chunk["context"],
                        "source_text": chunk["content"],
                    }
                )
        return records

    def _fallback_summary(self, text: str) -> str:
        """Build a concise, readable answer from chunk text."""
        normalized = " ".join((text or "").replace("@AIinMinutes", "").split())
        normalized = re.sub(r"Terms Explained:\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"Formula:\s*", "", normalized, flags=re.IGNORECASE)
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if len(sentence.strip()) >= 30
        ]
        return " ".join(sentences[:2])[:520]

    def _concepts_from_chunk(self, text: str, section_title: str) -> List[str]:
        """Extract a small set of clean concepts tied to a chunk."""
        concepts = []
        if section_title:
            concepts.append(section_title)
        for candidate in self._candidate_terms(text):
            if candidate not in concepts:
                concepts.append(candidate)
            if len(concepts) >= 4:
                break
        return concepts

    def _grounding_excerpt(self, text: str) -> str:
        """Return a short, clean excerpt copied from the source chunk."""
        sentences = self._extract_sentences(text)
        if sentences:
            return sentences[0][:220]
        return " ".join((text or "").split())[:220]

    def process_all_documents(self) -> int:
        """Process all documents and generate datasets."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get all unique document IDs
        cursor.execute("SELECT DISTINCT document_id FROM chunks")
        document_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Found {len(document_ids)} documents to process")

        total_samples = 0
        for doc_id in document_ids:
            batch = self.process_document(doc_id)
            total_samples += batch.sample_count

        logger.info(f"Dataset generation complete: {total_samples} total samples")

        return total_samples

    def _build_context(
        self, cursor, document_id: str, position: int, window: int = 1
    ) -> str:
        """Build context from surrounding chunks."""
        context_chunks = []

        # Previous chunk
        if position > 0:
            cursor.execute(
                """
                SELECT content FROM chunks 
                WHERE document_id = ? AND position_index = ?
            """,
                (document_id, position - 1),
        )
        row = cursor.fetchone()
        if row:
            previous = self._normalize_training_text(row[0][:500])
            if previous and not self._looks_administrative_or_noisy(previous):
                context_chunks.append(f"[Previous]: {previous}")

        # Next chunk
        cursor.execute(
            """
            SELECT content FROM chunks 
            WHERE document_id = ? AND position_index = ?
        """,
            (document_id, position + 1),
        )
        row = cursor.fetchone()
        if row:
            next_text = self._normalize_training_text(row[0][:500])
            if next_text and not self._looks_administrative_or_noisy(next_text):
                context_chunks.append(f"[Next]: {next_text}")

        return "\n".join(context_chunks)

    def _save_sample(self, cursor, sample: TrainingSample) -> None:
        """Save sample to database."""
        cursor.execute(
            """
            INSERT OR REPLACE INTO dataset_samples 
            (sample_id, document_id, chunk_id, sample_type, quality_score, is_exported, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
        """,
            (
                sample.id,
                sample.document_id,
                sample.chunk_id,
                sample.sample_type,
                sample.quality_score,
                sample.created_at,
            ),
        )

    def _export_batch(
        self, samples: List[TrainingSample], document_ids: List[str]
    ) -> None:
        """Export batch of samples to files."""
        if not samples:
            return

        batch_id = str(uuid.uuid4())
        base_filename = f"dataset_{batch_id}"

        # Prepare data
        qa_data = []
        concept_data = []

        for sample in samples:
            if sample.sample_type == "qa":
                qa_data.append(
                    {
                        "id": sample.id,
                        "messages": [
                            {"role": "user", "content": sample.input_text},
                            {"role": "assistant", "content": sample.output_text},
                        ],
                        "context": sample.context,
                        "quality": sample.quality_score,
                        "metadata": sample.metadata,
                    }
                )
            elif sample.sample_type == "concept":
                concept_data.append(
                    {
                        "id": sample.id,
                        "term": sample.input_text.replace("Define: ", ""),
                        "definition": sample.output_text,
                        "context": sample.context,
                        "quality": sample.quality_score,
                        "metadata": sample.metadata,
                    }
                )

        # Export Q&A pairs (JSONL format for fine-tuning)
        if qa_data:
            qa_file = self.dataset_dir / f"{base_filename}_qa.jsonl"
            with open(qa_file, "w", encoding="utf-8") as f:
                for item in qa_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Exported {len(qa_data)} Q&A samples to {qa_file}")

        # Export concepts (JSON format)
        if concept_data:
            concept_file = self.dataset_dir / f"{base_filename}_concepts.json"
            with open(concept_file, "w", encoding="utf-8") as f:
                json.dump(concept_data, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Exported {len(concept_data)} concept samples to {concept_file}"
            )

        # Export combined dataset (Parquet if available)
        try:
            import pandas as pd

            all_samples = []
            for sample in samples:
                all_samples.append(
                    {
                        "id": sample.id,
                        "type": sample.sample_type,
                        "input": sample.input_text,
                        "output": sample.output_text,
                        "context": sample.context,
                        "quality": sample.quality_score,
                        "document_id": sample.document_id,
                        "created_at": sample.created_at,
                    }
                )

            if all_samples:
                df = pd.DataFrame(all_samples)

                # Parquet export
                parquet_file = self.dataset_dir / f"{base_filename}_all.parquet"
                df.to_parquet(parquet_file, index=False)
                logger.info(f"Exported {len(all_samples)} samples to {parquet_file}")

                # CSV export
                csv_file = self.dataset_dir / f"{base_filename}_all.csv"
                df.to_csv(csv_file, index=False, encoding="utf-8")
                logger.info(f"Exported {len(all_samples)} samples to {csv_file}")
        except ImportError:
            logger.info("Pandas not available, skipping Parquet/CSV export")

        # Mark samples as exported
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        sample_ids = [s.id for s in samples]
        cursor.execute(
            f"""
            UPDATE dataset_samples 
            SET is_exported = 1 
            WHERE sample_id IN ({",".join("?" * len(sample_ids))})
        """,
            sample_ids,
        )
        conn.commit()
        conn.close()

        self.stats["total_exported"] += len(samples)

    def _export_document_dataset(
        self,
        batch_id: str,
        document,
        selected_chunks: List[Dict[str, Any]],
        samples: List[TrainingSample],
        generation_mode: str,
    ) -> List[str]:
        """Export a clean per-document JSON dataset and a companion JSONL file."""
        if not document or not samples:
            return []

        safe_name = self._safe_filename_stem(document.metadata.filename)
        json_path = self.dataset_dir / f"{safe_name}_{document.id}_dataset.json"
        jsonl_path = self.dataset_dir / f"{safe_name}_{document.id}_sft.jsonl"

        payload = {
            "dataset_id": batch_id,
            "document_id": document.id,
            "document_title": document.metadata.filename,
            "generated_at": datetime.now().isoformat(),
            "generation_mode": generation_mode,
            "trainability": {
                "is_directly_model_trainable": True,
                "format": "instruction_response_sft",
                "llm_curated": generation_mode.startswith("llm_"),
                "grounded_only": True,
            },
            "selected_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "position_index": chunk["position_index"],
                    "section_title": chunk["section_title"],
                    "word_count": chunk["word_count"],
                    "excerpt": self._grounding_excerpt(chunk["content"]),
                }
                for chunk in selected_chunks
            ],
            "samples": [
                {
                    "sample_id": sample.id,
                    "sample_type": sample.sample_type,
                    "chunk_id": sample.chunk_id,
                    "instruction": sample.input_text,
                    "response": sample.output_text,
                    "context": sample.context,
                    "quality_score": sample.quality_score,
                    "concepts": sample.metadata.get("concepts", []),
                    "grounding_excerpt": sample.metadata.get("grounding_excerpt", ""),
                    "section_title": sample.metadata.get("section_title", ""),
                }
                for sample in samples
            ],
        }

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for sample in samples:
                record = {
                    "sample_id": sample.id,
                    "messages": [
                        {"role": "user", "content": sample.input_text},
                        {"role": "assistant", "content": sample.output_text},
                    ],
                    "metadata": {
                        "document_id": sample.document_id,
                        "chunk_id": sample.chunk_id,
                        "sample_type": sample.sample_type,
                        "quality_score": sample.quality_score,
                        "concepts": sample.metadata.get("concepts", []),
                        "grounding_excerpt": sample.metadata.get("grounding_excerpt", ""),
                        "trainable": True,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._save_export_record(
            export_id=batch_id,
            document_id=document.id,
            export_path=str(json_path),
            jsonl_path=str(jsonl_path),
            sample_count=len(samples),
            selected_chunk_count=len(selected_chunks),
            generation_mode=generation_mode,
        )
        self.stats["total_exported"] += len(samples)
        logger.info("Exported %d samples to %s", len(samples), json_path)
        logger.info("Exported SFT JSONL to %s", jsonl_path)
        return [str(json_path), str(jsonl_path)]

    def _save_export_record(
        self,
        export_id: str,
        document_id: str,
        export_path: str,
        jsonl_path: str,
        sample_count: int,
        selected_chunk_count: int,
        generation_mode: str,
    ) -> None:
        """Persist export metadata for the UI and observability endpoints."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO dataset_exports
            (export_id, document_id, export_path, jsonl_path, sample_count, selected_chunk_count, generation_mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                export_id,
                document_id,
                export_path,
                jsonl_path,
                sample_count,
                selected_chunk_count,
                generation_mode,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def _safe_filename_stem(self, filename: str) -> str:
        """Create a filesystem-safe stem for dataset export filenames."""
        stem = Path(filename or "document").stem
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
        return stem[:60] or "document"

    def _log_generation(
        self,
        document_id: str,
        generated: int,
        exported: int,
        avg_quality: float,
        duration: float,
    ) -> None:
        """Log generation statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO dataset_generation_log 
            (document_id, samples_generated, samples_exported, avg_quality, duration_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                document_id,
                generated,
                exported,
                avg_quality,
                duration,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def _extract_json_array(self, response: str) -> Optional[List]:
        """Extract JSON array from LLM response."""
        import re
        import json

        text = (response or "").strip()

        # Remove markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Find array
        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1 and start < end:
            json_str = text[start : end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset generation statistics."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total samples
        cursor.execute("SELECT COUNT(*) as count FROM dataset_samples")
        total_samples = cursor.fetchone()["count"]

        # Exported samples
        cursor.execute(
            "SELECT COUNT(*) as count FROM dataset_samples WHERE is_exported = 1"
        )
        exported_samples = cursor.fetchone()["count"]

        # Average quality
        cursor.execute("SELECT AVG(quality_score) as avg FROM dataset_samples")
        avg_quality = cursor.fetchone()["avg"] or 0

        # By type
        cursor.execute("""
            SELECT sample_type, COUNT(*) as count 
            FROM dataset_samples 
            GROUP BY sample_type
        """)
        by_type = {row["sample_type"]: row["count"] for row in cursor.fetchall()}

        # Recent generations
        cursor.execute("""
            SELECT * FROM dataset_generation_log 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT document_id, export_path, jsonl_path, sample_count, selected_chunk_count, generation_mode, created_at
            FROM dataset_exports
            ORDER BY created_at DESC
            LIMIT 10
        """)
        recent_exports = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "total_samples": total_samples,
            "exported_samples": exported_samples,
            "pending_export": total_samples - exported_samples,
            "average_quality": round(avg_quality, 3),
            "by_type": by_type,
            "recent_generations": recent,
            "recent_exports": recent_exports,
            "settings": {
                "quality_threshold": self.quality_threshold,
                "dedup_threshold": self.dedup_threshold,
                "llm_sample_rate": self.llm_sample_rate,
                "batch_size": self.batch_size,
                "require_llm_for_export": self.require_llm_for_export,
                "background_postprocess_enabled": settings.dataset_background_postprocess_enabled,
            },
            "runtime_stats": self.stats,
        }

    def clear_cache(self) -> None:
        """Clear embedding cache to free memory."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def get_latest_export(self, document_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the most recent dataset export record."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if document_id:
            cursor.execute(
                """
                SELECT document_id, export_path, jsonl_path, sample_count, selected_chunk_count, generation_mode, created_at
                FROM dataset_exports
                WHERE document_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (document_id,),
            )
        else:
            cursor.execute(
                """
                SELECT document_id, export_path, jsonl_path, sample_count, selected_chunk_count, generation_mode, created_at
                FROM dataset_exports
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None


# Singleton instance
_dataset_factory = None


def get_dataset_factory() -> DatasetFactoryService:
    """Get singleton DatasetFactoryService instance."""
    global _dataset_factory
    if _dataset_factory is None:
        _dataset_factory = DatasetFactoryService()
    return _dataset_factory


async def continuous_dataset_generation() -> None:
    """
    Run dataset generation on a fixed interval while the feature is enabled.

    Keeping this in the service layer gives both application startup and router
    helpers a single public entry point.
    """
    factory = get_dataset_factory()

    logger.info("Starting continuous dataset generation background task")

    while settings.dataset_generation_enabled:
        try:
            logger.info("Starting scheduled dataset generation")
            total_samples = await asyncio.to_thread(factory.process_all_documents)
            logger.info(
                f"Scheduled dataset generation complete: {total_samples} samples generated"
            )
            await asyncio.sleep(settings.dataset_generation_interval_hours * 3600)
        except Exception as e:
            logger.error(f"Error in continuous dataset generation: {e}")
            await asyncio.sleep(300)
