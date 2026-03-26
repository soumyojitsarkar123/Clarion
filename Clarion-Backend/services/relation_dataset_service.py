"""
Relation Dataset Service - Logs mapping outputs for training data collection.

This service captures intermediate outputs from the knowledge mapping stage
to create a training dataset for relation validation experiments.
"""

import uuid
import sqlite3
import json
import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from pydantic import BaseModel, Field

from models.chunk import Chunk
from models.knowledge_map import KnowledgeMap, Relation
from utils.config import settings
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)

_DATASET_NOISE_TERMS = {
    "document",
    "question",
    "questions",
    "page",
    "attempt",
    "section",
    "semester",
    "examination",
    "schedule",
    "reporting",
}

_GENERIC_DATASET_TERMS = {
    "website",
    "frontend the website",
    "backend the website",
    "cse backend",
    "cse frontend",
    "name sec",
    "roll no",
    "enrollment number",
    "work assigned",
}


class RelationDatasetRecord(BaseModel):
    """A single relation record for the training dataset."""
    
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    relation_id: str
    
    concept_a: str
    concept_b: str
    relation_type: str
    
    llm_confidence: float = Field(ge=0.0, le=1.0)
    cooccurrence_score: Optional[float] = None
    semantic_similarity: Optional[float] = None
    
    chunk_context: str
    source_chunk_ids: List[str] = Field(default_factory=list)
    
    is_valid: Optional[bool] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class RelationDatasetService:
    """
    Service for logging relation extraction outputs to a dataset.
    
    This service is designed to be non-blocking - it queues records
    for async logging to avoid impacting pipeline performance.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.data_dir / "relation_dataset.db")
        self._init_database()
        self._pending_records: List[RelationDatasetRecord] = []
        self._lock = asyncio.Lock()
    
    def _init_database(self) -> None:
        """Initialize the relation dataset database."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relation_dataset (
                record_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                relation_id TEXT NOT NULL,
                concept_a TEXT NOT NULL,
                concept_b TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                llm_confidence REAL NOT NULL,
                cooccurrence_score REAL,
                semantic_similarity REAL,
                chunk_context TEXT NOT NULL,
                source_chunk_ids TEXT NOT NULL,
                is_valid INTEGER,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relation_doc 
            ON relation_dataset(document_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relation_type 
            ON relation_dataset(relation_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relation_valid 
            ON relation_dataset(is_valid)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Relation dataset initialized at %s", self.db_path)
    
    async def log_relation_batch(
        self,
        document_id: str,
        knowledge_map: KnowledgeMap,
        chunks: List[Chunk],
        embedding_similarities: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Log all relations from a knowledge map to the dataset.
        
        This method is async and non-blocking for the pipeline.
        
        Args:
            document_id: Document identifier
            knowledge_map: The knowledge map with extracted relations
            chunks: Source chunks for context
            embedding_similarities: Optional pre-computed embedding similarities
        
        Returns:
            Number of records logged
        """
        if not knowledge_map.relations:
            logger.info("No relations to log for document %s", document_id)
            return 0
        
        chunk_text_map = {chunk.chunk_id: chunk.content for chunk in chunks}
        concept_map = {concept.name: concept for concept in knowledge_map.concepts}
        
        records = []
        
        for relation in knowledge_map.relations:
            source_chunk_ids = self._collect_source_chunk_ids(
                relation.from_concept,
                relation.to_concept,
                concept_map,
            )
            chunk_context = self._build_relation_context(
                relation.from_concept,
                relation.to_concept,
                source_chunk_ids,
                chunks,
                chunk_text_map,
            )
            
            cooccurrence_score = self._calculate_cooccurrence(
                relation.from_concept,
                relation.to_concept,
                chunks
            )
            
            similarity = None
            if embedding_similarities:
                key = f"{relation.from_concept}|{relation.to_concept}"
                similarity = embedding_similarities.get(key)
            
            record = RelationDatasetRecord(
                document_id=document_id,
                relation_id=relation.id,
                concept_a=relation.from_concept,
                concept_b=relation.to_concept,
                relation_type=relation.relation_type.value if hasattr(relation.relation_type, 'value') else str(relation.relation_type),
                llm_confidence=relation.confidence or 0.5,
                cooccurrence_score=cooccurrence_score,
                semantic_similarity=similarity,
                chunk_context=chunk_context,
                source_chunk_ids=source_chunk_ids,
                is_valid=None,
                metadata={
                    "description": relation.description,
                    "original_confidence": relation.confidence,
                    "relation_confidence": relation.confidence,
                    "confidence_source": self._infer_confidence_source(relation),
                    "normalized_concept_a": self._normalize_concept_label(relation.from_concept),
                    "normalized_concept_b": self._normalize_concept_label(relation.to_concept),
                    "quality_flags": self._quality_flags(
                        relation.from_concept,
                        relation.to_concept,
                        chunk_context,
                        cooccurrence_score,
                    ),
                }
            )
            record.metadata["quality_score"] = self._quality_score(record)
            records.append(record)
        
        async with self._lock:
            self._pending_records.extend(records)
        
        await self._flush_records()
        
        logger.info(
            "Saved %d relation records for document %s",
            len(records), document_id
        )
        
        return len(records)
    
    def _calculate_cooccurrence(
        self,
        concept_a: str,
        concept_b: str,
        chunks: List[Chunk]
    ) -> float:
        """Calculate co-occurrence score based on chunk proximity."""
        if not chunks:
            return 0.0
        
        concept_a_lower = concept_a.lower()
        concept_b_lower = concept_b.lower()
        
        a_positions = []
        b_positions = []
        
        for i, chunk in enumerate(chunks):
            content_lower = chunk.content.lower()
            if concept_a_lower in content_lower:
                a_positions.append(i)
            if concept_b_lower in content_lower:
                b_positions.append(i)
        
        if not a_positions or not b_positions:
            return 0.0
        
        min_distance = min(
            abs(a - b) for a in a_positions for b in b_positions
        )
        
        max_distance = len(chunks) - 1
        if max_distance == 0:
            return 1.0
        
        cooccurrence = 1.0 - (min_distance / max_distance)
        return round(cooccurrence, 4)

    def _collect_source_chunk_ids(
        self,
        concept_a: str,
        concept_b: str,
        concept_map: Dict[str, Any],
    ) -> List[str]:
        """Collect source chunk ids for both relation concepts with stable ordering."""
        source_chunk_ids: List[str] = []
        for concept_name in (concept_a, concept_b):
            concept = concept_map.get(concept_name)
            if not concept:
                continue
            for chunk_id in concept.chunk_ids:
                if chunk_id and chunk_id not in source_chunk_ids:
                    source_chunk_ids.append(chunk_id)
        return source_chunk_ids[:6]

    def _build_relation_context(
        self,
        concept_a: str,
        concept_b: str,
        source_chunk_ids: List[str],
        chunks: List[Chunk],
        chunk_text_map: Dict[str, str],
    ) -> str:
        """Build a compact evidence-focused context snippet for one relation."""
        concept_a_lower = concept_a.lower()
        concept_b_lower = concept_b.lower()

        candidate_chunk_ids: List[str] = []
        for chunk_id in source_chunk_ids:
            if chunk_id in chunk_text_map and chunk_id not in candidate_chunk_ids:
                candidate_chunk_ids.append(chunk_id)

        for chunk in chunks:
            content_lower = chunk.content.lower()
            if (
                concept_a_lower in content_lower
                or concept_b_lower in content_lower
            ) and chunk.chunk_id not in candidate_chunk_ids:
                candidate_chunk_ids.append(chunk.chunk_id)

        ranked_chunk_ids = sorted(
            candidate_chunk_ids,
            key=lambda chunk_id: self._chunk_relevance_score(
                chunk_text_map.get(chunk_id, ""),
                concept_a_lower,
                concept_b_lower,
            ),
            reverse=True,
        )

        snippets: List[str] = []
        for chunk_id in ranked_chunk_ids[:4]:
            raw_text = chunk_text_map.get(chunk_id, "")
            if not raw_text:
                continue
            for snippet in self._extract_context_snippets(raw_text, concept_a, concept_b):
                if snippet and snippet not in snippets:
                    snippets.append(snippet)
                if len(snippets) >= 3:
                    break
            if len(snippets) >= 3:
                break

        context = " | ".join(snippets[:3]).strip()
        if len(context) > 520:
            context = context[:517].rsplit(" ", 1)[0].strip() + "..."
        return context

    def _chunk_relevance_score(self, text: str, concept_a_lower: str, concept_b_lower: str) -> float:
        """Score a chunk by how directly it supports the relation."""
        lowered = text.lower()
        score = 0.0
        if concept_a_lower in lowered:
            score += 1.0
        if concept_b_lower in lowered:
            score += 1.0
        if concept_a_lower in lowered and concept_b_lower in lowered:
            score += 2.0
        return score

    def _extract_context_snippets(self, text: str, concept_a: str, concept_b: str) -> List[str]:
        """Extract the most relevant cleaned snippets from raw chunk text."""
        cleaned_text = self._clean_context_text(text)
        if not cleaned_text:
            return []

        concept_a_lower = concept_a.lower()
        concept_b_lower = concept_b.lower()
        candidates: List[tuple[int, str]] = []

        parts = re.split(r"(?<=[.!?])\s+|\s+\|\s+|\n+", cleaned_text)
        for part in parts:
            snippet = self._clean_context_text(part)
            if not snippet or self._is_noisy_context(snippet):
                continue
            lowered = snippet.lower()
            score = 0
            if concept_a_lower in lowered:
                score += 1
            if concept_b_lower in lowered:
                score += 1
            if score == 2:
                score += 2
            if score == 0 and len(candidates) >= 2:
                continue
            candidates.append((score, snippet))

        if not candidates:
            fallback = self._clean_context_text(cleaned_text[:260])
            return [fallback] if fallback else []

        candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        selected: List[str] = []
        for _, snippet in candidates:
            if snippet not in selected:
                selected.append(snippet)
            if len(selected) >= 3:
                break
        return selected

    def _clean_context_text(self, text: str) -> str:
        """Normalize OCR-heavy context into a readable evidence snippet."""
        cleaned = str(text or "")
        cleaned = re.sub(r"\r\n?", "\n", cleaned)
        cleaned = re.sub(r"\|\s*\|+", "|", cleaned)
        cleaned = re.sub(r"(?<!\w)(?:[A-Za-z]\s+){3,}[A-Za-z](?!\w)", self._join_spaced_letters, cleaned)
        cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
        cleaned = re.sub(r"([.:;,\)])([A-Za-z])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" |")
        return cleaned

    def _join_spaced_letters(self, match: re.Match[str]) -> str:
        """Join OCR-spaced single-letter words into one token."""
        return match.group(0).replace(" ", "")

    def _is_noisy_context(self, text: str) -> bool:
        """Reject metadata-heavy or OCR-heavy context lines."""
        lowered = text.lower()
        if any(marker in lowered for marker in ["roll no", "enrollment number", "work assigned", "name sec"]):
            return True
        alpha_count = sum(char.isalpha() for char in text)
        digit_count = sum(char.isdigit() for char in text)
        if alpha_count and digit_count > alpha_count * 0.35:
            return True
        if len(re.findall(r"[A-Za-z][A-Za-z'-]+", text)) < 4:
            return True
        return False

    def _normalize_concept_label(self, value: str) -> str:
        """Create a stable normalized form for a concept label."""
        cleaned = re.sub(r"\s+", " ", str(value or "").strip().lower())
        cleaned = re.sub(r"[^a-z0-9\s-]+", "", cleaned)
        return cleaned

    def _infer_confidence_source(self, relation: Relation) -> str:
        """Label the provenance of the stored confidence score."""
        description = str(relation.description or "").lower()
        if "co-occurrence" in description or "cooccurrence" in description:
            return "heuristic_cooccurrence"
        return "relation_extractor"

    def _quality_flags(
        self,
        concept_a: str,
        concept_b: str,
        chunk_context: str,
        cooccurrence_score: Optional[float],
    ) -> List[str]:
        """Return simple review flags without changing the original record schema."""
        flags: List[str] = []
        normalized_a = self._normalize_concept_label(concept_a)
        normalized_b = self._normalize_concept_label(concept_b)

        if normalized_a in _GENERIC_DATASET_TERMS:
            flags.append("generic_concept_a")
        if normalized_b in _GENERIC_DATASET_TERMS:
            flags.append("generic_concept_b")
        if normalized_a == normalized_b:
            flags.append("duplicate_concepts")
        if cooccurrence_score is not None and cooccurrence_score <= 0:
            flags.append("weak_chunk_support")
        if not chunk_context:
            flags.append("missing_context")
        elif self._is_noisy_context(chunk_context):
            flags.append("noisy_context")
        return flags

    def _quality_score(self, record: RelationDatasetRecord) -> float:
        """Compute a lightweight quality score for export prioritization."""
        metadata = record.metadata or {}
        flags = metadata.get("quality_flags", [])
        score = 1.0
        score -= min(0.15 * len(flags), 0.45)
        score += min((record.cooccurrence_score or 0.0) * 0.25, 0.25)
        score += min(record.llm_confidence * 0.2, 0.2)
        if record.semantic_similarity is not None:
            score += min(record.semantic_similarity * 0.1, 0.1)
        return round(max(0.0, min(score, 1.0)), 3)
    
    async def _flush_records(self) -> None:
        """Flush pending records to database (async)."""
        await asyncio.sleep(0.1)
        
        async with self._lock:
            if not self._pending_records:
                return
            
            records_to_save = self._pending_records.copy()
            self._pending_records.clear()
        
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        
        for record in records_to_save:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO relation_dataset 
                    (record_id, document_id, relation_id, concept_a, concept_b, 
                     relation_type, llm_confidence, cooccurrence_score, 
                     semantic_similarity, chunk_context, source_chunk_ids,
                     is_valid, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.document_id,
                    record.relation_id,
                    record.concept_a,
                    record.concept_b,
                    record.relation_type,
                    record.llm_confidence,
                    record.cooccurrence_score,
                    record.semantic_similarity,
                    record.chunk_context,
                    json.dumps(record.source_chunk_ids),
                    record.is_valid,
                    record.created_at.isoformat(),
                    json.dumps(record.metadata)
                ))
            except Exception as e:
                logger.error("Error saving relation record: %s", e)
        
        conn.commit()
        conn.close()
        
        logger.info("Flushed %d relation records to dataset", len(records_to_save))

    def get_dataset(
        self,
        document_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        is_valid: Optional[bool] = None,
        limit: int = 1000
    ) -> List[RelationDatasetRecord]:
        """
        Retrieve relation dataset records.
        
        Args:
            document_id: Filter by document
            relation_type: Filter by relation type
            is_valid: Filter by validation status
            limit: Maximum records to return
        
        Returns:
            List of relation records
        """
        conn = sqlite_connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM relation_dataset WHERE 1=1"
        params = []
        
        if document_id:
            query += " AND document_id = ?"
            params.append(document_id)
        
        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)
        
        if is_valid is not None:
            query += " AND is_valid = ?"
            params.append(1 if is_valid else 0)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_record(row) for row in rows]
    
    def _row_to_record(self, row: sqlite3.Row) -> RelationDatasetRecord:
        """Convert database row to RelationDatasetRecord."""
        return RelationDatasetRecord(
            record_id=row["record_id"],
            document_id=row["document_id"],
            relation_id=row["relation_id"],
            concept_a=row["concept_a"],
            concept_b=row["concept_b"],
            relation_type=row["relation_type"],
            llm_confidence=row["llm_confidence"],
            cooccurrence_score=row["cooccurrence_score"],
            semantic_similarity=row["semantic_similarity"],
            chunk_context=row["chunk_context"],
            source_chunk_ids=json.loads(row["source_chunk_ids"]),
            is_valid=bool(row["is_valid"]) if row["is_valid"] is not None else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )
    
    def export_dataset(
        self,
        format: str = "json",
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export the relation dataset.
        
        Args:
            format: Export format (json, csv)
            document_id: Optional filter by document
        
        Returns:
            Exported data
        """
        records = self.get_dataset(document_id=document_id, limit=10000)
        
        if format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = [
                "record_id", "document_id", "relation_id",
                "concept_a", "concept_b", "relation_type",
                "llm_confidence", "cooccurrence_score", "semantic_similarity",
                "chunk_context", "quality_score", "quality_flags",
                "is_valid", "created_at"
            ]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                writer.writerow({
                    "record_id": record.record_id,
                    "document_id": record.document_id,
                    "relation_id": record.relation_id,
                    "concept_a": record.concept_a,
                    "concept_b": record.concept_b,
                    "relation_type": record.relation_type,
                    "llm_confidence": record.llm_confidence,
                    "cooccurrence_score": record.cooccurrence_score or "",
                    "semantic_similarity": record.semantic_similarity or "",
                    "chunk_context": self._clean_context_text(record.chunk_context)[:500],
                    "quality_score": self._quality_score(record),
                    "quality_flags": ", ".join((record.metadata or {}).get("quality_flags", [])),
                    "is_valid": record.is_valid if record.is_valid is not None else "",
                    "created_at": record.created_at.isoformat()
                })
            
            return {"format": "csv", "data": output.getvalue(), "count": len(records)}
        
        quality_summary = {
            "flagged_records": 0,
            "review_ready_records": 0,
        }
        json_records = []
        for r in records:
            metadata = r.metadata or {}
            quality_flags = metadata.get("quality_flags", [])
            quality_score = self._quality_score(r)
            if quality_flags:
                quality_summary["flagged_records"] += 1
            if quality_score >= 0.65 and not quality_flags:
                quality_summary["review_ready_records"] += 1
            json_records.append(
                {
                    "record_id": r.record_id,
                    "document_id": r.document_id,
                    "relation_id": r.relation_id,
                    "concept_a": r.concept_a,
                    "concept_b": r.concept_b,
                    "concept_a_normalized": metadata.get("normalized_concept_a") or self._normalize_concept_label(r.concept_a),
                    "concept_b_normalized": metadata.get("normalized_concept_b") or self._normalize_concept_label(r.concept_b),
                    "relation_type": r.relation_type,
                    "llm_confidence": r.llm_confidence,
                    "extraction_confidence": r.llm_confidence,
                    "confidence_source": metadata.get("confidence_source", "relation_extractor"),
                    "cooccurrence_score": r.cooccurrence_score,
                    "semantic_similarity": r.semantic_similarity,
                    "chunk_context": self._clean_context_text(r.chunk_context)[:500],
                    "source_chunk_ids": r.source_chunk_ids,
                    "quality_score": quality_score,
                    "quality_flags": quality_flags,
                    "relation_description": metadata.get("description"),
                    "is_valid": r.is_valid,
                    "created_at": r.created_at.isoformat(),
                }
            )

        return {
            "format": "json",
            "count": len(records),
            "quality_summary": quality_summary,
            "records": json_records,
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM relation_dataset")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) as labeled FROM relation_dataset WHERE is_valid IS NOT NULL")
        labeled = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT relation_type, COUNT(*) as count 
            FROM relation_dataset 
            GROUP BY relation_type
        """)
        type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT AVG(llm_confidence) as avg_conf FROM relation_dataset")
        avg_conf = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT AVG(cooccurrence_score) as avg_cooc FROM relation_dataset WHERE cooccurrence_score IS NOT NULL")
        avg_cooc = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_records": total,
            "labeled_records": labeled,
            "unlabeled_records": total - labeled,
            "relation_types": type_counts,
            "average_llm_confidence": round(avg_conf, 4),
            "average_cooccurrence": round(avg_cooc, 4)
        }
    
    def update_validation(
        self,
        record_id: str,
        is_valid: bool
    ) -> bool:
        """Update validation status for a record."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE relation_dataset SET is_valid = ? WHERE record_id = ?",
            (1 if is_valid else 0, record_id)
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def delete_document_records(self, document_id: str) -> int:
        """Delete all records for a document."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM relation_dataset WHERE document_id = ?",
            (document_id,)
        )
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted

    def export_document_snapshot(
        self,
        document_id: str,
        document_filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Write a human-readable JSON snapshot for one document into data/datasets."""
        export_data = self.export_dataset(format="json", document_id=document_id)
        records = export_data.get("records", [])
        if not records:
            return None

        safe_stem = self._safe_export_stem(document_filename or document_id)
        file_name = f"{safe_stem}_{document_id}_relations.json"
        output_path = settings.dataset_dir / file_name

        payload = {
            "document_id": document_id,
            "document_filename": document_filename,
            "exported_at": datetime.now().isoformat(),
            "count": len(records),
            "quality_summary": export_data.get("quality_summary", {}),
            "records": records,
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        output_path.write_text(content, encoding="utf-8")

        mirror_dir = self._workspace_dataset_dir()
        if mirror_dir and mirror_dir != settings.dataset_dir:
            mirror_dir.mkdir(parents=True, exist_ok=True)
            mirror_path = mirror_dir / file_name
            mirror_path.write_text(content, encoding="utf-8")
            logger.info("Exported dataset snapshot mirror to %s", mirror_path)
            return mirror_path

        logger.info("Exported dataset snapshot to %s", output_path)
        return output_path

    def _safe_export_stem(self, value: str) -> str:
        """Create a filesystem-safe filename stem."""
        stem = Path(value or "dataset").stem
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
        return cleaned or "dataset"

    def _workspace_dataset_dir(self) -> Optional[Path]:
        """Return the repo-level data/datasets directory when running from this workspace."""
        try:
            repo_root = Path(__file__).resolve().parents[2]
            return repo_root / "data" / "datasets"
        except Exception:
            return None
