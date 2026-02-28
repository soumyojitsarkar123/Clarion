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
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from pydantic import BaseModel, Field

from models.chunk import Chunk
from models.knowledge_map import KnowledgeMap, Relation
from utils.config import settings
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)


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
        
        records = []
        
        for relation in knowledge_map.relations:
            source_chunks = []
            context_parts = []
            
            for concept in knowledge_map.concepts:
                if concept.name == relation.from_concept or concept.name == relation.to_concept:
                    source_chunks.extend(concept.chunk_ids)
                    for cid in concept.chunk_ids[:2]:
                        if cid in chunk_text_map:
                            context_parts.append(chunk_text_map[cid][:300])
            
            chunk_context = " | ".join(context_parts) if context_parts else ""
            source_chunk_ids = list(set(source_chunks))[:5]
            
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
                    "original_confidence": relation.confidence
                }
            )
            records.append(record)
        
        async with self._lock:
            self._pending_records.extend(records)
        
        asyncio.create_task(self._flush_records())
        
        logger.info(
            "Queued %d relation records for document %s (async logging)",
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
                "chunk_context", "is_valid", "created_at"
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
                    "chunk_context": record.chunk_context[:500],
                    "is_valid": record.is_valid if record.is_valid is not None else "",
                    "created_at": record.created_at.isoformat()
                })
            
            return {"format": "csv", "data": output.getvalue(), "count": len(records)}
        
        return {
            "format": "json",
            "count": len(records),
            "records": [
                {
                    "record_id": r.record_id,
                    "document_id": r.document_id,
                    "relation_id": r.relation_id,
                    "concept_a": r.concept_a,
                    "concept_b": r.concept_b,
                    "relation_type": r.relation_type,
                    "llm_confidence": r.llm_confidence,
                    "cooccurrence_score": r.cooccurrence_score,
                    "semantic_similarity": r.semantic_similarity,
                    "chunk_context": r.chunk_context[:500],
                    "is_valid": r.is_valid,
                    "created_at": r.created_at.isoformat()
                }
                for r in records
            ]
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
