"""
Summary Service - generates structured summaries from documents.
"""

import uuid
import sqlite3
import json
from typing import List, Optional
import logging

from models.chunk import Chunk
from models.summary import StructuredSummary, SummarySection
from .knowledge_map_service import LLMInterface
from utils.config import settings

logger = logging.getLogger(__name__)


class SummaryService:
    """
    Service for generating structured hierarchical summaries.
    Creates summaries organized by conceptual hierarchy rather than paragraphs.
    """
    
    def __init__(self):
        self.llm = LLMInterface()
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
        
        overall_summary = self.llm.generate_summary(title, chunks)
        
        sections = self._generate_sections(chunks)
        
        summary = StructuredSummary(
            document_id=document_id,
            title=title,
            overall_summary=overall_summary,
            sections=sections,
            metadata={
                "chunk_count": len(chunks),
                "section_count": len(sections)
            }
        )
        
        self._save_summary(summary)
        
        logger.info(f"Summary generated with {len(sections)} sections")
        return summary
    
    def _generate_sections(self, chunks: List[Chunk]) -> List[SummarySection]:
        """Generate hierarchical sections from chunks."""
        sections = []
        
        current_main = None
        current_main_id = None
        
        for i, chunk in enumerate(chunks):
            section_title = chunk.section_title or f"Section {i+1}"
            
            is_subheading = (
                chunk.content.strip().startswith(('•', '-', '1.', '2.', '3.')) or
                len(chunk.content) < 200
            )
            
            if not is_subheading or current_main is None:
                current_main_id = str(uuid.uuid4())
                current_main = SummarySection(
                    id=current_main_id,
                    title=section_title,
                    level=1,
                    summary=self._summarize_section(chunk.content),
                    key_points=self._extract_key_points(chunk.content),
                    related_concepts=[],
                    child_sections=[]
                )
                sections.append(current_main)
            else:
                child_id = str(uuid.uuid4())
                child = SummarySection(
                    id=child_id,
                    title=section_title,
                    level=2,
                    summary=self._summarize_section(chunk.content),
                    key_points=self._extract_key_points(chunk.content),
                    related_concepts=[],
                    child_sections=[]
                )
                current_main.child_sections.append(child_id)
                sections.append(child)
        
        return sections
    
    def _summarize_section(self, text: str) -> str:
        """Generate a brief summary of section text."""
        if len(text) < 200:
            return text
        
        prompt = f"""Provide a brief 1-2 sentence summary of this text:

{text[:1000]}

Summary:"""
        
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.warning(f"Error generating section summary: {str(e)}")
            return text[:200] + "..."
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        prompt = f"""Extract 2-4 key points from this text as a JSON array of strings:

{text[:1500]}

Key Points (JSON array):"""
        
        try:
            response = self.llm.generate(prompt)
            points = json.loads(response)
            return points if isinstance(points, list) else []
        except Exception as e:
            logger.warning(f"Error extracting key points: {str(e)}")
            return []
    
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
