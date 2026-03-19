"""
Chunking Service - handles semantic document segmentation.
"""

import uuid
import re
import sqlite3
from typing import List, Optional, Tuple
import logging

from models.chunk import Chunk, ChunkCreate
from utils.config import settings
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Service for structure-aware document chunking.
    Chunks documents by semantic boundaries, preserving headings and structure.
    """
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.min_chunk_size = settings.min_chunk_size
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for chunks."""
        db_path = settings.data_dir / "clarion.db"
        self.db_path = str(db_path)
        
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                section_title TEXT,
                content TEXT NOT NULL,
                position_index INTEGER NOT NULL,
                word_count INTEGER NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _detect_headings(self, text: str) -> List[Tuple[str, int]]:
        """
        Detect headings in text with their positions.
        
        Args:
            text: Input text
        
        Returns:
            List of (heading_text, position) tuples
        """
        heading_patterns = [
            r'^(#{1,6})\s+(.+)$',
            r'^(\d+\.)+\s+(.+)$',
            r'^([A-Z][^.:]+):?$',
            r'^(?:CHAPTER|SECTION|PART)\s+\d+[:\s]+(.+)$',
        ]
        
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern in heading_patterns:
                match = re.match(pattern, line, re.MULTILINE | re.IGNORECASE)
                if match and match.lastindex:
                    heading_text = match.group(2) if match.lastindex >= 2 else match.group(1)
                    headings.append((heading_text.strip(), i))
                    break
        
        return headings
    
    def _is_heading_line(self, line: str) -> bool:
        """Check if a line appears to be a heading."""
        line = line.strip()
        
        if line.startswith('#'):
            return True
        
        if re.match(r'^\d+\.\s+[A-Z]', line):
            return True
        
        if re.match(r'^[A-Z][^.:]{5,50}:?$', line) and len(line.split()) <= 6:
            return True
        
        return False
    
    def _split_into_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        """
        Split text into sections based on detected headings.
        
        Args:
            text: Input text
        
        Returns:
            List of (section_title, section_content) tuples
        """
        lines = text.split('\n')
        sections = []
        current_title = None
        current_content = []
        
        for line in lines:
            if self._is_heading_line(line):
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_title, content))
                    current_content = []
                
                clean_title = line.lstrip('#').strip()
                clean_title = re.sub(r'^\d+\.\s+', '', clean_title)
                current_title = clean_title
            else:
                current_content.append(line)
        
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((current_title, content))
        
        if not sections:
            sections.append((None, text))
        
        return sections
    
    def _chunk_text(
        self,
        text: str,
        section_title: Optional[str],
        start_index: int
    ) -> List[Chunk]:
        """
        Chunk a section of text into smaller pieces.
        
        Args:
            text: Section text
            section_title: Section heading if detected
            start_index: Starting position index
        
        Returns:
            List of Chunk objects
        """
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id="",
                section_title=section_title,
                content=text,
                position_index=start_index,
                word_count=len(words)
            )
            chunks.append(chunk)
            return chunks
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            word_slice = words[i:i + self.chunk_size]
            chunk_text = ' '.join(word_slice)
            
            if len(chunk_text.strip()) < self.min_chunk_size:
                if chunks:
                    chunks[-1].content += ' ' + chunk_text
                    chunks[-1].word_count += len(word_slice)
                continue
            
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id="",
                section_title=section_title,
                content=chunk_text,
                position_index=start_index + i,
                word_count=len(word_slice)
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(
        self,
        document_id: str,
        text: str
    ) -> List[Chunk]:
        """
        Chunk a document into semantic pieces.
        
        Args:
            document_id: Parent document ID
            text: Document text content
        
        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking document {document_id}")
        
        sections = self._split_into_sections(text)
        
        all_chunks = []
        position_index = 0
        
        for section_title, section_text in sections:
            section_chunks = self._chunk_text(
                section_text,
                section_title,
                position_index
            )
            
            for chunk in section_chunks:
                chunk.document_id = document_id
            
            all_chunks.extend(section_chunks)
            position_index += len(section_text.split())
        
        self._save_chunks(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks for document {document_id}")
        return all_chunks
    
    def _save_chunks(self, chunks: List[Chunk]) -> None:
        """Save chunks to database."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk in chunks:
            cursor.execute("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, section_title, content, position_index, word_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.document_id,
                chunk.section_title,
                chunk.content,
                chunk.position_index,
                chunk.word_count
            ))
        
        conn.commit()
        conn.close()
    
    def get_chunks(self, document_id: str) -> List[Chunk]:
        """
        Retrieve all chunks for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            List of Chunk objects
        """
        conn = sqlite_connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chunks 
            WHERE document_id = ? 
            ORDER BY position_index
        """, (document_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        chunks = []
        for row in rows:
            chunks.append(Chunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                section_title=row["section_title"],
                content=row["content"],
                position_index=row["position_index"],
                word_count=row["word_count"]
            ))
        
        return chunks
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a specific chunk by ID."""
        conn = sqlite_connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            section_title=row["section_title"],
            content=row["content"],
            position_index=row["position_index"],
            word_count=row["word_count"]
        )
    
    def delete_chunks(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted chunks for document {document_id}")
