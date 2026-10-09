"""
Chunking Service - handles semantic document segmentation.
"""

import uuid
import re
import sqlite3
from collections import Counter
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
        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip())
        
        headings = []
        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            
            if self._is_heading_line(line, line_counts=line_counts):
                clean_title = line.lstrip('#').strip()
                clean_title = re.sub(r'^\d+(\.\d+)*\.?\s+', '', clean_title)
                headings.append((clean_title, i))
        
        return headings
    
    def _is_heading_line(self, line: str, line_counts: Optional[Counter] = None) -> bool:
        """Check if a line appears to be a heading, rejecting running headers and list items."""
        line = line.strip()
        if not line:
            return False
        
        # Exclude pure page numbers / page labels (e.g. "12", "Page 12", "Page 12 of 300", "- 12 -")
        if re.match(r'^(?:[-–—\s]*)?(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?(?:[-–—\s]*)?$', line, re.IGNORECASE):
            return False

        # Exclude copyright/editorial boilerplate from creating headings
        if re.match(r'^(?:copyright|all rights reserved|isbn|printed in|published by)\b', line, re.IGNORECASE):
            return False

        # Running headers: lines that repeat frequently across pages are headers/footers, not unique section headings
        if line_counts and line_counts.get(line, 0) >= 4:
            return False

        # Markdown headings (# Heading)
        if re.match(r'^#{1,6}\s+\S', line):
            return len(line.split()) <= 20
        
        # Numbered headings (1. Heading, 1.2 Heading, 1.2.3 Heading)
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', line):
            words = line.split()
            # If line ends with period/comma/semicolon/question mark/exclamation, it's a list item or sentence, not a heading
            if line[-1] in ('.', ',', ';', '?', '!'):
                return False
            if len(words) <= 12 and len(line) >= 3:
                return True
        
        # Explicit section labels (Chapter 1:, Section 2, Part I, Appendix A)
        if re.match(r'^(?:CHAPTER|SECTION|PART|APPENDIX)\s+[0-9A-Za-zIVXLCDM]+[:\s]+.+', line, re.IGNORECASE):
            words = line.split()
            if line[-1] in (',', ';'):
                return False
            if len(words) <= 15:
                return True
        
        return False
    
    def _split_into_sections(
        self,
        text: str,
        line_counts: Optional[Counter] = None
    ) -> List[Tuple[Optional[str], str]]:
        """
        Split text into sections based on detected headings.
        
        Args:
            text: Input text
            line_counts: Optional line frequency counter for running header suppression
        
        Returns:
            List of (section_title, section_content) tuples
        """
        lines = text.split('\n')
        if line_counts is None and len(lines) > 20:
            line_counts = Counter(l.strip() for l in lines if l.strip())

        sections = []
        current_title = None
        current_content = []
        
        for line in lines:
            if self._is_heading_line(line, line_counts=line_counts):
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_title, content))
                    current_content = []
                
                clean_title = line.lstrip('#').strip()
                clean_title = re.sub(r'^\d+(\.\d+)*\.?\s+', '', clean_title)
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
        
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            word_slice = words[i:i + self.chunk_size]
            chunk_text = ' '.join(word_slice)

            if len(word_slice) < self.min_chunk_size:
                if chunks:
                    chunks[-1].content += ' ' + chunk_text
                    chunks[-1].word_count = len(chunks[-1].content.split())
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

    def _merge_undersized_chunks(
        self,
        raw_chunks: List[Chunk],
    ) -> List[Chunk]:
        """
        Merge undersized chunks (< min_chunk_size) into neighbors to prevent micro-chunks.
        Guarantees that no chunk in a multi-chunk document is left with word_count < min_chunk_size.
        """
        if not raw_chunks:
            return []

        total_words = sum(chunk.word_count for chunk in raw_chunks)
        if total_words <= self.min_chunk_size:
            first = raw_chunks[0]
            if len(raw_chunks) > 1:
                first.content = "\n\n".join(c.content for c in raw_chunks)
                first.word_count = len(first.content.split())
            return [first]

        max_merged_size = int(self.chunk_size * 1.25)
        merged: List[Chunk] = []

        for chunk in raw_chunks:
            if not merged:
                merged.append(chunk)
                continue

            prev = merged[-1]
            if prev.word_count < self.min_chunk_size:
                prev.content = (prev.content.rstrip() + '\n\n' + chunk.content.lstrip()).strip()
                prev.word_count = len(prev.content.split())
                if not prev.section_title and chunk.section_title:
                    prev.section_title = chunk.section_title
            elif chunk.word_count < self.min_chunk_size:
                if (prev.word_count + chunk.word_count) <= max_merged_size:
                    prev.content = (prev.content.rstrip() + '\n\n' + chunk.content.lstrip()).strip()
                    prev.word_count = len(prev.content.split())
                else:
                    merged.append(chunk)
            else:
                merged.append(chunk)

        # Backwards merge for trailing micro-chunk
        while len(merged) > 1 and merged[-1].word_count < self.min_chunk_size:
            last = merged.pop()
            merged[-1].content = (merged[-1].content.rstrip() + '\n\n' + last.content.lstrip()).strip()
            merged[-1].word_count = len(merged[-1].content.split())

        return merged

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
        
        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip()) if len(lines) > 20 else None
        sections = self._split_into_sections(text, line_counts=line_counts)
        
        raw_chunks: List[Chunk] = []
        position_index = 0
        
        for section_title, section_text in sections:
            section_chunks = self._chunk_text(
                section_text,
                section_title,
                position_index
            )
            raw_chunks.extend(section_chunks)
            position_index += len(section_text.split())
        
        all_chunks = self._merge_undersized_chunks(raw_chunks)
        
        # Assign document_id and re-index positions
        current_pos = 0
        for chunk in all_chunks:
            chunk.document_id = document_id
            chunk.position_index = current_pos
            current_pos += chunk.word_count
        
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
