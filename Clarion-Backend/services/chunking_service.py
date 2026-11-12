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
from utils.file_handler import normalize_extracted_text
from utils.sqlite import connect as sqlite_connect
from utils.text_metrics import count_words
from services.provenance_migrations import apply_provenance_schema
from services.source_provenance import (
    CHUNK_PROVENANCE_VERSION,
    attach_page_references,
    map_normalized_interval,
)
import json

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
        apply_provenance_schema(self.db_path)
    
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
        start_index: int,
        normalized_start: Optional[int] = None,
        source_metadata: Optional[dict] = None,
        source_pages: Optional[list[dict]] = None,
        source_blocks: Optional[list[dict]] = None,
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
        word_spans = list(re.finditer(r"\S+", text))
        words = [match.group(0) for match in word_spans]
        chunks = []

        def provenance_for(local_start: int, local_end: int) -> dict:
            if normalized_start is None or not source_metadata:
                return {"location_available": False, "location_unavailable_reason": "source_offset_map_unavailable"}
            absolute_start = normalized_start + local_start
            absolute_end = normalized_start + local_end
            offset_map = source_metadata.get("offset_map") or {}
            clipped_runs = []
            for run in offset_map.get("runs", []):
                left = max(absolute_start, int(run["normalized_start"]))
                right = min(absolute_end, int(run["normalized_end"]))
                if left >= right:
                    continue
                item = {**run, "normalized_start": left, "normalized_end": right}
                if run.get("operation") == "copy":
                    item["source_start"] = int(run["source_start"]) + left - int(run["normalized_start"])
                    item["source_end"] = item["source_start"] + right - left
                clipped_runs.append(item)
            intervals, operations = map_normalized_interval(
                {"runs": clipped_runs}, absolute_start, absolute_end
            )
            page_intervals, page_numbers = attach_page_references(intervals, source_pages or [])
            block_refs = []
            for block in source_blocks or []:
                bstart, bend = block.get("source_char_start"), block.get("source_char_end")
                if bstart is not None and bend is not None and any(
                    int(span["source_start"]) < int(bend) and int(span["source_end"]) > int(bstart)
                    for span in intervals
                ):
                    block_refs.append({
                        "block_id": block.get("block_id"),
                        "block_type": block.get("block_type"),
                        "block_index": block.get("block_index"),
                        "source_char_start": bstart,
                        "source_char_end": bend,
                    })
            return {
                "version": CHUNK_PROVENANCE_VERSION,
                "location_kind": source_metadata.get("location_kind", "extracted_text"),
                "extraction_version": source_metadata.get("extraction_version"),
                "normalization_version": source_metadata.get("normalization_version"),
                "normalized_intervals": [{"start": absolute_start, "end": absolute_end}],
                "source_intervals": page_intervals,
                "page_references": page_numbers,
                "block_references": block_refs,
                "normalization_operations": operations,
                "location_available": bool(intervals or block_refs),
                "location_unavailable_reason": None if (intervals or block_refs) else "source_offset_map_unavailable",
                "content_mappings": [{
                    "content_start": 0,
                    "content_end": local_end - local_start,
                    "normalized_start": absolute_start,
                    "normalized_end": absolute_end,
                    "offset_map_runs": clipped_runs,
                    "page_boundaries": [
                        {key: page.get(key) for key in ("page_number", "source_char_start", "source_char_end")}
                        for page in (source_pages or [])
                    ],
                    "block_references": block_refs,
                    "location_kind": source_metadata.get("location_kind", "extracted_text"),
                }],
            }

        def build_chunk(word_start: int, word_end: int, position: int) -> Chunk:
            char_start = word_spans[word_start].start()
            char_end = word_spans[word_end - 1].end()
            chunk_id = str(uuid.uuid4())
            return Chunk(
                chunk_id=chunk_id,
                document_id="",
                section_title=section_title,
                content=text[char_start:char_end],
                position_index=position,
                word_count=word_end - word_start,
                source_provenance={**provenance_for(char_start, char_end), "_local_start": char_start, "_local_end": char_end},
                raw_chunk_ids=[chunk_id],
            )
        
        if len(words) <= self.chunk_size:
            if not words:
                return []
            chunk = build_chunk(0, len(words), start_index)
            chunks.append(chunk)
            return chunks
        
        step = max(1, self.chunk_size - self.chunk_overlap)
        last_covered_idx = 0
        for i in range(0, len(words), step):
            word_slice = words[i:i + self.chunk_size]
            chunk_text = ' '.join(word_slice)

            if len(word_slice) < self.min_chunk_size:
                if chunks:
                    if last_covered_idx < len(words):
                        local_start = chunks[-1].source_provenance.get("_local_start", 0)
                        extension_end = word_spans[-1].end()
                        chunks[-1].content = text[local_start:extension_end]
                        chunks[-1].word_count = count_words(chunks[-1].content)
                        chunks[-1].source_provenance = {
                            **provenance_for(local_start, extension_end),
                            "_local_start": local_start,
                            "_local_end": extension_end,
                        }
                        last_covered_idx = len(words)
                continue

            chunk = build_chunk(i, i + len(word_slice), start_index + i)
            chunks.append(chunk)
            last_covered_idx = min(len(words), i + len(word_slice))
        
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

        def merge_locations(left: Chunk, right: Chunk, right_content_offset: int) -> None:
            a, b = left.source_provenance or {}, right.source_provenance or {}
            intervals = list(a.get("source_intervals", [])) + list(b.get("source_intervals", []))
            intervals.sort(key=lambda x: (x.get("page_number") or 0, x.get("source_start", 0), x.get("source_end", 0)))
            unique_intervals = []
            for interval in intervals:
                if interval not in unique_intervals:
                    unique_intervals.append(interval)
            normalized = list(a.get("normalized_intervals", [])) + list(b.get("normalized_intervals", []))
            normalized.sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))
            merged_normalized = []
            for interval in normalized:
                if merged_normalized and interval.get("start") <= merged_normalized[-1].get("end"):
                    merged_normalized[-1]["end"] = max(merged_normalized[-1]["end"], interval.get("end", 0))
                elif interval not in merged_normalized:
                    merged_normalized.append(dict(interval))
            block_refs = {item.get("block_id"): item for item in a.get("block_references", []) + b.get("block_references", [])}
            content_mappings = list(a.get("content_mappings", []))
            for mapping in b.get("content_mappings", []):
                content_mappings.append({**mapping,
                    "content_start": mapping["content_start"] + right_content_offset,
                    "content_end": mapping["content_end"] + right_content_offset,
                })
            left.source_provenance = {
                "version": CHUNK_PROVENANCE_VERSION,
                "location_kind": a.get("location_kind") or b.get("location_kind"),
                "extraction_version": a.get("extraction_version") or b.get("extraction_version"),
                "normalization_version": a.get("normalization_version") or b.get("normalization_version"),
                "normalized_intervals": merged_normalized,
                "source_intervals": unique_intervals,
                "page_references": sorted(set(a.get("page_references", []) + b.get("page_references", []))),
                "block_references": list(block_refs.values()),
                "normalization_operations": sorted(set(a.get("normalization_operations", []) + b.get("normalization_operations", []))),
                "location_available": bool(a.get("location_available") or b.get("location_available")),
                "location_unavailable_reason": None if (a.get("location_available") or b.get("location_available")) else "source_offset_map_unavailable",
                "content_mappings": content_mappings,
                "_local_start": min(a.get("_local_start", 0), b.get("_local_start", 0)),
                "_local_end": max(a.get("_local_end", 0), b.get("_local_end", 0)),
            }
            left.raw_chunk_ids = list(dict.fromkeys((left.raw_chunk_ids or [left.chunk_id]) + (right.raw_chunk_ids or [right.chunk_id])))

        total_words = sum(chunk.word_count for chunk in raw_chunks)
        if total_words <= self.min_chunk_size:
            first = raw_chunks[0]
            if len(raw_chunks) > 1:
                for chunk in raw_chunks[1:]:
                    offset = len(first.content) + 2
                    first.content = first.content.rstrip() + "\n\n" + chunk.content.lstrip()
                    merge_locations(first, chunk, offset)
                first.word_count = count_words(first.content)
            return [first]

        max_merged_size = int(self.chunk_size * 1.25)
        merged: List[Chunk] = []

        for chunk in raw_chunks:
            if not merged:
                merged.append(chunk)
                continue

            prev = merged[-1]
            if prev.word_count < self.min_chunk_size:
                right_offset = len(prev.content.rstrip()) + 2 - len(chunk.content) + len(chunk.content.lstrip())
                prev.content = (prev.content.rstrip() + '\n\n' + chunk.content.lstrip()).strip()
                prev.word_count = count_words(prev.content)
                merge_locations(prev, chunk, right_offset)
                if not prev.section_title and chunk.section_title:
                    prev.section_title = chunk.section_title
            elif chunk.word_count < self.min_chunk_size:
                if (prev.word_count + chunk.word_count) <= max_merged_size:
                    right_offset = len(prev.content.rstrip()) + 2 - len(chunk.content) + len(chunk.content.lstrip())
                    prev.content = (prev.content.rstrip() + '\n\n' + chunk.content.lstrip()).strip()
                    prev.word_count = count_words(prev.content)
                    merge_locations(prev, chunk, right_offset)
                else:
                    merged.append(chunk)
            else:
                merged.append(chunk)

        # Backwards merge for trailing micro-chunk
        while len(merged) > 1 and merged[-1].word_count < self.min_chunk_size:
            last = merged.pop()
            right_offset = len(merged[-1].content.rstrip()) + 2 - len(last.content) + len(last.content.lstrip())
            merged[-1].content = (merged[-1].content.rstrip() + '\n\n' + last.content.lstrip()).strip()
            merged[-1].word_count = count_words(merged[-1].content)
            merge_locations(merged[-1], last, right_offset)

        return merged

    def chunk_document(
        self,
        document_id: str,
        text: str,
        source_metadata: Optional[dict] = None,
        source_pages: Optional[list[dict]] = None,
        source_blocks: Optional[list[dict]] = None,
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
        
        if not (source_metadata and source_metadata.get("offset_map")):
            text = normalize_extracted_text(text)
        
        lines = text.split('\n')
        line_counts = Counter(l.strip() for l in lines if l.strip()) if len(lines) > 20 else None
        sections = self._split_into_sections(text, line_counts=line_counts)
        
        raw_chunks: List[Chunk] = []
        position_index = 0
        
        search_from = 0
        for section_title, section_text in sections:
            section_start = text.find(section_text, search_from)
            if section_start < 0:
                section_start = None
            else:
                search_from = section_start + len(section_text)
            section_chunks = self._chunk_text(
                section_text,
                section_title,
                position_index,
                normalized_start=section_start,
                source_metadata=source_metadata,
                source_pages=source_pages,
                source_blocks=source_blocks,
            )
            raw_chunks.extend(section_chunks)
            position_index += count_words(section_text)
        
        all_chunks = self._merge_undersized_chunks(raw_chunks)
        
        # Assign document_id and re-index positions
        current_pos = 0
        for chunk in all_chunks:
            chunk.document_id = document_id
            chunk.position_index = current_pos
            current_pos += chunk.word_count
            chunk.source_provenance.pop("_local_start", None)
            chunk.source_provenance.pop("_local_end", None)
            self.check_token_length(chunk.content)
        
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
                (chunk_id, document_id, section_title, content, position_index, word_count,
                 source_provenance_json, raw_chunk_ids_json, provenance_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.document_id,
                chunk.section_title,
                chunk.content,
                chunk.position_index,
                chunk.word_count,
                json.dumps(chunk.source_provenance, ensure_ascii=False),
                json.dumps(chunk.raw_chunk_ids, ensure_ascii=False),
                chunk.source_provenance.get("version"),
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
                ,source_provenance=json.loads(row["source_provenance_json"] or "{}") if "source_provenance_json" in row.keys() and row["source_provenance_json"] else {}
                ,raw_chunk_ids=json.loads(row["raw_chunk_ids_json"] or "[]") if "raw_chunk_ids_json" in row.keys() and row["raw_chunk_ids_json"] else [row["chunk_id"]]
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
            ,source_provenance=json.loads(row["source_provenance_json"] or "{}") if "source_provenance_json" in row.keys() and row["source_provenance_json"] else {}
            ,raw_chunk_ids=json.loads(row["raw_chunk_ids_json"] or "[]") if "raw_chunk_ids_json" in row.keys() and row["raw_chunk_ids_json"] else [row["chunk_id"]]
        )
    
    def delete_chunks(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted chunks for document {document_id}")

    def check_token_length(self, text: str, max_tokens: int = 512) -> int:
        """
        Check a conservative subword estimate and flag chunks that need
        model-tokenizer segmentation before encoding. This is not an exact count.
        """
        words = text.split()
        estimated_tokens = int(len(words) * 1.35)
        if estimated_tokens > max_tokens or len(words) > max_tokens:
            logger.warning(
                f"Chunk with {len(words)} words (~{estimated_tokens} estimated subword tokens) "
                f"exceeds the {max_tokens} subword token limit baseline; model-aware embedding segmentation is required before encoding."
            )
        return estimated_tokens

