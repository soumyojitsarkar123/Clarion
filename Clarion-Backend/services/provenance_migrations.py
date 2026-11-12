"""Additive, versioned SQLite schema for source and embedding provenance."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from utils.sqlite import connect as sqlite_connect

PROVENANCE_SCHEMA_VERSION = 1


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _add_column(conn: sqlite3.Connection, table: str, name: str, declaration: str) -> None:
    if name not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def apply_provenance_schema(db_path: str | Path, *, fail_after_step: Optional[int] = None) -> None:
    """Apply schema v1 atomically; safe to call during each service initialization.

    ``fail_after_step`` is only used by isolated migration rollback tests.
    Existing rows receive NULL provenance so legacy data is never given invented
    page or source locations.
    """
    conn = sqlite_connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        applied = conn.execute("SELECT 1 FROM schema_migrations WHERE version=?", (PROVENANCE_SCHEMA_VERSION,)).fetchone()
        if applied:
            # Idempotent defensive check: older partial installs can be repaired
            # additively without changing or fabricating historical provenance.
            pass
        conn.execute("BEGIN IMMEDIATE")
        step = 0

        def checkpoint() -> None:
            nonlocal step
            step += 1
            if fail_after_step is not None and step >= fail_after_step:
                raise RuntimeError("Injected provenance migration failure")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                file_size INTEGER NOT NULL, file_type TEXT NOT NULL,
                upload_time TEXT NOT NULL, page_count INTEGER,
                word_count INTEGER, status TEXT NOT NULL,
                text_content TEXT, error_message TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_source_pages (
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                raw_text TEXT,
                source_char_start INTEGER,
                source_char_end INTEGER,
                extraction_method TEXT NOT NULL,
                extraction_version TEXT NOT NULL,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                location_available INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(document_id, page_number),
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        checkpoint()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_source_blocks (
                document_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                block_type TEXT NOT NULL,
                block_index INTEGER NOT NULL,
                raw_text TEXT,
                source_char_start INTEGER,
                source_char_end INTEGER,
                extraction_method TEXT NOT NULL,
                extraction_version TEXT NOT NULL,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY(document_id, block_id),
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        checkpoint()

        # Base table shapes also support creation when an older service has not
        # initialized its table yet; existing rows remain untouched.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                section_title TEXT, content TEXT NOT NULL,
                position_index INTEGER NOT NULL, word_count INTEGER NOT NULL
            )
        """)
        _add_column(conn, "chunks", "source_provenance_json", "TEXT")
        _add_column(conn, "chunks", "raw_chunk_ids_json", "TEXT")
        _add_column(conn, "chunks", "provenance_version", "TEXT")
        checkpoint()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                vector_index INTEGER NOT NULL
            )
        """)
        _add_column(conn, "embeddings", "embedding_model_id", "TEXT")
        _add_column(conn, "embeddings", "embedding_model_revision", "TEXT")
        checkpoint()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_segments (
                segment_id TEXT PRIMARY KEY,
                source_chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(source_chunk_id, segment_index)
            )
        """)
        _add_column(conn, "embedding_segments", "source_provenance_json", "TEXT")
        _add_column(conn, "embedding_segments", "token_count", "INTEGER")
        _add_column(conn, "embedding_segments", "embedding_model_id", "TEXT")
        _add_column(conn, "embedding_segments", "embedding_model_revision", "TEXT")
        _add_column(conn, "embedding_segments", "generation_version", "TEXT")
        checkpoint()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_segments_v2 (
                segment_id TEXT NOT NULL,
                embedding_model_id TEXT NOT NULL,
                embedding_model_revision TEXT NOT NULL,
                source_chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER,
                source_provenance_json TEXT,
                generation_version TEXT NOT NULL,
                PRIMARY KEY(embedding_model_id, embedding_model_revision, segment_id),
                UNIQUE(embedding_model_id, embedding_model_revision, source_chunk_id, segment_index)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_vector_map (
                document_id TEXT NOT NULL,
                embedding_model_id TEXT NOT NULL,
                embedding_model_revision TEXT NOT NULL,
                vector_id TEXT NOT NULL,
                vector_index INTEGER NOT NULL,
                segment_id TEXT,
                retained_chunk_id TEXT NOT NULL,
                raw_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
                source_provenance_json TEXT,
                PRIMARY KEY(document_id, embedding_model_id, embedding_model_revision, vector_id),
                UNIQUE(document_id, embedding_model_id, embedding_model_revision, vector_index)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_map_document_model ON embedding_vector_map(document_id, embedding_model_id, embedding_model_revision)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_segment_source_model ON embedding_segments_v2(source_chunk_id, embedding_model_id, embedding_model_revision)")
        checkpoint()
        _add_column(conn, "documents", "source_provenance_json", "TEXT")
        _add_column(conn, "documents", "source_file_path", "TEXT")
        _add_column(conn, "documents", "source_file_sha256", "TEXT")
        _add_column(conn, "documents", "extracted_page_count", "INTEGER")
        _add_column(conn, "documents", "text_bearing_page_count", "INTEGER")
        _add_column(conn, "documents", "raw_word_count", "INTEGER")
        checkpoint()
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, name) VALUES (?, ?)",
            (PROVENANCE_SCHEMA_VERSION, "source_provenance_v1"),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
