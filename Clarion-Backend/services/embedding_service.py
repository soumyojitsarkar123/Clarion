"""
Embedding Service - generates embeddings for document chunks.
"""

import sqlite3
import asyncio
import math
import hashlib
import struct
import os
import re
import json
from pathlib import Path
from typing import List, Optional, Any, Callable, Dict
import logging

from vectorstore import VectorStore
from models.chunk import Chunk
from utils.config import settings, initialize_compute_settings
from utils.sqlite import connect as sqlite_connect
from utils.text_metrics import count_words
from services.provenance_migrations import apply_provenance_schema
from services.source_provenance import EMBEDDING_PROVENANCE_VERSION, project_content_interval

logger = logging.getLogger(__name__)

# Module-level singleton model cache to prevent redundant model loads
_MODEL_CACHE: Dict[str, Any] = {}


class EmbeddingService:
    """
    Service for generating embeddings using SentenceTransformers.
    Manages embedding generation, vector caching, and FAISS vector storage.
    """
    
    def __init__(self):
        self.model_name = settings.embedding_model_name
        self.batch_size = settings.embedding_batch_size
        self.device = settings.embedding_device_actual or initialize_compute_settings()
        self.model: Optional[Any] = _MODEL_CACHE.get(self.model_name)
        self.model_revision: Optional[str] = self._resolve_model_revision(self.model) if self.model is not None else None
        self.active_model_id: Optional[str] = None
        self._segment_token_counts: Dict[str, Optional[int]] = {}
        # Canonical embedding model BAAI/bge-large-en-v1.5 produces 1024-dim vectors
        self.fallback_dimension = 1024
        self.using_fallback_embeddings = False
        self._segment_spans: Dict[str, tuple[int, int]] = {}
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for embeddings and chunk vector cache."""
        db_path = settings.data_dir / "clarion.db"
        self.db_path = str(db_path)
        
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                vector_index INTEGER NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                content_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                PRIMARY KEY (content_hash, model_name)
            )
        """)
        # A source chunk may require several model-safe embedding segments.  The
        # source chunk remains authoritative; this table is the durable mapping
        # from a vector id back to its source and exact character span.
        cursor.execute("""
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
        
        conn.commit()
        conn.close()
        apply_provenance_schema(self.db_path)
    
    def _load_model(self) -> None:
        """Load the sentence transformer model with singleton caching and CPU thread tuning."""
        if self.model is None and self.model_name in _MODEL_CACHE:
            self.model = _MODEL_CACHE[self.model_name]
            self.model_revision = self._resolve_model_revision(self.model)
            return

        if self.model is None:
            try:
                import torch
                if not torch.cuda.is_available():
                    threads = min(8, max(1, os.cpu_count() or 4))
                    torch.set_num_threads(threads)
            except Exception:
                pass

            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError(
                    "Failed to import sentence-transformers. Check dependency compatibility."
                ) from e
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                loaded = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
                logger.info("Embedding model loaded successfully from local cache")
            except Exception as local_error:
                logger.info(
                    "Embedding model not found in local cache; attempting remote load. Reason: %s",
                    local_error,
                )
                try:
                    loaded = SentenceTransformer(self.model_name, device=self.device)
                    logger.info("Embedding model loaded successfully")
                except Exception as remote_error:
                    raise RuntimeError(
                        "Embedding model could not be loaded. Ensure the model is cached locally "
                        "or provide network/Hugging Face access for the first download."
                    ) from remote_error

            self.model = loaded
            self.model_revision = self._resolve_model_revision(loaded)
            _MODEL_CACHE[self.model_name] = loaded

    def _resolve_model_revision(self, model: Any) -> Optional[str]:
        """Resolve a pinned Hugging Face commit from loaded config/cache when exposed."""
        candidates = []
        try:
            candidates.append(model._first_module().auto_model.config._commit_hash)
        except Exception:
            pass
        try:
            candidates.append(model._first_module().auto_model.name_or_path)
        except Exception:
            pass
        cache_roots = []
        if os.getenv("HUGGINGFACE_HUB_CACHE"):
            cache_roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
        if os.getenv("HF_HOME"):
            cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
        cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        repo_dir_name = "models--" + self.model_name.replace("/", "--")
        for cache_root in cache_roots:
            ref_path = cache_root / repo_dir_name / "refs" / "main"
            try:
                candidates.append(ref_path.read_text(encoding="ascii").strip())
            except OSError:
                continue
        for value in candidates:
            if not value:
                continue
            match = re.search(r"(?:snapshots[/\\])([0-9a-f]{8,64})(?:[/\\]|$)", str(value))
            if match:
                return match.group(1)
            if re.fullmatch(r"[0-9a-f]{8,64}", str(value)):
                return str(value)
        return None

    def _configured_model_id(self) -> str:
        if self.using_fallback_embeddings:
            return "clarion-hash-fallback-v1"
        if self.model is None:
            return "clarion-hash-fallback-v1"
        revision = self.model_revision or "revision-unavailable"
        return f"{self.model_name}@{revision}"

    def _hash_content(self, text: str) -> str:
        """Compute SHA-256 hash of text content for caching."""
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _get_cached_embeddings(self, texts: List[str], model_id: str) -> Dict[str, List[float]]:
        """Retrieve precomputed embeddings from SQLite cache."""
        cached: Dict[str, List[float]] = {}
        if not texts:
            return cached
        hashes = [self._hash_content(t) for t in texts]
        unique_hashes = list(set(hashes))
        try:
            conn = sqlite_connect(self.db_path)
            cursor = conn.cursor()
            for i in range(0, len(unique_hashes), 500):
                batch_hashes = unique_hashes[i : i + 500]
                placeholders = ",".join("?" for _ in batch_hashes)
                cursor.execute(
                    f"SELECT content_hash, vector_blob FROM chunk_embeddings WHERE model_name = ? AND content_hash IN ({placeholders})",
                    [model_id] + batch_hashes,
                )
                for ch_hash, blob in cursor.fetchall():
                    count = len(blob) // 4
                    cached[ch_hash] = list(struct.unpack(f"{count}f", blob))
            conn.close()
        except Exception as e:
            logger.warning(f"Error reading chunk embedding cache: {e}")
        return cached

    def _save_cached_embeddings(self, hash_to_vector: Dict[str, List[float]], model_id: str) -> None:
        """Store newly generated embeddings into SQLite cache."""
        if not hash_to_vector:
            return
        try:
            conn = sqlite_connect(self.db_path)
            cursor = conn.cursor()
            items = [
                (ch_hash, model_id, struct.pack(f"{len(vec)}f", *vec))
                for ch_hash, vec in hash_to_vector.items()
            ]
            cursor.executemany(
                "INSERT OR REPLACE INTO chunk_embeddings (content_hash, model_name, vector_blob) VALUES (?, ?, ?)",
                items,
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Error saving chunk embedding cache: {e}")
    
    def _model_token_limit(self) -> int:
        """Return the loaded model's real sequence limit, never a hard-coded guess."""
        self._load_model()
        limit = getattr(self.model, "get_max_seq_length", lambda: None)()
        if not isinstance(limit, int) or limit <= 0:
            limit = getattr(self.model, "max_seq_length", None)
        if not isinstance(limit, int) or limit <= 0:
            limit = getattr(getattr(self.model, "tokenizer", None), "model_max_length", None)
        if not isinstance(limit, int) or limit <= 0 or limit > 1_000_000:
            raise RuntimeError("Embedding model did not expose a usable max_seq_length")
        return limit

    def _token_offsets(self, text: str) -> List[tuple[int, int]]:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Embedding model tokenizer is unavailable")
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
        if offsets is None:
            raise RuntimeError("Tokenizer does not provide offset mappings; cannot safely preserve source spans")
        return [(int(start), int(end)) for start, end in offsets if end > start]

    def _ensure_offset_tokenizer(self) -> Any:
        """Use a fast tokenizer when the model's default tokenizer lacks offsets."""
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "is_fast", False):
            return tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, local_files_only=True)
        except Exception as local_error:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            except Exception as remote_error:
                raise RuntimeError("A fast tokenizer with offset mappings is required for lossless embedding segmentation") from remote_error
        self.model.tokenizer = tokenizer
        return tokenizer

    def prepare_embedding_segments(self, chunks: List[Chunk]) -> List[Chunk]:
        """Split source chunks at tokenizer offsets so every encode input fits the real model limit.

        Segment text uses non-overlapping source character spans. Original chunks are not
        modified, so their content remains recoverable and graph/mapping provenance stays
        attached to the original chunk id.
        """
        if not chunks:
            return []
        try:
            limit = self._model_token_limit()
        except Exception as error:
            # The deterministic fallback is not a transformer and has no sequence
            # ceiling. Retain the source chunks unchanged rather than pretending a
            # tokenizer-aware split occurred.
            logger.warning("Embedding model unavailable; source chunks will use fallback vectors: %s", error)
            return list(chunks)
        tokenizer = self._ensure_offset_tokenizer()
        special_count_method = getattr(tokenizer, "num_special_tokens_to_add", None)
        if callable(special_count_method):
            special_count = int(special_count_method(pair=False))
        else:
            special_count = len(tokenizer.encode("", add_special_tokens=True, truncation=False))
        payload_limit = limit - special_count
        if payload_limit <= 0:
            raise RuntimeError(f"Invalid embedding token limit {limit} (special tokens={special_count})")
        segments: List[Chunk] = []
        for chunk in chunks:
            offsets = self._token_offsets(chunk.content)
            if not offsets:
                continue
            first = 0
            segment_index = 0
            while first < len(offsets):
                candidate_end = min(len(offsets), first + payload_limit)
                # Include inter-token whitespace in the preceding segment. This
                # makes the ordered segment spans a non-overlapping, exact cover
                # of the original source string rather than merely a token cover.
                start = 0 if segment_index == 0 else offsets[first][0]
                while candidate_end > first:
                    end = offsets[candidate_end][0] if candidate_end < len(offsets) else len(chunk.content)
                    segment_text = chunk.content[start:end]
                    token_count = len(tokenizer.encode(segment_text, add_special_tokens=True, truncation=False))
                    if token_count <= limit:
                        break
                    candidate_end -= 1
                if candidate_end == first:
                    raise RuntimeError("A single tokenizer span exceeds the embedding model token limit")
                segment_id = f"{chunk.chunk_id}:seg:{segment_index}"
                self._segment_spans[segment_id] = (start, end)
                self._segment_token_counts[segment_id] = token_count
                source_map = project_content_interval(chunk.source_provenance, start, end)
                segment_provenance = {
                    "version": EMBEDDING_PROVENANCE_VERSION,
                    "source_chunk_id": chunk.chunk_id,
                    "retained_chunk_id": chunk.chunk_id,
                    "raw_chunk_ids": chunk.raw_chunk_ids or [chunk.chunk_id],
                    "segment_id": segment_id,
                    "document_id": chunk.document_id,
                    "segment_index": segment_index,
                    "location_kind": (chunk.source_provenance or {}).get("location_kind"),
                    "extraction_version": (chunk.source_provenance or {}).get("extraction_version"),
                    "normalization_version": (chunk.source_provenance or {}).get("normalization_version"),
                    **source_map,
                }
                segments.append(Chunk(
                    chunk_id=segment_id,
                    document_id=chunk.document_id,
                    section_title=chunk.section_title,
                    content=segment_text,
                    position_index=chunk.position_index,
                    word_count=count_words(segment_text),
                    source_provenance=segment_provenance,
                    raw_chunk_ids=chunk.raw_chunk_ids or [chunk.chunk_id],
                ))
                first = candidate_end
                segment_index += 1
        return segments

    def generate_embedding_segments(self, chunks: List[Chunk]) -> tuple[List[Chunk], List[List[float]]]:
        """Produce model-safe vector inputs and their vectors in identical order."""
        segments = self.prepare_embedding_segments(chunks)
        return segments, self._generate_safe_embeddings(segments)

    def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks, leveraging SQLite cache.
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            List of embedding vectors
        """
        _, embeddings = self.generate_embedding_segments(chunks)
        return embeddings

    def _generate_safe_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Encode only inputs already verified by ``prepare_embedding_segments``."""
        texts = [chunk.content for chunk in chunks]
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} chunks")

        try:
            self._load_model()
        except Exception as error:
            logger.warning("Embedding model load unavailable; cache/fallback identity selected: %s", error)
        configured_model_id = self._configured_model_id()
        active_model_id = (
            configured_model_id if self.model is not None else "clarion-hash-fallback-v1"
        )
        self.active_model_id = active_model_id
        # Legacy model-name-only cache entries are intentionally not reused as
        # versioned vectors because their revision cannot be established.
        cached_map = self._get_cached_embeddings(texts, active_model_id)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        results: List[Optional[List[float]]] = [None] * len(texts)

        for idx, text in enumerate(texts):
            h = self._hash_content(text)
            if h in cached_map:
                results[idx] = cached_map[h]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if not uncached_texts:
            self.using_fallback_embeddings = active_model_id == "clarion-hash-fallback-v1"
            return [r for r in results if r is not None]

        logger.info(
            f"Encoding {len(uncached_texts)} uncached chunks ({len(texts) - len(uncached_texts)} served from cache)"
        )

        encoding_fallback = False
        try:
            if self.model is None:
                raise RuntimeError("Embedding model unavailable")
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(uncached_texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embedding_list = embeddings.tolist()
            self.using_fallback_embeddings = False
        except Exception as error:
            logger.warning(
                "Embedding model unavailable; using deterministic fallback embeddings. Error: %s",
                error,
            )
            self.using_fallback_embeddings = True
            self.active_model_id = "clarion-hash-fallback-v1"
            encoding_fallback = True
            # Never mix cached vectors from one model with fallback vectors in
            # the same FAISS index.
            results = [None] * len(texts)
            fallback_cache = self._get_cached_embeddings(texts, self.active_model_id)
            uncached_indices = []
            uncached_texts = []
            for index, text in enumerate(texts):
                cached_vector = fallback_cache.get(self._hash_content(text))
                if cached_vector is None:
                    uncached_indices.append(index)
                    uncached_texts.append(text)
                else:
                    results[index] = cached_vector
            embedding_list = [self._fallback_embed_text(text) for text in uncached_texts]

        # Save newly computed embeddings to cache
        to_save: Dict[str, List[float]] = {}
        for orig_idx, vec, text in zip(uncached_indices, embedding_list, uncached_texts):
            results[orig_idx] = vec
            to_save[self._hash_content(text)] = vec

        final_model_id = "clarion-hash-fallback-v1" if encoding_fallback else active_model_id
        self.active_model_id = final_model_id
        self._save_cached_embeddings(to_save, final_model_id)
        return [r for r in results if r is not None]
    
    async def generate_embeddings_async(
        self,
        chunks: List[Chunk],
        batch_callback: Optional[Callable[[int, int], Any]] = None,
    ) -> List[List[float]]:
        """
        Asynchronously generate embeddings for a list of chunks in non-blocking batches.
        Yields control to the asyncio event loop between batches to keep the server responsive.
        
        Args:
            chunks: List of Chunk objects
            batch_callback: Async or sync callable(processed_count, total_count)
            
        Returns:
            List of embedding vectors
        """
        segments, embeddings = await asyncio.to_thread(self.generate_embedding_segments, chunks)
        if batch_callback:
            result = batch_callback(len(segments), len(segments))
            if asyncio.iscoroutine(result):
                await result
        return embeddings

    def create_vector_index(
        self,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> VectorStore:
        """
        Create FAISS index for document.
        
        Args:
            document_id: Document identifier
            chunks: List of chunks
            embeddings: Corresponding embeddings
        
        Returns:
            VectorStore instance
        """
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        model_id = self.active_model_id or self._configured_model_id()
        model_revision = self._model_revision_key(model_id)
        vector_store = VectorStore(document_id, model_id=model_id, model_revision=model_revision)
        vector_store.create_index(embeddings, chunk_ids)
        
        self._save_embedding_metadata(document_id, chunks, chunk_ids)
        
        logger.info(f"Created vector index for document {document_id}")
        return vector_store

    def get_embedding_segment(
        self, segment_id: str, model_id: Optional[str] = None, model_revision: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        conn = sqlite_connect(self.db_path)
        conn.row_factory = sqlite3.Row
        selected_model = model_id or self.active_model_id or self._configured_model_id()
        selected_revision = model_revision or (self.model_revision if self.model is not None else "1")
        row = conn.execute(
            "SELECT * FROM embedding_segments_v2 WHERE segment_id=? AND embedding_model_id=? AND embedding_model_revision=?",
            (segment_id, selected_model, selected_revision),
        ).fetchone()
        if row:
            result = dict(row)
            try:
                result["source_provenance"] = json.loads(result.pop("source_provenance_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                result["source_provenance"] = {}
            conn.close()
            return result
        # Do not fall back to an unscoped legacy segment when a specific model
        # identity was requested; IDs may be reused across model revisions.
        row = None
        if model_id is None and model_revision is None:
            row = conn.execute("SELECT * FROM embedding_segments WHERE segment_id = ?", (segment_id,)).fetchone()
        conn.close()
        result = dict(row) if row else None
        if result:
            try:
                result["source_provenance"] = json.loads(result.pop("source_provenance_json", None) or "{}")
            except (TypeError, json.JSONDecodeError):
                result["source_provenance"] = {}
        return result
    
    def _save_embedding_metadata(
        self,
        document_id: str,
        chunks: List[Chunk],
        chunk_ids: List[str],
    ) -> None:
        """Atomically persist the compatibility map and model-scoped vector chain."""
        model_id = self.active_model_id or self._configured_model_id()
        revision = self._model_revision_key(model_id)
        conn = sqlite_connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM embedding_vector_map WHERE document_id=? AND embedding_model_id=? AND embedding_model_revision=?",
                (document_id, model_id, revision),
            )
            conn.execute(
                "DELETE FROM embedding_segments_v2 WHERE document_id=? AND embedding_model_id=? AND embedding_model_revision=?",
                (document_id, model_id, revision),
            )
            conn.execute("DELETE FROM embeddings WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM embedding_segments WHERE document_id=?", (document_id,))
            for idx, (chunk, vector_id) in enumerate(zip(chunks, chunk_ids)):
                provenance = chunk.source_provenance or {}
                source_chunk_id = provenance.get("source_chunk_id")
                segment_index = None
                if ":seg:" in vector_id:
                    source_chunk_id, suffix = vector_id.rsplit(":seg:", 1)
                    segment_index = int(suffix)
                    char_start, char_end = self._segment_spans.get(vector_id, (0, len(chunk.content)))
                    conn.execute(
                        "INSERT OR REPLACE INTO embedding_segments_v2 "
                        "(segment_id, embedding_model_id, embedding_model_revision, source_chunk_id, document_id, "
                        "segment_index, char_start, char_end, content, token_count, source_provenance_json, generation_version) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (vector_id, model_id, revision, source_chunk_id, document_id, segment_index,
                         char_start, char_end, chunk.content, self._segment_token_counts.get(vector_id),
                         None, EMBEDDING_PROVENANCE_VERSION),
                    )
                    # Compatibility table is intentionally not authoritative across revisions.
                    conn.execute(
                        "INSERT OR REPLACE INTO embedding_segments "
                        "(segment_id, source_chunk_id, document_id, segment_index, char_start, char_end, content, "
                        "source_provenance_json, token_count, embedding_model_id, embedding_model_revision, generation_version) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (vector_id, source_chunk_id, document_id, segment_index, char_start, char_end, chunk.content,
                         None, self._segment_token_counts.get(vector_id), model_id, revision,
                         EMBEDDING_PROVENANCE_VERSION),
                    )
                else:
                    source_chunk_id = source_chunk_id or vector_id
                retained_chunk_id = source_chunk_id or vector_id
                raw_chunk_ids = chunk.raw_chunk_ids or [retained_chunk_id]
                stored_provenance = {
                    **provenance,
                    "document_id": document_id,
                    "retained_chunk_id": retained_chunk_id,
                    "raw_chunk_ids": raw_chunk_ids,
                    "segment_id": vector_id if segment_index is not None else None,
                    "vector_id": vector_id,
                    "embedding_model_id": model_id,
                    "embedding_model_revision": revision,
                }
                if segment_index is not None:
                    serialized = json.dumps(stored_provenance, ensure_ascii=False)
                    conn.execute(
                        "UPDATE embedding_segments_v2 SET source_provenance_json=? "
                        "WHERE segment_id=? AND embedding_model_id=? AND embedding_model_revision=?",
                        (serialized, vector_id, model_id, revision),
                    )
                    conn.execute(
                        "UPDATE embedding_segments SET source_provenance_json=? WHERE segment_id=?",
                        (serialized, vector_id),
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(chunk_id, document_id, vector_index, embedding_model_id, embedding_model_revision) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (vector_id, document_id, idx, model_id, revision),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO embedding_vector_map "
                    "(document_id, embedding_model_id, embedding_model_revision, vector_id, vector_index, segment_id, "
                    "retained_chunk_id, raw_chunk_ids_json, source_provenance_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (document_id, model_id, revision, vector_id, idx, vector_id if segment_index is not None else None, retained_chunk_id,
                     json.dumps(raw_chunk_ids, ensure_ascii=False), json.dumps(stored_provenance, ensure_ascii=False)),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _model_revision_key(self, model_id: str) -> str:
        return "1" if model_id == "clarion-hash-fallback-v1" else (self.model_revision or "revision-unavailable")
    
    def get_vector_store(self, document_id: str, *, load_model: bool = True) -> Optional[VectorStore]:
        """
        Get vector store for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            VectorStore instance if index exists
        """
        if load_model:
            try:
                self._load_model()
            except Exception:
                pass
        identities = []
        preferred = self._configured_model_id()
        identities.append((preferred, self._model_revision_key(preferred)))
        conn = sqlite_connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT embedding_model_id, embedding_model_revision "
                "FROM embedding_vector_map WHERE document_id=? ORDER BY embedding_model_id, embedding_model_revision",
                (document_id,),
            ).fetchall()
            identities.extend((row[0], row[1]) for row in rows if row[0] and row[1])
        finally:
            conn.close()
        for model_id, revision in dict.fromkeys(identities):
            vector_store = VectorStore(document_id, model_id=model_id, model_revision=revision)
            if vector_store._load_index() and self._vector_map_matches(
                document_id, model_id, revision, vector_store.chunk_ids
            ):
                self.active_model_id = model_id
                if model_id == "clarion-hash-fallback-v1":
                    self.using_fallback_embeddings = True
                return vector_store
        return None

    def _vector_map_matches(
        self, document_id: str, model_id: str, revision: str, vector_ids: List[str]
    ) -> bool:
        """Reject indexes with missing, stale, reordered, or mismatched DB maps."""
        conn = sqlite_connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT vector_index, vector_id FROM embedding_vector_map "
                "WHERE document_id=? AND embedding_model_id=? AND embedding_model_revision=? "
                "ORDER BY vector_index",
                (document_id, model_id, revision),
            ).fetchall()
        finally:
            conn.close()
        mapped_ids = [row[1] for row in rows]
        contiguous = all(int(row[0]) == index for index, row in enumerate(rows))
        if not contiguous or mapped_ids != list(vector_ids):
            logger.error(
                "Refusing inconsistent FAISS/database mapping for document %s, model %s@%s "
                "(FAISS=%d vectors, mapped=%d rows)",
                document_id, model_id, revision, len(vector_ids), len(mapped_ids),
            )
            return False
        return True
    
    def delete_embeddings(self, document_id: str) -> None:
        """Delete embeddings for a document."""
        conn = sqlite_connect(self.db_path)
        identities = []
        try:
            identities = conn.execute(
                "SELECT DISTINCT embedding_model_id, embedding_model_revision FROM embedding_vector_map WHERE document_id=?",
                (document_id,),
            ).fetchall()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM embedding_vector_map WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM embedding_segments_v2 WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM embeddings WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM embedding_segments WHERE document_id=?", (document_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        identities = list(identities or [])
        if not identities:
            model_id = self.active_model_id or self._configured_model_id()
            identities = [(model_id, self._model_revision_key(model_id))]
        for model_id, revision in identities:
            VectorStore(document_id, model_id=model_id, model_revision=revision).delete_index()
        # Remove the pre-versioned filename used by legacy indexes as well.
        VectorStore(document_id).delete_index()
        
        logger.info(f"Deleted embeddings for document {document_id}")
    
    def embed_query(self, query_text: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query_text: Query text
        
        Returns:
            Query embedding vector
        """
        if self.active_model_id == "clarion-hash-fallback-v1":
            self.using_fallback_embeddings = True
            return self._fallback_embed_text(query_text)
        try:
            self._load_model()
            expected = self.active_model_id
            if expected and expected != self._configured_model_id():
                raise RuntimeError("Loaded embedding model does not match the selected FAISS index")
            limit = self._model_token_limit()
            token_count = len(self.model.tokenizer.encode(query_text, add_special_tokens=True, truncation=False))
            if token_count > limit:
                raise ValueError(f"Query has {token_count} tokens; model limit is {limit}. Query chunking is not supported.")
            embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.using_fallback_embeddings = False
            return embedding[0].tolist()
        except Exception as error:
            if self.active_model_id and self.active_model_id != "clarion-hash-fallback-v1":
                raise RuntimeError(
                    f"The selected FAISS index requires {self.active_model_id}, but its query encoder is unavailable"
                ) from error
            logger.warning(
                "Query embedding model unavailable; using deterministic fallback embedding. Error: %s",
                error,
            )
            self.using_fallback_embeddings = True
            return self._fallback_embed_text(query_text)

    def _fallback_embed_text(self, text: str) -> List[float]:
        """
        Deterministic offline embedding fallback.

        Uses a signed hashing trick over tokens to produce a stable dense vector.
        This keeps pipeline behavior functional when the real model cannot be loaded.
        """
        vector = [0.0] * self.fallback_dimension
        tokens = re.findall(r"\b\w+\b", (text or "").lower())

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.fallback_dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]

        return vector

    def check_token_length(self, text: str, max_tokens: Optional[int] = None) -> int:
        """
        Check token length of text and log a warning if it exceeds max_tokens.
        Returns the subword token count.
        """
        token_count = 0
        try:
            if self.model is not None and hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                token_count = len(self.model.tokenizer.encode(text, add_special_tokens=True, truncation=False))
            else:
                token_count = int(len(text.split()) * 1.35)
        except Exception:
            token_count = int(len(text.split()) * 1.35)

        limit = max_tokens or (getattr(self.model, "max_seq_length", None) if self.model is not None else 512)
        if token_count > limit:
            logger.warning(
                f"Text token count ({token_count}) exceeds limit of {limit} subwords for the model. "
                "It must be segmented before embedding."
            )
        return token_count

