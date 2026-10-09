"""
Embedding Service - generates embeddings for document chunks.
"""

import sqlite3
import asyncio
import re
import math
import hashlib
import struct
import os
from typing import List, Optional, Any, Callable, Dict
import logging

from vectorstore import VectorStore
from models.chunk import Chunk
from utils.config import settings, initialize_compute_settings
from utils.sqlite import connect as sqlite_connect

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
        # Canonical embedding model BAAI/bge-large-en-v1.5 produces 1024-dim vectors
        self.fallback_dimension = 1024
        self.using_fallback_embeddings = False
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
        
        conn.commit()
        conn.close()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model with singleton caching and CPU thread tuning."""
        if self.model is None and self.model_name in _MODEL_CACHE:
            self.model = _MODEL_CACHE[self.model_name]
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
            _MODEL_CACHE[self.model_name] = loaded

    def _hash_content(self, text: str) -> str:
        """Compute SHA-256 hash of text content for caching."""
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _get_cached_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
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
                    [self.model_name] + batch_hashes,
                )
                for ch_hash, blob in cursor.fetchall():
                    count = len(blob) // 4
                    cached[ch_hash] = list(struct.unpack(f"{count}f", blob))
            conn.close()
        except Exception as e:
            logger.warning(f"Error reading chunk embedding cache: {e}")
        return cached

    def _save_cached_embeddings(self, hash_to_vector: Dict[str, List[float]]) -> None:
        """Store newly generated embeddings into SQLite cache."""
        if not hash_to_vector:
            return
        try:
            conn = sqlite_connect(self.db_path)
            cursor = conn.cursor()
            items = [
                (ch_hash, self.model_name, struct.pack(f"{len(vec)}f", *vec))
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
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks, leveraging SQLite cache.
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            List of embedding vectors
        """
        texts = [chunk.content for chunk in chunks]
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} chunks")

        # Check SQLite chunk embedding cache
        cached_map = self._get_cached_embeddings(texts)
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
            self.using_fallback_embeddings = False
            return [r for r in results if r is not None]

        logger.info(
            f"Encoding {len(uncached_texts)} uncached chunks ({len(texts) - len(uncached_texts)} served from cache)"
        )

        try:
            self._load_model()
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(uncached_texts) > 50,
                convert_to_numpy=True,
            )
            embedding_list = embeddings.tolist()
            self.using_fallback_embeddings = False
        except Exception as error:
            logger.warning(
                "Embedding model unavailable; using deterministic fallback embeddings. Error: %s",
                error,
            )
            self.using_fallback_embeddings = True
            embedding_list = [self._fallback_embed_text(text) for text in uncached_texts]

        # Save newly computed embeddings to cache
        to_save: Dict[str, List[float]] = {}
        for orig_idx, vec, text in zip(uncached_indices, embedding_list, uncached_texts):
            results[orig_idx] = vec
            to_save[self._hash_content(text)] = vec

        self._save_cached_embeddings(to_save)
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
        texts = [chunk.content for chunk in chunks]
        total = len(texts)
        if total == 0:
            return []

        logger.info(f"Generating embeddings asynchronously for {total} chunks")

        # Check SQLite cache
        cached_map = self._get_cached_embeddings(texts)
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
            self.using_fallback_embeddings = False
            if batch_callback:
                res = batch_callback(total, total)
                if asyncio.iscoroutine(res):
                    await res
            return [r for r in results if r is not None]

        try:
            if self.model is None:
                await asyncio.to_thread(self._load_model)

            all_new_embeddings: List[List[float]] = []
            batch_size = max(1, self.batch_size)

            def _encode_batch(batch_texts: List[str]) -> List[List[float]]:
                batch_res = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return batch_res.tolist()

            total_uncached = len(uncached_texts)
            for i in range(0, total_uncached, batch_size):
                batch_texts = uncached_texts[i : i + batch_size]
                batch_vectors = await asyncio.to_thread(_encode_batch, batch_texts)
                all_new_embeddings.extend(batch_vectors)

                if batch_callback:
                    current_done = len(texts) - total_uncached + len(all_new_embeddings)
                    res = batch_callback(current_done, total)
                    if asyncio.iscoroutine(res):
                        await res
                await asyncio.sleep(0)

            self.using_fallback_embeddings = False
        except Exception as error:
            logger.warning(
                "Embedding model unavailable or error during async embedding; using deterministic fallback: %s",
                error,
            )
            self.using_fallback_embeddings = True
            all_new_embeddings = [self._fallback_embed_text(text) for text in uncached_texts]
            if batch_callback:
                res = batch_callback(total, total)
                if asyncio.iscoroutine(res):
                    await res

        # Map and cache new embeddings
        to_save: Dict[str, List[float]] = {}
        for orig_idx, vec, text in zip(uncached_indices, all_new_embeddings, uncached_texts):
            results[orig_idx] = vec
            to_save[self._hash_content(text)] = vec

        self._save_cached_embeddings(to_save)
        return [r for r in results if r is not None]

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
        
        vector_store = VectorStore(document_id)
        vector_store.create_index(embeddings, chunk_ids)
        
        self._save_embedding_metadata(document_id, chunk_ids)
        
        logger.info(f"Created vector index for document {document_id}")
        return vector_store
    
    def _save_embedding_metadata(
        self,
        document_id: str,
        chunk_ids: List[str]
    ) -> None:
        """Save embedding metadata to database."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        for idx, chunk_id in enumerate(chunk_ids):
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (chunk_id, document_id, vector_index)
                VALUES (?, ?, ?)
            """, (chunk_id, document_id, idx))
        
        conn.commit()
        conn.close()
    
    def get_vector_store(self, document_id: str) -> Optional[VectorStore]:
        """
        Get vector store for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            VectorStore instance if index exists
        """
        vector_store = VectorStore(document_id)
        
        if vector_store._load_index():
            return vector_store
        
        return None
    
    def delete_embeddings(self, document_id: str) -> None:
        """Delete embeddings for a document."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM embeddings WHERE document_id = ?", (document_id,))
        
        conn.commit()
        conn.close()
        
        vector_store = VectorStore(document_id)
        vector_store.delete_index()
        
        logger.info(f"Deleted embeddings for document {document_id}")
    
    def embed_query(self, query_text: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query_text: Query text
        
        Returns:
            Query embedding vector
        """
        try:
            self._load_model()
            embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True
            )
            self.using_fallback_embeddings = False
            return embedding[0].tolist()
        except Exception as error:
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
