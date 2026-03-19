"""
Embedding Service - generates embeddings for document chunks.
"""

import sqlite3
import re
import math
import hashlib
from typing import List, Optional, Any
import logging

from vectorstore import VectorStore
from models.chunk import Chunk
from utils.config import settings
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using SentenceTransformers.
    Manages embedding generation and FAISS vector storage.
    """
    
    def __init__(self):
        self.model_name = settings.embedding_model_name
        self.batch_size = settings.embedding_batch_size
        self.device = settings.embedding_device
        self.model: Optional[Any] = None
        self.fallback_dimension = 384
        self.using_fallback_embeddings = False
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for embeddings."""
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
        
        conn.commit()
        conn.close()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError(
                    "Failed to import sentence-transformers. Check dependency compatibility."
                ) from e
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded successfully")
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            List of embedding vectors
        """
        texts = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")

        try:
            self._load_model()
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            embedding_list = embeddings.tolist()
            self.using_fallback_embeddings = False
            logger.info(f"Generated {len(embedding_list)} embeddings")
            return embedding_list
        except Exception as error:
            logger.warning(
                "Embedding model unavailable; using deterministic fallback embeddings. Error: %s",
                error,
            )
            self.using_fallback_embeddings = True
            return [self._fallback_embed_text(text) for text in texts]
    
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
