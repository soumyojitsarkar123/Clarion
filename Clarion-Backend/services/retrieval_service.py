"""
Retrieval Service - handles semantic search and retrieval.
"""

from typing import List, Optional, Tuple
import logging

from models.retrieval import RetrievalResult
from services.embedding_service import EmbeddingService
from services.chunking_service import ChunkingService
from vectorstore import VectorStore
from utils.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant document chunks.
    Handles semantic search using embeddings and FAISS.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chunking_service = ChunkingService()
        self.default_top_k = settings.default_top_k
        self.similarity_threshold = settings.similarity_threshold
    
    def retrieve(
        self,
        document_id: str,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            document_id: Document to search in
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of RetrievalResult objects
        """
        k = top_k or self.default_top_k
        
        vector_store = self.embedding_service.get_vector_store(document_id)
        
        if not vector_store:
            logger.warning(f"No vector index found for document {document_id}")
            return []
        
        query_embedding = self.embedding_service.embed_query(query)
        
        search_results = vector_store.search(query_embedding, k * 2)
        
        results = []
        chunks = {chunk.chunk_id: chunk for chunk in self.chunking_service.get_chunks(document_id)}
        
        for chunk_id, distance in search_results:
            segment = self.embedding_service.get_embedding_segment(
                chunk_id, vector_store.model_id, vector_store.model_revision
            )
            source_chunk_id = segment["source_chunk_id"] if segment else chunk_id
            if source_chunk_id not in chunks:
                continue
            chunk = chunks[source_chunk_id]
            
            score = 1.0 / (1.0 + distance)
            
            if score < self.similarity_threshold:
                continue
            
            provenance = (segment or {}).get("source_provenance") or chunk.source_provenance or {}
            result = RetrievalResult(
                chunk_id=chunk.chunk_id,
                content=segment["content"] if segment else chunk.content,
                score=score,
                section_title=chunk.section_title,
                position_index=chunk.position_index,
                document_id=document_id,
                page_references=provenance.get("page_references", []),
                source_intervals=provenance.get("source_intervals", []),
                segment_id=chunk_id,
                raw_chunk_ids=chunk.raw_chunk_ids or [chunk.chunk_id],
                embedding_model_id=vector_store.model_id,
            )
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    
    def retrieve_with_context(
        self,
        document_id: str,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[RetrievalResult], str]:
        """
        Retrieve chunks and build context string.
        
        Args:
            document_id: Document to search in
            query: Query text
            top_k: Number of results to return
        
        Returns:
            Tuple of (results, context_string)
        """
        results = self.retrieve(document_id, query, top_k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            header = f"[Chunk {i}]"
            if result.section_title:
                header += f" ({result.section_title})"
            header += f"\n{result.content}"
            context_parts.append(header)
        
        context = "\n\n".join(context_parts)
        
        return results, context
