"""
Confidence Scorer - Multi-factor confidence scoring for knowledge map elements.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np

from models.knowledge_map import Relation, Concept, RelationType
from models.chunk import Chunk
from models.graph import GraphEdgeType

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Calculate confidence scores for concepts and relations using multiple factors.
    
    Uses a weighted combination of:
    1. LLM confidence (direct from provider)
    2. Co-occurrence frequency in source chunks
    3. Semantic similarity between concepts
    4. Graph structural importance
    
    Example:
        scorer = ConfidenceScorer(embedding_service)
        score = scorer.score_relation(relation, chunks, graph)
    """
    
    # Default weights for composite score
    DEFAULT_WEIGHTS = {
        "llm_confidence": 0.30,
        "cooccurrence": 0.25,
        "semantic_similarity": 0.25,
        "structural_importance": 0.20
    }
    
    def __init__(
        self,
        embedding_service=None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize confidence scorer.
        
        Args:
            embedding_service: Service for generating embeddings (optional)
            weights: Custom weights for scoring factors
        """
        self.embedding_service = embedding_service
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, not 1.0. Normalizing.")
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def score_relation(
        self,
        relation: Relation,
        source_chunks: List[Chunk],
        concept_embeddings: Optional[Dict[str, List[float]]] = None,
        graph: Optional[Any] = None
    ) -> float:
        """
        Calculate composite confidence score for a relation.
        
        Args:
            relation: The relation to score
            source_chunks: Document chunks containing the concepts
            concept_embeddings: Pre-computed concept embeddings (optional)
            graph: Built graph for structural analysis (optional)
        
        Returns:
            Confidence score 0.0-1.0
        """
        try:
            scores = {}
            
            # Factor 1: LLM confidence
            scores["llm_confidence"] = self._extract_llm_confidence(relation)
            
            # Factor 2: Co-occurrence
            scores["cooccurrence"] = self._calculate_cooccurrence(
                relation.from_concept,
                relation.to_concept,
                source_chunks
            )
            
            # Factor 3: Semantic similarity
            if concept_embeddings:
                scores["semantic_similarity"] = self._calculate_semantic_similarity(
                    relation.from_concept,
                    relation.to_concept,
                    concept_embeddings
                )
            else:
                scores["semantic_similarity"] = 0.5  # Neutral if no embeddings
            
            # Factor 4: Structural importance
            if graph:
                scores["structural_importance"] = self._calculate_structural_importance(
                    relation, graph
                )
            else:
                scores["structural_importance"] = 0.5  # Neutral if no graph
            
            # Calculate weighted average
            composite = sum(
                scores[key] * self.weights[key]
                for key in self.weights.keys()
            )
            
            return round(max(0.0, min(1.0, composite)), 3)
            
        except Exception as e:
            logger.error(f"Failed to score relation {relation.id}: {e}")
            return 0.5  # Return neutral score on error
    
    def score_concept(
        self,
        concept: Concept,
        source_chunks: List[Chunk],
        graph: Optional[Any] = None
    ) -> float:
        """
        Calculate confidence score for a concept.
        
        Based on:
        - Frequency of mention in source text
        - Clarity of definition
        - Graph connectivity (if available)
        
        Args:
            concept: The concept to score
            source_chunks: Source document chunks
            graph: Knowledge graph (optional)
        
        Returns:
            Confidence score 0.0-1.0
        """
        try:
            scores = []
            
            # Factor 1: Frequency in source
            freq_score = self._calculate_concept_frequency(concept, source_chunks)
            scores.append(freq_score)
            
            # Factor 2: Has definition
            definition_score = 0.7 if concept.definition else 0.4
            scores.append(definition_score)
            
            # Factor 3: Graph connectivity
            if graph:
                node_id = f"concept_{concept.id}"
                if node_id in graph:
                    degree = graph.degree(node_id)
                    max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 1
                    connectivity_score = min(1.0, degree / max(max_degree, 1))
                    scores.append(connectivity_score)
            
            # Average scores
            return round(sum(scores) / len(scores), 3) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Failed to score concept {concept.id}: {e}")
            return 0.5
    
    def _extract_llm_confidence(self, relation: Relation) -> float:
        """Extract LLM confidence from relation metadata."""
        # If relation has explicit confidence, use it
        if hasattr(relation, 'confidence') and relation.confidence is not None:
            return float(relation.confidence)
        
        # Default to neutral if no confidence provided
        return 0.7
    
    def _calculate_cooccurrence(
        self,
        concept_a: str,
        concept_b: str,
        chunks: List[Chunk]
    ) -> float:
        """
        Calculate how often concepts appear together in chunks.
        
        Args:
            concept_a: First concept name
            concept_b: Second concept name
            chunks: Document chunks
        
        Returns:
            Score 0.0-1.0 based on co-occurrence frequency
        """
        if not chunks:
            return 0.0
        
        co_count = 0
        a_count = 0
        b_count = 0
        
        concept_a_lower = concept_a.lower()
        concept_b_lower = concept_b.lower()
        
        for chunk in chunks:
            text_lower = chunk.content.lower()
            
            has_a = concept_a_lower in text_lower
            has_b = concept_b_lower in text_lower
            
            if has_a:
                a_count += 1
            if has_b:
                b_count += 1
            if has_a and has_b:
                co_count += 1
        
        if co_count == 0:
            return 0.1  # Small non-zero for mention
        
        # Calculate Jaccard-like co-occurrence score
        # Score = co_count / min(a_count, b_count)
        min_count = min(a_count, b_count)
        if min_count == 0:
            return 0.1
        
        score = co_count / min_count
        
        # Normalize: saturate at 3+ co-occurrences
        return min(1.0, score)
    
    def _calculate_semantic_similarity(
        self,
        concept_a: str,
        concept_b: str,
        embeddings: Dict[str, List[float]]
    ) -> float:
        """
        Calculate semantic similarity using embeddings.
        
        Args:
            concept_a: First concept name
            concept_b: Second concept name
            embeddings: Dictionary mapping concept names to embedding vectors
        
        Returns:
            Cosine similarity score 0.0-1.0
        """
        try:
            # Find embeddings (case-insensitive)
            emb_a = None
            emb_b = None
            
            for name, emb in embeddings.items():
                if name.lower() == concept_a.lower():
                    emb_a = emb
                if name.lower() == concept_b.lower():
                    emb_b = emb
            
            if emb_a is None or emb_b is None:
                return 0.5  # Neutral if embeddings not found
            
            # Calculate cosine similarity
            vec_a = np.array(emb_a)
            vec_b = np.array(emb_b)
            
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.5
            
            similarity = dot_product / (norm_a * norm_b)
            
            # Normalize from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def _calculate_structural_importance(
        self,
        relation: Relation,
        graph: Any
    ) -> float:
        """
        Calculate structural importance of a relation in the graph.
        
        Based on:
        - Centrality of connected nodes
        - Edge betweenness (is it a bridge?)
        
        Args:
            relation: The relation
            graph: NetworkX graph
        
        Returns:
            Structural importance score 0.0-1.0
        """
        try:
            import networkx as nx
            
            # Find nodes in graph
            from_node = None
            to_node = None
            
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                if node_data.get("name", "").lower() == relation.from_concept.lower():
                    from_node = node_id
                if node_data.get("name", "").lower() == relation.to_concept.lower():
                    to_node = node_id
            
            if from_node is None or to_node is None:
                return 0.5
            
            # Get centrality scores
            pagerank_a = graph.nodes[from_node].get("pagerank", 0.5)
            pagerank_b = graph.nodes[to_node].get("pagerank", 0.5)
            
            # Average centrality of connected nodes
            avg_centrality = (pagerank_a + pagerank_b) / 2
            
            return avg_centrality
            
        except Exception as e:
            logger.warning(f"Structural importance calculation failed: {e}")
            return 0.5
    
    def _calculate_concept_frequency(
        self,
        concept: Concept,
        chunks: List[Chunk]
    ) -> float:
        """Calculate how frequently a concept appears in source text."""
        if not chunks or not concept.name:
            return 0.0
        
        mention_count = 0
        concept_lower = concept.name.lower()
        
        for chunk in chunks:
            text_lower = chunk.content.lower()
            mention_count += text_lower.count(concept_lower)
        
        # Normalize: more mentions = higher confidence, but saturate
        # 1 mention = 0.3, 5 mentions = 0.7, 10+ mentions = 1.0
        if mention_count == 0:
            return 0.1
        elif mention_count >= 10:
            return 1.0
        else:
            return 0.3 + (mention_count / 10) * 0.7
    
    def batch_score_relations(
        self,
        relations: List[Relation],
        source_chunks: List[Chunk],
        concept_embeddings: Optional[Dict[str, List[float]]] = None,
        graph: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Score multiple relations efficiently.
        
        Args:
            relations: List of relations to score
            source_chunks: Source document chunks
            concept_embeddings: Pre-computed embeddings
            graph: Knowledge graph
        
        Returns:
            Dictionary mapping relation IDs to scores
        """
        scores = {}
        
        for relation in relations:
            score = self.score_relation(
                relation,
                source_chunks,
                concept_embeddings,
                graph
            )
            scores[relation.id] = score
        
        return scores
    
    def get_score_breakdown(
        self,
        relation: Relation,
        source_chunks: List[Chunk],
        concept_embeddings: Optional[Dict[str, List[float]]] = None,
        graph: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of confidence score components.
        
        Args:
            relation: The relation
            source_chunks: Source chunks
            concept_embeddings: Embeddings
            graph: Graph
        
        Returns:
            Dictionary with component scores and final score
        """
        breakdown = {
            "relation_id": relation.id,
            "from_concept": relation.from_concept,
            "to_concept": relation.to_concept,
            "weights": self.weights,
            "components": {},
            "final_score": 0.0
        }
        
        # Calculate each component
        breakdown["components"]["llm_confidence"] = self._extract_llm_confidence(relation)
        breakdown["components"]["cooccurrence"] = self._calculate_cooccurrence(
            relation.from_concept,
            relation.to_concept,
            source_chunks
        )
        
        if concept_embeddings:
            breakdown["components"]["semantic_similarity"] = self._calculate_semantic_similarity(
                relation.from_concept,
                relation.to_concept,
                concept_embeddings
            )
        
        if graph:
            breakdown["components"]["structural_importance"] = self._calculate_structural_importance(
                relation, graph
            )
        
        # Calculate weighted final score
        final = sum(
            breakdown["components"][key] * self.weights[key]
            for key in breakdown["components"].keys()
        )
        
        breakdown["final_score"] = round(final, 3)
        
        return breakdown
