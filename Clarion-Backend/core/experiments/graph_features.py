"""
Graph Feature Extractor for Relation Validation Experiments

Extracts graph-structural features from knowledge graphs to enhance
relation validity prediction.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import networkx as nx
from networkx.readwrite import json_graph

from utils.config import settings
from utils.graph_store import graph_json_path, graph_pickle_path

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """
    Extracts graph-structural features for relation validation.
    
    Features extracted:
    - Node centrality of both concepts
    - Degree of each concept
    - Shortest path distance between concepts
    - Community membership similarity
    - Layer difference from hierarchy generator
    - Graph connectivity indicator
    """
    
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph
        self._centrality_cache: Optional[Dict] = None
        self._layers_cache: Optional[Dict[str, int]] = None
        self._communities_cache: Optional[Dict] = None
    
    def load_graph(self, document_id: str) -> bool:
        """
        Load graph for a document.
        
        Args:
            document_id: Document ID to load graph for
        
        Returns:
            True if graph loaded successfully
        """
        graph_path = graph_json_path(document_id)
        
        try:
            if graph_path.exists():
                data = json.loads(graph_path.read_text(encoding="utf-8"))
                self.graph = nx.DiGraph(
                    json_graph.node_link_graph(data, directed=True, multigraph=False)
                )
            else:
                legacy_path = graph_pickle_path(document_id)
                if not settings.allow_legacy_pickle_loading or not legacy_path.exists():
                    logger.warning(f"Graph not found for document {document_id}")
                    self.graph = None
                    return False
                with open(legacy_path, "rb") as f:
                    self.graph = pickle.load(f)
            
            logger.info(f"Loaded graph for document {document_id}: {self.graph.number_of_nodes()} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            self.graph = None
            return False
    
    def extract_features(self, concept_a: str, concept_b: str) -> Dict[str, float]:
        """
        Extract graph features for a relation between two concepts.
        
        Args:
            concept_a: First concept name
            concept_b: Second concept name
        
        Returns:
            Dictionary of feature name -> value
        """
        if self.graph is None:
            return self._empty_features()
        
        features = {}
        
        # 1. Node centrality
        features.update(self._get_centrality_features(concept_a, concept_b))
        
        # 2. Degree features
        features.update(self._get_degree_features(concept_a, concept_b))
        
        # 3. Shortest path distance
        features.update(self._get_path_features(concept_a, concept_b))
        
        # 4. Community features
        features.update(self._get_community_features(concept_a, concept_b))
        
        # 5. Layer features
        features.update(self._get_layer_features(concept_a, concept_b))
        
        # 6. Connectivity indicator
        features.update(self._get_connectivity_features(concept_a, concept_b))
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict when graph unavailable."""
        return {
            "centrality_a": 0.0,
            "centrality_b": 0.0,
            "centrality_diff": 0.0,
            "degree_a": 0,
            "degree_b": 0,
            "degree_diff": 0,
            "shortest_path": -1.0,
            "same_community": 0.0,
            "community_distance": -1.0,
            "layer_diff": -1.0,
            "layer_a": -1,
            "layer_b": -1,
            "is_connected": 0.0,
            "common_neighbors": 0,
            "graph_nodes": 0,
            "graph_edges": 0
        }
    
    def _get_centrality_features(self, concept_a: str, concept_b: str) -> Dict[str, float]:
        """Extract centrality features."""
        if self.graph is None:
            return {}
        
        # Compute centrality if not cached
        if self._centrality_cache is None:
            try:
                # Use degree centrality for speed
                self._centrality_cache = nx.degree_centrality(self.graph)
            except Exception as e:
                logger.warning(f"Failed to compute centrality: {e}")
                return {}
        
        # Find nodes matching concepts
        node_a = self._find_node(concept_a)
        node_b = self._find_node(concept_b)
        
        centrality_a = self._centrality_cache.get(node_a, 0.0) if node_a else 0.0
        centrality_b = self._centrality_cache.get(node_b, 0.0) if node_b else 0.0
        
        return {
            "centrality_a": centrality_a,
            "centrality_b": centrality_b,
            "centrality_diff": abs(centrality_a - centrality_b),
            "centrality_sum": centrality_a + centrality_b
        }
    
    def _get_degree_features(self, concept_a: str, concept_b: str) -> Dict[str, float]:
        """Extract degree features."""
        if self.graph is None:
            return {}
        
        node_a = self._find_node(concept_a)
        node_b = self._find_node(concept_b)
        
        degree_a = self.graph.degree(node_a) if node_a and node_a in self.graph else 0
        degree_b = self.graph.degree(node_b) if node_b and node_b in self.graph else 0
        
        return {
            "degree_a": float(degree_a),
            "degree_b": float(degree_b),
            "degree_diff": float(abs(degree_a - degree_b)),
            "degree_sum": float(degree_a + degree_b)
        }
    
    def _get_path_features(self, concept_a: str, concept_b: str) -> Dict[str, float]:
        """Extract shortest path features."""
        if self.graph is None:
            return {}
        
        node_a = self._find_node(concept_a)
        node_b = self._find_node(concept_b)
        
        if not node_a or not node_b:
            return {"shortest_path": -1.0}
        
        try:
            # Try directed path
            path_len = nx.shortest_path_length(self.graph, node_a, node_b)
            return {"shortest_path": float(path_len)}
        except nx.NetworkXNoPath:
            pass
        
        try:
            # Try undirected path (ignore direction)
            undirected = self.graph.to_undirected()
            path_len = nx.shortest_path_length(undirected, node_a, node_b)
            return {"shortest_path": float(path_len)}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"shortest_path": -1.0}
    
    def _get_community_features(self, concept_a: str, concept_b: str) -> Dict[str, float]:
        """Extract community features."""
        if self.graph is None:
            return {}
        
        # Compute communities if not cached
        if self._communities_cache is None:
            try:
                undirected = self.graph.to_undirected()
                if undirected.number_of_nodes() > 0:
                    communities = nx.community.louvain_communities(undirected)
                    self._communities_cache = {}
                    for i, comm in enumerate(communities):
                        for node in comm:
                            self._communities_cache[node] = i
            except Exception as e:
                logger.warning(f"Failed to compute communities: {e}")
                return {}
        
        if not self._communities_cache:
            return {"same_community": 0.0, "community_distance": -1.0}
        
        node_a = self._find_node(concept_a)
        node_b = self._find_node(concept_b)
        
        comm_a = self._communities_cache.get(node_a, -1)
        comm_b = self._communities_cache.get(node_b, -1)
        
        same_community = 1.0 if comm_a == comm_b and comm_a >= 0 else 0.0
        
        # Community distance (number of communities apart)
        community_distance = abs(comm_a - comm_b) if comm_a >= 0 and comm_b >= 0 else -1.0
        
        return {
            "same_community": same_community,
            "community_distance": float(community_distance) if community_distance >= 0 else -1.0
        }
    
    def _get_layer_features(self, concept_a: str, concept_b: str) -> Dict[str, float]:
        """Extract layer features from hierarchy."""
        if self.graph is None:
            return {}
        
        # Compute layers if not cached
        if self._layers_cache is None:
            self._layers_cache = self._compute_layers()
        
        node_a = self._find_node(concept_a)
        node_b = self._find_node(concept_b)
        
        layer_a = self._layers_cache.get(node_a, -1)
        layer_b = self._layers_cache.get(node_b, -1)
        
        layer_diff = abs(layer_a - layer_b) if layer_a >= 0 and layer_b >= 0 else -1.0
        
        return {
            "layer_a": float(layer_a) if layer_a >= 0 else -1.0,
            "layer_b": float(layer_b) if layer_b >= 0 else -1.0,
            "layer_diff": float(layer_diff) if layer_diff >= 0 else -1.0
        }
    
    def _compute_layers(self) -> Dict[str, int]:
        """Compute conceptual layers based on shortest path from root."""
        if self.graph is None:
            return {}
        
        layers = {}
        
        try:
            # Find root (document node)
            root_nodes = [
                n for n in self.graph.nodes()
                if self.graph.nodes[n].get("node_type") == "document"
            ]
            
            if not root_nodes:
                return {}
            
            root = root_nodes[0]
            
            # BFS to assign layers
            visited = set()
            queue = [(root, 0)]
            
            while queue:
                node, layer = queue.pop(0)
                
                if node in visited:
                    continue
                visited.add(node)
                layers[node] = layer
                
                for successor in self.graph.successors(node):
                    if successor not in visited:
                        queue.append((successor, layer + 1))
                        
        except Exception as e:
            logger.warning(f"Failed to compute layers: {e}")
        
        return layers
    
    def _get_connectivity_features(self, concept_a: str, concept_b: str) -> Dict[str, float]:
        """Extract connectivity features."""
        if self.graph is None:
            return {}
        
        node_a = self._find_node(concept_a)
        node_b = self._find_node(concept_b)
        
        # Check if connected (any path exists)
        if node_a and node_b:
            try:
                undirected = self.graph.to_undirected()
                is_connected = nx.has_path(undirected, node_a, node_b)
            except:
                is_connected = False
        else:
            is_connected = False
        
        # Count common neighbors
        common_neighbors = 0
        if node_a and node_b and node_a in self.graph and node_b in self.graph:
            neighbors_a = set(self.graph.predecessors(node_a)) | set(self.graph.successors(node_a))
            neighbors_b = set(self.graph.predecessors(node_b)) | set(self.graph.successors(node_b))
            common_neighbors = len(neighbors_a & neighbors_b)
        
        return {
            "is_connected": 1.0 if is_connected else 0.0,
            "common_neighbors": float(common_neighbors),
            "graph_nodes": float(self.graph.number_of_nodes()) if self.graph else 0,
            "graph_edges": float(self.graph.number_of_edges()) if self.graph else 0
        }
    
    def _find_node(self, concept_name: str) -> Optional[str]:
        """Find node ID for a concept name."""
        if self.graph is None:
            return None
        
        # Try exact match first
        for node in self.graph.nodes():
            node_label = self.graph.nodes[node].get("label", "").lower()
            if node_label == concept_name.lower():
                return node
        
        # Try partial match
        concept_lower = concept_name.lower()
        for node in self.graph.nodes():
            node_label = self.graph.nodes[node].get("label", "").lower()
            if concept_lower in node_label or node_label in concept_lower:
                return node
        
        return None
    
    def get_all_feature_names(self) -> List[str]:
        """Return all feature names for this extractor."""
        return [
            "centrality_a",
            "centrality_b", 
            "centrality_diff",
            "centrality_sum",
            "degree_a",
            "degree_b",
            "degree_diff",
            "degree_sum",
            "shortest_path",
            "same_community",
            "community_distance",
            "layer_a",
            "layer_b",
            "layer_diff",
            "is_connected",
            "common_neighbors",
            "graph_nodes",
            "graph_edges"
        ]


class GraphEnrichedFeatureExtractor:
    """
    Combines basic features with graph features for relation validation.
    """
    
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.basic_extractor = None  # We'll use the existing one
        self.graph_extractor = GraphFeatureExtractor(graph)
    
    def set_graph(self, graph: nx.DiGraph):
        """Set the graph for feature extraction."""
        self.graph_extractor = GraphFeatureExtractor(graph)
    
    def load_graph_for_document(self, document_id: str) -> bool:
        """Load graph for a specific document."""
        return self.graph_extractor.load_graph(document_id)
    
    def extract_all_features(
        self,
        basic_record: Dict,
        concept_a: str,
        concept_b: str
    ) -> Dict[str, float]:
        """
        Extract all features including graph features.
        
        Args:
            basic_record: Basic features from RelationFeatureExtractor
            concept_a: First concept name
            concept_b: Second concept name
        
        Returns:
            Combined feature dictionary
        """
        features = {}
        
        # Add basic features
        if basic_record:
            features.update(basic_record)
        
        # Add graph features
        graph_features = self.graph_extractor.extract_features(concept_a, concept_b)
        features.update(graph_features)
        
        # Add derived features
        features.update(self._compute_derived_features(features))
        
        return features
    
    def _compute_derived_features(self, features: Dict) -> Dict[str, float]:
        """Compute derived features from existing ones."""
        derived = {}
        
        # Combined confidence score
        llm_conf = features.get("llm_confidence", 0.5)
        cooc = features.get("cooccurrence_score", 0.0) or 0.0
        centrality = (features.get("centrality_a", 0.0) + features.get("centrality_b", 0.0)) / 2
        
        derived["graph_enhanced_confidence"] = (
            0.5 * llm_conf +
            0.25 * cooc +
            0.25 * min(centrality * 10, 1.0)
        )
        
        # Path-based confidence
        shortest_path = features.get("shortest_path", -1)
        if shortest_path > 0:
            derived["path_based_boost"] = 1.0 / (1.0 + shortest_path)
        else:
            derived["path_based_boost"] = 0.0
        
        # Community-based confidence  
        same_comm = features.get("same_community", 0.0)
        connected = features.get("is_connected", 0.0)
        derived["graph_structure_score"] = 0.5 * same_comm + 0.5 * connected
        
        return derived
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        basic_names = [
            "llm_confidence", "cooccurrence_score", "semantic_similarity",
            "relation_type_encoded", "context_length", "chunk_count",
            "conf_x_cooc", "conf_x_sem_sim"
        ]
        graph_names = self.graph_extractor.get_all_feature_names()
        derived_names = ["graph_enhanced_confidence", "path_based_boost", "graph_structure_score"]
        
        return basic_names + graph_names + derived_names
