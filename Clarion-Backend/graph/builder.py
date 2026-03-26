"""
Graph Builder - Constructs NetworkX knowledge graphs from extracted components.
"""

import uuid
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle

import networkx as nx
import numpy as np

from models.knowledge_map import KnowledgeMap, Concept, Relation, RelationType
from models.graph import (
    GraphNode, GraphEdge, GraphNodeType, GraphEdgeType,
    GraphMetrics, CytoscapeGraph, CytoscapeElement
)
from utils.config import settings

logger = logging.getLogger(__name__)


class GraphBuilderError(Exception):
    """Error in graph building process."""
    pass


class GraphBuilder:
    """
    Builds NetworkX DiGraph from extracted knowledge components.
    
    This class constructs a directed knowledge graph from concepts, relations,
    and hierarchical structure extracted from documents. It supports multiple
    relation types, centrality analysis, and various export formats.
    
    Attributes:
        graph: nx.DiGraph - The constructed NetworkX graph
        concept_nodes: Dict[str, str] - Maps concept names to node IDs
        relation_edges: Dict[str, str] - Maps relation IDs to edge IDs
        document_id: Optional[str] - Current document being processed
    """
    
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.concept_nodes: Dict[str, str] = {}
        self.relation_edges: Dict[str, str] = {}
        self.document_id: Optional[str] = None
        self._node_counter = 0
        self._edge_counter = 0
    
    def build_from_knowledge_map(
        self,
        knowledge_map: KnowledgeMap,
        calculate_metrics: bool = True
    ) -> nx.DiGraph:
        """
        Build complete knowledge graph from KnowledgeMap.
        
        Args:
            knowledge_map: Extracted knowledge map with concepts and relations
            calculate_metrics: Whether to calculate graph metrics
        
        Returns:
            Constructed NetworkX DiGraph
        
        Raises:
            GraphBuilderError: If graph construction fails
        """
        try:
            self.document_id = knowledge_map.document_id
            self.graph = nx.DiGraph()
            self.concept_nodes = {}
            self.relation_edges = {}
            
            logger.info(f"Building graph for document {self.document_id}")
            
            # 1. Add document root node
            self._add_document_node()
            
            # 2. Add topic and subtopic nodes
            self._add_topic_nodes(knowledge_map)
            
            # 3. Add concept nodes
            self._add_concept_nodes(knowledge_map.concepts)
            
            # 4. Add hierarchy edges (document → topics → subtopics → concepts)
            self._add_hierarchy_edges(knowledge_map)
            
            # 5. Add semantic relation edges
            self._add_relation_edges(knowledge_map.relations)
            
            # 6. Validate graph integrity
            is_valid, issues = self.validate_graph()
            if not is_valid:
                logger.warning(f"Graph validation issues: {issues}")
            
            # 7. Calculate metrics if requested
            if calculate_metrics:
                self._calculate_node_centrality()
                self._detect_communities()
            
            logger.info(
                f"Graph built successfully: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges"
            )
            
            return self.graph
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            raise GraphBuilderError(f"Graph construction failed: {e}") from e
    
    def _add_document_node(self) -> str:
        """Add root document node."""
        node_id = f"doc_{self.document_id}"
        self.graph.add_node(
            node_id,
            label="Document",
            node_type=GraphNodeType.DOCUMENT.value,
            document_id=self.document_id,
            is_root=True
        )
        return node_id
    
    def _add_topic_nodes(self, knowledge_map: KnowledgeMap) -> None:
        """Add main topic and subtopic nodes."""
        # Add main topics
        for topic in knowledge_map.main_topics:
            node_id = f"topic_{topic.id}"
            self.graph.add_node(
                node_id,
                label=topic.title,
                node_type=GraphNodeType.TOPIC.value,
                description=topic.description,
                concept_ids=topic.concept_ids,
                is_main_topic=True
            )
            
            # Link to document
            doc_id = f"doc_{self.document_id}"
            self.graph.add_edge(
                doc_id, node_id,
                relation_type=GraphEdgeType.HIERARCHY.value,
                weight=1.0
            )
        
        # Add subtopics
        for subtopic in knowledge_map.subtopics:
            node_id = f"subtopic_{subtopic.id}"
            self.graph.add_node(
                node_id,
                label=subtopic.title,
                node_type=GraphNodeType.SUBTOPIC.value,
                description=subtopic.description,
                parent_id=subtopic.parent_topic_id,
                concept_ids=subtopic.concept_ids,
                is_main_topic=False
            )
            
            # Link to parent topic
            parent_id = f"topic_{subtopic.parent_topic_id}"
            if parent_id in self.graph:
                self.graph.add_edge(
                    parent_id, node_id,
                    relation_type=GraphEdgeType.HIERARCHY.value,
                    weight=1.0
                )
    
    def _add_concept_nodes(self, concepts: List[Concept]) -> None:
        """Add concept nodes to graph."""
        for concept in concepts:
            node_id = f"concept_{concept.id}"
            
            self.graph.add_node(
                node_id,
                label=concept.name,
                node_type=GraphNodeType.CONCEPT.value,
                name=concept.name,
                definition=concept.definition,
                context=concept.context,
                chunk_ids=concept.chunk_ids
            )
            
            # Track concept name to node ID mapping
            self.concept_nodes[concept.name] = node_id
    
    def _add_hierarchy_edges(self, knowledge_map: KnowledgeMap) -> None:
        """Add edges connecting topics/subtopics to their concepts."""
        document_id = f"doc_{self.document_id}"
        assigned_concept_ids = set()

        # Connect main topics to concepts
        for topic in knowledge_map.main_topics:
            topic_id = f"topic_{topic.id}"
            for concept_id in topic.concept_ids:
                concept_node_id = f"concept_{concept_id}"
                if concept_node_id in self.graph:
                    assigned_concept_ids.add(concept_id)
                    self.graph.add_edge(
                        topic_id, concept_node_id,
                        relation_type=GraphEdgeType.HIERARCHY.value,
                        weight=0.8
                    )
        
        # Connect subtopics to concepts
        for subtopic in knowledge_map.subtopics:
            subtopic_id = f"subtopic_{subtopic.id}"
            for concept_id in subtopic.concept_ids:
                concept_node_id = f"concept_{concept_id}"
                if concept_node_id in self.graph:
                    assigned_concept_ids.add(concept_id)
                    self.graph.add_edge(
                        subtopic_id, concept_node_id,
                        relation_type=GraphEdgeType.HIERARCHY.value,
                        weight=0.9
                    )

        # If no trustworthy topic hierarchy exists, fall back to a clean
        # document -> concept structure instead of duplicating concept names as topics.
        if not knowledge_map.main_topics and not knowledge_map.subtopics:
            for concept in knowledge_map.concepts:
                concept_node_id = f"concept_{concept.id}"
                if concept_node_id in self.graph:
                    self.graph.add_edge(
                        document_id, concept_node_id,
                        relation_type=GraphEdgeType.HIERARCHY.value,
                        weight=0.7
                    )
            return

        # Concepts that were not attached to any heading still need a home in the hierarchy.
        for concept in knowledge_map.concepts:
            if concept.id in assigned_concept_ids:
                continue
            concept_node_id = f"concept_{concept.id}"
            if concept_node_id in self.graph:
                self.graph.add_edge(
                    document_id, concept_node_id,
                    relation_type=GraphEdgeType.HIERARCHY.value,
                    weight=0.65
                )
    
    def _add_relation_edges(self, relations: List[Relation]) -> None:
        """Add semantic relation edges between concepts."""
        for relation in relations:
            # Find node IDs for concepts
            from_node_id = self._find_concept_node(relation.from_concept)
            to_node_id = self._find_concept_node(relation.to_concept)
            
            if not from_node_id or not to_node_id:
                logger.warning(
                    f"Skipping relation: concepts not found - "
                    f"{relation.from_concept} -> {relation.to_concept}"
                )
                continue
            
            edge_id = f"rel_{relation.id}"
            
            # Map relation type
            edge_type = self._map_relation_type(relation.relation_type)
            
            # Add edge with attributes
            self.graph.add_edge(
                from_node_id,
                to_node_id,
                id=edge_id,
                relation_type=edge_type.value,
                confidence=relation.confidence,
                description=relation.description,
                weight=self._calculate_edge_weight(relation),
                validated=False  # Will be set during evaluation
            )
            
            self.relation_edges[relation.id] = edge_id
    
    def _find_concept_node(self, concept_name: str) -> Optional[str]:
        """Find node ID for a concept name."""
        if concept_name in self.concept_nodes:
            return self.concept_nodes[concept_name]
        
        # Try case-insensitive match
        concept_lower = concept_name.lower()
        for name, node_id in self.concept_nodes.items():
            if name.lower() == concept_lower:
                return node_id
        
        return None
    
    def _map_relation_type(self, relation_type: RelationType) -> GraphEdgeType:
        """Map KnowledgeMap relation type to GraphEdgeType."""
        mapping = {
            RelationType.PREREQUISITE: GraphEdgeType.PREREQUISITE,
            RelationType.DEFINITION: GraphEdgeType.DEFINITION,
            RelationType.EXPLANATION: GraphEdgeType.EXPLANATION,
            RelationType.CAUSE_EFFECT: GraphEdgeType.CAUSE_EFFECT,
            RelationType.EXAMPLE_OF: GraphEdgeType.EXAMPLE_OF,
            RelationType.SIMILAR_TO: GraphEdgeType.SIMILAR_TO,
            RelationType.PART_OF: GraphEdgeType.PART_OF,
            RelationType.DERIVES_FROM: GraphEdgeType.DERIVES_FROM,
        }
        return mapping.get(relation_type, GraphEdgeType.EXPLANATION)
    
    def _calculate_edge_weight(self, relation: Relation) -> float:
        """Calculate edge weight from relation attributes."""
        # Base weight from confidence
        weight = relation.confidence
        
        # Adjust based on relation type
        type_weights = {
            RelationType.PREREQUISITE: 1.0,
            RelationType.DEFINITION: 0.9,
            RelationType.EXPLANATION: 0.7,
            RelationType.CAUSE_EFFECT: 0.8,
            RelationType.EXAMPLE_OF: 0.6,
            RelationType.SIMILAR_TO: 0.5,
            RelationType.PART_OF: 0.8,
            RelationType.DERIVES_FROM: 0.7,
        }
        
        type_weight = type_weights.get(relation.relation_type, 0.5)
        return (weight + type_weight) / 2
    
    def _calculate_node_centrality(self) -> None:
        """Calculate and store centrality metrics for concept nodes."""
        if self.graph.number_of_nodes() == 0:
            return
        
        try:
            # PageRank centrality
            pagerank = nx.pagerank(self.graph, weight='weight')
            
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(
                self.graph, weight='weight'
            )
            
            # Closeness centrality
            closeness = nx.closeness_centrality(self.graph)
            
            # Update node attributes
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]['pagerank'] = pagerank.get(node_id, 0)
                self.graph.nodes[node_id]['betweenness'] = betweenness.get(node_id, 0)
                self.graph.nodes[node_id]['closeness'] = closeness.get(node_id, 0)
                
        except Exception as e:
            logger.warning(f"Failed to calculate centrality: {e}")
    
    def _detect_communities(self) -> None:
        """Detect communities in the concept graph."""
        try:
            # Only consider concept nodes for community detection
            concept_nodes = [
                n for n in self.graph.nodes()
                if self.graph.nodes[n].get('node_type') == GraphNodeType.CONCEPT.value
            ]
            
            if len(concept_nodes) < 3:
                return
            
            # Create subgraph of concepts
            concept_subgraph = self.graph.subgraph(concept_nodes)
            
            # Convert to undirected for community detection
            undirected = concept_subgraph.to_undirected()
            
            # Use Louvain community detection
            from networkx.algorithms import community
            communities = community.louvain_communities(undirected, seed=42)
            
            # Assign community IDs
            for comm_id, comm_nodes in enumerate(communities):
                for node_id in comm_nodes:
                    self.graph.nodes[node_id]['community'] = comm_id
                    
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
    
    def get_prerequisite_chain(
        self,
        concept_id: str,
        direction: str = "upstream"
    ) -> List[str]:
        """
        Get ordered prerequisite chain for a concept.
        
        Args:
            concept_id: Target concept node ID
            direction: "upstream" (prerequisites) or "downstream" (dependents)
        
        Returns:
            Ordered list of concept IDs
        """
        if concept_id not in self.graph:
            return []
        
        try:
            if direction == "upstream":
                # Get prerequisites (nodes that lead to target)
                # Filter by PREREQUISITE edges only
                prereq_edges = [
                    (u, v) for u, v, d in self.graph.in_edges(concept_id, data=True)
                    if d.get('relation_type') == GraphEdgeType.PREREQUISITE.value
                ]
                
                if not prereq_edges:
                    return []
                
                # Build chain using DFS
                chain = []
                visited = set()
                
                def dfs(node_id):
                    if node_id in visited:
                        return
                    visited.add(node_id)
                    
                    # Get prerequisites of this node
                    prereqs = [
                        u for u, v, d in self.graph.in_edges(node_id, data=True)
                        if d.get('relation_type') == GraphEdgeType.PREREQUISITE.value
                    ]
                    
                    for prereq in prereqs:
                        dfs(prereq)
                    
                    chain.append(node_id)
                
                # Start from target and work backwards
                for prereq, _ in prereq_edges:
                    dfs(prereq)
                
                return chain
                
            else:  # downstream
                # Get concepts that depend on this one
                dependent_edges = [
                    (u, v) for u, v, d in self.graph.out_edges(concept_id, data=True)
                    if d.get('relation_type') == GraphEdgeType.PREREQUISITE.value
                ]
                
                dependents = [v for _, v in dependent_edges]
                return dependents
                
        except Exception as e:
            logger.error(f"Failed to get prerequisite chain: {e}")
            return []
    
    def get_concept_clusters(
        self,
        algorithm: str = "louvain"
    ) -> List[List[str]]:
        """
        Detect concept clusters using community detection.
        
        Args:
            algorithm: "louvain", "label_prop", or "greedy_modularity"
        
        Returns:
            List of clusters (each is list of concept node IDs)
        """
        try:
            # Get concept nodes only
            concept_nodes = [
                n for n in self.graph.nodes()
                if self.graph.nodes[n].get('node_type') == GraphNodeType.CONCEPT.value
            ]
            
            if len(concept_nodes) < 3:
                return [concept_nodes]
            
            subgraph = self.graph.subgraph(concept_nodes).to_undirected()
            
            from networkx.algorithms import community
            
            if algorithm == "louvain":
                communities = community.louvain_communities(subgraph, seed=42)
            elif algorithm == "label_prop":
                communities = community.label_propagation_communities(subgraph)
                communities = [list(c) for c in communities]
            elif algorithm == "greedy_modularity":
                communities = community.greedy_modularity_communities(subgraph)
                communities = [list(c) for c in communities]
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            return communities
            
        except Exception as e:
            logger.error(f"Cluster detection failed: {e}")
            return []
    
    def get_learning_path(
        self,
        target_concept_id: str,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal learning path to reach a concept.
        
        Uses topological sort on prerequisite subgraph.
        
        Args:
            target_concept_id: Target concept to learn
            max_depth: Maximum path length
        
        Returns:
            Ordered list of concepts with metadata
        """
        chain = self.get_prerequisite_chain(target_concept_id, "upstream")
        
        if not chain:
            return []
        
        # Build path info
        path = []
        for i, concept_id in enumerate(chain[:max_depth]):
            node_data = self.graph.nodes[concept_id]
            
            path.append({
                "order": i + 1,
                "concept_id": concept_id,
                "concept_name": node_data.get("label", concept_id),
                "definition": node_data.get("definition"),
                "prerequisites_count": i,  # Number of prerequisites before this
                "difficulty": self._estimate_difficulty(concept_id, i, len(chain))
            })
        
        return path
    
    def _estimate_difficulty(
        self,
        concept_id: str,
        position: int,
        total_length: int
    ) -> str:
        """Estimate difficulty based on position and graph metrics."""
        node_data = self.graph.nodes[concept_id]
        
        # Factors
        pagerank = node_data.get("pagerank", 0)
        degree = self.graph.degree(concept_id)
        
        # Score
        score = (position / max(total_length, 1)) * 0.5 + \
                pagerank * 0.3 + \
                (degree / max(self.graph.number_of_nodes(), 1)) * 0.2
        
        if score < 0.33:
            return "foundation"
        elif score < 0.66:
            return "intermediate"
        else:
            return "advanced"
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """
        Validate graph integrity.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for orphaned concept nodes
        concept_nodes = [
            n for n in self.graph.nodes()
            if self.graph.nodes[n].get("node_type") == GraphNodeType.CONCEPT.value
        ]
        
        for node_id in concept_nodes:
            if self.graph.degree(node_id) == 0:
                issues.append(f"Orphaned concept node: {node_id}")
        
        # Check for self-loops
        self_loops = list(nx.selfloop_edges(self.graph))
        if self_loops:
            issues.append(f"Self-loops detected: {len(self_loops)}")
        
        # Check for cycles in prerequisite edges
        prereq_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relation_type") == GraphEdgeType.PREREQUISITE.value
        ]
        
        if prereq_edges:
            prereq_graph = nx.DiGraph()
            prereq_graph.add_edges_from(prereq_edges)
            try:
                cycles = list(nx.simple_cycles(prereq_graph))
                if cycles:
                    issues.append(f"Circular prerequisites detected: {len(cycles)} cycles")
            except:
                pass
        
        return len(issues) == 0, issues
    
    def calculate_metrics(self) -> GraphMetrics:
        """Calculate comprehensive graph metrics."""
        try:
            n_nodes = self.graph.number_of_nodes()
            n_edges = self.graph.number_of_edges()
            
            # Basic metrics
            density = nx.density(self.graph) if n_nodes > 1 else 0.0
            
            # Degree statistics
            if n_nodes > 0:
                degrees = [d for n, d in self.graph.degree()]
                avg_degree = np.mean(degrees)
                max_degree = max(degrees)
                min_degree = min(degrees)
            else:
                avg_degree = max_degree = min_degree = 0
            
            # Path analysis (only if connected)
            try:
                if nx.is_strongly_connected(self.graph) and n_nodes > 1:
                    diameter = nx.diameter(self.graph)
                    radius = nx.radius(self.graph)
                    avg_path = nx.average_shortest_path_length(self.graph)
                else:
                    diameter = radius = None
                    avg_path = None
            except:
                diameter = radius = avg_path = None
            
            # Clustering (on undirected version)
            try:
                undirected = self.graph.to_undirected()
                clustering = nx.average_clustering(undirected)
                transitivity = nx.transitivity(undirected)
            except:
                clustering = transitivity = None
            
            # Component analysis
            if n_nodes > 0:
                is_connected = nx.is_weakly_connected(self.graph)
                num_components = nx.number_weakly_connected_components(self.graph)
            else:
                is_connected = True
                num_components = 0
            
            return GraphMetrics(
                node_count=n_nodes,
                edge_count=n_edges,
                density=density,
                diameter=diameter,
                radius=radius,
                avg_path_length=avg_path,
                clustering_coefficient=clustering,
                transitivity=transitivity,
                is_connected=is_connected,
                num_components=num_components,
                avg_degree=avg_degree,
                max_degree=max_degree,
                min_degree=min_degree
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return GraphMetrics(
                node_count=0,
                edge_count=0,
                density=0.0,
                avg_degree=0,
                max_degree=0,
                min_degree=0
            )
