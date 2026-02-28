"""
Hierarchy Generator - Generate structured hierarchies from knowledge graphs.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

import networkx as nx

from models.graph import (
    ConceptLayer, PrerequisiteChain, TopicTreeNode,
    GraphNodeType, GraphEdgeType
)
from models.knowledge_map import KnowledgeMap

logger = logging.getLogger(__name__)


class HierarchyGenerator:
    """
    Generate topic trees, conceptual layers, and prerequisite chains.
    
    This class analyzes the knowledge graph structure to produce
    various hierarchical representations useful for learning and
    knowledge organization.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
    
    def generate_topic_tree(self) -> Optional[TopicTreeNode]:
        """
        Generate hierarchical topic tree from graph structure.
        
        Returns:
            Root TopicTreeNode with nested children
        """
        try:
            # Find document root
            doc_nodes = [
                n for n in self.graph.nodes()
                if self.graph.nodes[n].get("node_type") == GraphNodeType.DOCUMENT.value
            ]
            
            if not doc_nodes:
                logger.warning("No document node found in graph")
                return None
            
            root_id = doc_nodes[0]
            
            # Build tree recursively
            return self._build_tree_node(root_id)
            
        except Exception as e:
            logger.error(f"Topic tree generation failed: {e}")
            return None
    
    def _build_tree_node(self, node_id: str) -> TopicTreeNode:
        """Recursively build tree node with children."""
        node_data = self.graph.nodes[node_id]
        
        node = TopicTreeNode(
            id=node_id,
            title=node_data.get("label", node_id),
            description=node_data.get("description"),
            node_type=node_data.get("node_type", "unknown"),
            concept_ids=node_data.get("concept_ids", [])
        )
        
        # Find children (nodes connected by hierarchy edges)
        children = []
        for successor in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, successor)
            if edge_data and edge_data.get("relation_type") == GraphEdgeType.HIERARCHY.value:
                children.append(self._build_tree_node(successor))
        
        node.children = children
        return node
    
    def generate_conceptual_layers(
        self,
        max_layers: int = 5
    ) -> List[ConceptLayer]:
        """
        Organize concepts into learning layers based on prerequisites.
        
        Uses topological sorting to organize concepts from foundational
        to advanced based on prerequisite chains.
        
        Args:
            max_layers: Maximum number of layers to create
        
        Returns:
            List of ConceptLayer objects
        """
        try:
            # Get concept nodes
            concept_nodes = [
                n for n in self.graph.nodes()
                if self.graph.nodes[n].get("node_type") == GraphNodeType.CONCEPT.value
            ]
            
            if not concept_nodes:
                return []
            
            # Calculate prerequisite depth for each concept
            depths = {}
            for concept_id in concept_nodes:
                depths[concept_id] = self._calculate_prerequisite_depth(concept_id)
            
            # Group by depth
            depth_groups = defaultdict(list)
            for concept_id, depth in depths.items():
                depth_groups[depth].append(concept_id)
            
            # Create layers (may need to merge if too many)
            sorted_depths = sorted(depth_groups.keys())
            
            if len(sorted_depths) > max_layers:
                # Merge depths into max_layers bins
                import numpy as np
                bins = np.array_split(sorted_depths, max_layers)
                merged_groups = defaultdict(list)
                
                for i, bin_depths in enumerate(bins):
                    for depth in bin_depths:
                        merged_groups[i].extend(depth_groups[depth])
                
                depth_groups = merged_groups
                sorted_depths = sorted(merged_groups.keys())
            
            # Build layers
            layers = []
            layer_names = ["Foundation", "Intermediate", "Advanced", "Expert", "Specialized"]
            
            for i, depth in enumerate(sorted_depths[:max_layers]):
                concept_ids = depth_groups[depth]
                
                layer = ConceptLayer(
                    level=i + 1,
                    name=layer_names[i] if i < len(layer_names) else f"Layer {i+1}",
                    description=self._describe_layer(i, len(sorted_depths)),
                    concept_ids=concept_ids
                )
                layers.append(layer)
            
            return layers
            
        except Exception as e:
            logger.error(f"Conceptual layer generation failed: {e}")
            return []
    
    def _calculate_prerequisite_depth(self, concept_id: str) -> int:
        """
        Calculate how many levels of prerequisites a concept has.
        
        Returns:
            Depth level (0 = no prerequisites)
        """
        visited = set()
        max_depth = 0
        
        def dfs(node_id, current_depth):
            nonlocal max_depth
            
            if node_id in visited:
                return
            visited.add(node_id)
            
            max_depth = max(max_depth, current_depth)
            
            # Get prerequisites (incoming PREREQUISITE edges)
            prerequisites = [
                u for u, v, d in self.graph.in_edges(node_id, data=True)
                if d.get("relation_type") == GraphEdgeType.PREREQUISITE.value
            ]
            
            for prereq in prerequisites:
                dfs(prereq, current_depth + 1)
        
        dfs(concept_id, 0)
        return max_depth
    
    def _describe_layer(self, index: int, total: int) -> str:
        """Generate description for a conceptual layer."""
        descriptions = [
            "Core concepts with no prerequisites - start here",
            "Builds on foundation concepts - requires basic understanding",
            "Advanced concepts requiring solid foundation",
            "Expert-level concepts with complex dependencies",
            "Highly specialized cutting-edge topics"
        ]
        
        if index < len(descriptions):
            return descriptions[index]
        else:
            return f"Conceptual layer {index + 1} of {total}"
    
    def generate_prerequisite_chains(
        self,
        max_chains: int = 20,
        max_length: int = 10
    ) -> List[PrerequisiteChain]:
        """
        Identify all prerequisite chains in the document.
        
        A prerequisite chain is an ordered sequence of concepts where
        each concept depends on the previous one.
        
        Args:
            max_chains: Maximum number of chains to return
            max_length: Maximum length of each chain
        
        Returns:
            List of PrerequisiteChain objects
        """
        try:
            # Get concept nodes
            concept_nodes = [
                n for n in self.graph.nodes()
                if self.graph.nodes[n].get("node_type") == GraphNodeType.CONCEPT.value
            ]
            
            if not concept_nodes:
                return []
            
            # Build prerequisite subgraph
            prereq_edges = [
                (u, v) for u, v, d in self.graph.edges(data=True)
                if d.get("relation_type") == GraphEdgeType.PREREQUISITE.value
                and u in concept_nodes and v in concept_nodes
            ]
            
            prereq_graph = nx.DiGraph()
            prereq_graph.add_nodes_from(concept_nodes)
            prereq_graph.add_edges_from(prereq_edges)
            
            # Find all paths from foundation concepts to advanced concepts
            # Foundation: no incoming prerequisites
            # Advanced: no outgoing prerequisites (or many dependents)
            foundation_concepts = [
                n for n in concept_nodes
                if prereq_graph.in_degree(n) == 0
            ]
            
            advanced_concepts = [
                n for n in concept_nodes
                if prereq_graph.out_degree(n) == 0
                or prereq_graph.out_degree(n) > 2  # Many dependents
            ]
            
            chains = []
            chain_id = 0
            
            # Find paths from each foundation to each advanced concept
            for target in advanced_concepts:
                for source in foundation_concepts:
                    if source == target:
                        continue
                    
                    try:
                        # Find shortest path
                        path = nx.shortest_path(
                            prereq_graph, source, target
                        )
                        
                        if len(path) > 1 and len(path) <= max_length:
                            chain_id += 1
                            
                            chain = PrerequisiteChain(
                                chain_id=f"chain_{chain_id}",
                                target_concept_id=target,
                                concept_ids=path,
                                total_length=len(path),
                                estimated_difficulty=self._estimate_chain_difficulty(
                                    path
                                )
                            )
                            chains.append(chain)
                            
                            if len(chains) >= max_chains:
                                return chains
                                
                    except nx.NetworkXNoPath:
                        continue
            
            # Sort by length (longer chains first = more comprehensive)
            chains.sort(key=lambda x: x.total_length, reverse=True)
            
            return chains[:max_chains]
            
        except Exception as e:
            logger.error(f"Prerequisite chain generation failed: {e}")
            return []
    
    def _estimate_chain_difficulty(self, path: List[str]) -> str:
        """Estimate difficulty of a prerequisite chain."""
        length = len(path)
        
        # Consider average centrality
        centralities = []
        for node_id in path:
            cent = self.graph.nodes[node_id].get("pagerank", 0.5)
            centralities.append(cent)
        
        avg_cent = sum(centralities) / len(centralities) if centralities else 0.5
        
        # Score
        score = (length / 10) * 0.6 + avg_cent * 0.4
        
        if score < 0.33:
            return "foundation"
        elif score < 0.66:
            return "intermediate"
        else:
            return "advanced"
    
    def export_taxonomy(
        self,
        format: str = "skos"
    ) -> str:
        """
        Export hierarchy as standardized taxonomy.
        
        Args:
            format: "skos", "json", or "csv"
        
        Returns:
            Formatted taxonomy string
        """
        try:
            layers = self.generate_conceptual_layers()
            
            if format == "skos":
                return self._to_skos(layers)
            elif format == "json":
                import json
                return json.dumps(
                    {"layers": [layer.dict() for layer in layers]},
                    indent=2
                )
            elif format == "csv":
                return self._to_csv(layers)
            else:
                raise ValueError(f"Unknown format: {format}")
                
        except Exception as e:
            logger.error(f"Taxonomy export failed: {e}")
            return ""
    
    def _to_skos(self, layers: List[ConceptLayer]) -> str:
        """Convert to SKOS (Simple Knowledge Organization System) RDF."""
        lines = [
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"",
            "         xmlns:skos=\"http://www.w3.org/2004/02/skos/core#\">",
            ""
        ]
        
        for layer in layers:
            scheme_id = f"layer_{layer.level}"
            lines.append(f'  <skos:ConceptScheme rdf:about="#{scheme_id}">')
            lines.append(f'    <skos:prefLabel>{layer.name}</skos:prefLabel>')
            lines.append(f'    <skos:definition>{layer.description}</skos:definition>')
            lines.append('  </skos:ConceptScheme>')
            lines.append('')
            
            for concept_id in layer.concept_ids:
                node_data = self.graph.nodes[concept_id]
                concept_name = node_data.get("label", concept_id)
                
                lines.append(f'  <skos:Concept rdf:about="#{concept_id}">')
                lines.append(f'    <skos:prefLabel>{concept_name}</skos:prefLabel>')
                if node_data.get("definition"):
                    lines.append(f'    <skos:definition>{node_data["definition"]}</skos:definition>')
                lines.append(f'    <skos:inScheme rdf:resource="#{scheme_id}"/>')
                lines.append('  </skos:Concept>')
                lines.append('')
        
        lines.append('</rdf:RDF>')
        return '\n'.join(lines)
    
    def _to_csv(self, layers: List[ConceptLayer]) -> str:
        """Convert to CSV format."""
        lines = ["layer_level,layer_name,concept_id,concept_name,definition"]
        
        for layer in layers:
            for concept_id in layer.concept_ids:
                node_data = self.graph.nodes[concept_id]
                concept_name = node_data.get("label", concept_id)
                definition = node_data.get("definition", "").replace('"', '""')
                
                lines.append(
                    f'{layer.level},"{layer.name}",{concept_id},'
                    f'"{concept_name}","{definition}"'
                )
        
        return '\n'.join(lines)
