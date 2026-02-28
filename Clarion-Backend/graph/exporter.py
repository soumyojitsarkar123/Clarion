"""
Graph Exporter - Export knowledge graphs to various formats.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import pickle

import networkx as nx
import numpy as np

from models.graph import (
    GraphNode, GraphEdge, GraphMetrics, CytoscapeGraph,
    CytoscapeElement, GraphNodeType, GraphEdgeType
)

logger = logging.getLogger(__name__)


class GraphExportError(Exception):
    """Error during graph export."""
    pass


class GraphExporter:
    """
    Export NetworkX graphs to various formats for visualization and analysis.
    
    Supports:
    - Cytoscape.js JSON (web visualization)
    - GEXF (Gephi desktop tool)
    - GraphML (interoperability)
    - NetworkX pickle (Python)
    - D3.js JSON (custom visualizations)
    - Adjacency matrix (ML input)
    """
    
    def __init__(self):
        self.export_handlers = {
            "cytoscape": self.to_cytoscape_json,
            "gexf": self.to_gexf,
            "graphml": self.to_graphml,
            "pickle": self.to_networkx_pickle,
            "d3": self.to_d3_json,
            "adjacency": self.to_adjacency_matrix,
            "json": self.to_json_dict
        }
    
    def export(
        self,
        graph: nx.DiGraph,
        format: str,
        **kwargs
    ) -> Union[str, bytes, Dict]:
        """
        Export graph to specified format.
        
        Args:
            graph: NetworkX graph to export
            format: Export format name
            **kwargs: Format-specific options
        
        Returns:
            Exported data (type depends on format)
        """
        if format not in self.export_handlers:
            raise GraphExportError(f"Unknown format: {format}")
        
        try:
            return self.export_handlers[format](graph, **kwargs)
        except Exception as e:
            logger.error(f"Export to {format} failed: {e}")
            raise GraphExportError(f"Export failed: {e}") from e
    
    def to_cytoscape_json(
        self,
        graph: nx.DiGraph,
        include_metrics: bool = True,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export to Cytoscape.js JSON format.
        
        This format is optimized for web-based graph visualization
        using Cytoscape.js or compatible libraries.
        
        Args:
            graph: NetworkX graph
            include_metrics: Include graph metrics in output
            document_id: Document identifier
        
        Returns:
            Cytoscape.js compatible JSON structure
        """
        elements = {"nodes": [], "edges": []}
        
        # Export nodes
        for node_id, node_data in graph.nodes(data=True):
            node_element = {
                "data": {
                    "id": node_id,
                    "label": node_data.get("label", node_id),
                    "type": node_data.get("node_type", "unknown"),
                }
            }
            
            # Add optional fields if present
            for field in ["definition", "context", "centrality", "community", "layer"]:
                if field in node_data and node_data[field] is not None:
                    node_element["data"][field] = node_data[field]
            
            # Add metadata
            if "chunk_ids" in node_data:
                node_element["data"]["chunk_count"] = len(node_data["chunk_ids"])
            
            elements["nodes"].append(node_element)
        
        # Export edges
        for source, target, edge_data in graph.edges(data=True):
            edge_element = {
                "data": {
                    "id": edge_data.get("id", f"{source}_{target}"),
                    "source": source,
                    "target": target,
                    "relation_type": edge_data.get("relation_type", "unknown"),
                    "confidence": edge_data.get("confidence", 1.0),
                    "weight": edge_data.get("weight", 1.0),
                }
            }
            
            # Add optional fields
            if "description" in edge_data and edge_data["description"]:
                edge_element["data"]["description"] = edge_data["description"]
            
            if "validated" in edge_data:
                edge_element["data"]["validated"] = edge_data["validated"]
            
            elements["edges"].append(edge_element)
        
        result = {
            "format": "cytoscape",
            "generated_at": datetime.now().isoformat(),
            "elements": elements
        }
        
        if document_id:
            result["document_id"] = document_id
        
        if include_metrics:
            from graph.builder import GraphBuilder
            builder = GraphBuilder()
            builder.graph = graph
            result["metrics"] = builder.calculate_metrics().dict()
        
        return result
    
    def to_gexf(
        self,
        graph: nx.DiGraph,
        version: str = "1.2",
        encoding: str = "utf-8"
    ) -> str:
        """
        Export to GEXF (Graph Exchange XML Format).
        
        GEXF is the native format for Gephi and supports dynamic
        features, hierarchical structures, and rich metadata.
        
        Args:
            graph: NetworkX graph
            version: GEXF version
            encoding: Character encoding
        
        Returns:
            GEXF XML string
        """
        try:
            # Use NetworkX built-in GEXF export
            import io
            buffer = io.BytesIO()
            
            # Convert node attributes for GEXF compatibility
            gexf_graph = nx.DiGraph()
            
            for node_id, node_data in graph.nodes(data=True):
                # Convert complex types to strings
                gexf_data = {}
                for key, value in node_data.items():
                    if isinstance(value, (list, dict)):
                        gexf_data[key] = json.dumps(value)
                    else:
                        gexf_data[key] = value
                
                gexf_graph.add_node(node_id, **gexf_data)
            
            for source, target, edge_data in graph.edges(data=True):
                gexf_data = {}
                for key, value in edge_data.items():
                    if isinstance(value, (list, dict)):
                        gexf_data[key] = json.dumps(value)
                    else:
                        gexf_data[key] = value
                
                gexf_graph.add_edge(source, target, **gexf_data)
            
            nx.write_gexf(gexf_graph, buffer, version=version)
            return buffer.getvalue().decode(encoding)
            
        except Exception as e:
            logger.error(f"GEXF export failed: {e}")
            raise GraphExportError(f"GEXF export failed: {e}") from e
    
    def to_graphml(self, graph: nx.DiGraph) -> str:
        """
        Export to GraphML format.
        
        GraphML is a comprehensive XML format supported by many
        graph analysis tools including yEd, Cytoscape, and Gephi.
        
        Args:
            graph: NetworkX graph
        
        Returns:
            GraphML XML string
        """
        try:
            import io
            buffer = io.BytesIO()
            
            # NetworkX write_graphml expects string path or file-like
            nx.write_graphml(graph, buffer)
            return buffer.getvalue().decode("utf-8")
            
        except Exception as e:
            logger.error(f"GraphML export failed: {e}")
            raise GraphExportError(f"GraphML export failed: {e}") from e
    
    def to_networkx_pickle(
        self,
        graph: nx.DiGraph,
        protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> bytes:
        """
        Export to NetworkX native pickle format.
        
        This preserves all NetworkX-specific attributes and is
        suitable for Python interoperability and caching.
        
        Args:
            graph: NetworkX graph
            protocol: Pickle protocol version
        
        Returns:
            Pickled graph bytes
        """
        try:
            return pickle.dumps(graph, protocol=protocol)
        except Exception as e:
            logger.error(f"Pickle export failed: {e}")
            raise GraphExportError(f"Pickle export failed: {e}") from e
    
    def to_d3_json(
        self,
        graph: nx.DiGraph,
        node_size_attr: str = "centrality",
        link_weight_attr: str = "weight"
    ) -> Dict[str, Any]:
        """
        Export to D3.js force-directed graph format.
        
        This format is optimized for D3.js visualizations with
        force simulation, custom styling, and interactivity.
        
        Args:
            graph: NetworkX graph
            node_size_attr: Node attribute for sizing
            link_weight_attr: Edge attribute for link strength
        
        Returns:
            D3.js compatible JSON structure
        """
        nodes = []
        links = []
        
        # Build nodes
        for node_id, node_data in graph.nodes(data=True):
            node = {
                "id": node_id,
                "name": node_data.get("label", node_id),
                "group": node_data.get("node_type", "unknown"),
            }
            
            # Add size attribute
            if node_size_attr in node_data:
                node["size"] = node_data[node_size_attr]
            
            # Add additional properties
            for key in ["definition", "community", "layer"]:
                if key in node_data:
                    node[key] = node_data[key]
            
            nodes.append(node)
        
        # Build links
        for source, target, edge_data in graph.edges(data=True):
            link = {
                "source": source,
                "target": target,
                "value": edge_data.get(link_weight_attr, 1.0),
                "type": edge_data.get("relation_type", "unknown")
            }
            
            if "confidence" in edge_data:
                link["confidence"] = edge_data["confidence"]
            
            links.append(link)
        
        return {
            "nodes": nodes,
            "links": links
        }
    
    def to_adjacency_matrix(
        self,
        graph: nx.DiGraph,
        relation_type: Optional[str] = None,
        weight_attr: str = "weight"
    ) -> np.ndarray:
        """
        Export to adjacency matrix (NumPy array).
        
        Useful for machine learning, graph neural networks, and
        mathematical analysis.
        
        Args:
            graph: NetworkX graph
            relation_type: Filter by specific relation type
            weight_attr: Edge attribute to use as weights
        
        Returns:
            NumPy adjacency matrix
        """
        try:
            # Get concept nodes only (for cleaner matrix)
            concept_nodes = [
                n for n in graph.nodes()
                if graph.nodes[n].get("node_type") == GraphNodeType.CONCEPT.value
            ]
            
            if not concept_nodes:
                return np.array([])
            
            # Create subgraph if filtering by relation type
            if relation_type:
                edges = [
                    (u, v, d) for u, v, d in graph.edges(data=True)
                    if d.get("relation_type") == relation_type
                    and u in concept_nodes and v in concept_nodes
                ]
                subgraph = nx.DiGraph()
                subgraph.add_nodes_from(concept_nodes)
                for u, v, d in edges:
                    subgraph.add_edge(u, v, **d)
            else:
                subgraph = graph.subgraph(concept_nodes)
            
            # Get adjacency matrix
            nodes_list = list(subgraph.nodes())
            adj_matrix = nx.to_numpy_array(
                subgraph,
                nodelist=nodes_list,
                weight=weight_attr
            )
            
            return adj_matrix
            
        except Exception as e:
            logger.error(f"Adjacency matrix export failed: {e}")
            raise GraphExportError(f"Adjacency matrix export failed: {e}") from e
    
    def to_json_dict(
        self,
        graph: nx.DiGraph,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Export to generic JSON dictionary.
        
        Simple format suitable for general-purpose storage and
        interoperability.
        
        Args:
            graph: NetworkX graph
            include_metadata: Include graph metadata
        
        Returns:
            JSON-serializable dictionary
        """
        # Nodes
        nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_dict = {"id": node_id}
            node_dict.update(node_data)
            nodes.append(node_dict)
        
        # Edges
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            edge_dict = {
                "source": source,
                "target": target
            }
            edge_dict.update(edge_data)
            edges.append(edge_dict)
        
        result = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": nodes,
            "links": edges
        }
        
        if include_metadata:
            result["metadata"] = {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "generated_at": datetime.now().isoformat()
            }
        
        return result
    
    def save_to_file(
        self,
        graph: nx.DiGraph,
        filepath: Path,
        format: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Export graph to file.
        
        Args:
            graph: NetworkX graph
            filepath: Output file path
            format: Export format (inferred from extension if None)
            **kwargs: Export options
        
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        
        # Infer format from extension
        if format is None:
            ext = filepath.suffix.lower()
            format_map = {
                ".gexf": "gexf",
                ".graphml": "graphml",
                ".pkl": "pickle",
                ".pickle": "pickle",
                ".json": "cytoscape"
            }
            format = format_map.get(ext, "cytoscape")
        
        # Export data
        data = self.export(graph, format, **kwargs)
        
        # Write to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, bytes):
            filepath.write_bytes(data)
        elif isinstance(data, str):
            filepath.write_text(data, encoding="utf-8")
        else:
            # JSON serializable
            filepath.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8"
            )
        
        logger.info(f"Graph exported to {filepath} (format: {format})")
        return filepath
