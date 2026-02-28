"""
Graph Analyzer - Advanced graph analysis algorithms and metrics.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import networkx as nx
import numpy as np

from models.graph import GraphMetrics

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """
    Analyze knowledge graphs using graph-theoretic algorithms.
    
    Provides:
    - Centrality analysis (PageRank, betweenness, closeness)
    - Community detection (Louvain, label propagation)
    - Path analysis (shortest paths, diameter)
    - Structural analysis (clustering, connectivity)
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._metrics_cache: Optional[GraphMetrics] = None
        self._centrality_cache: Dict[str, Dict[str, float]] = {}
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive graph metrics.
        
        Returns:
            Dictionary with all metric categories
        """
        return {
            "basic": self.get_basic_metrics(),
            "centrality": self.get_centrality_metrics(),
            "clustering": self.get_clustering_metrics(),
            "connectivity": self.get_connectivity_metrics(),
            "path_analysis": self.get_path_analysis()
        }
    
    def get_basic_metrics(self) -> Dict[str, Any]:
        """
        Basic graph statistics.
        
        Returns:
            Dictionary with node count, edge count, density, degrees
        """
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        
        metrics = {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "density": nx.density(self.graph) if n_nodes > 1 else 0.0
        }
        
        if n_nodes > 0:
            degrees = [d for n, d in self.graph.degree()]
            metrics["avg_degree"] = np.mean(degrees)
            metrics["max_degree"] = max(degrees)
            metrics["min_degree"] = min(degrees)
            metrics["median_degree"] = np.median(degrees)
        else:
            metrics.update({
                "avg_degree": 0,
                "max_degree": 0,
                "min_degree": 0,
                "median_degree": 0
            })
        
        return metrics
    
    def get_centrality_metrics(
        self,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate centrality metrics for key concept identification.
        
        Args:
            top_n: Number of top nodes to return for each metric
        
        Returns:
            Dictionary with centrality scores and top nodes
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        try:
            # PageRank
            pagerank = nx.pagerank(self.graph, weight='weight')
            
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(
                self.graph, weight='weight'
            )
            
            # Closeness centrality
            closeness = nx.closeness_centrality(self.graph)
            
            # Degree centrality
            degree_cent = nx.degree_centrality(self.graph)
            
            # Eigenvector centrality (for undirected version)
            try:
                undirected = self.graph.to_undirected()
                eigenvector = nx.eigenvector_centrality(
                    undirected, weight='weight', max_iter=1000
                )
            except:
                eigenvector = {}
            
            # Get top nodes for each metric
            def get_top_nodes(metric_dict, n=top_n):
                sorted_nodes = sorted(
                    metric_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                return [
                    {"node_id": node_id, "score": score}
                    for node_id, score in sorted_nodes[:n]
                ]
            
            return {
                "pagerank": {
                    "scores": pagerank,
                    "top_nodes": get_top_nodes(pagerank)
                },
                "betweenness": {
                    "scores": betweenness,
                    "top_nodes": get_top_nodes(betweenness)
                },
                "closeness": {
                    "scores": closeness,
                    "top_nodes": get_top_nodes(closeness)
                },
                "degree": {
                    "scores": degree_cent,
                    "top_nodes": get_top_nodes(degree_cent)
                },
                "eigenvector": {
                    "scores": eigenvector,
                    "top_nodes": get_top_nodes(eigenvector) if eigenvector else []
                }
            }
            
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")
            return {}
    
    def get_clustering_metrics(self) -> Dict[str, Any]:
        """
        Clustering and community structure analysis.
        
        Returns:
            Dictionary with clustering coefficients and communities
        """
        try:
            undirected = self.graph.to_undirected()
            
            metrics = {
                "clustering_coefficient": nx.average_clustering(undirected),
                "transitivity": nx.transitivity(undirected)
            }
            
            # Community detection
            from networkx.algorithms import community
            
            # Louvain communities
            try:
                louvain_comms = community.louvain_communities(undirected, seed=42)
                metrics["louvain_communities"] = [
                    list(comm) for comm in louvain_comms
                ]
                metrics["num_communities"] = len(louvain_comms)
                
                # Modularity
                partition = {}
                for i, comm in enumerate(louvain_comms):
                    for node in comm:
                        partition[node] = i
                
                modularity = community.modularity(
                    undirected, louvain_comms, weight='weight'
                )
                metrics["modularity"] = modularity
                
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")
                metrics["louvain_communities"] = []
                metrics["num_communities"] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Clustering metrics failed: {e}")
            return {}
    
    def get_connectivity_metrics(self) -> Dict[str, Any]:
        """
        Graph connectivity analysis.
        
        Returns:
            Dictionary with connectivity statistics
        """
        try:
            metrics = {
                "is_strongly_connected": nx.is_strongly_connected(self.graph),
                "is_weakly_connected": nx.is_weakly_connected(self.graph),
                "num_strongly_connected_components": nx.number_strongly_connected_components(
                    self.graph
                ),
                "num_weakly_connected_components": nx.number_weakly_connected_components(
                    self.graph
                )
            }
            
            # Strongly connected components
            sccs = list(nx.strongly_connected_components(self.graph))
            metrics["strongly_connected_components"] = [
                list(scc) for scc in sccs
            ]
            
            # Largest component
            if sccs:
                largest_scc = max(sccs, key=len)
                metrics["largest_component_size"] = len(largest_scc)
                metrics["largest_component_percentage"] = (
                    len(largest_scc) / self.graph.number_of_nodes() * 100
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Connectivity metrics failed: {e}")
            return {}
    
    def get_path_analysis(self) -> Dict[str, Any]:
        """
        Path and distance analysis.
        
        Returns:
            Dictionary with path metrics
        """
        try:
            metrics = {}
            
            # Only calculate if graph is reasonably sized and connected
            if self.graph.number_of_nodes() < 2:
                return metrics
            
            # Weakly connected analysis
            if nx.is_weakly_connected(self.graph):
                undirected = self.graph.to_undirected()
                
                try:
                    metrics["diameter"] = nx.diameter(undirected)
                    metrics["radius"] = nx.radius(undirected)
                    metrics["average_shortest_path_length"] = (
                        nx.average_shortest_path_length(undirected)
                    )
                except:
                    pass
            
            # All pairs shortest paths (sample if too large)
            nodes = list(self.graph.nodes())
            if len(nodes) > 100:
                # Sample nodes
                import random
                random.seed(42)
                sample_nodes = random.sample(nodes, 100)
            else:
                sample_nodes = nodes
            
            # Calculate path statistics
            path_lengths = []
            for source in sample_nodes[:50]:  # Limit pairs
                try:
                    lengths = nx.single_source_shortest_path_length(
                        self.graph.to_undirected(), source
                    )
                    path_lengths.extend(lengths.values())
                except:
                    pass
            
            if path_lengths:
                metrics["avg_path_length_sampled"] = np.mean(path_lengths)
                metrics["max_path_length_sampled"] = max(path_lengths)
                metrics["path_length_distribution"] = self._distribution(path_lengths)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
            return {}
    
    def _distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate distribution histogram."""
        if not values:
            return {}
        
        hist = defaultdict(int)
        for v in values:
            hist[int(v)] += 1
        
        return dict(hist)
    
    def identify_key_concepts(
        self,
        metric: str = "pagerank",
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify most important concepts using centrality.
        
        Args:
            metric: Centrality metric to use
            top_n: Number of concepts to return
        
        Returns:
            List of key concepts with scores
        """
        try:
            centrality_data = self.get_centrality_metrics(top_n=top_n)
            
            if metric not in centrality_data:
                return []
            
            top_nodes = centrality_data[metric]["top_nodes"]
            
            # Enrich with node data
            enriched = []
            for node_info in top_nodes:
                node_id = node_info["node_id"]
                score = node_info["score"]
                
                node_data = self.graph.nodes[node_id]
                
                enriched.append({
                    "concept_id": node_id,
                    "concept_name": node_data.get("label", node_id),
                    "score": score,
                    "definition": node_data.get("definition"),
                    "centrality_metric": metric
                })
            
            return enriched
            
        except Exception as e:
            logger.error(f"Key concept identification failed: {e}")
            return []
    
    def find_bridges(self) -> List[Tuple[str, str]]:
        """
        Find bridge edges whose removal would disconnect the graph.
        
        Returns:
            List of bridge edge tuples
        """
        try:
            undirected = self.graph.to_undirected()
            bridges = list(nx.bridges(undirected))
            return bridges
        except:
            return []
    
    def get_node_influence(self, node_id: str) -> Dict[str, float]:
        """
        Calculate influence metrics for a specific node.
        
        Args:
            node_id: Node to analyze
        
        Returns:
            Dictionary with influence metrics
        """
        if node_id not in self.graph:
            return {}
        
        try:
            # Local metrics
            degree = self.graph.degree(node_id)
            in_degree = self.graph.in_degree(node_id)
            out_degree = self.graph.out_degree(node_id)
            
            # Reach (nodes within 2 hops)
            try:
                reach = nx.single_source_shortest_path_length(
                    self.graph, node_id, cutoff=2
                )
                reach_count = len(reach) - 1  # Exclude self
            except:
                reach_count = 0
            
            return {
                "degree": degree,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "reach_2hop": reach_count
            }
            
        except Exception as e:
            logger.error(f"Influence calculation failed for {node_id}: {e}")
            return {}
