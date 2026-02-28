"""
Graph models for knowledge graph representation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphNodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    DOCUMENT = "document"
    TOPIC = "topic"
    SUBTOPIC = "subtopic"
    CONCEPT = "concept"


class GraphEdgeType(str, Enum):
    """Types of semantic relationships in the graph."""
    PREREQUISITE = "prerequisite"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    CAUSE_EFFECT = "cause-effect"
    EXAMPLE_OF = "example-of"
    SIMILAR_TO = "similar-to"
    PART_OF = "part-of"
    DERIVES_FROM = "derives-from"
    HIERARCHY = "hierarchy"


class GraphNode(BaseModel):
    """A node in the knowledge graph."""
    
    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Display label")
    node_type: GraphNodeType = Field(..., description="Type of node")
    
    # Concept-specific fields
    definition: Optional[str] = Field(None, description="Concept definition")
    context: Optional[str] = Field(None, description="Source context")
    chunk_ids: List[str] = Field(default_factory=list, description="Source chunk IDs")
    
    # Hierarchy fields
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    level: int = Field(0, description="Hierarchy level")
    
    # Analysis fields
    centrality: Optional[float] = Field(None, description="PageRank centrality score")
    community: Optional[int] = Field(None, description="Community cluster ID")
    layer: Optional[str] = Field(None, description="Conceptual layer")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True


class GraphEdge(BaseModel):
    """An edge in the knowledge graph representing a relationship."""
    
    id: str = Field(..., description="Unique edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation_type: GraphEdgeType = Field(..., description="Type of relationship")
    
    # Confidence and validation
    confidence: float = Field(1.0, description="Confidence score 0-1", ge=0.0, le=1.0)
    validated: bool = Field(False, description="Whether relation is validated")
    
    # Description
    description: Optional[str] = Field(None, description="Human-readable description")
    
    # Source tracking
    chunk_ids: List[str] = Field(default_factory=list, description="Supporting chunk IDs")
    
    # Computed weight
    weight: float = Field(1.0, description="Edge weight for algorithms")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True


class GraphMetrics(BaseModel):
    """Metrics for a knowledge graph."""
    
    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")
    density: float = Field(..., description="Graph density 0-1")
    
    # Connectivity
    diameter: Optional[int] = Field(None, description="Longest shortest path")
    radius: Optional[int] = Field(None, description="Minimum eccentricity")
    avg_path_length: Optional[float] = Field(None, description="Average path length")
    
    # Clustering
    clustering_coefficient: Optional[float] = Field(None, description="Global clustering coefficient")
    transitivity: Optional[float] = Field(None, description="Graph transitivity")
    
    # Component analysis
    is_connected: bool = Field(True, description="Whether graph is connected")
    num_components: int = Field(1, description="Number of connected components")
    
    # Degree statistics
    avg_degree: float = Field(..., description="Average node degree")
    max_degree: int = Field(..., description="Maximum node degree")
    min_degree: int = Field(..., description="Minimum node degree")


class CytoscapeElement(BaseModel):
    """Single element for Cytoscape.js format."""
    
    data: Dict[str, Any] = Field(..., description="Element data")
    
    # Optional position for fixed layouts
    position: Optional[Dict[str, float]] = Field(None, description="Node position {x, y}")
    
    # Optional styling
    classes: Optional[str] = Field(None, description="CSS classes")


class CytoscapeGraph(BaseModel):
    """Complete graph in Cytoscape.js format."""
    
    elements: Dict[str, List[CytoscapeElement]] = Field(
        default_factory=lambda: {"nodes": [], "edges": []},
        description="Graph elements"
    )


class ConceptLayer(BaseModel):
    """A layer in the conceptual hierarchy."""
    
    level: int = Field(..., description="Layer level (1 = foundation)")
    name: str = Field(..., description="Layer name")
    description: str = Field(..., description="Layer description")
    concept_ids: List[str] = Field(default_factory=list, description="Concepts in this layer")


class PrerequisiteChain(BaseModel):
    """A chain of prerequisite concepts."""
    
    chain_id: str = Field(..., description="Unique chain identifier")
    target_concept_id: str = Field(..., description="Target concept to learn")
    concept_ids: List[str] = Field(..., description="Ordered list of concept IDs")
    total_length: int = Field(..., description="Number of concepts in chain")
    estimated_difficulty: str = Field("medium", description="Difficulty level")


class TopicTreeNode(BaseModel):
    """A node in the topic tree hierarchy."""
    
    id: str = Field(..., description="Node identifier")
    title: str = Field(..., description="Topic/subtopic title")
    description: Optional[str] = Field(None, description="Description")
    node_type: str = Field(..., description="'topic' or 'subtopic'")
    concept_ids: List[str] = Field(default_factory=list, description="Associated concepts")
    children: List["TopicTreeNode"] = Field(default_factory=list, description="Child nodes")


class HierarchyExport(BaseModel):
    """Complete hierarchy export."""
    
    document_id: str = Field(..., description="Document identifier")
    hierarchy_type: str = Field(..., description="Type of hierarchy")
    created_at: str = Field(..., description="Creation timestamp")
    data: Dict[str, Any] = Field(..., description="Hierarchy data")


class GraphExportRequest(BaseModel):
    """Request to export a graph."""
    
    format: str = Field("cytoscape", description="Export format")
    include_metrics: bool = Field(True, description="Include graph metrics")


class GraphAnalysisResult(BaseModel):
    """Result of graph analysis."""
    
    document_id: str = Field(..., description="Document identifier")
    metrics: GraphMetrics = Field(..., description="Graph metrics")
    
    # Centrality analysis
    centrality: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Centrality scores by metric"
    )
    
    # Community detection
    communities: List[List[str]] = Field(
        default_factory=list,
        description="Detected communities"
    )
    
    # Key concepts
    key_concepts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most important concepts"
    )
    
    # Prerequisite analysis
    foundation_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts with no prerequisites"
    )
    advanced_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts with many dependents"
    )
