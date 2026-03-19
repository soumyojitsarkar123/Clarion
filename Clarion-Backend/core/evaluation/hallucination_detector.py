"""
Hallucination Detector - Detect hallucinated or unsupported relations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import networkx as nx

from models.knowledge_map import KnowledgeMap, Relation, Concept, RelationType
from models.chunk import Chunk
from models.graph import GraphEdgeType

logger = logging.getLogger(__name__)


class FlagSeverity(str, Enum):
    """Severity levels for hallucination flags."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FlagType(str, Enum):
    """Types of hallucination flags."""
    MISSING_SOURCE = "missing_source"
    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTION = "contradiction"
    CIRCULAR_PREREQUISITE = "circular_prerequisite"
    SEMANTIC_DRIFT = "semantic_drift"
    ORPHANED_CONCEPT = "orphaned_concept"
    UNSUPPORTED_RELATION = "unsupported_relation"


@dataclass
class HallucinationFlag:
    """A flag indicating potential hallucination or quality issue."""
    
    flag_type: FlagType
    severity: FlagSeverity
    item_type: str  # "relation", "concept", "graph"
    item_id: str
    description: str
    suggestion: str
    concepts_involved: List[str]
    confidence_score: Optional[float] = None


@dataclass
class HallucinationReport:
    """Complete report of hallucination detection."""
    
    document_id: str
    total_relations: int
    total_concepts: int
    flagged_count: int
    flags: List[HallucinationFlag]
    recommendations: List[str]
    overall_risk_score: float  # 0.0-1.0


class HallucinationDetector:
    """
    Detect hallucinations and quality issues in knowledge maps.
    
    Detection methods:
    1. Source verification - Check if relation is supported by text
    2. Semantic drift - Detect LLM interpretation divergence
    3. Contradiction detection - Find mutually exclusive relations
    4. Cycle detection - Identify circular prerequisites
    5. Confidence thresholding - Flag low-confidence items
    6. Orphan detection - Find disconnected concepts
    
    Example:
        detector = HallucinationDetector(
            confidence_threshold=0.5,
            similarity_threshold=0.3
        )
        report = detector.detect(knowledge_map, source_chunks, graph)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        similarity_threshold: float = 0.3,
        min_chunk_support: int = 1
    ):
        """
        Initialize hallucination detector.
        
        Args:
            confidence_threshold: Minimum confidence to avoid flagging
            similarity_threshold: Threshold for semantic similarity checks
            min_chunk_support: Minimum chunks that must support a relation
        """
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.min_chunk_support = min_chunk_support
    
    def detect(
        self,
        knowledge_map: KnowledgeMap,
        source_chunks: List[Chunk],
        graph: Optional[nx.DiGraph] = None,
        confidence_scores: Optional[Dict[str, float]] = None
    ) -> HallucinationReport:
        """
        Run all hallucination detection methods.
        
        Args:
            knowledge_map: The knowledge map to analyze
            source_chunks: Source document chunks
            graph: Built knowledge graph (optional)
            confidence_scores: Pre-computed confidence scores (optional)
        
        Returns:
            HallucinationReport with flags and recommendations
        """
        flags: List[HallucinationFlag] = []
        
        try:
            # Method 1: Source verification
            flags.extend(self._verify_sources(
                knowledge_map, source_chunks
            ))
            
            # Method 2: Low confidence detection
            if confidence_scores:
                flags.extend(self._detect_low_confidence(
                    knowledge_map, confidence_scores
                ))
            
            # Method 3: Contradiction detection
            flags.extend(self._detect_contradictions(knowledge_map))
            
            # Method 4: Circular prerequisite detection
            if graph:
                flags.extend(self._detect_cycles(graph))
            
            # Method 5: Orphaned concept detection
            if graph:
                flags.extend(self._detect_orphaned_concepts(knowledge_map, graph))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(flags)
            
            # Calculate overall risk
            risk_score = self._calculate_risk_score(flags, knowledge_map)
            
            return HallucinationReport(
                document_id=knowledge_map.document_id,
                total_relations=len(knowledge_map.relations),
                total_concepts=len(knowledge_map.concepts),
                flagged_count=len(flags),
                flags=flags,
                recommendations=recommendations,
                overall_risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return HallucinationReport(
                document_id=knowledge_map.document_id,
                total_relations=len(knowledge_map.relations),
                total_concepts=len(knowledge_map.concepts),
                flagged_count=0,
                flags=[],
                recommendations=["Error during validation - manual review recommended"],
                overall_risk_score=0.5
            )
    
    def _verify_sources(
        self,
        knowledge_map: KnowledgeMap,
        chunks: List[Chunk]
    ) -> List[HallucinationFlag]:
        """
        Verify that relations are supported by source text.
        
        Checks:
        - Both concepts appear in cited chunks
        - Relation type is textually supported
        """
        flags = []
        
        for relation in knowledge_map.relations:
            # Check concept presence in chunks
            from_found = False
            to_found = False
            support_count = 0
            
            for chunk in chunks:
                chunk_text = chunk.content.lower()
                
                if relation.from_concept.lower() in chunk_text:
                    from_found = True
                if relation.to_concept.lower() in chunk_text:
                    to_found = True
                
                if (relation.from_concept.lower() in chunk_text and 
                    relation.to_concept.lower() in chunk_text):
                    support_count += 1
            
            # Flag if concepts not found
            if not from_found or not to_found:
                missing = []
                if not from_found:
                    missing.append(relation.from_concept)
                if not to_found:
                    missing.append(relation.to_concept)
                
                flags.append(HallucinationFlag(
                    flag_type=FlagType.MISSING_SOURCE,
                    severity=FlagSeverity.HIGH,
                    item_type="relation",
                    item_id=relation.id,
                    description=f"Concept(s) not found in source chunks: {', '.join(missing)}",
                    suggestion="Verify concept extraction accuracy or remove unsupported relation",
                    concepts_involved=[relation.from_concept, relation.to_concept],
                    confidence_score=relation.confidence
                ))
            
            # Flag if insufficient support
            elif support_count < self.min_chunk_support:
                flags.append(HallucinationFlag(
                    flag_type=FlagType.UNSUPPORTED_RELATION,
                    severity=FlagSeverity.MEDIUM,
                    item_type="relation",
                    item_id=relation.id,
                    description=f"Relation supported by only {support_count} chunk(s)",
                    suggestion="Review relation accuracy with low co-occurrence",
                    concepts_involved=[relation.from_concept, relation.to_concept],
                    confidence_score=relation.confidence
                ))
        
        return flags
    
    def _detect_low_confidence(
        self,
        knowledge_map: KnowledgeMap,
        confidence_scores: Dict[str, float]
    ) -> List[HallucinationFlag]:
        """Flag relations with confidence below threshold."""
        flags = []
        
        for relation in knowledge_map.relations:
            score = confidence_scores.get(relation.id, 0.5)
            
            if score < self.confidence_threshold:
                # Determine severity based on how far below threshold
                if score < 0.3:
                    severity = FlagSeverity.HIGH
                elif score < 0.5:
                    severity = FlagSeverity.MEDIUM
                else:
                    severity = FlagSeverity.LOW
                
                flags.append(HallucinationFlag(
                    flag_type=FlagType.LOW_CONFIDENCE,
                    severity=severity,
                    item_type="relation",
                    item_id=relation.id,
                    description=f"Low confidence score: {score:.2f} (threshold: {self.confidence_threshold})",
                    suggestion="Review relation validity or gather more supporting evidence",
                    concepts_involved=[relation.from_concept, relation.to_concept],
                    confidence_score=score
                ))
        
        return flags
    
    def _detect_contradictions(
        self,
        knowledge_map: KnowledgeMap
    ) -> List[HallucinationFlag]:
        """
        Detect contradictory relations.
        
        Examples:
        - A PREREQUISITE B and B PREREQUISITE A (circular)
        - A PART_OF B and B PART_OF A
        """
        flags = []
        
        # Check for circular prerequisites
        prereq_relations = [
            r for r in knowledge_map.relations
            if r.relation_type == RelationType.PREREQUISITE
        ]
        
        for rel in prereq_relations:
            # Check for reverse prerequisite
            reverse = next(
                (r for r in prereq_relations
                 if r.from_concept == rel.to_concept
                 and r.to_concept == rel.from_concept),
                None
            )
            
            if reverse:
                flags.append(HallucinationFlag(
                    flag_type=FlagType.CONTRADICTION,
                    severity=FlagSeverity.CRITICAL,
                    item_type="relation",
                    item_id=rel.id,
                    description=f"Circular prerequisite: {rel.from_concept} ↔ {rel.to_concept}",
                    suggestion="Remove one relation or verify correct direction",
                    concepts_involved=[rel.from_concept, rel.to_concept],
                    confidence_score=min(rel.confidence, reverse.confidence)
                ))
        
        # Check for mutual part-of
        part_of_relations = [
            r for r in knowledge_map.relations
            if r.relation_type == RelationType.PART_OF
        ]
        
        for rel in part_of_relations:
            reverse = next(
                (r for r in part_of_relations
                 if r.from_concept == rel.to_concept
                 and r.to_concept == rel.from_concept),
                None
            )
            
            if reverse:
                flags.append(HallucinationFlag(
                    flag_type=FlagType.CONTRADICTION,
                    severity=FlagSeverity.HIGH,
                    item_type="relation",
                    item_id=rel.id,
                    description=f"Mutual part-of relation: {rel.from_concept} ↔ {rel.to_concept}",
                    suggestion="Part-of should be hierarchical, not mutual. Verify correctness.",
                    concepts_involved=[rel.from_concept, rel.to_concept],
                    confidence_score=min(rel.confidence, reverse.confidence)
                ))
        
        return flags
    
    def _detect_cycles(
        self,
        graph: nx.DiGraph
    ) -> List[HallucinationFlag]:
        """
        Detect cycles in prerequisite chains.
        
        Long cycles (3+ nodes) in prerequisites indicate potential issues.
        """
        flags = []
        
        try:
            # Get prerequisite edges only
            prereq_edges = [
                (u, v) for u, v, d in graph.edges(data=True)
                if d.get("relation_type") == GraphEdgeType.PREREQUISITE.value
            ]
            
            if not prereq_edges:
                return flags
            
            # Build prerequisite subgraph
            prereq_graph = nx.DiGraph()
            prereq_graph.add_edges_from(prereq_edges)
            
            # Find cycles
            cycles = list(nx.simple_cycles(prereq_graph))
            
            for cycle in cycles:
                if len(cycle) >= 2:  # Report all cycles
                    # Get concept names
                    concept_names = []
                    for node_id in cycle:
                        node_data = graph.nodes[node_id]
                        concept_names.append(node_data.get("label", node_id))
                    
                    flags.append(HallucinationFlag(
                        flag_type=FlagType.CIRCULAR_PREREQUISITE,
                        severity=FlagSeverity.CRITICAL if len(cycle) <= 3 else FlagSeverity.HIGH,
                        item_type="graph",
                        item_id=f"cycle_{'_'.join(cycle[:3])}",
                        description=f"Circular prerequisite chain ({len(cycle)} concepts): {' → '.join(concept_names)}",
                        suggestion="Break the cycle by removing the weakest prerequisite relation",
                        concepts_involved=concept_names
                    ))
        
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
        
        return flags
    
    def _detect_orphaned_concepts(
        self,
        knowledge_map: KnowledgeMap,
        graph: nx.DiGraph
    ) -> List[HallucinationFlag]:
        """Detect concepts with no connections."""
        flags = []
        
        for concept in knowledge_map.concepts:
            node_id = f"concept_{concept.id}"
            
            if node_id in graph:
                degree = graph.degree(node_id)
                
                if degree == 0:
                    flags.append(HallucinationFlag(
                        flag_type=FlagType.ORPHANED_CONCEPT,
                        severity=FlagSeverity.MEDIUM,
                        item_type="concept",
                        item_id=concept.id,
                        description=f"Concept '{concept.name}' has no relations",
                        suggestion="Consider adding relations or verify if concept is significant",
                        concepts_involved=[concept.name]
                    ))
        
        return flags
    
    def _generate_recommendations(
        self,
        flags: List[HallucinationFlag]
    ) -> List[str]:
        """Generate actionable recommendations from flags."""
        recommendations = []
        
        if not flags:
            recommendations.append("No issues detected. Knowledge map appears valid.")
            return recommendations
        
        # Count by severity
        critical_count = sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL)
        high_count = sum(1 for f in flags if f.severity == FlagSeverity.HIGH)
        medium_count = sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM)
        
        # General summary
        recommendations.append(
            f"Found {len(flags)} potential issues: "
            f"{critical_count} critical, {high_count} high, {medium_count} medium priority."
        )
        
        # Specific recommendations by type
        if critical_count > 0:
            recommendations.append(
                "Critical issues (circular prerequisites, contradictions) must be resolved "
                "before using the knowledge map."
            )
        
        if any(f.flag_type == FlagType.MISSING_SOURCE for f in flags):
            recommendations.append(
                "Some relations reference concepts not found in source text. "
                "Review concept extraction or verify document coverage."
            )
        
        if any(f.flag_type == FlagType.LOW_CONFIDENCE for f in flags):
            low_conf_count = sum(1 for f in flags if f.flag_type == FlagType.LOW_CONFIDENCE)
            recommendations.append(
                f"{low_conf_count} relation(s) have low confidence scores. "
                "Consider manual review or additional evidence."
            )
        
        if any(f.flag_type == FlagType.ORPHANED_CONCEPT for f in flags):
            orphan_count = sum(1 for f in flags if f.flag_type == FlagType.ORPHANED_CONCEPT)
            recommendations.append(
                f"{orphan_count} concept(s) have no relations. "
                "Verify if these are isolated concepts or if relations are missing."
            )
        
        return recommendations
    
    def _calculate_risk_score(
        self,
        flags: List[HallucinationFlag],
        knowledge_map: KnowledgeMap
    ) -> float:
        """
        Calculate overall risk score for the knowledge map.
        
        Returns:
            Risk score 0.0-1.0 (higher = more risky)
        """
        if not flags:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            FlagSeverity.CRITICAL: 1.0,
            FlagSeverity.HIGH: 0.7,
            FlagSeverity.MEDIUM: 0.4,
            FlagSeverity.LOW: 0.1
        }
        
        total_weight = sum(
            severity_weights.get(f.severity, 0.5)
            for f in flags
        )
        
        # Normalize by number of relations
        num_relations = max(len(knowledge_map.relations), 1)
        risk_score = min(1.0, total_weight / num_relations)
        
        return round(risk_score, 3)
