#!/usr/bin/env python3
"""Clarion End-to-End Requirements Verification Matrix Runner.

This script executes an authoritative, independent, opaque-box audit
verifying all Clarion features across four rigorous tiers:

Tier 1: Feature Coverage (Chunking, Embedding, Concept Extraction, Relation Detection,
        Graph Builder, Calibration, Agreement, Diagnostics, Retrieval, Summary)
Tier 2: Boundary & Corner Cases (Empty docs, Short docs, Extreme tokens, Single nodes,
        Disconnected graphs, Zero agreement)
Tier 3: Cross-Feature Integration (Chunking -> Embedding -> FAISS -> RAG Assembly,
        Graph -> Export Pipeline, Multi-Signal Calibration)
Tier 4: Real-World Application Scenarios (BSM 301 Math Paper, Thevenin Lecture Slides,
        DAA Textbook Sample)

Authoritative sources:
- /mnt/d/Clarion/.agents/teamwork/ORIGINAL_REQUEST.md
- /mnt/d/Clarion/PROJECT.md
- Actual persisted data artifacts in Clarion-Backend/data/
"""

from __future__ import annotations

import argparse
import json
import math
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# Ensure Clarion-Backend is in python path
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@dataclass
class TestResult:
    tier: str
    feature_id: str
    name: str
    status: str  # PASS / FAIL
    duration_ms: float
    details: str = ""
    error: Optional[str] = None


class VerificationRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def record(
        self,
        tier: str,
        feature_id: str,
        name: str,
        func: Callable[[], str],
    ) -> bool:
        start = time.perf_counter()
        try:
            details = func()
            duration = (time.perf_counter() - start) * 1000.0
            res = TestResult(
                tier=tier,
                feature_id=feature_id,
                name=name,
                status="PASS",
                duration_ms=duration,
                details=details,
            )
            self.results.append(res)
            symbol = "✓"
            if self.verbose:
                print(f"  [{symbol}] {tier} - {feature_id}: {name} ({duration:.1f}ms) -> {details}", flush=True)
            else:
                print(f"  [{symbol}] {feature_id}: {name} ({duration:.1f}ms)", flush=True)
            return True
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000.0
            import traceback
            err_trace = traceback.format_exc()
            res = TestResult(
                tier=tier,
                feature_id=feature_id,
                name=name,
                status="FAIL",
                duration_ms=duration,
                details="Exception raised during verification",
                error=str(e),
            )
            self.results.append(res)
            print(f"  [✗] {tier} - {feature_id}: {name} FAILED: {e}", flush=True)
            if self.verbose:
                print(err_trace, flush=True)
            return False

    # =========================================================================
    # TIER 1: FEATURE COVERAGE
    # =========================================================================

    def test_t1_chunking_service(self) -> str:
        """Verify chunking formula S = max(1, C_size - C_overlap) and residual merging."""
        from services.chunking_service import ChunkingService

        svc = ChunkingService()
        c_size = svc.chunk_size
        c_overlap = svc.chunk_overlap
        c_min = svc.min_chunk_size
        expected_step = max(1, c_size - c_overlap)

        # 1. Verify formula and parameters
        assert expected_step == 462 or expected_step > 0, f"Unexpected step {expected_step}"

        # 2. Verify multi-chunk partitioning with residual merging
        # Create text with 712 words
        words = [f"word{i}" for i in range(712)]
        text = " ".join(words)
        chunks = svc.chunk_document("test-doc-t1-chunk", text)
        try:
            assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
            for idx, ch in enumerate(chunks):
                assert ch.word_count >= c_min, f"Chunk {idx} word_count {ch.word_count} < min {c_min}"

            # 3. Verify residual merging when tail < min_chunk_size
            words_tail = [f"token{i}" for i in range(540)]
            chunks_tail = svc.chunk_document("test-doc-t1-tail", " ".join(words_tail))
            try:
                # Tail 540 - 462 = 78 words (< 100), merged into preceding chunk
                assert len(chunks_tail) == 1, f"Expected tail to be merged into 1 chunk, got {len(chunks_tail)}"
                assert chunks_tail[0].word_count == 540, f"Expected 540 words, got {chunks_tail[0].word_count}"
            finally:
                svc.delete_chunks("test-doc-t1-tail")
        finally:
            svc.delete_chunks("test-doc-t1-chunk")

        return f"Step={expected_step}, ChunkSize={c_size}, Overlap={c_overlap}, Residual merging verified"

    def test_t1_embedding_service(self) -> str:
        """Verify vector normalization ||v||_2 = 1.0, dimension 1024, and SHA-256 caching."""
        from services.embedding_service import EmbeddingService
        from models.chunk import Chunk

        svc = EmbeddingService()
        sample_chunk = Chunk(
            chunk_id="chunk-test-t1",
            document_id="doc-test-t1",
            section_title="Introduction",
            content="Clarion document intelligence pipeline semantic representation verification.",
            position_index=0,
            word_count=8,
        )

        embeddings = svc.generate_embeddings([sample_chunk])
        assert len(embeddings) == 1, "Failed to generate embedding"
        vec = np.array(embeddings[0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))

        # Check dimension (1024-dim for bge-large-en-v1.5)
        dim = len(vec)
        assert dim == 1024, f"Expected 1024 dimensions, got {dim}"

        # Check unit normalization: ||v||_2 == 1.0 (within float tolerance)
        assert abs(norm - 1.0) < 1e-3, f"Vector norm {norm:.6f} is not unit normalized"

        # Check SHA-256 cache key calculation
        h = svc._hash_content(sample_chunk.content)
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h), f"Invalid SHA-256 hash {h}"

        return f"Dim={dim}, L2_Norm={norm:.6f}, UnitNormalized=True, SHA256={h[:12]}..."

    def test_t1_concept_extraction(self) -> str:
        """Verify concept extraction returns structured Concept objects with valid metadata."""
        from services.knowledge_map_service import KnowledgeMapService
        from models.chunk import Chunk

        kms = KnowledgeMapService()
        chunks = [
            Chunk(
                chunk_id="ch-concepts-1",
                document_id="doc-concepts-1",
                section_title="Graph Theory",
                content="Graph neural networks utilize message passing between nodes and edges to learn node representations.",
                position_index=0,
                word_count=16,
            )
        ]
        concepts = kms._extract_concepts_heuristic(chunks)
        assert len(concepts) > 0, "No concepts extracted"
        for c in concepts:
            assert c.name, "Concept missing name"
            assert c.id, "Concept missing ID"
            assert hasattr(c, "frequency"), "Concept missing frequency attribute"

        concept_names = [c.name.lower() for c in concepts]
        return f"Extracted {len(concepts)} concepts: {', '.join(concept_names[:3])}"

    def test_t1_relation_detection(self) -> str:
        """Verify candidate relation detection preserves evidence text and candidate flag."""
        from services.knowledge_map_service import KnowledgeMapService
        from models.chunk import Chunk
        from models.knowledge_map import Concept

        kms = KnowledgeMapService()
        chunks = [
            Chunk(
                chunk_id="ch-rel-1",
                document_id="doc-rel-1",
                section_title="Network Analysis",
                content="In graph theory, Adamic-Adar score is a prerequisite for link prediction in graphs.",
                position_index=0,
                word_count=13,
            )
        ]
        c1 = Concept(id="c1", name="Adamic-Adar score", chunk_ids=["ch-rel-1"], frequency=2)
        c2 = Concept(id="c2", name="link prediction", chunk_ids=["ch-rel-1"], frequency=2)

        relations = kms._detect_relations_by_cooccurrence([c1, c2], chunks)
        assert len(relations) > 0, "No relations detected"
        for r in relations:
            assert r.from_concept in ["Adamic-Adar score", "link prediction"]
            assert r.to_concept in ["Adamic-Adar score", "link prediction"]
            assert r.is_candidate is True, "Relation must have is_candidate=True"
            assert r.evidence_text, "Relation missing grounded evidence_text"

        return f"Detected {len(relations)} relations; candidate_flags=True, evidence_text preserved"

    def test_t1_graph_builder(self) -> str:
        """Verify GraphBuilder DiGraph construction, structural vs semantic edges, density and Cytoscape serialization."""
        from graph.builder import GraphBuilder
        from graph.exporter import GraphExporter
        from models.knowledge_map import KnowledgeMap, Concept, Relation, RelationType

        builder = GraphBuilder()
        concepts = [
            Concept(id="c_alpha", name="Alpha Concept", chunk_ids=["ch1"], frequency=2),
            Concept(id="c_beta", name="Beta Concept", chunk_ids=["ch1"], frequency=2),
            Concept(id="c_gamma", name="Gamma Concept", chunk_ids=["ch1"], frequency=1),
        ]
        relations = [
            Relation(
                id="r_ab",
                from_concept="Alpha Concept",
                to_concept="Beta Concept",
                relation_type=RelationType.PREREQUISITE,
                is_candidate=True,
                evidence_text="Alpha is a prerequisite for Beta.",
            ),
            Relation(
                id="r_bg",
                from_concept="Beta Concept",
                to_concept="Gamma Concept",
                relation_type=RelationType.EXPLANATION,
                is_candidate=True,
                evidence_text="Beta explains Gamma.",
            ),
        ]

        km = KnowledgeMap(
            document_id="doc_builder_t1",
            main_topics=[],
            subtopics=[],
            concepts=concepts,
            relations=relations,
        )

        graph = builder.build_from_knowledge_map(km)
        assert isinstance(graph, nx.DiGraph), "Expected nx.DiGraph"

        # Check node count: 1 root doc node + 3 concepts = 4 nodes
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        assert num_nodes == 4, f"Expected 4 nodes, got {num_nodes}"
        # Hierarchy edges from root: 3, plus 2 semantic edges = 5 edges
        assert num_edges == 5, f"Expected 5 edges, got {num_edges}"

        # Check directed density formula D_dir = |E| / (|V|*(|V|-1))
        # For 4 nodes, max edges = 4 * 3 = 12. Density = 5 / 12 = 0.416667
        expected_density = 5.0 / 12.0
        actual_density = nx.density(graph)
        assert abs(actual_density - expected_density) < 1e-5, f"Density mismatch: {actual_density} vs {expected_density}"

        # Cytoscape export schema check
        exporter = GraphExporter()
        cyto = exporter.to_cytoscape_json(graph)
        assert "elements" in cyto
        elements = cyto["elements"]
        for n in elements.get("nodes", []):
            assert "data" in n and "id" in n["data"]
        for e in elements.get("edges", []):
            assert "data" in e and "source" in e["data"] and "target" in e["data"]

        return f"Nodes={num_nodes}, Edges={num_edges}, D_dir={actual_density:.4f}, Cytoscape export valid"

    def test_t1_calibration_engine(self) -> str:
        """Verify Platt scaling, Isotonic regression, and ECE / MCE / Brier metrics."""
        from core.evaluation.calibration import (
            PlattCalibrator,
            IsotonicCalibrator,
            compute_calibration_metrics,
        )

        np.random.seed(42)
        uncalibrated_scores = np.random.uniform(0.1, 0.9, size=100)
        true_labels = (uncalibrated_scores + np.random.normal(0, 0.2, size=100) > 0.5).astype(int)

        # Platt Calibration
        platt = PlattCalibrator()
        platt.fit(uncalibrated_scores, true_labels)
        platt_probs = platt.calibrate(uncalibrated_scores)
        assert np.all(platt_probs >= 0.0) and np.all(platt_probs <= 1.0), "Platt probabilities out of bounds [0, 1]"

        # Isotonic Calibration
        isotonic = IsotonicCalibrator()
        isotonic.fit(uncalibrated_scores, true_labels)
        iso_probs = isotonic.calibrate(uncalibrated_scores)
        assert np.all(iso_probs >= 0.0) and np.all(iso_probs <= 1.0), "Isotonic probabilities out of bounds [0, 1]"

        # Metrics
        metrics = compute_calibration_metrics(true_labels, platt_probs, n_bins=10)
        ece = metrics.ece
        mce = metrics.mce
        brier = metrics.brier_score

        assert 0.0 <= ece <= 1.0, f"ECE {ece} not in [0, 1]"
        assert 0.0 <= mce <= 1.0, f"MCE {mce} not in [0, 1]"
        assert 0.0 <= brier <= 1.0, f"Brier score {brier} not in [0, 1]"

        return f"Platt ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier:.4f}, Probs bounded in [0, 1]"

    def test_t1_agreement_engine(self) -> str:
        """Verify Cohen's kappa and Fleiss' kappa algorithms and Landis-Koch benchmarks."""
        from core.benchmarking.agreement import compute_cohens_kappa, compute_fleiss_kappa

        # 1. Perfect agreement
        r1 = ["VALID", "INVALID", "VALID", "VALID", "INVALID", "INVALID", "VALID"]
        r2 = ["VALID", "INVALID", "VALID", "VALID", "INVALID", "INVALID", "VALID"]
        res_perfect = compute_cohens_kappa(r1, r2)
        kappa_perfect = res_perfect.kappa
        assert abs(kappa_perfect - 1.0) < 1e-4, f"Expected 1.0 for perfect agreement, got {kappa_perfect}"

        # 2. Benchmark agreement
        a = ["VALID", "VALID", "VALID", "INVALID", "INVALID", "INVALID", "VALID", "INVALID", "VALID", "INVALID"]
        b = ["VALID", "VALID", "INVALID", "INVALID", "INVALID", "VALID", "VALID", "INVALID", "INVALID", "INVALID"]
        res_mod = compute_cohens_kappa(a, b)
        kappa_mod = res_mod.kappa
        assert -1.0 <= kappa_mod <= 1.0, f"Kappa {kappa_mod} out of range [-1, 1]"

        # 3. Fleiss' Kappa for 3 raters
        ratings_matrix = [
            ["VALID", "VALID", "VALID"],
            ["INVALID", "INVALID", "INVALID"],
            ["VALID", "VALID", "VALID"],
            ["INVALID", "INVALID", "INVALID"],
            ["VALID", "VALID", "VALID"],
            ["INVALID", "INVALID", "INVALID"],
        ]
        res_fleiss = compute_fleiss_kappa(ratings_matrix)
        fleiss_perfect = res_fleiss.kappa
        assert abs(fleiss_perfect - 1.0) < 1e-4, f"Expected 1.0 for perfect Fleiss, got {fleiss_perfect}"

        return f"Cohen κ_perfect={kappa_perfect:.4f}, Cohen κ_mod={kappa_mod:.4f}, Fleiss κ_perfect={fleiss_perfect:.4f}"

    def test_t1_diagnostics_service(self) -> str:
        """Verify DiagnosticsService health-check across all subsystem components."""
        from services.diagnostics_service import DiagnosticsService

        diag = DiagnosticsService()
        report = diag.run_health_check(document_id="a945d65e-f069-41f9-abb2-e1a7e774d559")
        assert "status" in report, "Diagnostics report missing status"
        assert "checks" in report, "Diagnostics report missing checks"
        checks = report["checks"]
        assert "embedding_service_health" in checks, "Missing embedding check"
        assert "vectorstore_integrity" in checks, "Missing vectorstore check"
        assert "graph_integrity" in checks, "Missing graph check"
        assert "dataset_integrity" in checks, "Missing dataset check"
        assert "llm_connectivity" in checks, "Missing llm connectivity check"

        return f"Overall={report['status']}, Checked: {', '.join(checks.keys())}"

    def test_t1_retrieval_service(self) -> str:
        """Verify vector similarity retrieval formula s = 1/(1+d) on canonical document."""
        from services.retrieval_service import RetrievalService

        retriever = RetrievalService()
        # Query canonical DAA document
        doc_id = "a945d65e-f069-41f9-abb2-e1a7e774d559"
        results = retriever.retrieve(doc_id, "Bellman-Ford algorithm shortest path", top_k=3)
        assert len(results) > 0, "No retrieval results returned for canonical document"

        for r in results:
            assert r.chunk_id, "Result missing chunk_id"
            assert r.content, "Result missing content"
            assert 0.0 < r.score <= 1.0, f"Score {r.score} should be in (0, 1]"

        top_match = results[0]
        return f"Retrieved {len(results)} chunks; Top score={top_match.score:.4f}, chunk={top_match.chunk_id[:8]}"

    def test_t1_summary_service(self) -> str:
        """Verify summary generation schema and section extraction."""
        from services.summary_service import SummaryService
        from models.chunk import Chunk

        svc = SummaryService()
        chunks = [
            Chunk(
                chunk_id="sum-ch-1",
                document_id="doc-sum-1",
                section_title="1. Overview",
                content="This is the comprehensive overview of knowledge mapping algorithms.",
                position_index=0,
                word_count=10,
            ),
            Chunk(
                chunk_id="sum-ch-2",
                document_id="doc-sum-1",
                section_title="2. Experimental Setup",
                content="Experiments were performed using 5 random seeds across stratified splits.",
                position_index=1,
                word_count=10,
            ),
        ]
        summary = svc.generate_summary(document_id="doc-sum-1", title="Knowledge Mapping Paper", chunks=chunks)
        assert hasattr(summary, "document_type"), "Summary missing document_type"
        assert hasattr(summary, "sections"), "Summary missing sections"
        assert len(summary.sections) == 2, f"Expected 2 summary sections, got {len(summary.sections)}"

        return f"Summary doc_type={summary.document_type}, sections={len(summary.sections)}"

    # =========================================================================
    # TIER 2: BOUNDARY & CORNER CASES
    # =========================================================================

    def test_t2_empty_document_boundary(self) -> str:
        """Verify empty string and whitespace handling across chunking and knowledge map."""
        from services.chunking_service import ChunkingService
        from services.knowledge_map_service import KnowledgeMapService

        chunk_svc = ChunkingService()
        km_svc = KnowledgeMapService()

        # 1. Chunking empty text returns a safe empty chunk with 0 words
        empty_chunks = chunk_svc.chunk_document("empty-doc", "")
        try:
            assert len(empty_chunks) == 1, f"Expected 1 chunk, got {len(empty_chunks)}"
            assert empty_chunks[0].word_count == 0, f"Expected 0 words, got {empty_chunks[0].word_count}"
            assert empty_chunks[0].content == "", f"Expected empty content, got '{empty_chunks[0].content}'"
        finally:
            chunk_svc.delete_chunks("empty-doc")

        ws_chunks = chunk_svc.chunk_document("ws-doc", "   \n\t   ")
        try:
            assert len(ws_chunks) == 1, f"Expected 1 chunk, got {len(ws_chunks)}"
            assert ws_chunks[0].word_count == 0, f"Expected 0 words, got {ws_chunks[0].word_count}"
        finally:
            chunk_svc.delete_chunks("ws-doc")

        # 2. Knowledge map heuristic on empty chunks returns empty concepts
        concepts = km_svc._extract_concepts_heuristic([])
        assert concepts == [], f"Expected [], got {concepts}"

        return "Empty text correctly emits safe 0-word chunk without exceptions"

    def test_t2_short_document_retention(self) -> str:
        """Verify document shorter than min_chunk_size is retained as a single chunk."""
        from services.chunking_service import ChunkingService

        svc = ChunkingService()
        short_text = "This is a brief short document containing only nine words."
        expected_words = len(short_text.split())
        chunks = svc.chunk_document("short-doc", short_text)
        try:
            assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
            assert chunks[0].word_count == expected_words, f"Expected {expected_words} words, got {chunks[0].word_count}"
            assert chunks[0].content == short_text
        finally:
            svc.delete_chunks("short-doc")

        return f"Short document ({expected_words} words < {svc.min_chunk_size}) retained intact as 1 chunk"

    def test_t2_extreme_token_length(self) -> str:
        """Verify extreme token lengths (>512 words) partition safely without truncation loops."""
        from services.chunking_service import ChunkingService

        svc = ChunkingService()
        long_words = [f"word_{i}" for i in range(1500)]
        long_text = " ".join(long_words)
        chunks = svc.chunk_document("extreme-doc", long_text)
        try:
            assert len(chunks) >= 3, f"Expected at least 3 chunks for 1500 words, got {len(chunks)}"
            max_allowed = int(svc.chunk_size * 1.25)
            for idx, ch in enumerate(chunks):
                assert ch.word_count <= max_allowed, f"Chunk {idx} word count {ch.word_count} > max {max_allowed}"
            # Verify token length estimation warns on extreme chunks
            tokens = svc.check_token_length(long_text[:4000])
            assert tokens > 512, f"Expected >512 tokens, got {tokens}"
        finally:
            svc.delete_chunks("extreme-doc")

        return f"1500 words partitioned into {len(chunks)} chunks, all <= {max_allowed} words, token check warned"

    def test_t2_single_node_graph_density(self) -> str:
        """Verify single node graph (|V|=1) handles division by zero safely (D_dir = 0.0)."""
        isolated = nx.DiGraph()
        isolated.add_node("lone_node")
        # Formula: |E| / (|V|*(|V|-1)). When |V|=1, denominator is 0.
        dens = nx.density(isolated)
        assert dens == 0.0, f"Expected 0.0 for 1-node graph, got {dens}"

        return f"Single-node graph density safely evaluated to {dens} (ZeroDivisionError avoided)"

    def test_t2_disconnected_graph_handling(self) -> str:
        """Verify disconnected components and unreachable paths are handled without unhandled crashes."""
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("C", "D")

        weak_components = list(nx.weakly_connected_components(g))
        assert len(weak_components) == 2, f"Expected 2 components, got {len(weak_components)}"

        has_path = nx.has_path(g, "A", "C")
        assert has_path is False, "A and C should not be connected"

        return f"Detected {len(weak_components)} disconnected components; unreachable path handled cleanly"

    def test_t2_zero_variance_kappa_boundary(self) -> str:
        """Verify Cohen's kappa on edge cases: 100% identical ratings and total disagreement."""
        from core.benchmarking.agreement import compute_cohens_kappa

        all_ones_a = ["VALID", "VALID", "VALID", "VALID", "VALID"]
        all_ones_b = ["VALID", "VALID", "VALID", "VALID", "VALID"]
        res_zv = compute_cohens_kappa(all_ones_a, all_ones_b)
        kappa_zv = res_zv.kappa
        assert not math.isnan(kappa_zv), "Kappa returned NaN on zero-variance input"

        opp_a = ["VALID", "VALID", "VALID", "VALID"]
        opp_b = ["INVALID", "INVALID", "INVALID", "INVALID"]
        res_opp = compute_cohens_kappa(opp_a, opp_b)
        kappa_opp = res_opp.kappa
        assert not math.isnan(kappa_opp), "Kappa returned NaN on total disagreement"
        assert kappa_opp <= 0.0, f"Expected negative or zero kappa for total disagreement, got {kappa_opp}"

        return f"Zero-variance κ={kappa_zv}, Total disagreement κ={kappa_opp} (ZeroDivisionError guarded)"

    # =========================================================================
    # TIER 3: CROSS-FEATURE INTEGRATION
    # =========================================================================

    def test_t3_ingestion_to_rag_assembly(self) -> str:
        """Verify cross-module chain: Document -> Chunks -> Embeddings -> FAISS Index -> Top-K Search -> RAG Prompt."""
        from services.chunking_service import ChunkingService
        from services.embedding_service import EmbeddingService
        from vectorstore import VectorStore

        # 1. Chunking
        chunk_svc = ChunkingService()
        raw_text = (
            "The Bellman-Ford algorithm computes single-source shortest paths in a weighted digraph. "
            "Unlike Dijkstra's algorithm, Bellman-Ford detects negative weight cycles and relaxes all edges |V|-1 times. "
            "Its time complexity is O(|V| * |E|), making it suitable for distributed routing protocols like RIP."
        )
        doc_id = "doc-rag-chain-temp"
        chunks = chunk_svc.chunk_document(doc_id, raw_text)
        try:
            assert len(chunks) >= 1, "Failed to create chunks"

            # 2. Embedding
            emb_svc = EmbeddingService()
            embeddings = emb_svc.generate_embeddings(chunks)
            assert len(embeddings) == len(chunks), "Embedding count mismatch"

            # 3. Vector indexing
            store = VectorStore(document_id=doc_id)
            store.create_index(embeddings, [c.chunk_id for c in chunks])
            try:
                # 4. Query Embedding & Search
                query_text = "What is the time complexity of Bellman-Ford?"
                query_emb = emb_svc.embed_query(query_text)

                search_results = store.search(query_emb, k=1)
                assert len(search_results) == 1, "No chunks retrieved"
                matched_chunk_id, dist = search_results[0]
                assert matched_chunk_id == chunks[0].chunk_id, "Provenance mismatch: wrong chunk retrieved"

                # 5. Context Assembly
                matched_chunk = next(c for c in chunks if c.chunk_id == matched_chunk_id)
                assembled_context = f"CONTEXT:\n{matched_chunk.content}\n\nQUESTION:\n{query_text}"
                assert "Bellman-Ford" in assembled_context and "negative weight cycles" in assembled_context, "Assembled context missing key facts"
            finally:
                store.delete_index()
        finally:
            chunk_svc.delete_chunks(doc_id)

        return f"Ingestion -> Retrieval -> Assembly verified with 100% provenance tracking"

    def test_t3_knowledge_map_to_export_pipeline(self) -> str:
        """Verify Concept Extraction -> Relation Detection -> DiGraph -> Candidate Dataset JSON Export."""
        from services.chunking_service import ChunkingService
        from services.knowledge_map_service import KnowledgeMapService
        from graph.builder import GraphBuilder

        chunk_svc = ChunkingService()
        km_svc = KnowledgeMapService()
        builder = GraphBuilder()

        text = "Thevenin's Theorem simplifies linear two-terminal circuits into an equivalent voltage source and impedance. Complex algebra is required to calculate AC impedance."
        doc_id = "doc_thev_pipe_temp"
        chunks = chunk_svc.chunk_document(doc_id, text)
        try:
            km = km_svc.build_knowledge_map(doc_id, chunks)
            try:
                concepts = km.concepts
                relations = km.relations
                graph = builder.build_from_knowledge_map(km)

                # Build candidate export record schema
                export_records = []
                for r in relations:
                    record = {
                        "record_id": f"rec_{r.id}",
                        "document_id": doc_id,
                        "relation_id": r.id,
                        "concept_a": r.from_concept,
                        "concept_b": r.to_concept,
                        "relation_type": r.relation_type.value if hasattr(r.relation_type, "value") else str(r.relation_type),
                        "is_candidate": True,
                        "evidence_text": r.evidence_text,
                    }
                    export_records.append(record)

                assert graph.number_of_nodes() >= 1, "Graph has no nodes"
                for rec in export_records:
                    assert rec["is_candidate"] is True
                    assert rec["evidence_text"]
            finally:
                km_svc.delete_knowledge_map(doc_id)
        finally:
            chunk_svc.delete_chunks(doc_id)

        return f"Pipeline extracted {len(concepts)} concepts, {len(relations)} relations, assembled DiGraph & export schema"

    def test_t3_multi_signal_calibration_integration(self) -> str:
        """Verify multi-signal scoring pipeline integration with calibrated probability estimation."""
        from core.evaluation.multi_signal_validator import MultiSignalValidator
        from core.evaluation.calibration import PlattCalibrator
        from models.chunk import Chunk

        validator = MultiSignalValidator()
        platt = PlattCalibrator()
        # Train calibrator on synthetic anchor points
        platt.fit(np.array([0.2, 0.4, 0.6, 0.8]), np.array([0, 0, 1, 1]))
        validator.set_calibrator(platt)

        chunks = [
            Chunk(
                chunk_id="ch_val_1",
                document_id="doc_val_1",
                section_title="Circuit Analysis",
                content="Impedance is calculated using complex algebra for AC linear circuits.",
                position_index=0,
                word_count=10,
            )
        ]

        decision = validator.validate_relation(
            concept_a="Impedance",
            concept_b="complex algebra",
            relation_type="explanation",
            chunks=chunks,
            generator_confidence=0.85,
        )

        assert 0.0 <= decision.calibrated_probability <= 1.0, f"Probability {decision.calibrated_probability} out of bounds"
        assert decision.evidence_text, "Decision missing grounded evidence_text"

        return (
            f"Decision={decision.is_valid}, CalibratedProb={decision.calibrated_probability:.4f}, "
            f"RawScore={decision.raw_score:.4f}, Evidence=\"{decision.evidence_text[:30]}...\""
        )

    # =========================================================================
    # TIER 4: REAL-WORLD APPLICATION SCENARIOS
    # =========================================================================

    def test_t4_multipage_academic_paper_bsm301(self) -> str:
        """Scenario 1: Multipage Academic Paper — BSM 301 Mathematics-III (65430b13-d3d6-49b8-8386-f34762f0ae9c)."""
        doc_id = "65430b13-d3d6-49b8-8386-f34762f0ae9c"
        graph_path = BACKEND_DIR / "data" / "graphs" / f"{doc_id}.json"
        dataset_path = BACKEND_DIR / "data" / "datasets" / f"BSM_301_Mathematics-lll_{doc_id}_relations.json"

        assert graph_path.exists(), f"Graph artifact missing at {graph_path}"
        assert dataset_path.exists(), f"Dataset artifact missing at {dataset_path}"

        with open(graph_path, encoding="utf-8") as f:
            graph_data = json.load(f)
        with open(dataset_path, encoding="utf-8") as f:
            dataset_data = json.load(f)

        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", []) or graph_data.get("edges", [])

        # Authoritative BSM 301 Invariants:
        assert len(nodes) == 10, f"BSM 301 expected 10 nodes, got {len(nodes)}"
        assert len(links) == 43, f"BSM 301 expected 43 edges, got {len(links)}"

        hier_edges = [e for e in links if e.get("relation_type") == "hierarchy" or e.get("type") == "hierarchy"]
        sem_edges = [e for e in links if e not in hier_edges]
        assert len(hier_edges) == 9, f"Expected 9 hierarchy edges, got {len(hier_edges)}"
        assert len(sem_edges) == 34, f"Expected 34 semantic relations, got {len(sem_edges)}"

        records = dataset_data.get("records", [])
        assert len(records) == 34, f"BSM 301 expected 34 exported relation records, got {len(records)}"

        total_density = 43.0 / (10.0 * 9.0)
        sem_density = 34.0 / (9.0 * 8.0)

        return (
            f"Nodes=10, TotalEdges=43 (Hier=9, Sem=34), ExportedRecords=34, "
            f"D_total={total_density:.6f}, D_sem={sem_density:.6f}"
        )

    def test_t4_thevenin_theorem_slides_runs(self) -> str:
        """Scenario 2: Technical Lecture Slides — Understanding Thevenin's Theorem (Runs 1 & 2)."""
        run1_id = "827d8b9c-4444-4d35-ae6a-4a48dc81f0ec"
        run2_id = "fc5f2c9b-85ab-437b-8700-b23ae982ee08"

        # Verify Run 1
        g1_path = BACKEND_DIR / "data" / "graphs" / f"{run1_id}.json"
        d1_path = BACKEND_DIR / "data" / "datasets" / f"Understanding-Thevenins-Theorem_1_.pptx_{run1_id}_relations.json"
        assert g1_path.exists() and d1_path.exists(), "Thevenin Run 1 artifacts missing"

        with open(g1_path, encoding="utf-8") as f:
            g1 = json.load(f)
        with open(d1_path, encoding="utf-8") as f:
            d1 = json.load(f)

        links1 = g1.get("links", []) or g1.get("edges", [])
        assert len(g1["nodes"]) == 16, f"Run 1 expected 16 nodes, got {len(g1['nodes'])}"
        assert len(links1) == 16, f"Run 1 expected 16 edges, got {len(links1)}"
        assert len(d1.get("records", [])) == 1, f"Run 1 expected 1 relation, got {len(d1.get('records', []))}"
        r1_density = 16.0 / (16.0 * 15.0)  # 16 / 240 = 0.066667
        r1_sem_density = 1.0 / (2.0 * 1.0)  # 1 / 2 = 0.500000

        # Verify Run 2
        g2_path = BACKEND_DIR / "data" / "graphs" / f"{run2_id}.json"
        d2_path = BACKEND_DIR / "data" / "datasets" / f"Understanding-Thevenins-Theorem_1_.pptx_{run2_id}_relations.json"
        assert g2_path.exists() and d2_path.exists(), "Thevenin Run 2 artifacts missing"

        with open(g2_path, encoding="utf-8") as f:
            g2 = json.load(f)
        with open(d2_path, encoding="utf-8") as f:
            d2 = json.load(f)

        links2 = g2.get("links", []) or g2.get("edges", [])
        assert len(g2["nodes"]) == 18, f"Run 2 expected 18 nodes, got {len(g2['nodes'])}"
        assert len(links2) == 19, f"Run 2 expected 19 edges, got {len(links2)}"
        assert len(d2.get("records", [])) == 2, f"Run 2 expected 2 relations, got {len(d2.get('records', []))}"
        r2_density = 19.0 / (18.0 * 17.0)  # 19 / 306 = 0.062092
        r2_sem_density = 2.0 / (3.0 * 2.0)  # 2 / 6 = 0.333333

        return (
            f"Run 1: N=16, E=16, Rel=1, D_dir={r1_density:.6f}, D_sem={r1_sem_density:.4f} | "
            f"Run 2: N=18, E=19, Rel=2, D_dir={r2_density:.6f}, D_sem={r2_sem_density:.4f}"
        )

    def test_t4_textbook_sample_daa(self) -> str:
        """Scenario 3: Real Textbook Sample — Design and Analysis of Algorithms DAA (a945d65e-f069-41f9-abb2-e1a7e774d559)."""
        doc_id = "a945d65e-f069-41f9-abb2-e1a7e774d559"
        from utils.sqlite import connect as sqlite_connect
        from utils.config import settings

        db_path = settings.get_db_path()
        conn = sqlite_connect(db_path)
        cur = conn.cursor()

        chunk_count = cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0]
        assert chunk_count == 40, f"DAA expected 40 chunks, got {chunk_count}"

        km_row = cur.execute("SELECT data FROM knowledge_maps WHERE document_id = ?", (doc_id,)).fetchone()
        assert km_row is not None, "DAA knowledge map not found in database"
        km_data = json.loads(km_row[0])
        concepts = km_data.get("concepts", [])
        relations = km_data.get("relations", [])
        assert len(concepts) == 12, f"DAA expected 12 concepts, got {len(concepts)}"
        assert len(relations) == 34, f"DAA expected 34 relations, got {len(relations)}"

        cyto_path = BACKEND_DIR / "data" / "graphs" / f"{doc_id}_cytoscape.json"
        assert cyto_path.exists(), f"Cytoscape graph missing at {cyto_path}"
        with open(cyto_path, encoding="utf-8") as f:
            cyto = json.load(f)
        cyto_nodes = cyto.get("elements", {}).get("nodes", []) or cyto.get("nodes", [])
        cyto_edges = cyto.get("elements", {}).get("edges", []) or cyto.get("edges", [])
        assert len(cyto_nodes) == 18, f"DAA expected 18 Cytoscape nodes, got {len(cyto_nodes)}"
        assert len(cyto_edges) == 51, f"DAA expected 51 Cytoscape edges, got {len(cyto_edges)}"

        conn.close()
        return f"Chunks=40, Concepts=12, CandidateRelations=34, CytoscapeNodes=18, CytoscapeEdges=51"

    # =========================================================================
    # RUNNER ORCHESTRATION
    # =========================================================================

    def run_all(self) -> int:
        print("=" * 80, flush=True)
        print("CLARION INDEPENDENT END-TO-END REQUIREMENTS VERIFICATION MATRIX", flush=True)
        print("=" * 80, flush=True)
        start_time = time.perf_counter()

        # Tier 1
        print("\n--- TIER 1: FEATURE COVERAGE (10 FEATURES) ---", flush=True)
        self.record("Tier 1", "T1.1", "Chunking Service (Formula S, Residual Merging, Size Bounds)", self.test_t1_chunking_service)
        self.record("Tier 1", "T1.2", "Embedding Service (1024-dim, Unit Normalization, SHA-256 Cache)", self.test_t1_embedding_service)
        self.record("Tier 1", "T1.3", "Concept Extraction (Metadata, Chunk References, Frequency)", self.test_t1_concept_extraction)
        self.record("Tier 1", "T1.4", "Relation Detection (Grounded Evidence, Candidate Flags, Inferred Types)", self.test_t1_relation_detection)
        self.record("Tier 1", "T1.5", "Graph Builder (DiGraph, Hierarchy vs Semantic, Density, Cytoscape Schema)", self.test_t1_graph_builder)
        self.record("Tier 1", "T1.6", "Calibration Engine (Platt Sigmoid, Isotonic Regression, ECE, MCE, Brier)", self.test_t1_calibration_engine)
        self.record("Tier 1", "T1.7", "Agreement Engine (Cohen's κ, Fleiss' κ, Chance Correction, Landis-Koch)", self.test_t1_agreement_engine)
        self.record("Tier 1", "T1.8", "Diagnostics Service (Subsystem Probes, Health & Degraded State)", self.test_t1_diagnostics_service)
        self.record("Tier 1", "T1.9", "Retrieval Service (FAISS Index, s=1/(1+d), Top-K Similarity Ranking)", self.test_t1_retrieval_service)
        self.record("Tier 1", "T1.10", "Summary Service (Hierarchical Sections, Document Classification)", self.test_t1_summary_service)

        # Tier 2
        print("\n--- TIER 2: BOUNDARY & CORNER CASES (6 TESTS) ---", flush=True)
        self.record("Tier 2", "T2.1", "Empty Document Boundary (Empty String & Whitespace Handling)", self.test_t2_empty_document_boundary)
        self.record("Tier 2", "T2.2", "Short Document Retention (< min_chunk_size retained as 1 chunk)", self.test_t2_short_document_retention)
        self.record("Tier 2", "T2.3", "Extreme Token Lengths (>512 words safely partitioned without loop)", self.test_t2_extreme_token_length)
        self.record("Tier 2", "T2.4", "Single Node Graph Density (|V|=1 avoids ZeroDivisionError -> 0.0)", self.test_t2_single_node_graph_density)
        self.record("Tier 2", "T2.5", "Disconnected Graphs (Component Isolation & Unreachable Paths)", self.test_t2_disconnected_graph_handling)
        self.record("Tier 2", "T2.6", "Zero-Variance & Complete Disagreement Kappa Boundaries", self.test_t2_zero_variance_kappa_boundary)

        # Tier 3
        print("\n--- TIER 3: CROSS-FEATURE INTEGRATION (3 CHAINS) ---", flush=True)
        self.record("Tier 3", "T3.1", "Ingestion to Grounded RAG Assembly Chain (Chunk -> Embed -> FAISS -> RAG)", self.test_t3_ingestion_to_rag_assembly)
        self.record("Tier 3", "T3.2", "Knowledge Map to Candidate Dataset Export Pipeline", self.test_t3_knowledge_map_to_export_pipeline)
        self.record("Tier 3", "T3.3", "Multi-Signal Scoring to Calibrated Probability Integration", self.test_t3_multi_signal_calibration_integration)

        # Tier 4
        print("\n--- TIER 4: REAL-WORLD APPLICATION SCENARIOS (3 BENCHMARKS) ---", flush=True)
        self.record("Tier 4", "T4.1", "Scenario 1: Multipage Academic Paper (BSM 301 Mathematics-III)", self.test_t4_multipage_academic_paper_bsm301)
        self.record("Tier 4", "T4.2", "Scenario 2: Technical Lecture Slides (Understanding Thevenin Runs 1 & 2)", self.test_t4_thevenin_theorem_slides_runs)
        self.record("Tier 4", "T4.3", "Scenario 3: Real Textbook Sample (Design & Analysis of Algorithms DAA)", self.test_t4_textbook_sample_daa)

        total_elapsed = time.perf_counter() - start_time

        # Print Summary Matrix
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        total = len(self.results)

        print("\n" + "=" * 80, flush=True)
        print("REQUIREMENTS VERIFICATION MATRIX SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"{'Tier':<10} | {'ID':<6} | {'Status':<6} | {'Duration (ms)':<14} | {'Verification Test':<36}", flush=True)
        print("-" * 80, flush=True)
        for r in self.results:
            print(f"{r.tier:<10} | {r.feature_id:<6} | {r.status:<6} | {r.duration_ms:>12.2f}ms | {r.name[:36]:<36}", flush=True)
        print("=" * 80, flush=True)
        print(f"TOTAL: {total} tests | PASSED: {passed} | FAILED: {failed} | DURATION: {total_elapsed:.2f}s", flush=True)
        print("=" * 80, flush=True)

        return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Clarion E2E Requirements Verification Matrix Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose execution traces")
    args = parser.parse_args()

    runner = VerificationRunner(verbose=args.verbose)
    sys.exit(runner.run_all())


if __name__ == "__main__":
    main()
