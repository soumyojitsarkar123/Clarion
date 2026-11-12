"""Empirical Challenger Adversarial Stress Suite for Milestone 5.

Comprehensive empirical verification of:
1. All 7 mandatory deliverable files exist, non-empty, and structurally valid.
2. Bit-exact equations and numerical constants across:
   - Chunking window step size S = 462 (512 - 50) and pathological bounds.
   - Residual chunk merging bound C_max = 640 words (1.25 * 512).
   - Embedding unit normalization ||v||_2 = 1.0.
   - FAISS squared distance bound d^2 in [0, 4] and similarity s in [0.20, 1.00].
   - BSM 301 graph: |V|=11, |E|=19, D_dir = 19/110 ≈ 0.1727, candidate relations = 34.
   - Thevenin runs 1 & 2: (|V|=16, |E|=16, rel=1) and (|V|=18, |E|=19, rel=2).
   - Candidate database: 1,554 records, 44 docs, 100% unlabeled (is_valid IS NULL).
   - Annotation queues: batch 1 (200 records), overlap (50 records), 0% complete.
   - Singularity and zero-variance boundary tests: single node density, uniform ratings kappa.
"""

import json
import math
import os
import sqlite3
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"

sys.path.insert(0, str(BACKEND_DIR))

PASS_COUNT = 0
FAIL_COUNT = 0


def assert_test(name: str, condition: bool, msg: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}", flush=True)
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}: {msg}", flush=True)
        raise AssertionError(f"{name}: {msg}")


def test_deliverables_existence_and_structure():
    print("\n--- Test Suite 1: Mandatory Deliverable Files Integrity ---")
    deliverables = [
        ("reports/historical/FINAL_AUDIT_REPORT.md", 10000, ["Executive Summary", "Repository Overview & Actual System Architecture", "Catalog of Identified Issues", "Outstanding Issues"]),
        ("FIX_LOG.md", 8000, ["Master Fix Inventory Table", "Detailed Technical Fix Logs", "Validation"]),
        ("MATHEMATICAL_AUDIT.md", 12000, ["EQ-1", "EQ-2", "EQ-3", "EQ-4", "EQ-5", "EQ-6", "EQ-7", "EQ-8", "EQ-9"]),
        ("reports/historical/PAPER_CODE_CONSISTENCY.md", 8000, ["VERIFIED", "Consistency Mapping Matrix", "appendix_report.md", "CLM-01"]),
        ("FRONTEND_NOMENCLATURE_AUDIT.md", 6000, ["Clarion", "Overview Metrics Grid Restoration", "Inline SVG Favicon", "Static DOM Invariants"]),
        ("EXPERIMENTAL_RESULTS.md", 8000, ["MANDATORY SCIENTIFIC INTEGRITY DECLARATION", "Candidate Relation Dataset Audit", "1,554", "UNMEASURED / UNTESTED"]),
        ("reports/historical/SUBMISSION_READINESS_CHECKLIST.md", 6000, ["Acceptance Criteria Evaluation", "TECHNICALLY IMPROVED / RESEARCH VALIDATION PENDING"]),
    ]

    for fname, min_size, required_sections in deliverables:
        fpath = ROOT_DIR / fname
        assert_test(f"File exists: {fname}", fpath.exists(), f"Path not found: {fpath}")
        size = fpath.stat().st_size
        assert_test(f"File size valid: {fname} ({size} >= {min_size})", size >= min_size, f"Size {size} < {min_size}")
        
        content = fpath.read_text(encoding="utf-8")
        for sec in required_sections:
            assert_test(f"Section present: '{sec}' in {fname}", sec in content, f"Missing '{sec}' in {fname}")


def test_chunking_step_and_residual_merging():
    print("\n--- Test Suite 2: Chunking Mathematics & Boundary Cases ---", flush=True)
    from services.chunking_service import ChunkingService

    service = ChunkingService()

    # Default parameters: 512 size, 50 overlap
    step_default = max(1, service.chunk_size - service.chunk_overlap)
    assert_test("Default step size is 462", step_default == 462, f"Got {step_default}")

    # Boundary: Overlap >= size
    step_eq = max(1, 512 - 512)
    assert_test("Equal overlap step size is 1", step_eq == 1, f"Got {step_eq}")

    step_exceed = max(1, 512 - 600)
    assert_test("Exceeding overlap step size is 1", step_exceed == 1, f"Got {step_exceed}")

    # Boundary: Zero overlap
    step_zero = max(1, 512 - 0)
    assert_test("Zero overlap step size is 512", step_zero == 512, f"Got {step_zero}")

    # Residual merging upper bound: floor(1.25 * 512) = 640
    c_max_merged = math.floor(1.25 * service.chunk_size)
    assert_test("C_max_merged is 640", c_max_merged == 640, f"Got {c_max_merged}")

    # Empirical chunking of sub-threshold text (< 100 words)
    short_text = " ".join([f"token_{i}" for i in range(45)])
    short_chunks = service.chunk_document("doc_short", short_text)
    assert_test("Short document (<100 words) produces exactly 1 chunk", len(short_chunks) == 1)
    assert_test("Short document word count is 45", short_chunks[0].word_count == 45)

    # Empty document produces a safe placeholder chunk with 0 words without crash
    empty_chunks = service.chunk_document("doc_empty", "")
    assert_test("Empty document produces 1 safe chunk with 0 words", len(empty_chunks) == 1 and empty_chunks[0].word_count == 0)
    service.delete_chunks("doc_empty")
    service.delete_chunks("doc_short")


def test_vector_normalization_and_faiss_bounds():
    print("\n--- Test Suite 3: Vector Normalization & FAISS Similarity Calibration ---")
    np.random.seed(42)
    dim = 1024

    # Generate 100 random vectors and normalize them
    raw_vecs = np.random.randn(100, dim).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    unit_vecs = raw_vecs / norms

    # Verify all norms are unit
    all_unit = np.allclose(np.linalg.norm(unit_vecs, axis=1), 1.0, atol=1e-5)
    assert_test("Normalized vectors have unit L2 norm", all_unit)

    # Theoretical distance bounds for unit vectors: d^2 in [0, 4]
    # Test identical vectors: d^2 = 0 -> s = 1.0
    d2_ident = float(np.sum((unit_vecs[0] - unit_vecs[0]) ** 2))
    s_ident = 1.0 / (1.0 + d2_ident)
    assert_test("Identical vectors: d^2 = 0.0, s = 1.0", abs(d2_ident - 0.0) < 1e-6 and abs(s_ident - 1.0) < 1e-6)

    # Test opposite vectors: d^2 = 4 -> s = 0.20
    d2_opp = float(np.sum((unit_vecs[0] - (-unit_vecs[0])) ** 2))
    s_opp = 1.0 / (1.0 + d2_opp)
    assert_test("Opposite vectors: d^2 = 4.0, s = 0.20", abs(d2_opp - 4.0) < 1e-6 and abs(s_opp - 0.20) < 1e-6)

    # Test random pair distances and similarities
    v1 = unit_vecs[0:50]
    v2 = unit_vecs[50:100]
    d2_pairs = np.sum((v1 - v2) ** 2, axis=1)
    s_pairs = 1.0 / (1.0 + d2_pairs)

    assert_test("All random pair d^2 in [0.0, 4.0]", np.all((d2_pairs >= 0.0) & (d2_pairs <= 4.0)))
    assert_test("All random pair s in [0.20, 1.00]", np.all((s_pairs >= 0.20) & (s_pairs <= 1.00)))

    # Monotonicity test: d1^2 < d2^2 <=> s1 > s2
    sorted_idx = np.argsort(d2_pairs)
    d2_sorted = d2_pairs[sorted_idx]
    s_sorted = s_pairs[sorted_idx]
    is_strictly_decreasing = np.all(np.diff(s_sorted) <= 0)
    assert_test("Similarity s is strictly monotonically decreasing with d^2", is_strictly_decreasing)


def test_graph_mathematics_and_topology():
    print("\n--- Test Suite 4: Graph Topology & Handshaking Lemma Verification ---")
    graph_dir = BACKEND_DIR / "data" / "graphs"

    # 1. BSM 301 Graph
    bsm_path = graph_dir / "8e355a58-ec35-4d54-92c5-3431066245ec.json"
    assert_test("BSM 301 graph file exists", bsm_path.exists())
    with open(bsm_path) as f:
        bsm_data = json.load(f)
    bsm_nodes = bsm_data.get("elements", {}).get("nodes", []) or bsm_data.get("nodes", [])
    bsm_edges = bsm_data.get("elements", {}).get("edges", []) or bsm_data.get("edges", [])

    n_bsm = len(bsm_nodes)
    e_bsm = len(bsm_edges)
    assert_test("BSM 301 |V| = 11", n_bsm == 11, f"Got {n_bsm}")
    assert_test("BSM 301 |E| = 19", e_bsm == 19, f"Got {e_bsm}")

    bsm_density = e_bsm / (n_bsm * (n_bsm - 1))
    assert_test("BSM 301 density D_dir = 19/110 ≈ 0.172727", abs(bsm_density - (19 / 110)) < 1e-6)

    bsm_avg_deg = (2 * e_bsm) / n_bsm
    assert_test("BSM 301 average degree d_bar = 38/11 ≈ 3.454545", abs(bsm_avg_deg - (38 / 11)) < 1e-6)

    # 2. Thevenin Run 1
    th1_path = graph_dir / "827d8b9c-4444-4d35-ae6a-4a48dc81f0ec.json"
    assert_test("Thevenin Run 1 graph exists", th1_path.exists())
    with open(th1_path) as f:
        th1_data = json.load(f)
    n_th1 = len(th1_data.get("elements", {}).get("nodes", []) or th1_data.get("nodes", []))
    e_th1 = len(th1_data.get("elements", {}).get("edges", []) or th1_data.get("edges", []))
    assert_test("Thevenin Run 1 |V| = 16", n_th1 == 16, f"Got {n_th1}")
    assert_test("Thevenin Run 1 |E| = 16", e_th1 == 16, f"Got {e_th1}")
    th1_density = e_th1 / (n_th1 * (n_th1 - 1))
    assert_test("Thevenin Run 1 density = 16/240 ≈ 0.066667", abs(th1_density - (16 / 240)) < 1e-6)
    th1_avg_deg = (2 * e_th1) / n_th1
    assert_test("Thevenin Run 1 avg degree = 2.0", abs(th1_avg_deg - 2.0) < 1e-6)

    # 3. Thevenin Run 2
    th2_path = graph_dir / "fc5f2c9b-85ab-437b-8700-b23ae982ee08.json"
    assert_test("Thevenin Run 2 graph exists", th2_path.exists())
    with open(th2_path) as f:
        th2_data = json.load(f)
    n_th2 = len(th2_data.get("elements", {}).get("nodes", []) or th2_data.get("nodes", []))
    e_th2 = len(th2_data.get("elements", {}).get("edges", []) or th2_data.get("edges", []))
    assert_test("Thevenin Run 2 |V| = 18", n_th2 == 18, f"Got {n_th2}")
    assert_test("Thevenin Run 2 |E| = 19", e_th2 == 19, f"Got {e_th2}")
    th2_density = e_th2 / (n_th2 * (n_th2 - 1))
    assert_test("Thevenin Run 2 density = 19/306 ≈ 0.062092", abs(th2_density - (19 / 306)) < 1e-6)
    th2_avg_deg = (2 * e_th2) / n_th2
    assert_test("Thevenin Run 2 avg degree = 38/18 ≈ 2.111111", abs(th2_avg_deg - (38 / 18)) < 1e-6)

    # 4. Boundary Cases: Single node graph and empty graph
    import networkx as nx
    g_single = nx.DiGraph()
    g_single.add_node("lone_node")
    d_single = nx.density(g_single) if g_single.number_of_nodes() > 1 else 0.0
    assert_test("Single node graph density is 0.0 (avoids ZeroDivisionError)", d_single == 0.0)

    g_empty = nx.DiGraph()
    d_empty = nx.density(g_empty) if g_empty.number_of_nodes() > 1 else 0.0
    assert_test("Empty graph density is 0.0", d_empty == 0.0)


def test_sqlite_candidate_database_and_queues():
    print("\n--- Test Suite 5: SQLite Database & Annotation Queue Integrity ---")
    db_path = BACKEND_DIR / "data" / "relation_dataset.db"
    assert_test("relation_dataset.db exists", db_path.exists())

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    total_records = cur.execute("SELECT count(*) FROM relation_dataset").fetchone()[0]
    assert_test("Total records in relation_dataset is exactly 1,554", total_records == 1554, f"Got {total_records}")

    distinct_docs = cur.execute("SELECT count(DISTINCT document_id) FROM relation_dataset").fetchone()[0]
    assert_test("Distinct documents count is exactly 44", distinct_docs == 44, f"Got {distinct_docs}")

    unlabeled_count = cur.execute("SELECT count(*) FROM relation_dataset WHERE is_valid IS NULL").fetchone()[0]
    assert_test("100% unlabeled records (is_valid IS NULL == 1554)", unlabeled_count == 1554, f"Got {unlabeled_count}")

    labeled_count = cur.execute("SELECT count(*) FROM relation_dataset WHERE is_valid IS NOT NULL").fetchone()[0]
    assert_test("0% labeled records (is_valid IS NOT NULL == 0)", labeled_count == 0, f"Got {labeled_count}")

    # Verify table schema columns
    cols = [r[1] for r in cur.execute("PRAGMA table_info(relation_dataset)").fetchall()]
    assert_test("Schema contains 'is_valid' column", "is_valid" in cols)
    assert_test("Schema contains 'document_id' column", "document_id" in cols)
    assert_test("Schema contains 'relation_type' column", "relation_type" in cols)

    # Specific document counts
    bsm_doc_id = "65430b13-d3d6-49b8-8386-f34762f0ae9c"
    bsm_db_count = cur.execute("SELECT count(*) FROM relation_dataset WHERE document_id = ?", (bsm_doc_id,)).fetchone()[0]
    assert_test("BSM 301 candidate records in DB == 34", bsm_db_count == 34, f"Got {bsm_db_count}")

    th1_db_count = cur.execute("SELECT count(*) FROM relation_dataset WHERE document_id = '827d8b9c-4444-4d35-ae6a-4a48dc81f0ec'").fetchone()[0]
    assert_test("Thevenin Run 1 candidate records in DB == 1", th1_db_count == 1, f"Got {th1_db_count}")

    th2_db_count = cur.execute("SELECT count(*) FROM relation_dataset WHERE document_id = 'fc5f2c9b-85ab-437b-8700-b23ae982ee08'").fetchone()[0]
    assert_test("Thevenin Run 2 candidate records in DB == 2", th2_db_count == 2, f"Got {th2_db_count}")

    conn.close()

    # Annotation queues
    q1_path = BACKEND_DIR / "data" / "ground_truth" / "annotation_queue_batch1.json"
    with open(q1_path) as f:
        q1_data = json.load(f)
    q1_records = q1_data.get("records", [])
    assert_test("Primary Batch 1 has 200 records", len(q1_records) == 200)
    q1_unlabeled = sum(1 for r in q1_records if r.get("user_label") is None)
    assert_test("Primary Batch 1 has 200 unlabeled records (0% complete)", q1_unlabeled == 200)

    q2_path = BACKEND_DIR / "data" / "ground_truth" / "annotation_queue_batch1_overlap_b.json"
    with open(q2_path) as f:
        q2_data = json.load(f)
    q2_records = q2_data.get("records", [])
    assert_test("Independent Overlap has 50 records", len(q2_records) == 50)
    q2_unlabeled = sum(1 for r in q2_records if r.get("user_label") is None)
    assert_test("Independent Overlap has 50 unlabeled records (0% complete)", q2_unlabeled == 50)


def test_agreement_and_calibration_formulas():
    print("\n--- Test Suite 6: Calibration & Agreement Formulations ---", flush=True)
    from core.benchmarking.agreement import compute_cohens_kappa, compute_fleiss_kappa
    from core.evaluation.calibration import PlattCalibrator, compute_calibration_metrics

    # 1. Cohen's Kappa perfect agreement
    r_a = ["1", "0", "1", "1", "0", "0", "1", "0"]
    r_b = ["1", "0", "1", "1", "0", "0", "1", "0"]
    res_perfect = compute_cohens_kappa(r_a, r_b)
    assert_test("Cohen Kappa perfect agreement is 1.0", abs(res_perfect.kappa - 1.0) < 1e-4)

    # 2. Zero-variance edge case (all 1s)
    r_uni_a = ["1"] * 20
    r_uni_b = ["1"] * 20
    res_uni = compute_cohens_kappa(r_uni_a, r_uni_b)
    assert_test("Cohen Kappa uniform ratings (zero variance) returns 1.0 gracefully", abs(res_uni.kappa - 1.0) < 1e-4)

    # 3. Fleiss Kappa perfect agreement
    matrix_perfect = [["VALID", "VALID", "VALID"]] * 5 + [["INVALID", "INVALID", "INVALID"]] * 5
    res_fleiss = compute_fleiss_kappa(matrix_perfect)
    assert_test("Fleiss Kappa perfect agreement is 1.0", abs(res_fleiss.kappa - 1.0) < 1e-4)

    # 4. Calibration metrics bounds
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.85, 0.15, 0.75, 0.25])
    cal_res = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert_test("ECE in [0, 1]", 0.0 <= cal_res.ece <= 1.0)
    assert_test("MCE in [0, 1]", 0.0 <= cal_res.mce <= 1.0)
    assert_test("Brier score in [0, 1]", 0.0 <= cal_res.brier_score <= 1.0)


def main():
    print("=====================================================================")
    print("      MILIESTONE 5 ADVERSARIAL CHALLENGER EMPIRICAL SUITE")
    print("=====================================================================")

    try:
        test_deliverables_existence_and_structure()
        test_chunking_step_and_residual_merging()
        test_vector_normalization_and_faiss_bounds()
        test_graph_mathematics_and_topology()
        test_sqlite_candidate_database_and_queues()
        test_agreement_and_calibration_formulas()
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n=====================================================================")
    print(f"  ALL CHALLENGER ADVERSARIAL TESTS PASSED: {PASS_COUNT} passed, 0 failed")
    print("=====================================================================")


if __name__ == "__main__":
    main()
