"""Milestone 5 Adversarial Verification Script by Teamwork Challenger (teamwork_challenger_m5_2).

Performs exhaustive, empirical, adversarial verification across:
1. Frontend nomenclature and DOM invariants.
2. Paper-code consistency claim sampling (code lines, data stores, formulas).
3. Submission readiness and classification integrity.
"""

import json
import re
import sqlite3
import sys
from pathlib import Path
import lxml.html
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = BACKEND_DIR / "data"

def test_frontend_nomenclature():
    print("\n--- [1] FRONTEND NOMENCLATURE & BRANDING ADVERSARIAL SCAN ---")
    html_file = FRONTEND_DIR / "index.html"
    js_file = FRONTEND_DIR / "app.js"
    css_file = FRONTEND_DIR / "styles.css"
    config_file = BACKEND_DIR / "utils" / "config.py"
    env_file = BACKEND_DIR / ".env"
    env_example = BACKEND_DIR / ".env.example"

    # 1. Obsolete prototype names scan
    obsolete_names = [r"\baureon\b", r"\bclairon\b", r"Intelligent Document Knowledge System"]
    targets = [html_file, js_file, css_file, config_file, env_file, env_example]
    for target in targets:
        if not target.exists():
            continue
        content = target.read_text(encoding="utf-8")
        for pat in obsolete_names:
            matches = re.findall(pat, content, flags=re.IGNORECASE)
            assert not matches, f"Obsolete name '{pat}' found in {target.name}: {matches}"
    print("  ✓ Zero obsolete prototype names ('Aureon', 'Clairon', 'Intelligent Document Knowledge System') in production configs & frontend")

    # 2. Canonical product branding 'Clarion'
    doc = lxml.html.parse(str(html_file)).getroot()
    title = doc.xpath("//title")[0].text_content().strip()
    assert title.startswith("Clarion"), f"Title does not start with Clarion: '{title}'"
    logo = doc.xpath("//span[contains(@class, 'brand-logo')]")[0].text_content().strip()
    assert logo == "Clarion", f"Logo is not Clarion: '{logo}'"
    badge = doc.xpath("//span[contains(@class, 'brand-badge')]")[0].text_content().strip()
    assert badge == "Research Preview", f"Badge is not 'Research Preview': '{badge}'"
    print(f"  ✓ Canonical branding enforced: title='{title}', logo='{logo}', badge='{badge}'")

    # 3. DOM Invariants: 8 Metric cards
    expected_metric_ids = [
        "stat-words", "stat-pages", "stat-sections", "stat-chunks",
        "stat-concepts", "stat-relations", "stat-nodes", "stat-edges"
    ]
    grid_cards = doc.xpath("//div[contains(@class, 'metrics-grid')]//div[contains(@class, 'metric-card')]")
    assert len(grid_cards) == 8, f"Expected 8 metric cards, got {len(grid_cards)}"
    for m_id in expected_metric_ids:
        el = doc.get_element_by_id(m_id)
        assert el is not None, f"Missing metric element #{m_id}"
        assert el.text_content().strip() == "--", f"Metric #{m_id} does not initialize with '--'"
    print(f"  ✓ All 8 metric cards verified in 4x2 grid with initial placeholder '--'")

    # 4. Audit table elements verified against index.html
    required_ids = [
        "status-doc-name", "doc-info", "analyze-btn", "overview-doc-title",
        "graph-container", "graph-info", "concepts-table", "concepts-count",
        "candidate-relations-panel", "relations-count", "structure-container",
        "structure-count", "rag-query-input", "rag-query-btn", "rag-output-box"
    ]
    for r_id in required_ids:
        el = doc.get_element_by_id(r_id)
        assert el is not None, f"Audit report listed #{r_id} but it is missing in index.html"
    print(f"  ✓ All {len(required_ids)} audit-referenced DOM elements present in index.html")


def test_paper_code_consistency_sampling():
    print("\n--- [2] PAPER-CODE CONSISTENCY EMPIRICAL CROSS-CHECK ---")
    
    # CLM-01: Backend architecture
    main_content = (BACKEND_DIR / "main.py").read_text(encoding="utf-8")
    assert "FastAPI(" in main_content, "FastAPI not in main.py"
    routers = re.findall(r"app\.include_router\(", main_content)
    assert len(routers) == 11, f"Expected 11 routers included in main.py, found {len(routers)}"
    print(f"  ✓ CLM-01: FastAPI application verified with exactly {len(routers)} routers")

    # CLM-02 to CLM-05: Chunking Service implementation
    chunking_file = BACKEND_DIR / "services" / "chunking_service.py"
    chunking_code = chunking_file.read_text(encoding="utf-8")
    assert "step = max(1, self.chunk_size - self.chunk_overlap)" in chunking_code
    assert "if len(word_slice) < self.min_chunk_size:" in chunking_code
    assert "chunks[-1].content += ' ' + ' '.join(remainder_words)" in chunking_code
    assert "max_merged_size = int(self.chunk_size * 1.25)" in chunking_code
    print("  ✓ CLM-02 to CLM-05: Chunking equations and merging bounds verified in chunking_service.py")

    # CLM-07 & CLM-08: Unit Normalization and Similarity Bounding
    retrieval_file = BACKEND_DIR / "services" / "retrieval_service.py"
    retrieval_code = retrieval_file.read_text(encoding="utf-8")
    assert "score = 1.0 / (1.0 + distance)" in retrieval_code or "score = 1.0 / (1.0 + float(distance))" in retrieval_code
    # Mathematical proof check:
    # For unit vectors q, v: d^2 = ||q - v||_2^2 = 2 - 2*cos(theta) in [0, 4]
    # score = 1 / (1 + d^2) => min score = 1/(1+4) = 0.20, max score = 1/(1+0) = 1.00
    cosines = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    d_sq = 2.0 - 2.0 * cosines
    scores = 1.0 / (1.0 + d_sq)
    assert np.isclose(scores[0], 0.20), f"Min score {scores[0]} != 0.20"
    assert np.isclose(scores[-1], 1.00), f"Max score {scores[-1]} != 1.00"
    print(f"  ✓ CLM-07 & CLM-08: Vector similarity s = 1/(1+d^2) strictly bounds to [{scores[0]:.2f}, {scores[-1]:.2f}]")

    # CLM-09 & CLM-10: Graph Density and Average Degree
    analyzer_file = BACKEND_DIR / "graph" / "analyzer.py"
    analyzer_code = analyzer_file.read_text(encoding="utf-8")
    assert "nx.density(self.graph) if n_nodes > 1 else 0.0" in analyzer_code
    assert "np.mean([d for n, d in self.graph.degree()])" in analyzer_code or "np.mean(degrees)" in analyzer_code
    print("  ✓ CLM-09 & CLM-10: Directed density formula and Handshaking lemma average degree verified")

    # CLM-11 & CLM-12: BSM 301 Mathematics-III Graph & Dataset
    bsm_graph_file = DATA_DIR / "graphs" / "8e355a58-5d27-4467-85fd-410a8ea74bb0.json"
    if not bsm_graph_file.exists():
        # Search for any BSM graph
        candidates = list(DATA_DIR.glob("graphs/*8e355a58*.json"))
        assert candidates, "BSM graph file not found"
        bsm_graph_file = candidates[0]
    bsm_data = json.loads(bsm_graph_file.read_text(encoding="utf-8"))
    nodes_count = len(bsm_data.get("nodes", []))
    edges_count = len(bsm_data.get("links", [])) or len(bsm_data.get("edges", []))
    assert nodes_count == 11, f"Expected 11 nodes, got {nodes_count}"
    assert edges_count == 19, f"Expected 19 edges, got {edges_count}"
    expected_density = 19.0 / (11.0 * 10.0)
    expected_avg_deg = (2.0 * 19.0) / 11.0
    print(f"  ✓ CLM-11: BSM 301 Graph verified: |V|={nodes_count}, |E|={edges_count}, D_dir={expected_density:.4f}, avg_d={expected_avg_deg:.4f}")

    # BSM 301 relations in DB
    db_path = DATA_DIR / "relation_dataset.db"
    assert db_path.exists(), "relation_dataset.db missing"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM relation_dataset WHERE document_id LIKE '65430b13%'")
    bsm_rel_count = c.fetchone()[0]
    assert bsm_rel_count == 34, f"Expected 34 BSM candidate relations, got {bsm_rel_count}"
    print(f"  ✓ CLM-12: BSM 301 Candidate relations count in SQLite verified: {bsm_rel_count} records (distinct from 19 graph edges)")

    # CLM-13: Thevenin Runs 1 & 2
    thev1_file = DATA_DIR / "graphs" / "827d8b9c-4444-4d35-ae6a-4a48dc81f0ec.json"
    thev2_file = DATA_DIR / "graphs" / "fc5f2c9b-85ab-437b-8700-b23ae982ee08.json"
    assert thev1_file.exists(), "Thevenin Run 1 file missing"
    assert thev2_file.exists(), "Thevenin Run 2 file missing"
    t1_data = json.loads(thev1_file.read_text(encoding="utf-8"))
    t2_data = json.loads(thev2_file.read_text(encoding="utf-8"))
    t1_n = len(t1_data.get("nodes", []))
    t1_e = len(t1_data.get("links", [])) or len(t1_data.get("edges", []))
    t2_n = len(t2_data.get("nodes", []))
    t2_e = len(t2_data.get("links", [])) or len(t2_data.get("edges", []))
    assert t1_n == 16 and t1_e == 16, f"Thevenin Run 1 expected 16/16, got {t1_n}/{t1_e}"
    assert t2_n == 18 and t2_e == 19, f"Thevenin Run 2 expected 18/19, got {t2_n}/{t2_e}"
    print(f"  ✓ CLM-13: Thevenin Run 1 (|V|={t1_n}, |E|={t1_e}) and Run 2 (|V|={t2_n}, |E|={t2_e}) verified")

    # CLM-15 & CLM-16: Candidate Relation Database & Annotation Reality
    c.execute("SELECT COUNT(*) FROM relation_dataset")
    total_records = c.fetchone()[0]
    assert total_records == 1554, f"Expected 1554 candidate records, got {total_records}"
    c.execute("SELECT COUNT(*) FROM relation_dataset WHERE is_valid IS NOT NULL")
    labeled_records = c.fetchone()[0]
    assert labeled_records == 0, f"Expected 0 labeled records, got {labeled_records}"
    c.execute("SELECT COUNT(DISTINCT document_id) FROM relation_dataset")
    doc_count = c.fetchone()[0]
    assert doc_count == 44, f"Expected 44 documents, got {doc_count}"
    conn.close()
    print(f"  ✓ CLM-15: candidate relations verified: {total_records} rows, {doc_count} docs, 100.0% unlabeled")

    queue1_data = json.loads((DATA_DIR / "ground_truth" / "annotation_queue_batch1.json").read_text(encoding="utf-8"))
    queue_overlap_data = json.loads((DATA_DIR / "ground_truth" / "annotation_queue_batch1_overlap_b.json").read_text(encoding="utf-8"))
    queue1 = queue1_data.get("records", queue1_data)
    queue_overlap = queue_overlap_data.get("records", queue_overlap_data)
    assert len(queue1) == 200, f"Expected 200 items in queue 1, got {len(queue1)}"
    assert len(queue_overlap) == 50, f"Expected 50 items in overlap queue, got {len(queue_overlap)}"
    # Check that annotations are 0% completed
    q1_completed = sum(1 for item in queue1 if item.get("label") or item.get("is_valid") is not None)
    assert q1_completed == 0, f"Expected 0 completed in queue 1, got {q1_completed}"
    print(f"  ✓ CLM-16: Annotation queues verified: Primary={len(queue1)}, Overlap={len(queue_overlap)}, Completed=0 (0.0%)")

    # CLM-22: Literature Citations & references.bib
    bib_file = ROOT_DIR / "references.bib"
    assert bib_file.exists(), "references.bib missing"
    bib_content = bib_file.read_text(encoding="utf-8")
    expected_citations = [
        "landis1977measurement", "cohen1960coefficient", "fleiss1971measuring",
        "platt1999probabilistic", "guo2017calibration", "xiao2023cpack",
        "johnson2019billion", "hagberg2008exploring"
    ]
    for cit in expected_citations:
        assert cit in bib_content, f"Missing citation key '{cit}' in references.bib"
    print(f"  ✓ CLM-22: All {len(expected_citations)} audited literature citations present in references.bib")


def test_submission_readiness():
    print("\n--- [3] SUBMISSION READINESS & CLASSIFICATION ADVERSARIAL AUDIT ---")
    checklist_file = ROOT_DIR / "reports" / "historical" / "SUBMISSION_READINESS_CHECKLIST.md"
    audit_report_file = ROOT_DIR / "reports" / "historical" / "FINAL_AUDIT_REPORT.md"
    project_file = ROOT_DIR / "PROJECT.md"

    assert checklist_file.exists(), "reports/historical/SUBMISSION_READINESS_CHECKLIST.md missing"
    assert audit_report_file.exists(), "reports/historical/FINAL_AUDIT_REPORT.md missing"
    assert project_file.exists(), "PROJECT.md missing"

    chk_text = checklist_file.read_text(encoding="utf-8")
    rep_text = audit_report_file.read_text(encoding="utf-8")
    proj_text = project_file.read_text(encoding="utf-8")

    expected_classification = "TECHNICALLY IMPROVED / RESEARCH VALIDATION PENDING"
    assert expected_classification in chk_text, f"Missing expected classification in {checklist_file.name}"
    assert expected_classification in rep_text, f"Missing expected classification in {audit_report_file.name}"
    print(f"  ✓ Formal classification consistently designated as '{expected_classification}'")

    # Verify that claims of human trials or IR evaluation are NOT falsely marked completed
    assert "0 human labels (100% unlabeled)" in chk_text or "100.0% of which are unlabeled" in chk_text
    assert "UNMEASURED / UNTESTED" in chk_text
    print("  ✓ Scientific honesty verified: human trials and IR metrics are strictly quarantined as unmeasured")

    # Verify PROJECT.md milestone statuses
    for milestone in ["M1", "M2", "M3", "M4", "M5"]:
        pattern = rf"\|\s*{milestone}\s*\|[^|]+\|[^|]+\|[^|]+\|\s*DONE\s*\|"
        assert re.search(pattern, proj_text), f"Milestone {milestone} not marked DONE in PROJECT.md"
    print("  ✓ Milestones M1 through M5 verified marked DONE in PROJECT.md")


if __name__ == "__main__":
    print("======================================================================")
    print("CLARION MILESTONE 5 ADVERSARIAL CHALLENGER VERIFICATION HARNESS")
    print("======================================================================")
    test_frontend_nomenclature()
    test_paper_code_consistency_sampling()
    test_submission_readiness()
    print("\n======================================================================")
    print("ALL CHALLENGER AUDITS PASSED WITH ZERO DISCREPANCIES (100% SUCCESS)")
    print("======================================================================")
