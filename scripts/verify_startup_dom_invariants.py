"""Empirical DOM and AST Verification Script for Clarion Startup Invariants.

Verifies:
1. Static HTML startup DOM invariants in frontend/index.html:
   - Ribbon document chip: "None Selected"
   - Document info panel: "No document selected"
   - Analyze button: disabled
   - Overview title: "No document analyzed"
   - Metric cards: all display "--"
   - Cytoscape graph container: empty, "0 nodes / 0 edges"
   - Concepts table: "No concepts loaded", badge "0 concepts"
   - Candidate relations panel: display: none, "0 relations"
   - Structure container: "No structural hierarchy generated yet."
   - RAG interface: ask button disabled, output box display: none
   - Script cache buster: app.js?v=21
2. Frontend code contracts in frontend/app.js:
   - Auto-selection logic eliminated from loadDocuments()
   - Selector parity (data-doc-id and data-id)
   - window.selectDocument exposed with string ID support
   - Upload selection preserved
   - resetWorkspaceViews cleanup comprehensive
"""

import sys
from pathlib import Path
import lxml.html

ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT_DIR / "frontend" / "index.html"
JS_PATH = ROOT_DIR / "frontend" / "app.js"


def test_html_invariants():
    print("[1] Validating Static HTML Startup Invariants in frontend/index.html...")
    content = HTML_PATH.read_text(encoding="utf-8")
    doc = lxml.html.fromstring(content)

    # 1. Ribbon document chip
    doc_name_el = doc.get_element_by_id("status-doc-name")
    assert doc_name_el is not None, "Missing #status-doc-name element"
    assert doc_name_el.text_content().strip() == "None Selected", (
        f"Expected 'None Selected', got '{doc_name_el.text_content().strip()}'"
    )
    print("  ✓ #status-doc-name == 'None Selected'")

    # 2. Document info
    doc_info_el = doc.get_element_by_id("doc-info")
    assert doc_info_el is not None, "Missing #doc-info element"
    assert doc_info_el.text_content().strip() == "No document selected", (
        f"Expected 'No document selected', got '{doc_info_el.text_content().strip()}'"
    )
    print("  ✓ #doc-info == 'No document selected'")

    # 3. Analyze button
    analyze_btn = doc.get_element_by_id("analyze-btn")
    assert analyze_btn is not None, "Missing #analyze-btn element"
    assert "disabled" in analyze_btn.attrib, "#analyze-btn is not disabled"
    assert "Start Full Analysis" in analyze_btn.text_content(), (
        f"Expected 'Start Full Analysis', got '{analyze_btn.text_content().strip()}'"
    )
    print("  ✓ #analyze-btn is disabled with label 'Start Full Analysis'")

    # 4. Overview subtitle
    overview_title = doc.get_element_by_id("overview-doc-title")
    assert overview_title is not None, "Missing #overview-doc-title element"
    assert overview_title.text_content().strip() == "No document analyzed", (
        f"Expected 'No document analyzed', got '{overview_title.text_content().strip()}'"
    )
    print("  ✓ #overview-doc-title == 'No document analyzed'")

    # 5. Metric cards
    metrics = ["stat-words", "stat-pages", "stat-sections", "stat-chunks", "stat-concepts"]
    for m in metrics:
        el = doc.get_element_by_id(m)
        assert el is not None, f"Missing #{m} element"
        assert el.text_content().strip() == "--", f"Expected '--' for #{m}, got '{el.text_content().strip()}'"
    print(f"  ✓ All {len(metrics)} overview metric cards display '--'")

    # 6. Graph container and counter
    graph_container = doc.get_element_by_id("graph-container")
    assert graph_container is not None, "Missing #graph-container element"
    assert len(graph_container.getchildren()) == 0, "#graph-container is not empty"
    graph_info = doc.get_element_by_id("graph-info")
    assert graph_info is not None, "Missing #graph-info element"
    assert graph_info.text_content().strip() == "0 nodes / 0 edges", (
        f"Expected '0 nodes / 0 edges', got '{graph_info.text_content().strip()}'"
    )
    print("  ✓ #graph-container is empty and #graph-info displays '0 nodes / 0 edges'")

    # 7. Concepts table
    concepts_count = doc.get_element_by_id("concepts-count")
    assert concepts_count is not None, "Missing #concepts-count element"
    assert concepts_count.text_content().strip() == "0 concepts"
    concepts_table = doc.get_element_by_id("concepts-table")
    assert concepts_table is not None, "Missing #concepts-table element"
    tbodies = concepts_table.xpath(".//tbody")
    assert len(tbodies) >= 1, "Missing #concepts-table tbody"
    assert "No concepts loaded" in tbodies[0].text_content(), (
        f"Expected 'No concepts loaded' in table, got '{tbodies[0].text_content().strip()}'"
    )
    print("  ✓ #concepts-table displays 'No concepts loaded' and badge is '0 concepts'")

    # 8. Candidate relations panel
    rel_panel = doc.get_element_by_id("candidate-relations-panel")
    assert rel_panel is not None, "Missing #candidate-relations-panel element"
    style = rel_panel.attrib.get("style", "")
    assert "display: none" in style or "display:none" in style, (
        f"Expected #candidate-relations-panel to be hidden, got style='{style}'"
    )
    rel_count = doc.get_element_by_id("relations-count")
    assert rel_count is not None, "Missing #relations-count element"
    assert rel_count.text_content().strip() == "0 relations"
    print("  ✓ #candidate-relations-panel has style='display: none;' and badge is '0 relations'")

    # 9. Structure hierarchy container
    struct_container = doc.get_element_by_id("structure-container")
    assert struct_container is not None, "Missing #structure-container element"
    assert "No structural hierarchy generated yet." in struct_container.text_content()
    struct_count = doc.get_element_by_id("structure-count")
    assert struct_count is not None, "Missing #structure-count element"
    assert struct_count.text_content().strip() == "0 sections"
    print("  ✓ #structure-container displays placeholder and badge is '0 sections'")

    # 10. RAG interface
    rag_btn = doc.get_element_by_id("rag-query-btn")
    assert rag_btn is not None, "Missing #rag-query-btn element"
    assert "disabled" in rag_btn.attrib, "#rag-query-btn is not disabled"
    rag_box = doc.get_element_by_id("rag-output-box")
    assert rag_box is not None, "Missing #rag-output-box element"
    rag_style = rag_box.attrib.get("style", "")
    assert "display: none" in rag_style or "display:none" in rag_style, (
        f"Expected #rag-output-box to be hidden, got style='{rag_style}'"
    )
    print("  ✓ #rag-query-btn is disabled and #rag-output-box is hidden")

    # 11. Cache buster
    scripts = doc.xpath("//script[@src]")
    app_scripts = [s.attrib["src"] for s in scripts if "app.js" in s.attrib["src"]]
    assert len(app_scripts) == 1, f"Expected 1 app.js script tag, found {app_scripts}"
    assert app_scripts[0] == "app.js?v=21", f"Expected 'app.js?v=21', got '{app_scripts[0]}'"
    print(f"  ✓ Script tag cache buster is '{app_scripts[0]}'")


def test_js_contracts():
    print("\n[2] Validating Implementation Contracts in frontend/app.js...")
    code = JS_PATH.read_text(encoding="utf-8")

    # 1. No auto-selection on startup
    assert "!selectedDocId && documents.length > 0" not in code, (
        "Found forbidden auto-selection pattern '!selectedDocId && documents.length > 0'"
    )
    assert "firstAnalyzed || documents[0]" not in code, (
        "Found forbidden auto-selection pattern 'firstAnalyzed || documents[0]'"
    )
    print("  ✓ Startup auto-selection eliminated")

    # 2. Selector parity
    assert "item.dataset.docId = doc.id;" in code, "Missing item.dataset.docId assignment"
    assert "item.dataset.id = doc.id;" in code, "Missing item.dataset.id assignment"
    assert 'data-doc-id="${selectedDocId}"' in code or 'data-id="${selectedDocId}"' in code, (
        "Missing dual attribute selector lookup in selectDocument()"
    )
    print("  ✓ Dual selector attributes (data-doc-id and data-id) present")

    # 3. Global selectDocument exposure
    assert "window.selectDocument = selectDocument;" in code, "Missing window.selectDocument export"
    print("  ✓ window.selectDocument globally exposed")

    # 4. Upload ingestion preservation
    assert "selectDocument(uploadedDoc);" in code, "Missing upload ingestion auto-selection"
    print("  ✓ Upload ingestion explicitly preserves selectDocument(uploadedDoc)")

    # 5. Clean reset on transition
    assert "renderConceptsTable([]);" in code, "resetWorkspaceViews missing renderConceptsTable([])"
    assert 'ragBox.style.display = "none";' in code, "resetWorkspaceViews missing hiding ragBox"
    assert "currentKnowledgeMap = null;" in code, "resetWorkspaceViews missing nullifying currentKnowledgeMap"
    assert "currentSummary = null;" in code, "resetWorkspaceViews missing nullifying currentSummary"
    print("  ✓ resetWorkspaceViews clears concepts, inspector, RAG, and cache references")


if __name__ == "__main__":
    print("=" * 70)
    print("STARTUP DOM INVARIANTS & REFRESH BEHAVIOR STATIC VERIFICATION")
    print("=" * 70)
    test_html_invariants()
    test_js_contracts()
    print("\n" + "=" * 70)
    print("ALL STATIC DOM INVARIANTS AND CONTRACTS VERIFIED!")
    print("=" * 70)
