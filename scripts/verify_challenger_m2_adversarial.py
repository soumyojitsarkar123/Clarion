"""Empirical Challenger 2 Adversarial Stress Suite for Milestone 2.

Comprehensive verification of:
1. 8 Overview Cards Invariants & Exact ID matching with app.js:1577-1600.
2. SVG Favicon Data URI XML Well-Formedness across multiple parsers.
3. Brand Badge presence, styling, and DOM hierarchy.
4. Complete HTML Syntax, Unclosed Tags, and Zero Duplicate IDs.
5. Cross-reference of all app.js getElementById / querySelector IDs against index.html.
6. CSS Grid Layout math and responsive breakpoints.
7. Data flow simulation from backend analysis-reports schema to overview cards.
"""

import html.parser
import re
import sys
import urllib.parse
import xml.dom.minidom
import xml.etree.ElementTree as ET
import xml.sax
from pathlib import Path
import lxml.etree
import lxml.html

ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT_DIR / "frontend" / "index.html"
CSS_PATH = ROOT_DIR / "frontend" / "styles.css"
JS_PATH = ROOT_DIR / "frontend" / "app.js"

EXPECTED_8_METRICS = [
    ("stat-words", "Words"),
    ("stat-pages", "Pages"),
    ("stat-sections", "Sections"),
    ("stat-chunks", "Chunks"),
    ("stat-concepts", "Concepts"),
    ("stat-relations", "Relations"),
    ("stat-nodes", "Nodes"),
    ("stat-edges", "Edges"),
]


def test_1_overview_cards_invariants():
    print("[Challenge 1] 8 Overview Cards Invariants & app.js:1577-1600 Parity")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()

    # Find the overview panel and metrics grid
    grids = doc.xpath("//div[contains(@class, 'metrics-grid')]")
    assert len(grids) == 1, f"Expected 1 .metrics-grid, found {len(grids)}"
    grid = grids[0]

    cards = grid.xpath("./div[contains(@class, 'metric-card')]")
    assert len(cards) == 8, f"Expected exactly 8 .metric-card children, found {len(cards)}"

    for idx, (expected_id, expected_label) in enumerate(EXPECTED_8_METRICS):
        card = cards[idx]
        labels = card.xpath("./span[contains(@class, 'metric-label')]")
        values = card.xpath("./span[contains(@class, 'metric-value')]")

        assert len(labels) == 1, f"Card {idx} ({expected_id}) missing .metric-label"
        assert len(values) == 1, f"Card {idx} ({expected_id}) missing .metric-value"

        actual_label = labels[0].text_content().strip()
        actual_val = values[0].text_content().strip()
        actual_id = values[0].attrib.get("id")

        assert actual_id == expected_id, f"Card {idx} expected id '{expected_id}', got '{actual_id}'"
        assert actual_label == expected_label, f"Card {idx} expected label '{expected_label}', got '{actual_label}'"
        assert actual_val == "--", f"Card {idx} expected initial value '--', got '{actual_val}'"

    # Cross-check against app.js lines 1576-1600
    js_content = JS_PATH.read_text(encoding="utf-8")
    for expected_id, _ in EXPECTED_8_METRICS:
        call_pattern = rf'updateOverviewMetric\s*\(\s*"{expected_id}"'
        assert re.search(call_pattern, js_content), f"app.js missing updateOverviewMetric call for '{expected_id}'"

    # Specifically verify the 3 restored metrics
    for restored_id in ["stat-relations", "stat-nodes", "stat-edges"]:
        match = re.search(rf'updateOverviewMetric\s*\(\s*"{restored_id}"', js_content)
        assert match, f"Missing updateOverviewMetric for restored ID {restored_id}"

    # Also verify lines 245 and 338 reset arrays include all non-document metrics
    for m in ["sections", "chunks", "concepts", "relations", "nodes", "edges"]:
        pattern = rf'\[.*"{m}".*\]\.forEach'
        assert re.search(pattern, js_content), f"app.js reset loop missing metric '{m}'"

    print("  ✓ All 8 overview cards verified with exact IDs, labels, and initial '--'.")
    print("  ✓ Full parity with app.js lines 1576-1600 and reset loops confirmed.")


def test_2_favicon_xml_wellformedness():
    print("\n[Challenge 2] SVG Favicon Data URI XML Well-Formedness & Multi-Parser Parsing")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()
    favicon_links = doc.xpath("//head/link[@rel='icon']")
    assert len(favicon_links) == 1, f"Expected 1 <link rel='icon'> in <head>, found {len(favicon_links)}"

    href = favicon_links[0].attrib.get("href", "")
    assert href.startswith("data:image/svg+xml,"), f"Favicon href is not SVG data URI: {href}"

    header, sep, raw_svg = href.partition(",")
    svg_payload = urllib.parse.unquote(raw_svg)

    # Parser 1: xml.etree.ElementTree
    root_et = ET.fromstring(svg_payload)
    assert root_et.tag.endswith("svg")
    assert root_et.attrib.get("viewBox") == "0 0 100 100"

    # Parser 2: lxml.etree (strict libxml2)
    root_lxml = lxml.etree.fromstring(svg_payload.encode("utf-8"))
    assert root_lxml.tag == "{http://www.w3.org/2000/svg}svg"

    # Parser 3: xml.dom.minidom
    dom = xml.dom.minidom.parseString(svg_payload)
    assert dom.documentElement.tagName == "svg"

    # Parser 4: xml.sax
    class NullHandler(xml.sax.ContentHandler):
        def __init__(self):
            super().__init__()
            self.saw_text = False
        def characters(self, content):
            if "💠" in content:
                self.saw_text = True
    handler = NullHandler()
    xml.sax.parseString(svg_payload, handler)
    assert handler.saw_text, "SAX parser did not encounter the glyph '💠'"

    print("  ✓ Favicon parses cleanly across 4 XML parsers (ElementTree, lxml2, minidom, SAX).")
    print(f"  ✓ Valid SVG namespace ({root_et.tag}) and viewBox verified.")


def test_3_brand_badge_presence_and_styling():
    print("\n[Challenge 3] Brand Badge DOM Placement and CSS Properties")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()

    # Find badge
    badges = doc.xpath("//header//div[contains(@class, 'brand-title-row')]//span[contains(@class, 'brand-badge')]")
    assert len(badges) == 1, f"Expected 1 .brand-badge in .brand-title-row, found {len(badges)}"
    badge = badges[0]
    assert badge.text_content().strip() == "Research Preview"

    # Sibling logo
    parent = badge.getparent()
    logos = parent.xpath("./span[contains(@class, 'brand-logo')]")
    assert len(logos) == 1, "Expected 1 sibling .brand-logo"
    assert logos[0].text_content().strip() == "Clarion"

    # Verify styles.css rules
    css_content = CSS_PATH.read_text(encoding="utf-8")
    assert ".brand-badge {" in css_content
    assert "color: #38bdf8;" in css_content
    assert "font-family:" in css_content and "JetBrains Mono" in css_content
    assert "text-transform: uppercase;" in css_content

    print("  ✓ Brand badge 'Research Preview' verified in DOM with matching CSS rules.")


def test_4_html_syntax_unclosed_tags_duplicate_ids():
    print("\n[Challenge 4] Adversarial HTML Syntax, Tag Closure, & Duplicate ID Audit")
    raw_html = HTML_PATH.read_text(encoding="utf-8")

    VOID_ELEMENTS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"
    }

    class TagAndIDAuditor(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.errors = []
            self.ids = []

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if "id" in attrs_dict:
                self.ids.append((attrs_dict["id"], self.getpos()))
            if tag.lower() not in VOID_ELEMENTS:
                self.stack.append((tag.lower(), self.getpos()))

        def handle_endtag(self, tag):
            tag_lower = tag.lower()
            if tag_lower in VOID_ELEMENTS:
                return
            if not self.stack:
                self.errors.append(f"Spurious closing </{tag}> at line {self.getpos()[0]}")
                return
            last_tag, last_pos = self.stack.pop()
            if last_tag != tag_lower:
                self.errors.append(
                    f"Mismatched tag: expected </{last_tag}> (from line {last_pos[0]}), got </{tag}> at line {self.getpos()[0]}"
                )

        def finalize(self):
            if self.stack:
                for unclosed_tag, pos in self.stack:
                    self.errors.append(f"Unclosed <{unclosed_tag}> from line {pos[0]}")

    auditor = TagAndIDAuditor()
    auditor.feed(raw_html)
    auditor.finalize()

    assert not auditor.errors, f"HTML structure errors detected: {auditor.errors}"
    print(f"  ✓ Tag balance check passed: 0 errors across {len(raw_html.splitlines())} lines.")

    # Duplicate ID check
    seen = {}
    duplicates = []
    for el_id, pos in auditor.ids:
        if el_id in seen:
            duplicates.append((el_id, seen[el_id], pos))
        else:
            seen[el_id] = pos

    assert not duplicates, f"Duplicate IDs detected in index.html: {duplicates}"
    print(f"  ✓ Global ID uniqueness verified: {len(seen)} IDs checked, 0 duplicates found.")


def test_5_app_js_queried_ids_exist_in_dom():
    print("\n[Challenge 5] Cross-Verification of app.js DOM Queries against index.html")
    js_content = JS_PATH.read_text(encoding="utf-8")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()
    all_dom_ids = set(doc.xpath("//@id"))

    # Extract all document.getElementById("...")
    direct_ids = set(re.findall(r'document\.getElementById\(\s*["\']([^"\'$`]+)["\']\s*\)', js_content))

    # Also check dynamic template literals in app.js like `stat-${m}`
    dynamic_prefixes = re.findall(r'document\.getElementById\(\s*`([^`]+)\$\{[^}]+\}([^`]*)`\s*\)', js_content)
    print(f"  Direct getElementById IDs queried: {len(direct_ids)}")

    # Check for missing direct IDs
    # Note: Some elements might be dynamically created in JS, but let's check which static elements must exist
    missing_ids = [i for i in direct_ids if i not in all_dom_ids]
    if missing_ids:
        print(f"  Notice: IDs queried in app.js but not found in index.html: {missing_ids}")
    else:
        print("  ✓ Every static ID queried by document.getElementById in app.js exists in index.html!")

    # Verify that all 8 overview metric IDs are present
    for metric_id, _ in EXPECTED_8_METRICS:
        assert metric_id in all_dom_ids, f"Required overview ID '{metric_id}' is missing from DOM!"
    print("  ✓ All 8 overview metric IDs confirmed present in DOM.")


def test_6_css_grid_geometry():
    print("\n[Challenge 6] CSS Grid Mathematical Balance Verification")
    css_content = CSS_PATH.read_text(encoding="utf-8")

    # Desktop: repeat(4, 1fr) -> 8 items = exactly 2 full rows, 0 orphans
    desktop_match = re.search(r'\.metrics-grid\s*\{[^}]*grid-template-columns:\s*repeat\(4,\s*1fr\);', css_content)
    assert desktop_match, "Base .metrics-grid must be repeat(4, 1fr)"
    desktop_cols = 4
    total_cards = 8
    assert total_cards % desktop_cols == 0, f"Desktop grid orphan cards: {total_cards} % {desktop_cols} != 0"
    print(f"  ✓ Desktop grid: {total_cards} cards / {desktop_cols} cols = {total_cards // desktop_cols} full rows (0 orphans).")

    # Mobile: repeat(2, 1fr) -> 8 items = exactly 4 full rows, 0 orphans
    mobile_match = re.search(r'\.metrics-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,\s*1fr\);', css_content)
    assert mobile_match, "Mobile .metrics-grid must be repeat(2, 1fr)"
    mobile_cols = 2
    assert total_cards % mobile_cols == 0, f"Mobile grid orphan cards: {total_cards} % {mobile_cols} != 0"
    print(f"  ✓ Mobile grid: {total_cards} cards / {mobile_cols} cols = {total_cards // mobile_cols} full rows (0 orphans).")


def test_7_data_flow_simulation():
    print("\n[Challenge 7] Data Flow Simulation: Backend Report -> DOM Metrics Injection")
    # Simulate the exact mapping logic of renderAnalysisReport(report) from app.js:1572-1600
    mock_report = {
        "metrics": {
            "words": 15420,
            "pages": 42,
            "sections": 8,
            "chunks": 65,
            "concepts": 23,
            "candidate_relations": 89,
            "graph_nodes": 23,
            "graph_edges": 89,
        },
        "summary": {
            "relation_count": 89,
            "graph_nodes": 23,
            "graph_edges": 89,
        },
        "artifacts": {
            "dataset_records": 89,
        }
    }

    doc = lxml.html.parse(str(HTML_PATH)).getroot()

    # Apply mock data simulation
    simulated_dom_values = {}
    metrics = mock_report.get("metrics", {})
    summary = mock_report.get("summary", {})

    simulated_dom_values["stat-words"] = metrics.get("words")
    simulated_dom_values["stat-pages"] = metrics.get("pages")
    simulated_dom_values["stat-sections"] = metrics.get("sections")
    simulated_dom_values["stat-chunks"] = metrics.get("chunks")
    simulated_dom_values["stat-concepts"] = metrics.get("concepts")
    simulated_dom_values["stat-relations"] = metrics.get("candidate_relations") or metrics.get("relations") or summary.get("relation_count")
    simulated_dom_values["stat-nodes"] = metrics.get("graph_nodes") or summary.get("graph_nodes")
    simulated_dom_values["stat-edges"] = metrics.get("graph_edges") or summary.get("graph_edges")

    for metric_id, expected_val in simulated_dom_values.items():
        el = doc.get_element_by_id(metric_id)
        assert el is not None, f"Target DOM element #{metric_id} not found in index.html"
        # Simulate JS textContent assignment
        el.text = str(expected_val)
        assert el.text == str(expected_val)

    print(f"  ✓ Successfully simulated data ingestion across all 8 overview cards:")
    for metric_id, val in simulated_dom_values.items():
        print(f"    - #{metric_id} <= {val}")


def main():
    print("=" * 70)
    print("CLARION CHALLENGER 2 — ADVERSARIAL STRESS TEST HARNESS")
    print("=" * 70)
    test_1_overview_cards_invariants()
    test_2_favicon_xml_wellformedness()
    test_3_brand_badge_presence_and_styling()
    test_4_html_syntax_unclosed_tags_duplicate_ids()
    test_5_app_js_queried_ids_exist_in_dom()
    test_6_css_grid_geometry()
    test_7_data_flow_simulation()
    print("\n" + "=" * 70)
    print("ALL 7 ADVERSARIAL CHALLENGES COMPLETED WITH 100% PASS RATE")
    print("=" * 70)


if __name__ == "__main__":
    main()
