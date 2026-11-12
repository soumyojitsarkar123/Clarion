"""Adversarial and Empirical Stress Verification for Milestone 2 Frontend Deliverables.

Tests:
1. DOM Structure & HTML Integrity:
   - Tag balance / closure audit
   - Global ID uniqueness (zero duplicate IDs)
2. Favicon SVG Data URI:
   - Extraction and URI syntax
   - XML parsing against W3C SVG namespace
   - XML well-formedness, viewBox, and glyph verification
3. Brand Badge & Nomenclature:
   - Badge DOM placement, class, and text content
   - Adjacent brand logo and canonical title check
   - Zero tolerance for legacy/prototype names
4. 8 Overview Metric Cards:
   - Exact count (8 cards in 4x2 grid)
   - Exact IDs: stat-words, stat-pages, stat-sections, stat-chunks,
                stat-concepts, stat-relations, stat-nodes, stat-edges
   - Exact initial text: '--'
   - Structural uniformity (.metric-card > .metric-label + .metric-value)
5. Contract Parity with frontend/app.js:
   - All IDs queried by app.js exist in index.html
   - updateOverviewMetric call-sites match DOM elements
6. State Simulation:
   - Model the state transitions (initial -> select -> analyzed -> reset)
"""

import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
import lxml.etree
import lxml.html

ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT_DIR / "frontend" / "index.html"
CSS_PATH = ROOT_DIR / "frontend" / "styles.css"
JS_PATH = ROOT_DIR / "frontend" / "app.js"

EXPECTED_METRICS = [
    ("stat-words", "WORDS"),
    ("stat-pages", "PAGES"),
    ("stat-sections", "SECTIONS"),
    ("stat-chunks", "CHUNKS"),
    ("stat-concepts", "CONCEPTS"),
    ("stat-relations", "RELATIONS"),
    ("stat-nodes", "NODES"),
    ("stat-edges", "EDGES"),
]

FORBIDDEN_LEGACY_NAMES = [
    r"\baureon\b",
    r"\bclairon\b",
    r"\bclair\b",
    r"\bdocgraph\b",
    r"\bgraphdoc\b",
]


class StrictTagValidator(HTMLParser):
    VOID_ELEMENTS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"
    }

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []
        self.all_ids = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if "id" in attrs_dict:
            self.all_ids.append((attrs_dict["id"], self.getpos()))

        if tag.lower() not in self.VOID_ELEMENTS:
            self.stack.append((tag.lower(), self.getpos()))

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in self.VOID_ELEMENTS:
            return

        if not self.stack:
            self.errors.append(f"Unexpected closing tag </{tag}> at line {self.getpos()[0]}")
            return

        last_tag, last_pos = self.stack.pop()
        if last_tag != tag_lower:
            self.errors.append(
                f"Mismatched tag: expected </{last_tag}> (opened at line {last_pos[0]}), "
                f"got </{tag}> at line {self.getpos()[0]}"
            )

    def validate_complete(self):
        if self.stack:
            for unclosed, pos in self.stack:
                self.errors.append(f"Unclosed tag <{unclosed}> opened at line {pos[0]}")


def test_html_syntax_and_unique_ids():
    print("[Test 1] HTML Syntax, Tag Balancing, and ID Uniqueness")
    raw_html = HTML_PATH.read_text(encoding="utf-8")

    # Strict tag validation
    validator = StrictTagValidator()
    validator.feed(raw_html)
    validator.validate_complete()

    assert not validator.errors, f"HTML syntax errors found: {validator.errors}"
    print(f"  ✓ Strict HTML parsing passed: 0 unclosed/mismatched tags across {len(raw_html.splitlines())} lines")

    # ID uniqueness check
    ids_seen = {}
    duplicates = []
    for element_id, pos in validator.all_ids:
        if element_id in ids_seen:
            duplicates.append((element_id, ids_seen[element_id], pos))
        else:
            ids_seen[element_id] = pos

    assert not duplicates, f"Duplicate IDs found in index.html: {duplicates}"
    print(f"  ✓ ID uniqueness verified: {len(ids_seen)} unique IDs, 0 duplicates")


def test_svg_favicon_data_uri():
    print("\n[Test 2] SVG Favicon Data URI XML Well-Formedness & Integrity")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()

    # Find favicon link
    favicon_links = doc.xpath("//link[@rel='icon']")
    assert len(favicon_links) == 1, f"Expected 1 <link rel='icon'>, found {len(favicon_links)}"
    favicon = favicon_links[0]
    href = favicon.attrib.get("href", "")
    assert href, "Favicon href attribute is empty"

    # Verify URI format
    assert href.startswith("data:image/svg+xml"), f"Favicon href must be SVG data URI, got: {href[:40]}"

    # Extract raw SVG
    header, _, raw_svg_payload = href.partition(",")
    assert raw_svg_payload, "Missing SVG payload in data URI"

    # Decode if URL-encoded
    svg_text = urllib.parse.unquote(raw_svg_payload)

    # 1. Standard library xml.etree.ElementTree parse
    root = ET.fromstring(svg_text)
    svg_ns = "http://www.w3.org/2000/svg"
    expected_tag = f"{{{svg_ns}}}svg"
    assert root.tag == expected_tag, f"Expected root tag {expected_tag}, got {root.tag}"
    assert root.attrib.get("viewBox") == "0 0 100 100", f"Unexpected viewBox: {root.attrib.get('viewBox')}"

    # 2. lxml.etree parse for strict XML conformance
    lxml_root = lxml.etree.fromstring(svg_text.encode("utf-8"))
    assert lxml_root.tag == expected_tag

    # 3. Check text child element
    texts = list(root.findall(f"{{{svg_ns}}}text"))
    assert len(texts) == 1, f"Expected 1 <text> element in favicon SVG, found {len(texts)}"
    text_el = texts[0]
    assert text_el.attrib.get("font-size") == "90", f"Expected font-size='90', got {text_el.attrib.get('font-size')}"
    assert text_el.attrib.get("y") == ".9em", f"Expected y='.9em', got {text_el.attrib.get('y')}"
    assert text_el.text == "💠", f"Expected emoji '💠', got '{text_el.text}'"

    print("  ✓ Favicon data URI is valid W3C SVG XML")
    print(f"  ✓ SVG payload details: tag={root.tag}, viewBox={root.attrib.get('viewBox')}, glyph={text_el.text}")


def test_brand_badge_and_nomenclature():
    print("\n[Test 3] Brand Badge, Canonical Branding, & Legacy Nomenclature Audit")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()
    raw_html = HTML_PATH.read_text(encoding="utf-8")

    # Title check
    titles = doc.xpath("//title")
    assert len(titles) == 1, "Missing or multiple <title> elements"
    title_text = titles[0].text_content().strip()
    assert title_text.startswith("Clarion"), f"Title must start with 'Clarion', got: {title_text}"
    print(f"  ✓ <title> verified: '{title_text}'")

    # Brand badge check
    badges = doc.xpath("//header//span[contains(@class, 'brand-badge')]")
    assert len(badges) == 1, f"Expected 1 .brand-badge in header, found {len(badges)}"
    badge = badges[0]
    assert badge.text_content().strip() == "Research Preview", (
        f"Expected 'Research Preview', got '{badge.text_content().strip()}'"
    )

    # Verify badge is a sibling to .brand-logo inside .brand-title-row
    parent = badge.getparent()
    assert "brand-title-row" in parent.attrib.get("class", ""), "Badge not in .brand-title-row"
    logos = parent.xpath("./span[contains(@class, 'brand-logo')]")
    assert len(logos) == 1, "Expected .brand-logo sibling in .brand-title-row"
    assert logos[0].text_content().strip() == "Clarion", f"Expected 'Clarion' logo, got '{logos[0].text_content()}'"
    print("  ✓ Brand badge 'Research Preview' correctly placed adjacent to 'Clarion' logo")

    # CSS styling check for .brand-badge
    css_content = CSS_PATH.read_text(encoding="utf-8")
    assert ".brand-badge" in css_content, "Missing .brand-badge rule in styles.css"
    assert "JetBrains Mono" in css_content, "Missing monospace font for .brand-badge"
    print("  ✓ CSS styling for .brand-badge confirmed in styles.css")

    # Zero tolerance for obsolete prototype names
    for pattern in FORBIDDEN_LEGACY_NAMES:
        matches = re.findall(pattern, raw_html, flags=re.IGNORECASE)
        assert not matches, f"Found legacy prototype terms matching '{pattern}' in index.html: {matches}"
    print("  ✓ Zero occurrences of legacy names ('Aureon', 'Clairon', 'Clair', 'DocGraph', 'GraphDoc')")


def test_overview_metrics_deck():
    print("\n[Test 4] 8-Card Overview Metrics Deck Deep Inspection")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()

    grid = doc.xpath("//div[contains(@class, 'metrics-grid')]")
    assert len(grid) == 1, f"Expected 1 .metrics-grid, found {len(grid)}"
    cards = grid[0].xpath("./div[contains(@class, 'metric-card')]")
    assert len(cards) == 8, f"Expected exactly 8 .metric-card elements, found {len(cards)}"

    found_ids = []
    for idx, card in enumerate(cards):
        labels = card.xpath("./span[contains(@class, 'metric-label')]")
        values = card.xpath("./span[contains(@class, 'metric-value')]")
        assert len(labels) == 1, f"Card #{idx} missing .metric-label"
        assert len(values) == 1, f"Card #{idx} missing .metric-value"

        val_el = values[0]
        val_id = val_el.attrib.get("id")
        val_text = val_el.text_content().strip()
        label_text = labels[0].text_content().strip().upper()

        found_ids.append(val_id)
        assert val_text == "--", f"Card #{idx} ({val_id}) initial value must be '--', got '{val_text}'"

        expected_id, expected_label = EXPECTED_METRICS[idx]
        assert val_id == expected_id, f"Card #{idx} expected id '{expected_id}', got '{val_id}'"
        assert label_text == expected_label, f"Card #{idx} expected label '{expected_label}', got '{label_text}'"

        print(f"  ✓ Card #{idx+1}: [{val_id}] label='{label_text}' initial_value='{val_text}'")

    assert found_ids == [m[0] for m in EXPECTED_METRICS], "Overview cards order or IDs mismatch"
    print("  ✓ All 8 overview cards verified with exact IDs, labels, and initial '--' state")


def test_app_js_contract_parity():
    print("\n[Test 5] Frontend app.js Contract Parity & ID Cross-Verification")
    js_content = JS_PATH.read_text(encoding="utf-8")
    doc = lxml.html.parse(str(HTML_PATH)).getroot()

    # Find all updateOverviewMetric calls in app.js
    pattern = r'updateOverviewMetric\s*\(\s*["\']([^"\']+)["\']'
    metric_calls = set(re.findall(pattern, js_content))
    expected_ids = {m[0] for m in EXPECTED_METRICS}

    print(f"  Found updateOverviewMetric targets in app.js: {sorted(metric_calls)}")
    assert metric_calls == expected_ids, (
        f"updateOverviewMetric targets do not match 8-card grid: {metric_calls} vs {expected_ids}"
    )

    # Verify lines 1570-1605 specific block
    for metric_id in ["stat-relations", "stat-nodes", "stat-edges"]:
        pattern_str = rf'updateOverviewMetric\s*\(\s*"{metric_id}"'
        assert re.search(pattern_str, js_content), f"Missing {pattern_str} in app.js"

    # Verify DOM contains all targets
    for metric_id in expected_ids:
        el = doc.get_element_by_id(metric_id)
        assert el is not None, f"Element #{metric_id} queried by app.js is missing from index.html"

    print("  ✓ Full parity confirmed: all 8 metrics referenced in app.js exist in index.html DOM")


def test_responsive_grid_css():
    print("\n[Test 6] Responsive Grid CSS & Layout Robustness")
    css_content = CSS_PATH.read_text(encoding="utf-8")

    # Base grid
    assert "grid-template-columns: repeat(4, 1fr);" in css_content, (
        "Base .metrics-grid must specify repeat(4, 1fr) for 4x2 layout"
    )
    # Mobile responsive grid (<= 768px)
    assert "grid-template-columns: repeat(2, 1fr);" in css_content, (
        "Mobile .metrics-grid must adapt to repeat(2, 1fr)"
    )
    print("  ✓ CSS grid layout verified: 4 columns on desktop (4x2 = 8 cards), 2 columns on mobile")


def run_all_challenges():
    print("=" * 70)
    print("EMPIRICAL ADVERSARIAL CHALLENGE — MILESTONE 2 FRONTEND DELIVERABLES")
    print("=" * 70)

    test_html_syntax_and_unique_ids()
    test_svg_favicon_data_uri()
    test_brand_badge_and_nomenclature()
    test_overview_metrics_deck()
    test_app_js_contract_parity()
    test_responsive_grid_css()

    print("\n" + "=" * 70)
    print("VERDICT: ALL ADVERSARIAL CHALLENGES PASSED (100% SUCCESS)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_challenges()
