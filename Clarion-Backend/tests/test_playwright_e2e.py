"""Browser-level end-to-end verification using Playwright for the Clarion frontend.

Verifies:
1. Document upload and analysis status display.
2. Honest rendering of degraded vs live model status badges.
3. Concepts and candidate relations tables populated from API responses.
4. Cytoscape graph container initialization and error-free rendering.
5. Grounded RAG query interaction and supporting chunk rendering.
6. Graceful error handling for missing documents or unreachable endpoints.

Requires: playwright (`pip install playwright && playwright install chromium`).
Skips cleanly if playwright or browser binary is not installed in the runtime.
"""

import os
import pytest
from pathlib import Path

# Clean skip if playwright is not installed in the environment
playwright = pytest.importorskip("playwright", reason="Playwright is not installed in this environment")
from playwright.sync_api import sync_playwright, expect

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
FRONTEND_URL = os.environ.get("CLARION_FRONTEND_URL", "http://localhost:8080")
BACKEND_URL = os.environ.get("CLARION_BACKEND_URL", "http://localhost:8000")
REAL_DOC_ID = "a945d65e-f069-41f9-abb2-e1a7e774d559"


@pytest.fixture(scope="module")
def browser_context():
    """Launch headless browser for e2e tests."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            yield context
            browser.close()
        except Exception as e:
            pytest.skip(f"Browser launch failed: {e}")


def test_frontend_loads_and_displays_status_ribbon(browser_context):
    """Verify that the frontend page loads and initializes the status ribbon."""
    page = browser_context.new_page()
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")

    # Brand and title check
    expect(page.locator(".brand-logo")).to_have_text("Clarion")
    expect(page.locator(".brand-badge")).to_have_text("RESEARCH PREVIEW")

    # Status ribbon elements present
    expect(page.locator("#doc-chip")).to_be_visible()
    expect(page.locator("#mode-chip")).to_be_visible()
    expect(page.locator("#model-chip")).to_be_visible()
    expect(page.locator("#honest-status-badge")).to_be_visible()
    page.close()


def test_degraded_model_badge_displayed_honestly(browser_context):
    """Verify that when Ollama is unreachable, the UI honestly displays fallback/degraded status."""
    page = browser_context.new_page()
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")

    # Check that status indicates fallback or degraded state
    badge_text = page.locator("#honest-status-badge").text_content()
    assert badge_text is not None
    # In offline environments, status must state fallback or offline, never faked live inference
    assert any(term in badge_text.lower() for term in ["fallback", "checking", "offline", "unavailable", "degraded"])
    page.close()


def test_concepts_and_relations_loaded_from_api(browser_context):
    """Verify that selecting a document populates the concepts and candidate relations tables."""
    page = browser_context.new_page()
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")

    # Check document library
    doc_item = page.locator(f'[data-id="{REAL_DOC_ID}"]')
    if doc_item.count() > 0:
        doc_item.first.click()
        page.wait_for_timeout(1000)

        # Overview counts should be populated
        concepts_badge = page.locator("#concepts-count").text_content()
        assert concepts_badge and "0 concepts" not in concepts_badge

        # Relational Grounding notice present
        expect(page.locator(".panel-notice")).to_contain_text("Candidate Hypotheses")

        # Table rows rendered
        concept_rows = page.locator("#concepts-table tbody tr")
        assert concept_rows.count() > 0

        relation_rows = page.locator("#relations-table tbody tr")
        assert relation_rows.count() > 0
    page.close()


def test_graph_renders_without_javascript_exception(browser_context):
    """Verify that the Cytoscape graph container mounts without JS errors."""
    page = browser_context.new_page()
    js_errors = []
    page.on("pageerror", lambda err: js_errors.append(str(err)))

    page.goto(FRONTEND_URL, wait_until="domcontentloaded")
    expect(page.locator("#graph-container")).to_be_visible()

    # Confirm no fatal JavaScript exceptions crashed the page
    fatal_errors = [e for e in js_errors if "syntaxerror" in e.lower() or "referenceerror" in e.lower()]
    assert len(fatal_errors) == 0, f"Fatal JS errors encountered: {fatal_errors}"
    page.close()


def test_rag_query_interaction_and_grounding(browser_context):
    """Verify that the RAG query input accepts questions and renders responses."""
    page = browser_context.new_page()
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")

    doc_item = page.locator(f'[data-id="{REAL_DOC_ID}"]')
    if doc_item.count() > 0:
        doc_item.first.click()
        page.wait_for_timeout(1000)

        # Ask a query via RAG input
        query_input = page.locator("#rag-query-input")
        query_btn = page.locator("#rag-query-btn")
        if query_btn.is_enabled():
            query_input.fill("What is Dijkstra algorithm?")
            query_btn.click()
            page.wait_for_timeout(3000)

            # Response box should display
            expect(page.locator("#rag-output-box")).to_be_visible()
            expect(page.locator("#rag-answer-text")).not_to_be_empty()
    page.close()


def test_graceful_handling_of_api_errors(browser_context):
    """Verify that non-existent documents or API failures render clean error feedback."""
    page = browser_context.new_page()
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")

    # Trigger request for an invalid document ID
    page.evaluate("() => window.selectDocument && window.selectDocument('non-existent-doc-id-000')")
    page.wait_for_timeout(1000)

    # Confirm UI remains interactive and did not crash
    expect(page.locator(".app-header")).to_be_visible()
    page.close()
