"""Helper utilities for Clarion E2E opaque-box test suites."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import lxml.html

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_HTML = ROOT_DIR / "frontend" / "index.html"
FRONTEND_APP_JS = ROOT_DIR / "frontend" / "app.js"


def make_test_document_payload(
    doc_id: str = "doc-test-123",
    title: str = "Computational Complexity Analysis",
    content: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a standard document ingestion/analysis test payload."""
    if content is None:
        content = (
            "Computational complexity theory focuses on classifying computational problems "
            "according to their inherent difficulty. P is the class of decision problems solvable "
            "by a deterministic Turing machine in polynomial time. NP is the class solvable in "
            "polynomial time by a non-deterministic Turing machine. NP-complete problems are the "
            "hardest problems in NP to which all other problems in NP can be reduced in polynomial time."
        )
    return {
        "document_id": doc_id,
        "title": title,
        "content": content,
        "size": len(content.encode("utf-8")),
        "word_count": len(content.split()),
    }


def parse_frontend_html(html_path: Optional[Path] = None) -> lxml.html.HtmlElement:
    """Parse frontend/index.html using lxml.html."""
    target_path = html_path or FRONTEND_HTML
    content = target_path.read_text(encoding="utf-8")
    return lxml.html.fromstring(content)


def get_dom_element_text(doc: lxml.html.HtmlElement, element_id: str) -> str:
    """Retrieve stripped text content of a DOM element by id."""
    el = doc.get_element_by_id(element_id)
    assert el is not None, f"DOM element with id '{element_id}' not found"
    return el.text_content().strip()


def run_node_eval(js_code: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute a Node.js snippet and return parsed stdout or result dict.

    Returns:
        {"stdout": str, "stderr": str, "returncode": int, "json": Any}
    """
    proc = subprocess.run(
        ["node", "-e", js_code],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ROOT_DIR),
    )
    result: Dict[str, Any] = {
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
        "json": None,
    }
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            result["json"] = json.loads(proc.stdout.strip())
        except Exception:
            pass
    return result


def evaluate_frontend_format_duration(duration_seconds: Any) -> str:
    """Call the frontend formatDuration function via Node.js on frontend/app.js.

    If formatDuration is not defined in app.js, raises AssertionError reporting
    the implementation bug according to R1/F4 specifications.
    """
    val_json = json.dumps(duration_seconds)
    script = f"""
    const fs = require('fs');
    const path = require('path');
    const appJsPath = path.resolve('{FRONTEND_APP_JS.as_posix()}');
    const appCode = fs.readFileSync(appJsPath, 'utf8');

    // Run in lightweight sandbox
    const vm = require('vm');
    const sandbox = {{
        console: console,
        document: {{ getElementById: () => ({{ innerHTML: '', textContent: '', appendChild: () => {{}} }}) }},
        window: {{}},
    }};
    vm.createContext(sandbox);
    try {{
        vm.runInContext(appCode, sandbox);
    }} catch (e) {{
        // Ignore initialization errors from missing browser DOM
    }}

    if (typeof sandbox.formatDuration === 'function') {{
        const res = sandbox.formatDuration({val_json});
        console.log(JSON.stringify({{ success: true, formatted: res }}));
    }} else {{
        const match = appCode.match(/function\\s+formatDuration\\s*\\(/);
        if (!match) {{
            console.log(JSON.stringify({{ success: false, error: 'formatDuration is not defined in frontend/app.js' }}));
        }} else {{
            console.log(JSON.stringify({{ success: false, error: 'formatDuration not accessible in global context' }}));
        }}
    }}
    """
    res = run_node_eval(script)
    if res["returncode"] != 0:
        raise RuntimeError(f"Node execution failed: {res['stderr']}")
    data = res.get("json") or {}
    if not data.get("success"):
        raise AssertionError(data.get("error", "formatDuration failed in frontend/app.js"))
    return data.get("formatted", "")


def simulate_render_pipeline_status(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simulate frontend renderPipelineStatus(payload) and return parsed table rows."""
    payload_json = json.dumps(payload)
    script = f"""
    const fs = require('fs');
    const path = require('path');
    const appJsPath = path.resolve('{FRONTEND_APP_JS.as_posix()}');
    const appCode = fs.readFileSync(appJsPath, 'utf8');

    const vm = require('vm');
    const mockElements = {{
        'timeline-table-body': {{ innerHTML: '' }},
        'telem-llm-model': {{ textContent: '' }},
        'telem-analysis-mode': {{ textContent: '' }},
        'status-doc-name': {{ textContent: '' }},
        'pipeline-summary': {{ textContent: '' }},
        'honest-status-badge': {{ textContent: '' }},
        'honest-chip': {{ className: '' }},
        'status-mode': {{ textContent: '' }},
        'mode-chip': {{ className: '' }},
    }};
    const sandbox = {{
        console: console,
        document: {{
            addEventListener: () => {{}},
            getElementById: (id) => mockElements[id] || {{
                innerHTML: '',
                textContent: '',
                appendChild: () => {{}},
                classList: {{ add: () => {{}}, remove: () => {{}} }},
                style: {{ width: '0%' }},
            }},
            querySelectorAll: () => [],
            querySelector: () => null,
        }},
        window: {{ location: {{ protocol: 'http:' }} }},
    }};
    vm.createContext(sandbox);
    try {{
        vm.runInContext(appCode, sandbox);
    }} catch (e) {{
        // ignore initialization errors
    }}

    if (typeof sandbox.renderPipelineStatus === 'function') {{
        sandbox.renderPipelineStatus({payload_json});
        console.log(JSON.stringify({{
            success: true,
            tableHtml: mockElements['timeline-table-body'].innerHTML,
            mockElements: mockElements,
        }}));
    }} else {{
        console.log(JSON.stringify({{ success: false, error: 'renderPipelineStatus is not defined' }}));
    }}
    """
    res = run_node_eval(script)
    if res["returncode"] != 0:
        raise RuntimeError(f"Node render simulation failed: {res['stderr']}")
    data = res.get("json") or {}
    if not data.get("success"):
        raise AssertionError(data.get("error", "renderPipelineStatus failed"))

    html = data.get("tableHtml", "")
    if not html.strip() or "No stage telemetry recorded" in html:
        return []

    doc = lxml.html.fromstring(f"<table><tbody>{html}</tbody></table>")
    rows = []
    for tr in doc.xpath(".//tr"):
        cells = [td.text_content().strip() for td in tr.xpath(".//td")]
        titles = [td.attrib.get("title", "") for td in tr.xpath(".//td")]
        rows.append({
            "cells": cells,
            "titles": titles,
            "html": lxml.html.tostring(tr).decode("utf-8"),
        })
    return rows


def assert_l2_normalized(vector: List[float], tol: float = 1e-4) -> None:
    """Assert Euclidean norm of vector equals 1.0 within tolerance."""
    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) <= tol, f"Vector norm {norm} is not 1.0 (tol={tol})"
