"""Capture current repository accounting and final engineering gate decisions."""
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
ARTIFACTS = ROOT / "research_artifacts"
RUN = ARTIFACTS / "demo_runs" / "clarion-final-demo-2026-11-12"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.rstrip("\r\n")


def category(path):
    p = path.replace("\\", "/")
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    if p.startswith("docs/") and p.lower().endswith(".pdf"):
        return "Raw source document corpus"
    if p.startswith("docs/archive/packages/"):
        return "Historical archive integrity and checksum evidence"
    if p.startswith("research/data/"):
        return "Raw or processed research dataset and source register"
    if p == "research/references.bib":
        return "Research reference bibliography"
    if p.startswith("research_artifacts/") and any(token in p for token in ("presnapshot", "presnapshot", "final-freeze", "frozen-presnapshot", "readiness-presnapshot", "provenance-presnapshot")):
        return "Recovery material and historical evidence"
    if p.startswith("research_artifacts/isolated-provenance-e2e-"):
        return "Historical isolated provenance validation evidence"
    if p.startswith("research_artifacts/isolated_pdf_index_"):
        return "Processed isolated document-index validation output"
    if p.startswith("research_artifacts/figures/") or p.startswith("research_artifacts/tables/"):
        return "Figures or tables"
    if p.startswith("research_artifacts/annotation/"):
        return "Research annotation protocol or template"
    if p.startswith("research_artifacts/demo_runs/"):
        if "/pytest-tmp/" in p or "/data/" in p or "/logs/" in p:
            return "Temporary/generated output or ignored local demo runtime"
        return "Raw live-demo validation evidence"
    if p.startswith("research_artifacts/raw/"):
        return "Historical/raw experimental or validation evidence"
    if p.startswith("research_artifacts/processed/"):
        return "Processed evidence or machine-readable technical specification"
    if p.startswith("research_artifacts/scripts/") or p.startswith("scripts/") or p.startswith("research/scripts/"):
        return "Reproducibility or validation script"
    if p.startswith("Clarion-Backend/tests/") or p.startswith("tests/"):
        return "Essential test code"
    if p.startswith("docs/") and p.lower().endswith(".pdf"):
        return "Raw source document corpus"
    if p.startswith("research/protocols/") or p.startswith("research/annotation/"):
        return "Essential research protocol"
    if p.startswith("research/manuscript/") or p.startswith("research/reports/") or p.startswith("reports/"):
        return "Research-authored manuscript/report retained untouched"
    if p.startswith("Clarion-Backend/") or p.startswith("frontend/") or p.startswith(".github/"):
        return "Essential application source or CI configuration"
    if p.endswith(".md"):
        return "Developer documentation or repository guidance"
    return "Unresolved; requires owner review"


head = git("rev-parse", "HEAD")
branch = git("branch", "--show-current")
status_lines = git("-c", "core.quotePath=false", "status", "--porcelain=v1", "--untracked-files=all").splitlines()
modified, deleted, untracked, staged = [], [], [], []
for line in status_lines:
    code, path = line[:2], line[3:]
    if "?" in code:
        untracked.append({"path": path, "category": category(path)})
    else:
        if "M" in code:
            modified.append(path)
        if "D" in code:
            deleted.append(path)
        if code[0] in "MADRCU":
            staged.append(path)
groups = Counter(item["category"] for item in untracked)
demo = json.loads((RUN / "live-demo-verification.json").read_text(encoding="utf-8"))
suite_result = {
    "linux_full_suite": {"status": "passed", "result": "327 passed, 2 skipped", "scope_note": "Fresh Linux run covered the implementation changes before the final opt-out guard test was added; the latest guard test is included in the following Windows full-suite run.", "log": "research_artifacts/raw/backend-full-suite-linux-py312-final-2026-11-12.log"},
    "windows_full_suite_latest": {"status": "passed", "result": "328 passed, 2 skipped", "exit_code": 0, "collected_test_functions": 329},
    "focused_windows_regression": {"status": "passed", "result": "39 passed", "scope": ["test_relation_dataset_service.py", "test_text_metrics.py", "test_graph.py", "test_knowledge_map_service.py", "test_api.py"]}
}
gates = {
    "repository_integrity": "PASS WITH PRESERVED WORKTREE CHANGES: all non-ignored untracked paths categorized; two tracked deletions reconcile to preserved relocated originals and recovery copies; 88 existing researcher-authored report/manuscript files left untouched.",
    "engineering_regression": "PASS: full Linux suite passed for implementation; latest full Windows suite passed after the final guard change.",
    "document_statistics_correctness": "PASS FOR ONE REAL PDF: independent PDF metadata/text, API and database agree; frontend display not browser-verified.",
    "mind_map_correctness": "PARTIAL: live model generated five concepts and one evidence-backed candidate relation; graph has 11 nodes, 11 edges, no orphan edges, no duplicate concept names or directed relation pairs. Panel UX not browser-verified.",
    "frontend_demonstration_readiness": "PARTIAL: live backend workflow and API path work; browser process exits before page rendering, so screenshots and frontend interactions are unverified.",
    "research_measurement_readiness": "PARTIAL: word-count spec, metrics API, schemas, annotation templates and live audit script exist; multi-document and stage-duration coverage still required.",
    "research_evaluation_validity": "BLOCKED: no human-verified retrieval qrels or graph gold labels; no quality metrics are claimed."
}
result = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "head": head,
    "branch": branch,
    "commit_created": False,
    "release_zip_created": False,
    "research_manuscript_or_academic_reports_written": False,
    "git_status_counts": {"modified_tracked": len(modified), "deleted_tracked": len(deleted), "untracked_nonignored": len(untracked), "staged": len(staged)},
    "modified_tracked_paths": modified,
    "deleted_tracked_paths": deleted,
    "untracked_paths": untracked,
    "untracked_category_counts": dict(groups),
    "unresolved_untracked_paths": [item["path"] for item in untracked if item["category"].startswith("Unresolved")],
    "ignored_runtime_policy": [".gitignore now excludes isolated demo runtime data, pytest temp outputs and logs; local databases and vector indexes remain on disk for reproducibility but are not Git candidates."],
    "recovery_evidence": {"snapshot_directory": "research_artifacts/engineering-final-blocker-resolution-2026-11-12/", "previous_manifest_verification": "research_artifacts/processed/final_freeze_inventory_2026-11-12.json", "historical_snapshot_preserved": True},
    "data_integrity": demo,
    "test_results": suite_result,
    "gates": gates,
    "known_unreconciled_items": ["Browser screenshot/interaction evidence is unavailable because both Chromium and Edge exited during startup.", "The completed live analysis wrote a unique dataset mirror under workspace data; its byte-identical copy is preserved under the isolated demo evidence directory and the workspace copy was removed.", "Two probe-created default DB audit rows and empty null-only migration schema were reversed exactly; document/chunk counts were preserved. See research_artifacts/processed/default-database-probe-rollback-2026-11-12.json."],
    "evidence_paths": ["research_artifacts/demo_runs/clarion-final-demo-2026-11-12/live-demo-verification.json", "research_artifacts/demo_runs/clarion-final-demo-2026-11-12/query-response.json", "research_artifacts/demo_runs/clarion-final-demo-2026-11-12/knowledge-map.json", "research_artifacts/demo_runs/clarion-final-demo-2026-11-12/evidence/graph-output-cytoscape.json", "research_artifacts/processed/document-metrics-spec-2026-11-12.json", "research_artifacts/processed/measurement-collection-schema-2026-11-12.json"]
}
path = ARTIFACTS / "processed" / "final-engineering-handoff-2026-11-12.json"
path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"path": str(path.relative_to(ROOT)), "head": head, "git_status_counts": result["git_status_counts"], "untracked_category_counts": result["untracked_category_counts"], "unresolved_untracked_paths": result["unresolved_untracked_paths"], "gates": gates}, indent=2, ensure_ascii=False))
