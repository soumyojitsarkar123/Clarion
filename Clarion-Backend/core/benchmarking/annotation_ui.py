"""
Interactive Human Annotation Interfaces for Clarion Knowledge Graph Relation Validation.

Provides two safe, local, dependency-free annotation interfaces:
1. InteractiveTerminalAnnotator: Console/terminal-based step-by-step annotation.
2. AnnotationWebServer: Lightweight local HTTP server serving a single-page web UI.

Data Safety Guarantees:
- Automatic timestamped backup (.bak_<timestamp>) created prior to modification.
- Immutable candidate metadata (record_id, document_id, concepts, chunks, scores) can NEVER be altered.
- Atomic file writing (writes to .tmp then os.replace) prevents corruption on interrupt.
- In-memory validation prevents advancing on invalid or incomplete fields.
- 100% independent execution for Batch A and Batch B.
"""

from __future__ import annotations

import datetime
import html
import http.server
import json
import logging
import os
import shutil
import socketserver
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from models.ground_truth import ErrorCategory, ValidityLabel

logger = logging.getLogger(__name__)

ALLOWED_LABELS = [e.value for e in ValidityLabel]
ALLOWED_ERROR_CATEGORIES = [e.value for e in ErrorCategory if e.value != "NONE"]

IMMUTABLE_FIELDS = (
    "record_id",
    "document_id",
    "relation_id",
    "concept_a",
    "concept_b",
    "relation_type",
    "chunk_context",
    "source_chunk_ids",
    "llm_confidence",
    "cooccurrence_score",
    "semantic_similarity",
    "existing_db_is_valid",
)

HUMAN_EDITABLE_FIELDS = (
    "label",
    "error_category",
    "annotator_id",
    "confidence_rating",
    "adjudicated_evidence_text",
    "notes",
)


def create_backup(file_path: Path) -> Path:
    """Create a timestamped backup before any write operation."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_name(f"{file_path.name}.bak_{ts}")
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created safety backup: {backup_path}")
    return backup_path


def atomic_save_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Safely write JSON using atomic replace to prevent corrupt files."""
    tmp_path = file_path.with_name(f"{file_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, file_path)


def validate_single_annotation(
    label: Optional[str],
    error_category: Optional[str],
    annotator_id: Optional[str],
    confidence_rating: Optional[int],
    evidence_text: Optional[str],
) -> Tuple[bool, List[str]]:
    """Validate human annotation input against protocol rules."""
    errors = []
    if not label or label.strip().upper() not in ALLOWED_LABELS:
        errors.append(f"Label must be one of: {ALLOWED_LABELS}")
    else:
        lbl_upper = label.strip().upper()
        if lbl_upper == "INVALID":
            if not error_category or error_category.strip().upper() not in ALLOWED_ERROR_CATEGORIES:
                errors.append(
                    f"INVALID relations require an error category from: {ALLOWED_ERROR_CATEGORIES}"
                )

    if not annotator_id or not annotator_id.strip():
        errors.append("Annotator ID is mandatory (e.g. initials or code).")
    elif annotator_id.strip().lower() in ("unknown", "none", "placeholder", "null"):
        errors.append("Placeholder annotator ID is not allowed.")

    if confidence_rating is None:
        errors.append("Confidence rating (1-5) is mandatory.")
    else:
        try:
            cr = int(confidence_rating)
            if not (1 <= cr <= 5):
                errors.append("Confidence rating must be between 1 and 5.")
        except (ValueError, TypeError):
            errors.append("Confidence rating must be an integer between 1 and 5.")

    if not evidence_text or not evidence_text.strip():
        errors.append("Verbatim evidence text citing the document context is mandatory.")

    return len(errors) == 0, errors


# ===========================================================================
# 1. Interactive Terminal Annotator
# ===========================================================================

class InteractiveTerminalAnnotator:
    """Lightweight console-based annotation session with full navigation and auto-save."""

    def __init__(self, file_path: str, annotator_id: Optional[str] = None):
        self.path = Path(file_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.path}")

        # Create immediate backup
        self.backup_path = create_backup(self.path)

        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.records: List[Dict[str, Any]] = self.data.get("records", [])
        if not self.records:
            raise ValueError(f"No records found in {self.path}")

        self.annotator_id = annotator_id or ""
        self.current_idx = 0
        # Fast-forward to first unannotated record
        for i, r in enumerate(self.records):
            if not r.get("label"):
                self.current_idx = i
                break

    def _save(self) -> None:
        atomic_save_json(self.path, self.data)

    def _stats(self) -> Tuple[int, int, Dict[str, int]]:
        total = len(self.records)
        completed = 0
        counts: Dict[str, int] = {"VALID": 0, "INVALID": 0, "AMBIGUOUS": 0}
        for r in self.records:
            lbl = r.get("label")
            if lbl:
                completed += 1
                counts[str(lbl).upper()] = counts.get(str(lbl).upper(), 0) + 1
        return total, completed, counts

    def run(self) -> None:
        """Run terminal interactive loop."""
        print("\n" + "=" * 70)
        print("  CLARION HUMAN RELATION ANNOTATION CONSOLE")
        print("=" * 70)
        print(f"File: {self.path.name}")
        print(f"Backup created at: {self.backup_path.name}")
        total, completed, counts = self._stats()
        print(f"Progress: {completed}/{total} ({round(completed/total*100, 1)}%) | "
              f"VALID: {counts['VALID']} | INVALID: {counts['INVALID']} | AMBIGUOUS: {counts['AMBIGUOUS']}")

        if not self.annotator_id:
            while not self.annotator_id.strip():
                try:
                    self.annotator_id = input("\nEnter your Annotator ID (e.g. initials 'AM1'): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    return

        while True:
            r = self.records[self.current_idx]
            total, completed, counts = self._stats()
            pct = round(completed / total * 100, 1)

            print("\n" + "-" * 70)
            print(f"RECORD [{self.current_idx + 1} / {total}] — Overall Completed: {completed}/{total} ({pct}%)")
            print(f"Candidate ID : {r['record_id']}")
            print(f"Document ID  : {r['document_id']}")
            print("-" * 70)
            print(f"  HEAD (concept_a) : {r['concept_a']}")
            print(f"  PREDICATE        : {r['relation_type']}")
            print(f"  TAIL (concept_b) : {r['concept_b']}")
            print("-" * 70)
            print("DOCUMENT EVIDENCE CONTEXT (chunk_context):")
            print(f"  {r['chunk_context']}")
            print("-" * 70)
            print("[NOTICE] Automated scores (llm_confidence, cooccurrence, semantic_sim) are INFORMATIONAL ONLY.")
            print("         Decisions must be based SOLELY on the document text above.")
            print("-" * 70)

            cur_lbl = r.get("label")
            if cur_lbl:
                cur_err = r.get("error_category")
                cur_ann = r.get("annotator_id")
                print(f"Current Annotation: [{cur_lbl}] (error: {cur_err or 'NONE'}) by {cur_ann}")
            else:
                print("Current Annotation: [UNANNOTATED]")

            print("\nCommands:")
            print("  [V] VALID     [I] INVALID     [A] AMBIGUOUS")
            print("  [P] Previous  [N] Next        [U] Next Unannotated  [J] Jump  [Q] Save & Quit")

            try:
                cmd = input("\nEnter choice: ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print("\nSaving progress and exiting...")
                self._save()
                break

            if cmd == "Q":
                self._save()
                print(f"\nProgress safely saved to {self.path.name}. Goodbye!")
                break
            elif cmd == "P":
                if self.current_idx > 0:
                    self.current_idx -= 1
                else:
                    print("Already at first record.")
                continue
            elif cmd == "N":
                if self.current_idx < total - 1:
                    self.current_idx += 1
                else:
                    print("Already at last record.")
                continue
            elif cmd == "U":
                found = False
                for idx in range(self.current_idx + 1, total):
                    if not self.records[idx].get("label"):
                        self.current_idx = idx
                        found = True
                        break
                if not found:
                    for idx in range(0, self.current_idx):
                        if not self.records[idx].get("label"):
                            self.current_idx = idx
                            found = True
                            break
                if not found:
                    print("All records have been annotated!")
                continue
            elif cmd == "J":
                try:
                    target = int(input(f"Enter record number (1-{total}): ").strip())
                    if 1 <= target <= total:
                        self.current_idx = target - 1
                    else:
                        print("Invalid record number.")
                except ValueError:
                    print("Invalid input.")
                continue
            elif cmd in ("V", "I", "A"):
                lbl = {"V": "VALID", "I": "INVALID", "A": "AMBIGUOUS"}[cmd]
                err_cat = "NONE"
                if lbl == "INVALID":
                    print("\nSelect INVALID Error Category:")
                    for idx, cat in enumerate(ALLOWED_ERROR_CATEGORIES, 1):
                        print(f"  {idx}. {cat}")
                    cat_choice = None
                    while cat_choice not in range(1, len(ALLOWED_ERROR_CATEGORIES) + 1):
                        try:
                            cat_raw = input("Enter category number (1-5): ").strip()
                            cat_choice = int(cat_raw)
                        except ValueError:
                            pass
                    err_cat = ALLOWED_ERROR_CATEGORIES[cat_choice - 1]

                # Confidence
                conf = 5
                try:
                    conf_raw = input("Confidence rating [1-5] (default 5): ").strip()
                    if conf_raw:
                        conf = int(conf_raw)
                        if not (1 <= conf <= 5):
                            conf = 5
                except ValueError:
                    conf = 5

                # Evidence Text
                default_ev = r.get("chunk_context", "")
                print("\nEnter verbatim evidence sentence(s).")
                print(f"[Press Enter to accept full chunk_context]:")
                try:
                    ev_input = input("Evidence: ").strip()
                except (EOFError, KeyboardInterrupt):
                    ev_input = ""
                ev_text = ev_input if ev_input else default_ev

                # Notes
                try:
                    notes_input = input("Optional notes (or press Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    notes_input = ""

                # Validate
                ok, errs = validate_single_annotation(lbl, err_cat, self.annotator_id, conf, ev_text)
                if not ok:
                    print(f"Validation Error: {errs}")
                    continue

                # Apply human edits strictly to editable fields
                r["label"] = lbl
                r["error_category"] = err_cat if lbl == "INVALID" else "NONE"
                r["annotator_id"] = self.annotator_id
                r["confidence_rating"] = conf
                r["adjudicated_evidence_text"] = ev_text
                r["notes"] = notes_input if notes_input else None

                # Atomic save on every record
                self._save()
                print(f"Saved Record {self.current_idx + 1} as [{lbl}]")

                # Advance to next record
                if self.current_idx < total - 1:
                    self.current_idx += 1


# ===========================================================================
# 2. Local Web Server Annotation Interface
# ===========================================================================

WEB_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clarion Human Relation Annotation Console</title>
<style>
  :root {
    --bg: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --primary: #3b82f6;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }
  .container { max-width: 900px; margin: 0 auto; }
  header { margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 12px; }
  h1 { font-size: 1.3rem; font-weight: 700; color: #fff; }
  .badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
  .badge-valid { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
  .badge-invalid { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
  .badge-ambiguous { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }
  .progress-bar-container { background: #334155; border-radius: 6px; height: 10px; overflow: hidden; margin-bottom: 20px; }
  .progress-bar-fill { background: var(--primary); height: 100%; width: 0%; transition: width 0.3s ease; }
  .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
  .card-title { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.05em; }
  .relation-display { display: flex; align-items: center; justify-content: space-between; background: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid var(--border); margin-bottom: 10px; }
  .concept-box { font-size: 1.15rem; font-weight: 700; color: #38bdf8; max-width: 40%; word-break: break-word; }
  .predicate-arrow { display: flex; flex-direction: column; align-items: center; font-size: 0.8rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; padding: 0 16px; }
  .predicate-arrow::after { content: "──────►"; color: var(--primary); font-size: 1.1rem; }
  .evidence-box { background: #0b1120; border-left: 4px solid var(--primary); padding: 14px; font-size: 0.95rem; border-radius: 4px; color: #e2e8f0; white-space: pre-wrap; word-break: break-word; max-height: 220px; overflow-y: auto; }
  .warning-banner { background: rgba(245, 158, 11, 0.15); border: 1px solid var(--warning); border-radius: 6px; padding: 12px; margin-bottom: 20px; font-size: 0.82rem; color: #fde68a; }
  .warning-banner strong { color: #fff; }
  .btn-group { display: flex; gap: 10px; margin-bottom: 15px; }
  .btn { flex: 1; padding: 12px; border-radius: 6px; border: 1px solid transparent; font-weight: 600; cursor: pointer; transition: all 0.15s ease; font-size: 0.95rem; text-align: center; }
  .btn-outline { background: transparent; border-color: var(--border); color: var(--text); }
  .btn-outline:hover { background: #334155; }
  .btn-valid { background: rgba(16, 185, 129, 0.15); border-color: var(--success); color: var(--success); }
  .btn-valid.active, .btn-valid:hover { background: var(--success); color: #fff; }
  .btn-invalid { background: rgba(239, 68, 68, 0.15); border-color: var(--danger); color: var(--danger); }
  .btn-invalid.active, .btn-invalid:hover { background: var(--danger); color: #fff; }
  .btn-ambiguous { background: rgba(245, 158, 11, 0.15); border-color: var(--warning); color: var(--warning); }
  .btn-ambiguous.active, .btn-ambiguous:hover { background: var(--warning); color: #fff; }
  .form-group { margin-bottom: 14px; }
  label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px; }
  select, input[type="text"], input[type="number"], textarea { width: 100%; background: #0f172a; border: 1px solid var(--border); border-radius: 6px; padding: 10px; color: #fff; font-size: 0.9rem; }
  select:focus, input:focus, textarea:focus { border-color: var(--primary); outline: none; }
  textarea { min-height: 80px; resize: vertical; }
  .nav-bar { display: flex; justify-content: space-between; align-items: center; margin-top: 20px; }
  .toast { position: fixed; bottom: 20px; right: 20px; background: #1e293b; border: 1px solid var(--primary); padding: 12px 20px; border-radius: 6px; color: #fff; display: none; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); z-index: 100; font-size: 0.9rem; }
  .metadata-tag { font-size: 0.75rem; color: var(--text-muted); }
  .copy-btn { background: #334155; border: none; border-radius: 4px; color: #94a3b8; font-size: 0.75rem; padding: 3px 8px; cursor: pointer; float: right; }
  .copy-btn:hover { color: #fff; }
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Clarion Relation Annotation Console</h1>
      <span class="metadata-tag" id="file-label">Batch: loading...</span>
    </div>
    <div id="stats-summary" style="font-size:0.85rem; font-weight:600;">
      0 / 0 completed
    </div>
  </header>

  <div class="progress-bar-container">
    <div class="progress-bar-fill" id="progress-bar"></div>
  </div>

  <div class="warning-banner">
    <strong>SCIENTIFIC INTEGRITY GUARD:</strong> Base your decision <strong>SOLELY</strong> on the provided document evidence below. Automated similarity and confidence scores are purely informational and must not influence your verdict.
  </div>

  <div class="card">
    <div class="card-title">Candidate Relation (Record <span id="record-num">1</span> of <span id="record-total">200</span>)</div>
    <div class="relation-display">
      <div class="concept-box" id="concept-a">Head Concept</div>
      <div class="predicate-arrow" id="relation-type">explanation</div>
      <div class="concept-box" id="concept-b" style="text-align:right;">Tail Concept</div>
    </div>
    <div style="font-size:0.75rem; color:var(--text-muted); display:flex; justify-content:space-between;">
      <span>Record ID: <code id="record-id">...</code></span>
      <span>Doc ID: <code id="doc-id">...</code></span>
    </div>
  </div>

  <div class="card">
    <div class="card-title">
      Source Document Evidence
      <button class="copy-btn" onclick="copyContextToEvidence()">Copy to Evidence Field</button>
    </div>
    <div class="evidence-box" id="chunk-context">Loading evidence...</div>
  </div>

  <div class="card">
    <div class="card-title">Human Adjudication Decision</div>
    <div class="form-group">
      <label>1. Validity Assessment *</label>
      <div class="btn-group">
        <button type="button" class="btn btn-valid" id="btn-v" onclick="selectLabel('VALID')">VALID [V]</button>
        <button type="button" class="btn btn-invalid" id="btn-i" onclick="selectLabel('INVALID')">INVALID [I]</button>
        <button type="button" class="btn btn-ambiguous" id="btn-a" onclick="selectLabel('AMBIGUOUS')">AMBIGUOUS [A]</button>
      </div>
    </div>

    <div class="form-group" id="err-cat-group" style="display:none;">
      <label>2. Invalid Reason Category *</label>
      <select id="error-category">
        <option value="UNSUPPORTED_ASSERTION">UNSUPPORTED_ASSERTION (Evidence does not support claim)</option>
        <option value="DIRECTION_INVERSION">DIRECTION_INVERSION (Relation arrow is reversed)</option>
        <option value="TYPE_MISMATCH">TYPE_MISMATCH (Relation type is incorrect)</option>
        <option value="SPURIOUS_COOCCURRENCE">SPURIOUS_COOCCURRENCE (Concepts co-occur incidentally)</option>
        <option value="NOISE_ENTITY">NOISE_ENTITY (One or both entities are garbled/noise)</option>
      </select>
    </div>

    <div style="display:flex; gap:14px;">
      <div class="form-group" style="flex:1;">
        <label>Annotator ID *</label>
        <input type="text" id="annotator-id" placeholder="e.g. AM1" />
      </div>
      <div class="form-group" style="flex:1;">
        <label>Confidence Rating (1-5) *</label>
        <select id="confidence-rating">
          <option value="5">5 - Completely Certain</option>
          <option value="4">4 - High Confidence</option>
          <option value="3">3 - Moderate Confidence</option>
          <option value="2">2 - Low Confidence</option>
          <option value="1">1 - Speculative</option>
        </select>
      </div>
    </div>

    <div class="form-group">
      <label>Verbatim Evidence Citation *</label>
      <textarea id="evidence-text" placeholder="Copy the minimal sentence(s) from above that justify your decision."></textarea>
    </div>

    <div class="form-group">
      <label>Optional Notes / Rationale</label>
      <input type="text" id="notes" placeholder="Optional notes for adjudication review." />
    </div>

    <div class="nav-bar">
      <div>
        <button type="button" class="btn btn-outline" onclick="prevRecord()">← Previous</button>
        <button type="button" class="btn btn-outline" onclick="nextRecord()">Next →</button>
        <button type="button" class="btn btn-outline" onclick="nextUnannotated()">Next Unannotated ⏭</button>
      </div>
      <button type="button" class="btn btn-valid" style="flex:none; padding:12px 30px;" onclick="saveCurrent(true)">Save & Next ➔</button>
    </div>
  </div>
</div>

<div class="toast" id="toast">Record saved.</div>

<script>
let state = {
  currentIndex: 0,
  totalRecords: 0,
  records: [],
  selectedLabel: null,
};

async function loadData() {
  const res = await fetch('/api/records');
  const d = await res.json();
  state.records = d.records;
  state.totalRecords = d.records.length;
  document.getElementById('file-label').textContent = 'Batch: ' + d.filename;
  
  // Restore saved annotator_id from localStorage if available
  const savedAnn = localStorage.getItem('clarion_annotator_id');
  if (savedAnn) {
    document.getElementById('annotator-id').value = savedAnn;
  }

  // Find first unannotated record
  let startIdx = 0;
  for (let i = 0; i < state.records.length; i++) {
    if (!state.records[i].label) {
      startIdx = i;
      break;
    }
  }
  showRecord(startIdx);
}

function showRecord(idx) {
  if (idx < 0 || idx >= state.totalRecords) return;
  state.currentIndex = idx;
  const r = state.records[idx];

  document.getElementById('record-num').textContent = (idx + 1);
  document.getElementById('record-total').textContent = state.totalRecords;
  document.getElementById('concept-a').textContent = r.concept_a;
  document.getElementById('concept-b').textContent = r.concept_b;
  document.getElementById('relation-type').textContent = r.relation_type;
  document.getElementById('record-id').textContent = r.record_id.slice(0, 13) + '...';
  document.getElementById('doc-id').textContent = r.document_id.slice(0, 13) + '...';
  document.getElementById('chunk-context').textContent = r.chunk_context;

  // Populate fields
  selectLabel(r.label || null);
  if (r.error_category && r.error_category !== 'NONE') {
    document.getElementById('error-category').value = r.error_category;
  } else {
    document.getElementById('error-category').value = 'UNSUPPORTED_ASSERTION';
  }

  if (r.annotator_id) {
    document.getElementById('annotator-id').value = r.annotator_id;
  }
  document.getElementById('confidence-rating').value = r.confidence_rating || '5';
  document.getElementById('evidence-text').value = r.adjudicated_evidence_text || '';
  document.getElementById('notes').value = r.notes || '';

  updateStats();
}

function selectLabel(lbl) {
  state.selectedLabel = lbl;
  document.getElementById('btn-v').classList.toggle('active', lbl === 'VALID');
  document.getElementById('btn-i').classList.toggle('active', lbl === 'INVALID');
  document.getElementById('btn-a').classList.toggle('active', lbl === 'AMBIGUOUS');
  document.getElementById('err-cat-group').style.display = (lbl === 'INVALID') ? 'block' : 'none';
}

function copyContextToEvidence() {
  const ctx = document.getElementById('chunk-context').textContent;
  document.getElementById('evidence-text').value = ctx;
}

function updateStats() {
  let completed = 0;
  for (let r of state.records) {
    if (r.label) completed++;
  }
  const pct = Math.round((completed / state.totalRecords) * 100);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('stats-summary').textContent = `${completed} / ${state.totalRecords} completed (${pct}%)`;
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(() => { t.style.display = 'none'; }, 2000);
}

async function saveCurrent(advance = false) {
  const annId = document.getElementById('annotator-id').value.trim();
  const evText = document.getElementById('evidence-text').value.trim();
  const lbl = state.selectedLabel;
  const errCat = (lbl === 'INVALID') ? document.getElementById('error-category').value : 'NONE';
  const conf = parseInt(document.getElementById('confidence-rating').value, 10);
  const notes = document.getElementById('notes').value.trim();

  if (!lbl) {
    alert('Please select a validity label (VALID, INVALID, or AMBIGUOUS).');
    return false;
  }
  if (!annId) {
    alert('Please enter your Annotator ID.');
    return false;
  }
  if (!evText) {
    alert('Please provide verbatim evidence text supporting your decision.');
    return false;
  }

  // Save annotator ID for convenience
  localStorage.setItem('clarion_annotator_id', annId);

  const payload = {
    label: lbl,
    error_category: errCat,
    annotator_id: annId,
    confidence_rating: conf,
    adjudicated_evidence_text: evText,
    notes: notes || null,
  };

  const res = await fetch(`/api/record/${state.currentIndex}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const respData = await res.json();
  if (respData.success) {
    state.records[state.currentIndex] = respData.record;
    showToast(`Record ${state.currentIndex + 1} saved successfully!`);
    updateStats();
    if (advance && state.currentIndex < state.totalRecords - 1) {
      showRecord(state.currentIndex + 1);
    }
    return true;
  } else {
    alert('Save failed: ' + respData.error);
    return false;
  }
}

function prevRecord() {
  if (state.currentIndex > 0) showRecord(state.currentIndex - 1);
}

function nextRecord() {
  if (state.currentIndex < state.totalRecords - 1) showRecord(state.currentIndex + 1);
}

function nextUnannotated() {
  for (let i = state.currentIndex + 1; i < state.totalRecords; i++) {
    if (!state.records[i].label) {
      showRecord(i);
      return;
    }
  }
  for (let i = 0; i < state.currentIndex; i++) {
    if (!state.records[i].label) {
      showRecord(i);
      return;
    }
  }
  alert('All records have been annotated!');
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      saveCurrent(true);
    }
    return;
  }
  if (e.key === 'v' || e.key === 'V') selectLabel('VALID');
  if (e.key === 'i' || e.key === 'I') selectLabel('INVALID');
  if (e.key === 'a' || e.key === 'A') selectLabel('AMBIGUOUS');
  if (e.key === 'ArrowRight' || e.key === 'n' || e.key === 'N') nextRecord();
  if (e.key === 'ArrowLeft' || e.key === 'p' || e.key === 'P') prevRecord();
});

loadData();
</script>
</body>
</html>
"""


class LocalAnnotationHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP handler serving the annotation single-page app and REST endpoints."""

    def __init__(self, *args, file_path: Path, **kwargs):
        self.file_path = file_path
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP request logging
        pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = WEB_HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/records":
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            resp = {
                "filename": self.file_path.name,
                "total": len(data.get("records", [])),
                "records": data.get("records", []),
            }
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/record/"):
            try:
                idx_str = parsed.path.split("/")[-1]
                idx = int(idx_str)
            except ValueError:
                self._send_json({"success": False, "error": "Invalid record index"}, 400)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(body)
            except Exception as exc:
                self._send_json({"success": False, "error": f"Invalid JSON payload: {exc}"}, 400)
                return

            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            records = data.get("records", [])
            if idx < 0 or idx >= len(records):
                self._send_json({"success": False, "error": f"Index {idx} out of range"}, 400)
                return

            rec = records[idx]

            # Validate
            lbl = payload.get("label")
            err_cat = payload.get("error_category")
            ann_id = payload.get("annotator_id")
            conf = payload.get("confidence_rating")
            ev_text = payload.get("adjudicated_evidence_text")
            notes = payload.get("notes")

            ok, errs = validate_single_annotation(lbl, err_cat, ann_id, conf, ev_text)
            if not ok:
                self._send_json({"success": False, "error": "; ".join(errs)}, 400)
                return

            # Apply ONLY human-editable fields
            rec["label"] = str(lbl).strip().upper()
            rec["error_category"] = str(err_cat).strip().upper() if str(lbl).strip().upper() == "INVALID" else "NONE"
            rec["annotator_id"] = str(ann_id).strip()
            rec["confidence_rating"] = int(conf)
            rec["adjudicated_evidence_text"] = str(ev_text).strip()
            rec["notes"] = str(notes).strip() if notes else None

            # Atomically save
            atomic_save_json(self.file_path, data)

            self._send_json({"success": True, "record": rec}, 200)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReusableThreadingServer(socketserver.ThreadingTCPServer):
    """Multi-threaded TCP server with immediate socket reuse and daemon threads."""

    allow_reuse_address = True
    daemon_threads = True


def run_web_server(file_path: str, port: int = 8080, host: str = "127.0.0.1") -> None:
    """Launch local web server for interactive browser-based human annotation."""
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Annotation file not found: {p}")

    backup_path = create_backup(p)
    print("\n" + "=" * 70)
    print("  CLARION HUMAN RELATION ANNOTATION WEB CONSOLE")
    print("=" * 70)
    print(f"File   : {p.name}")
    print(f"Backup : {backup_path.name}")
    print(f"URL    : http://{host}:{port}/")
    print("=" * 70)
    print("Open the URL in your browser to begin annotating.")
    print("Press Ctrl+C in this terminal to stop the server.\n", flush=True)

    handler_factory = lambda *args, **kwargs: LocalAnnotationHTTPHandler(*args, file_path=p, **kwargs)

    import signal
    original_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    except (ValueError, AttributeError):
        pass

    try:
        with ReusableThreadingServer((host, port), handler_factory) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down web annotation server.", flush=True)
    finally:
        try:
            signal.signal(signal.SIGTERM, original_sigterm)
        except (ValueError, AttributeError):
            pass
