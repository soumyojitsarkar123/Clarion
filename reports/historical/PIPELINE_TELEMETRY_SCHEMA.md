## Final verification addendum — 2026-11-11

This current addendum supersedes status statements recorded earlier in this report. Source identity: branch `main`; base and HEAD `ebca6ed0125c7131c05c945620b23a1970844b92`; tracked-diff SHA-256 `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`.

Current results: backend suite **304 passed, 2 skipped, 0 failed** in 90.31s; targeted backend contracts **19 passed**; frontend timeline contracts **13 passed**; `node --check frontend/app.js` passed. The ordinary suite’s two skips remain recorded as skips in that run. Separately, the two live Ollama tests passed when run with host-daemon access; they are not included in the ordinary suite’s pass count.

Chromium rendered the actual frontend against deterministic stub API responses. It verified seven canonical stages, requested labels and duration formats, unavailable `--` metrics, no duplicate `llm_analysis`, and completed/running/failed/pending states. This verifies frontend rendering, not live-backend browser E2E; live-backend browser E2E remains unverified.

Live Ollama `qwen3.5:4b` smoke validation completed two chunks and four schema-validated records. Request durations were 8.252s and 15.370s in this run. Strict export succeeded; a separate controlled fault-injection run verified retry, partial accounting, strict-export refusal (HTTP 409), and resume with checkpoint reuse. This is smoke and reliability evidence, not a controlled performance benchmark or a claim of general performance improvement.

SHA-256 checks for all three production databases matched before and after isolated integration/full-suite testing. Overall project classification: **PARTIALLY VERIFIED**.

---

# PIPELINE TELEMETRY SCHEMA

## Current remediation revalidation (2026-11-11)

The current serializer retains the seven canonical stages and genuine base metrics. Pending/unmeasured derived counts are omitted or null; throughput is null when duration or count is unavailable/invalid, while a measured zero count with a valid positive duration remains `0.0`. `analysis_mode` and `model` remain available on mapping even when pending. Snapshot `Clarion_Verification_Snapshot_2026-11-11` is based on `ebca6ed0125c7131c05c945620b23a1970844b92`; tracked-diff SHA-256 is `5FCFC6610885DDF2B9936F82E57E1B7D0DAD4DA7CB0BB3223D8BEF40668873CB`. Current serializer/recovery/persistence and integration checks passed. Full backend suite: 304 passed, 2 skipped, 0 failed in 90.31s. Chromium validated the actual timeline UI using stub API data; live-backend browser E2E remains unverified. Separate live Ollama smoke and dataset generation passed; controlled performance benchmarking remains incomplete.

**Version**: 2.0  
**Date**: 2026-11-11  
**Git Commit**: `ebca6ed0125c7131c05c945620b23a1970844b92`  
**Status**: AUTHORITATIVE — Backend drives frontend rendering

---

## 1. Authoritative Stage Registry

The canonical pipeline stage sequence is defined in `Clarion-Backend/services/pipeline_observability.py`:

```python
PIPELINE_STAGES = [
    "ingestion",
    "chunking",
    "embedding",
    "mapping",
    "graph_building",
    "evaluation",
    "hierarchy",
]
```

**Stage count**: 7  
**Excluded**: `llm_analysis` (removed — was a synthetic presentation artifact cloned from mapping)

---

## 2. Pipeline Status API Response Schema

**Endpoint**: `GET /system/status/{document_id}` (or embedded in `/upload` polling)

```jsonc
{
  "document_id": "string",
  "status": "pending | running | completed | failed | degraded",
  "analysis_mode": "llm | deterministic_fallback",
  "active_model": "string | null",
  "total_duration_sec": "number | null",   // null until pipeline completes
  "started_at": "ISO-8601 string",
  "completed_at": "ISO-8601 string | null",
  "stages": {
    // Keyed by stage identifier from PIPELINE_STAGES
    "ingestion": { /* StageRecord */ },
    "chunking":  { /* StageRecord */ },
    "embedding": { /* StageRecord */ },
    "mapping":   { /* StageRecord */ },
    "graph_building": { /* StageRecord */ },
    "evaluation": { /* StageRecord */ },
    "hierarchy":  { /* StageRecord */ }
  }
}
```

---

## 3. StageRecord Schema

```jsonc
{
  "name": "string",             // matches PIPELINE_STAGES key
  "status": "pending | running | completed | failed | fallback",
  "progress": 0,                // 0–100 integer; 0 when pending
  "start_time": "ISO-8601 | null",
  "end_time": "ISO-8601 | null",
  "duration_sec": "number | null",  // null until stage completes; NEVER coerced to 0.0
  "input_size": "number | null",    // null if not yet recorded
  "output_size": "number | null",   // null if not yet recorded
  "quality_metrics": {
    // Stage-specific supplementary metrics (see §4)
  },
  "errors": ["string"]          // empty array when no errors
}
```

> [!IMPORTANT]
> `duration_sec`, `input_size`, and `output_size` are **`null`** (JSON null) for stages
> that have not yet executed. They are **never** coerced to `0.0` or `0` for pending stages.
> The frontend must render `null` as `--`, not as `"0.00s"` or `"0"`.

---

## 4. Stage-Specific Quality Metrics

### ingestion
| Field | Type | Description |
|-------|------|-------------|
| `char_count` | int\|null | Total extracted character count |

### chunking
| Field | Type | Description |
|-------|------|-------------|
| `chunk_count` | int | Number of chunks produced (also in `output_size`) |

### embedding
| Field | Type | Description |
|-------|------|-------------|
| `throughput_chunks_sec` | float | Chunks embedded per second |
| `device` | string | Compute device: `"cpu"` or `"cuda"` |
| `embedding_dim` | int\|null | Embedding dimension (1024 for BAAI/bge-large-en-v1.5) |

### mapping
| Field | Type | Description |
|-------|------|-------------|
| `analysis_mode` | string | `"llm"` or `"deterministic_fallback"` |
| `model` | string | Active model name (e.g., `"qwen2.5:7b"`) |
| `relation_count` | int\|null | Number of relations extracted |

### graph_building
| Field | Type | Description |
|-------|------|-------------|
| `node_count` | int | Knowledge graph node count |
| `edge_count` | int | Knowledge graph edge count |

### evaluation
| Field | Type | Description |
|-------|------|-------------|
| `candidates_evaluated` | int\|null | Number of candidate relations evaluated |
| `risk_flags` | int\|null | Number of hallucination/risk flags detected (0 = no flags) |

> [!NOTE]
> `evaluation.output_size = 0` means **zero risk flags detected**, not zero items processed.
> The input (candidate count) and output (flag count) represent different quantities.

### hierarchy
| Field | Type | Description |
|-------|------|-------------|
| `hierarchy_levels` | int\|null | Number of hierarchy levels generated |
| `depth` | int\|null | Alias for hierarchy_levels |

> [!NOTE]
> `hierarchy.input_size` represents **total graph elements** (nodes + edges), not nodes alone.
> `hierarchy.output_size` represents **hierarchy depth/levels**, not element count.

---

## 5. Frontend Display Mapping

The frontend `resolveStageMetrics(stageName, s)` function maps backend fields to display labels:

| Stage | Input Label | Input Source | Output Label | Output Source |
|-------|-------------|--------------|--------------|---------------|
| ingestion | File bytes | `s.input_size` | Characters extracted | `s.output_size \|\| qm.char_count` |
| chunking | Characters | `s.input_size` | Chunks produced | `s.chunk_count \|\| s.output_size` |
| embedding | Chunks embedded | `s.input_size \|\| qm.chunk_count` | Vectors (dims) | `s.output_size + qm.embedding_dim` |
| mapping | Chunks analysed | `s.input_size` | Relations extracted | `s.output_size \|\| qm.relation_count` |
| graph_building | Relations mapped | `s.input_size` | Nodes / Edges | `s.node_count + "/" + s.edge_count` |
| evaluation | Candidate relations evaluated | `s.input_size \|\| qm.candidates_evaluated` | Risk flags detected | `s.output_size \|\| qm.risk_flags` |
| hierarchy | Graph elements processed | `s.input_size \|\| qm.node_count + qm.edge_count` | Hierarchy levels generated | `s.output_size \|\| qm.hierarchy_levels` |

---

## 6. Duration Formatting Rules

Implemented in `frontend/app.js` as `formatStageDuration(durSec)`:

| Condition | Display |
|-----------|---------|
| `null` or `undefined` | `"--"` |
| `>= 0ms` and `< 1000ms` | `"{N}ms"` (rounded to nearest ms) |
| `>= 1000ms` | `"{N.NN}s"` (2 decimal places) |

Examples:
- `null` → `"--"` (never `"0.00s"`)
- `0.003` → `"3ms"`
- `0.125` → `"125ms"`
- `1.25` → `"1.25s"`
- `26.85` → `"26.85s"`

---

## 7. Stage Status Values

| Value | Meaning |
|-------|---------|
| `"pending"` | Stage has not begun |
| `"running"` | Stage is actively executing |
| `"completed"` | Stage finished successfully |
| `"failed"` | Stage encountered an unrecoverable error |
| `"fallback"` | Stage completed using deterministic fallback (mapping stage only) |

---

## 8. Schema Versioning

This document defines **Schema Version 2.0**.  
Changes from Version 1.x:
- Removed `llm_analysis` synthetic stage
- Changed `duration_sec` default from `0.0` to `null`
- Changed `input_size` default from `0` to `null`
- Changed `output_size` default from `0` to `null`
- Added `name` field to each StageRecord
- Added `start_time`, `end_time`, `progress`, `errors` to all StageRecords
- Added `quality_metrics` passthrough (previously stripped)

**API backward compatibility**: The `stages` dict keys are unchanged. Consumers testing `duration_sec == 0.0` for "not executed" must be updated to test `duration_sec is None`.
