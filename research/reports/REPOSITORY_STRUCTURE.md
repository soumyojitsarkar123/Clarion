# Repository research structure

Research workspace: [`research/`](../README.md). Existing research evidence remains under [`research_artifacts/`](../../research_artifacts/README.md) to preserve prior evidence paths and snapshots.

| Required area | Current location/state |
|---|---|
| raw | `research_artifacts/raw/`; original PDFs remain in `docs/` |
| processed | `research_artifacts/processed/` |
| protocols | `research/protocols/` |
| annotation | `research_artifacts/annotation/`; labels not yet collected |
| experiments | `research_artifacts/` run-specific folders; no controlled retrieval/performance run |
| results | `research/reports/` and legacy root reports; numerical outcome tables limited to prior descriptive outputs |
| figures/tables | `research_artifacts/figures/`, `research_artifacts/tables/` |
| scripts | `research_artifacts/scripts/`; no relocation to avoid altering traceability |
| reports | `research/reports/`, with earlier reports retained at repository root |
| manuscript | `research/manuscript/`; evidence-limited working draft |

The existing codebase and reports contain extensive pre-existing worktree modifications. No consolidation move was attempted. Cleanup inventory/log and pre-research snapshot remain at the repository root and under `research_artifacts/2026-11-11_research-validation-presnapshot/` respectively. Current `.gitignore` additions were limited to disposable caches; research data and PDFs are not ignored.
