# Final repository structure

The root now contains only the project entry points and policy files: `README.md`, `PROJECT.md`, `CLARION_CURRENT_STATUS.md`, `CONTRIBUTING.md`, and `LICENSE`.

## Maintained areas

- `Clarion-Backend/` — API, services, models, configuration, and backend tests.
- `frontend/` — browser client.
- `docs/` — source documents and retained historical ZIP packages under `docs/archive/packages/`.
- `research/` — protocols, evidence indexes, datasets, bibliography, and archived research documents.
- `research_artifacts/` — raw/processed observations, scripts, inventories, and verified recovery snapshots.
- `reports/current/` — active software readiness and cleanup reports.
- `reports/historical/` — prior audits, logs, and recovery evidence.
- `scripts/`, `tests/` — repository-level checks and cross-project tests.

## Data and preservation

Source PDFs, runtime databases, model environments, research outputs, historical packages, and recovery snapshots are retained. Only two verified exact duplicates were removed: the extracted documentation folder whose files matched its retained ZIP, and the backend copy of a recovery patch whose SHA-256 matched the archived canonical copy. No Git reset, clean, or checkout operation was used.

For authoritative status and remaining readiness gates, see [CLARION_CURRENT_STATUS.md](../../CLARION_CURRENT_STATUS.md) and [SOFTWARE_RELEASE_READINESS.md](SOFTWARE_RELEASE_READINESS.md). The cleanup inventory and action log are [here](REPOSITORY_CLEANUP_INVENTORY.md) and [here](REPOSITORY_CLEANUP_LOG.md).
