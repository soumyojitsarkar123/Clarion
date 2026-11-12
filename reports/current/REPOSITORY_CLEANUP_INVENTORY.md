# Repository cleanup inventory

**Scope:** post-organization working tree, excluding virtual environments, caches, Git internals, and the two inventory CSVs themselves. The machine-readable inventory records included paths, byte sizes, SHA-256, status, and classification.

- Final inventory: [`repository_inventory_post_cleanup.csv`](../../research_artifacts/processed/repository_inventory_post_cleanup.csv), 1,577 files.
- ZIP member inventory: [`archive_content_inventory_post_cleanup.csv`](../../research_artifacts/processed/archive_content_inventory_post_cleanup.csv), 671 entries; all retained packages passed integrity checks.
- Frozen pre-clean exact worktree: [`engineering-frozen-presnapshot-2026-11-11`](../../research_artifacts/engineering-frozen-presnapshot-2026-11-11/README.md).
- Documentation and verification package: [`Clarion_Clean_Repository_Documentation_and_Verification_2026-11-11.zip`](../../docs/archive/packages/Clarion_Clean_Repository_Documentation_and_Verification_2026-11-11.zip) with adjacent SHA-256 file. The self-referential inventory CSVs are kept in the repository and omitted from this ZIP.

## Disposition

Preserved source PDFs, application and research databases, runtime environments, models/caches, benchmark records, research files, historical reports, both distinct validated legacy ZIP packages, and recovery snapshots. Moved current reports to `reports/current/`, older reports to `reports/historical/`, bibliography and evidence tables to `research/`, and ZIP deliverables to `docs/archive/packages/`.

Removed only two verified exact duplicates after the frozen snapshot: the extracted documentation-consolidation folder matched its retained ZIP entry-for-entry by SHA-256, and the backend recovery patch matched the archived patch byte-for-byte. No Git reset, clean, checkout, or broad cache/database/source deletion was used.

Virtual environments and model caches remain on disk and are excluded from the file inventory; temporary pytest output was removed after verification attempts.

