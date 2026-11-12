# Security and recovery verification

**Classification: NOT VERIFIED for release.** Isolated additive schema rollback and fail-closed model/index checks passed. No security review, production migration rehearsal, backup restore, or failure-injection campaign was completed.

- Migration tables/columns are additive and transactional. Fresh schema, legacy row preservation without fabricated provenance, idempotent application, and injected mid-migration rollback are covered in isolated SQLite tests.
- FAISS loading verifies model/revision identity and rejects an ordered vector-ID map mismatch. Query encoding refuses to substitute a deterministic fallback vector when a real-model index is selected but its model is unavailable.
- Reindex/vector-map and deletion behavior is unit-tested with an isolated fake index; actual isolated FAISS create/search and map consistency passed for the 100-vector PDF run.
- Authentication/authorization, rate limits, upload abuse controls, secrets handling, SQLite restore from backup, disk-full/database failure, interrupted FAISS writes, and production rollback were not reviewed or simulated.
- Production DB was not accessed by the experiments. Current byte hashes are recorded for traceability; a pre-run SHA baseline was not captured, so no before/after equality claim is made.
