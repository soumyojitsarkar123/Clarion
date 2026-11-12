# Security and operational readiness

**Classification: NOT VERIFIED for deployment.** This is a code-path review and limited isolated migration test, not a penetration test or security certification.

## Code-path observations

- Upload validates the filename extension, non-empty payload, and configured size after reading the entire upload into memory. It does not verify file signatures/MIME content before parsing.
- The inspected upload, query, and document routes do not show authentication or authorization enforcement. Treat the current API as suitable only for a trusted local environment unless a protected reverse proxy is separately configured and validated.
- A configured upload size limit exists, but request-body streaming limits, rate limiting, and concurrent-upload resource bounds were not demonstrated.
- Filenames are stored as metadata and export paths use sanitization in dataset services; traversal/security behavior was not tested adversarially end to end in this run.
- Sensitive-prompt/source logging and secret handling were not comprehensively audited. Provider/API keys are configuration inputs; no secret scan or production log review was performed.
- SQLite services create tables on startup. The dataset generation audit table now has an additive, idempotent migration tested on an isolated old-schema DB. A whole-database production migration framework, rollback rehearsal, or concurrent migration test is not established.
- A clean isolated migration test completed and passed. A production database copy/backup restore test, `integrity_check` after restore, corruption recovery, and restart-after-partial-indexing test were not performed.

## Operational gates

| Control | Status |
|---|---|
| Authentication and authorization | NOT VERIFIED; no enforcement observed in inspected routes |
| Upload extension/size checks | PARTIALLY VERIFIED by code inspection; MIME/signature, streaming, and adversarial tests incomplete |
| Path traversal | PARTIALLY VERIFIED by safe export-stem helpers; upload-to-export test incomplete |
| Rate limiting / concurrency control | NOT VERIFIED |
| Secret protection / sensitive logging | NOT VERIFIED comprehensively |
| Isolated additive dataset audit migration | VERIFIED by one passing test |
| Production schema migration/rollback | NOT VERIFIED |
| Backup restore and corruption recovery | NOT VERIFIED |
| Shutdown/restart and concurrent uploads | NOT VERIFIED |
| Formal security certification | NOT CLAIMED |

Do not classify Clarion as production-secure or operationally recoverable until the unverified controls are exercised in a disposable deployment and restore environment.
