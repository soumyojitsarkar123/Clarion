# Database Migration Report

## Status: PARTIALLY VERIFIED

`embedding_segments` is created with `CREATE TABLE IF NOT EXISTS`; existing databases remain readable. New segment rows include a deterministic segment ID, source chunk ID, document ID, ordinal, character span, and segment content.

Historical vector indexes contain original chunk IDs and no `embedding_segments` rows. Retrieval supports those as legacy entries, but this means old and new indexing strategies must not coexist for the same document. Reindex every existing document with `delete_embeddings(document_id)` followed by processing before enabling segment indexing in production. `delete_embeddings` removes both embedding metadata and segment provenance, then deletes the vector index, providing the rollback boundary. Back up `clarion.db` and `data/vectorstore/<document-id>*` before reindexing.

No production database artifacts were modified during validation; the test used a temporary database.
