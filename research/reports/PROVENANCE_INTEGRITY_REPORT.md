# Provenance integrity report

**Status: PARTIALLY VERIFIED at segment-to-chunk; NOT IMPLEMENTED at PDF-source level.**

## Current evidence

The isolated corpus run produced 170 embedding segments and 170 FAISS vectors. Existing embedding code stores `source_chunk_id`, `segment_index`, and `char_start/char_end`; offsets refer to the input chunk string. Existing tests/report claim exact segment reconstruction and SQLite/FAISS alignment in selected fixtures. This supports an internal segment-to-chunk integrity property only.

## Missing coordinate links

| Transition | Current representation | Result |
|---|---|---|
| PDF → page extraction | Parser loops pages but joins nonempty text; page boundary/source offset map is not persisted. | Missing page-local mapping. |
| extracted → normalized text | Regex normalization rewrites text and returns only a string. | No transform map; source offsets cannot be recovered reliably. |
| normalized → raw/retained chunk | Chunker splits by words/sections, merges undersized chunks; `position_index` is word-like and adjusted cumulatively. | No source character interval; overlap/merge complicates a scalar interval. |
| retained chunk → segment | Segment `char_start/end` local to chunk text. | Verified local span schema; not PDF coordinates. |
| segment → vector | SQLite segment identity aligns with FAISS vector metadata in isolated indexing. | Alignment verified for corpus run. |
| vector → retrieved evidence | Retrieval resolves segment and parent chunk, returns segment text and parent section/position. | No page or original source interval in result. |

## Integrity invariants for a future implementation

Every source span should include immutable source SHA-256, page index (1-based), coordinate convention, extraction backend/version, half-open page-local character interval, and text hash. Normalization should retain a mapping relation, not a presumed one-to-one offset; rewrites may map an output interval to multiple source intervals or be explicitly marked ambiguous. Chunk merges and overlaps must retain interval sets. Segment coordinates must be composed through that mapping. Retrieval results must surface the source interval and quote, and a validator must verify the quote against the page extraction.

## Status by claim

- SQLite/FAISS count and ordering alignment: **VERIFIED** for the isolated run.
- Segment character spans relative to their source chunk: **PARTIALLY VERIFIED** by selected fixtures and code schema.
- Complete text reconstruction for all corpus segments: **PARTIALLY VERIFIED**; prior fixture-level tests do not substitute for a corpus-wide source comparison.
- PDF page-to-retrieved-span traceability: **NOT IMPLEMENTED**.
- Visual glyph coordinates or OCR ground truth: **NOT EVALUATED**.
- End-to-end provenance method as a contribution: **NOT ESTABLISHED**.

The earlier root `PROVENANCE_AND_RETRIEVAL_VALIDATION.md` uses “complete” and “100%” language for selected test contracts. Read it as historical test evidence, not proof of PDF source coordinates; its broader phrasing is not supported by the present schema.
