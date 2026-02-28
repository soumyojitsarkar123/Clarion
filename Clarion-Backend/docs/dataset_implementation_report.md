# Relation Dataset Collection - Implementation Report

**Date:** February 20, 2026  
**Status:** VERIFIED WORKING

---

## Overview

This report documents the implementation of automatic logging of intermediate mapping outputs to create a training dataset for relation validation experiments.

---

## Verified Test Results

### Dataset API Tests

| Test | Status | Details |
|------|--------|---------|
| Export JSON | PASS | HTTP 200, returns dataset |
| Export CSV | PASS | HTTP 200, downloadable CSV |
| Get Relations | PASS | HTTP 200, paginated list |
| Dataset Stats | PASS | Returns counts/averages |
| Filter by Type | PASS | Filters correctly |
| Health Check | PASS | Returns healthy |

### Dataset Logging Tests

| Test | Status | Details |
|------|--------|---------|
| Async Logging | PASS | Non-blocking queue |
| Record Creation | PASS | All fields populated |
| Co-occurrence Score | PASS | Calculated from chunks |
| Export Format | PASS | JSON and CSV working |

---

## Implementation Summary

### 1. RelationDatasetService (`services/relation_dataset_service.py`)

**Features:**
- Async non-blocking logging (doesn't slow pipeline)
- Captures during mapping stage:
  - extracted concepts
  - inferred relations  
  - chunk context used
  - LLM confidence
  - co-occurrence score (calculated)
  - embedding similarity (when available)
- SQLite storage at `data/relation_dataset.db`

### 2. Dataset Storage Format

```sql
CREATE TABLE relation_dataset (
    record_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    concept_a TEXT NOT NULL,
    concept_b TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    llm_confidence REAL NOT NULL,
    cooccurrence_score REAL,
    semantic_similarity REAL,
    chunk_context TEXT NOT NULL,
    source_chunk_ids TEXT NOT NULL,
    is_valid INTEGER,
    created_at TEXT NOT NULL,
    metadata TEXT
);
```

### 3. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dataset/relations/export` | GET | Export dataset (JSON/CSV) |
| `/dataset/relations` | GET | List relation records |
| `/dataset/relations/stats` | GET | Dataset statistics |
| `/dataset/relations/validate` | PATCH | Update validation label |
| `/dataset/relations/{doc_id}` | DELETE | Delete document records |

---

## Dataset Fields

| Field | Type | Description |
|-------|------|-------------|
| record_id | string | Unique record ID |
| document_id | string | Source document |
| relation_id | string | Relation ID from knowledge map |
| concept_a | string | First concept |
| concept_b | string | Second concept |
| relation_type | string | Type (prerequisite, part-of, etc.) |
| llm_confidence | float | LLM confidence (0-1) |
| cooccurrence_score | float | Chunk proximity score |
| semantic_similarity | float | Embedding similarity |
| chunk_context | string | Source text context |
| source_chunk_ids | list | Source chunk IDs |
| is_valid | bool | Validation label (nullable) |
| created_at | datetime | Creation timestamp |

---

## Usage Examples

### Export Dataset (JSON)
```bash
GET /dataset/relations/export?document_id=<doc_id>&format=json
```

### Export Dataset (CSV)
```bash
GET /dataset/relations/export?format=csv
```

### Get Statistics
```bash
GET /dataset/relations/stats
```

Response:
```json
{
  "total_records": 150,
  "labeled_records": 45,
  "unlabeled_records": 105,
  "relation_types": {
    "prerequisite": 30,
    "part-of": 45,
    "similar-to": 35,
    ...
  },
  "average_llm_confidence": 0.82,
  "average_cooccurrence": 0.65
}
```

### Update Validation Label
```bash
PATCH /dataset/relations/validate
Content-Type: application/json

{
  "record_id": "uuid-here",
  "is_valid": true
}
```

---

## Integration with Pipeline

The dataset logging is automatically triggered during the mapping stage:

1. Pipeline executes mapping stage
2. Knowledge map with relations is created
3. `RelationDatasetService.log_relation_batch()` is called (async, non-blocking)
4. Records are queued and flushed to database asynchronously
5. Pipeline continues without waiting

This ensures dataset collection does not impact pipeline performance.

---

## Files Created/Modified

1. **`services/relation_dataset_service.py`** (Created)
   - RelationDatasetService class
   - RelationDatasetRecord model
   - Async logging methods
   - Export functionality

2. **`routers/dataset.py`** (Created)
   - Dataset export endpoints
   - Statistics endpoints
   - Validation update endpoints

3. **`services/processing_pipeline.py`** (Modified)
   - Added RelationDatasetService integration
   - Added dataset logging call in mapping stage

4. **`routers/__init__.py`** (Modified)
   - Added dataset_router

5. **`main.py`** (Modified)
   - Registered dataset router

---

## Conclusion

The relation dataset collection system is **VERIFIED WORKING** and provides:

- Automatic logging during mapping stage
- Non-blocking async operations
- Complete dataset fields for training
- JSON and CSV export formats
- Filtering and statistics
- Validation labeling capability

The system is ready for relation validation experiments.
