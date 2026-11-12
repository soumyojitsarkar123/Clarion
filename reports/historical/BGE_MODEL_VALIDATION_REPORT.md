# BGE Model Validation Report

## Status: PARTIALLY VERIFIED

Command: `.\.venv-validation\Scripts\python.exe scripts\validate_bge_segmentation.py`

- Model: `BAAI/bge-large-en-v1.5`, loaded locally on CPU.
- Tokenizer/model input limit: 512 tokens, including special tokens.
- Embedding dimension: 1024.
- Under-limit input: 511 tokens, one segment.
- Exact-limit input: 512 tokens, one segment.
- Over-limit input: 513 tokens, two segments of 512 and 3 tokens.
- Unicode/punctuation input: 1,302 tokens, three segments of 512, 512, and 282 tokens.
- Every nonempty segment was embedded by the actual BGE model and was within the limit.

The run exposed and fixed two real implementation defects: use of a missing tokenizer API for special-token counting, and inability to obtain offset mappings from the model's default tokenizer. The service now selects a fast tokenizer and dynamically reduces a candidate boundary when a token offset ends within a wordpiece.

Empty and whitespace-only sources produce zero embedding segments intentionally. They must be rejected or omitted at document-ingestion validation; they are not silently embedded as fake content.
