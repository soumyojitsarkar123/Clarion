# Human Relationship Annotation Protocol

## Status

Protocol artifacts exist, but no completed annotations were found. Current source: `Clarion-Backend/data/ground_truth/annotation_queue_batch1.json` and `annotation_queue_batch1_overlap_b.json`. The primary batch declares 200 records, stratified by document and relation type with seed 42. The overlap batch declares 50 records; all 50 relation record IDs are in the primary batch and are intended for independent second annotation. Backend `relation_dataset` currently has 1,562 rows and all 1,562 `is_valid` values are NULL. Queue exports are not gold labels.

## Annotation procedure

1. Annotators judge only the source passage in `chunk_context`; hide `llm_confidence`, cooccurrence and semantic similarity during judgment.
2. Assign `VALID`, `INVALID`, or `AMBIGUOUS`. `AMBIGUOUS` is not forced into binary metrics until adjudication rules are defined.
3. For invalid relations select the closest error category; include an evidence excerpt and concise rationale. Do not infer facts absent from the source text.
4. Two annotators independently label the 50-record overlap without discussing items. Resolve disagreements only after preserving both original labels. Keep an adjudication record separate from raw annotations.
5. Pilot the instructions on a small subset, record unclear cases, then version the protocol before the full batch. Never tune labels to attain a target kappa.

## Existing schema and limitations

Existing labels: `VALID`, `INVALID`, `AMBIGUOUS`. Existing error categories: `NONE`, `UNSUPPORTED_ASSERTION`, `DIRECTION_INVERSION`, `TYPE_MISMATCH`, `SPURIOUS_COOCCURRENCE`, `NOISE_ENTITY`. The schema does not explicitly distinguish insufficient/incomplete evidence or an incorrect entity; clarify whether these map to existing categories or add versioned categories before annotation. Existing record fields include relation/document IDs, entities, relation type, source chunk IDs/context, label, error category, annotator ID, confidence rating, adjudicated evidence and notes. Add annotation timestamp in a versioned capture format; do not mutate original source queues.

## Agreement and metrics

For two independent annotators, report the 3x3 confusion matrix, class marginals, observed agreement, expected agreement and Cohen's kappa with an interval only when justified by the sample/design. The 50 overlap items are the independent unit; report document clustering and the small overlap sample limitation. If missing labels or more annotators materially change assumptions, assess Krippendorff's alpha. Report adjudicated VALID/INVALID/AMBIGUOUS counts separately. Compute precision/recall/F1 only after a documented binary mapping and adjudication, with TP/FP/FN definitions and class imbalance.

No labels, confusion matrix, kappa, precision, recall or F1 are currently available. Annotation is **NOT EVALUATED**; the quality experiment is **BLOCKED**.
