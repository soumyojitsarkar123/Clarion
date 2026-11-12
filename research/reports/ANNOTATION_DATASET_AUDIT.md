# Annotation dataset audit

**Status: BLOCKED for quality estimates; source candidates audited.**

Canonical backend relation dataset: 1,562 candidate rows; 1,562 have `is_valid = NULL`. An alternate root database contains a separate 37-candidate collection and must not be silently merged. Existing annotation exports include 200 primary and 50 overlap IDs; all 50 overlap IDs also occur in the primary set. These are queued assignments/templates, not completed independent annotations. No annotator labels, adjudications, or agreement values were found in the reviewed artifacts.

The current label guidance includes VALID / INVALID / AMBIGUOUS and error classes such as unsupported assertion, direction inversion, type mismatch, spurious co-occurrence, and noisy entity. It needs an explicit distinction for contradictory, incomplete, and insufficient evidence and rules for relation direction, multi-sentence evidence, abstention, and source quote requirements. Freeze a versioned guideline before annotation. Assign independent annotators blind to model confidence; do not treat the overlapped exports as independent until annotation logs demonstrate separate human work.

Required next steps: reconcile DB identity and hashes; deduplicate by immutable candidate ID; randomize a stratified sample with recorded seed; recruit two annotators; train on a disjoint pilot; freeze guideline; independently label; adjudicate disagreements; preserve raw labels and adjudicated labels separately. Agreement is **NOT EVALUATED** because no independent labels exist.
