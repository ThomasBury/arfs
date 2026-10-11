# 03 — Selector contracts and remaining selectors

**Status:** Reviewed; findings open. **Outcome:** [Completed report](03-selector-contracts.md) and [executable evidence](evidence/03-selector-contracts/checks.py). Fourteen new findings and the inherited export failure; 73 checks in each saved environment, plus isolated summary checks with styling dependencies. No library fixes applied.

**Purpose:** Establish whether thresholds, support masks, selected columns, and
pipeline summaries mean what the public API says, and review `VariableImportance`
and collinearity elimination beyond their previously reviewed helper functions.

**Read first**

- [Queue and shared review procedure](README.md).
- [BaseThresholdSelector](../src/arfs/feature_selection/base.py).
- [Unsupervised selectors](../src/arfs/feature_selection/unsupervised.py).
- [VariableImportance](../src/arfs/feature_selection/variable_importance.py).
- [Summary integration](../src/arfs/feature_selection/summary.py) and
  [public exports](../src/arfs/feature_selection/__init__.py).
- [Existing threshold tests](../tests/test_featselect.py) and the source cells in
  [basic feature selection](../docs/notebooks/basic_feature_selection.ipynb).

**Review questions and checks**

1. Use hand-calculable frames to check missing ratios, unique counts, categorical
   cardinality, and both sides of every threshold, including equality. Cover
   constant/all-missing columns, no columns selected, and invalid thresholds.
   Resolve the documented semantics before calling a boundary a bug.
2. Check `fit`, `fit_transform`, `transform`, `get_support`, and feature names
   agree. Exercise unfitted use, repeated fit, parameter updates, reordered or
   missing columns, duplicate names, and supported input types. Verify index,
   dtype, input preservation, and weight forwarding through a real pipeline.
3. Trace collinearity elimination with negative associations, asymmetric
   associations, ties, and disconnected features. Use small matrices or frames
   whose retained set can be justified; do not repeat unit 1A's full statistics
   review. Check that support metadata describes the actual output.
4. Review `VariableImportance` aggregation, normalization, cumulative threshold
   selection, zero/negative/nonfinite importance, task choice, weights, and
   category handling. Include a small real fit and a controlled arithmetic
   reference; separate importance calculation errors from selection errors.
5. Exercise `make_fs_summary` with a real multi-step pipeline, a transformer
   between selectors, and an empty selection. Ensure rows refer to the correct
   original features and each stage's actual decisions. Check public exports.

**Known findings to carry forward**

- [Unit 0](00-baseline.md): missing export, misleading threshold tests, and
  broken synthetic fixture. They remain findings, not assumed repairs.
- [Unit 1A](01a-statistical-foundations.md): association/relevance defects may
  affect collinearity results. Cross-reference the shared cause.
- [Unit 2](02-preprocessing.md): ordinal output metadata and input/state
  defects can affect these pipelines.

**Deliverables and completion**

- Report: `reviews/03-selector-contracts.md`.
- Evidence: `reviews/evidence/03-selector-contracts/`.
- Account for each question above with evidence or a precise remaining limit.
  Separate existing defects from new findings, and state any unresolved API
  decision. Follow the queue's completion/update procedure.
- Next queued item: [04 — training and benchmarks](04-training-and-benchmarks.todo.md).

**Boundary:** Do not repair selectors or repeat the all-relevant/Lasso selection
screens. Training-wrapper internals belong to item 04; trace them here only as
needed to explain selector behavior.
