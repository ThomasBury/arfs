# 04 — GBM training, prediction, persistence, and benchmarks

**Status:** Reviewed; findings open. **Outcome:** [Completed report](04-training-and-benchmarks.md) and [executable evidence](evidence/04-training-and-benchmarks/checks.py). Fifteen new findings; 71 checks in each saved environment, including weighted native references and fresh-process model reloads. Strict and unmasked logs distinguish the newer environment's inherited GrootCV blocker. No library fixes applied.

**Purpose:** Check whether the training wrapper preserves the intended learning
task, data split, weights, prediction meaning, and saved model behavior, and
whether its benchmarking helpers make valid comparisons.

**Read first**

- [Queue and shared review procedure](README.md).
- [GradientBoosting and splitting/training helpers](../src/arfs/gbm.py).
- [Benchmarking helpers](../src/arfs/benchmark.py).
- LightForest/model-detection helpers in [utils.py](../src/arfs/utils.py).
- Call sites in [VariableImportance](../src/arfs/feature_selection/variable_importance.py)
  and [TreeDiscretizer](../src/arfs/preprocessing.py).
- Relevant cells in [regression](../docs/notebooks/arfs_regression.ipynb),
  [classification](../docs/notebooks/arfs_classification.ipynb), and
  [weighted non-normal losses](../docs/notebooks/arfs_non_normal_loss_and_sample_weight.ipynb).

**Review questions and checks**

1. Trace row positions through train/validation splitting for features, targets,
   weights, groups, and initial scores. Test non-default indexes, invalid lengths,
   stratification, and group isolation. Check how simultaneous split options are
   resolved and whether train-only fitting and validation remain distinct.
2. Compare small weighted regression and classification fits with direct native
   LightGBM references. Check task/objective/metric interpretation, categorical
   input, early stopping, custom evaluation, and parameter precedence. Verify
   supported initial-score behavior and document any backend-specific boundary.
3. Test labels versus probabilities versus raw scores for supported objectives,
   single-row predictions, unfitted calls, arrays/dataframes, and reordered or
   missing features. Check repeat fits do not inherit obsolete state or mutate
   caller parameters. Do not assume sklearn contracts for a wrapper that does
   not advertise them.
4. Save and reload a small trusted model in a fresh process under `/tmp` and
   compare predictions and required metadata. Inspect error handling, output
   paths, and overwrite behavior. Record the trusted-input boundary for joblib;
   do not load unknown artifacts or overwrite existing user files.
5. Trace `sklearn_pimp_bench` and `compare_varimp`: fitting versus evaluation
   data, weights in both places, task-specific scoring, estimator reuse, random
   state, and correspondence between reported importance and feature names.
   Confirm an issue with a small independent reference before reporting it.

**Known findings to carry forward**

- [Unit 1B](01b-all-relevant.md): weighted permutation importance and related
  shared paths have existing findings. Check which callers are actually affected.
- [Unit 2](02-preprocessing.md): tree parameter drift and class-label grouping
  are already demonstrated. Inspect the shared wrapper without counting those
  same discretizer defects twice.
- Read item 03's report once available for selector-specific dependencies.

**Deliverables and completion**

- Report: `reviews/04-training-and-benchmarks.md`.
- Evidence: `reviews/evidence/04-training-and-benchmarks/`.
- Cover training, prediction, persistence, and benchmark interpretation; record
  supported scenarios actually exercised and precise limits. Follow the queue's
  completion/update procedure.
- Next queued item: [05 — sampling and parallel execution](05-sampling-and-parallel.todo.md).

**Boundary:** No wrapper rewrite, model tuning project, or claim of comparative
model superiority from a smoke fixture. GPU/optional backends need a documented
support boundary if unavailable, not fabricated passing results.
