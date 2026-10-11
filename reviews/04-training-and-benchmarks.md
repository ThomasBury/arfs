This review covers **GBM training, prediction, persistence, and benchmarks**, item 04 in the queue. It adds a report and executable evidence. No library source, established tests, dependencies, lockfile, CI, or earlier reports were changed.

Reviewed commit: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11. Scope includes `gbm.py`, `benchmark.py`, LightForest/model-detection helpers, direct selector/discretizer callers, and relevant notebook source. Expected use is CPU tabular modelling with pandas or documented NumPy inputs, optional observation weights, and trusted local model files.

**Result: 15 new findings; remediation remains open.** Prioritize group isolation, prediction schemas, invalid training inputs, and initial-score state/persistence. Ordinary weighted fits agree with independent native LightGBM references. The standalone permutation benchmark correctly weights both fitting and scoring; the previously reported selector permutation-weight defect does not apply to that helper.

**Evidence and environment**

The [71 checks](evidence/04-training-and-benchmarks/checks.py) use unit 0's installed wheels, from temporary working directories outside the checkout. Fresh hashes verify six installed modules against the reviewed source. Both saved environments retain `setuptools==80.9.0`, unit 0's import workaround; these runs do not establish clean-install success. “Newer” means the saved snapshot, not a fresh dependency resolution.

| Snapshot | Python | pandas | NumPy | sklearn | LightGBM | Matplotlib | Strict markers | Markers disabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen `locked-env` | 3.12.7 | 2.2.3 | 2.0.2 | 1.6.1 | 4.6.0 | 3.9.4 | 41 passed, 30 xfailed; exit 0 | 30 failed, 41 passed; exit 1 |
| Newer `wheel-env` | 3.12.7 | 3.0.6 | 2.5.4 | 1.9.1 | 4.7.0 | 3.11.2 | 40 passed, 31 xfailed; exit 0 | 31 failed, 40 passed; exit 1 |

Environment identities, source hashes, and exact dependency pins: [frozen](evidence/04-training-and-benchmarks/frozen-environment.json), [frozen pins](evidence/04-training-and-benchmarks/frozen-environment.requirements.txt), [newer](evidence/04-training-and-benchmarks/current-environment.json), [newer pins](evidence/04-training-and-benchmarks/current-environment.requirements.txt). Commands, execution times, return codes, and evidence hashes: [frozen runs](evidence/04-training-and-benchmarks/frozen-runs.json), [newer runs](evidence/04-training-and-benchmarks/current-runs.json).

Logs: [frozen checks](evidence/04-training-and-benchmarks/frozen-checks.txt), [frozen unmasked failures](evidence/04-training-and-benchmarks/frozen-failures.txt), [newer checks](evidence/04-training-and-benchmarks/current-checks.txt), [newer unmasked failures](evidence/04-training-and-benchmarks/current-failures.txt). Expected-failure markers are strict: an unexpected pass fails the evidence run. The newer GrootCV comparison is blocked by unit 0's pandas read-only shadow-array defect, rather than reaching F14; the marker and report distinguish that inherited failure.

Training and reload checks use real models. Weighted benchmark checks observe the real permutation calculation and substitute only the boxplot boundary to isolate F13. The vocabulary check observes a real encoder fit and compares it with an independent train-only encoder. Comparison checks observe real selector/model fits while disabling plotting. No ARFS implementation is patched to make a failing path pass.

**Confirmed behavior and support boundaries**

| Surface | What the evidence establishes |
| --- | --- |
| Splits and row alignment | Plain, stratified, and grouped splits separately match sklearn references. Features, targets, weights, and initial scores are sliced by row position with `iloc`, including unrelated non-default Series indexes. This is positional alignment, not an index join. Separate group splitting isolates groups. Length mismatches are rejected. F1 covers simultaneous options. |
| Train versus validation | The wrapper uses a fixed 80/20 split and split seed 42; it does not refit on all rows after early stopping. Returned validation features match the reference. A controlled metric that improves on training and worsens on validation stops at iteration 1 after the ten-round patience period, demonstrating validation-based stopping. No independent test set is created. |
| Weighted training | Regression, binary, multiclass, and Poisson fits reproduce predictions, best iteration, and validation metrics from independently constructed native datasets. An unweighted fit differs in every fixture, so the comparison can detect lost weights. Caller frames, targets, weights, and parameter dictionaries are preserved. |
| Objective and metric handling | Canonical classification objectives return labels or probabilities as requested. Default metrics for regression, binary, multiclass one-versus-all, and Tweedie match native defaults. A weighted custom evaluation function matches native `feval`; its training and validation weights remain aligned. A callable squared-error objective matches native output and leaves the backend objective as `none`. It has no automatic classification link/label inference. F3 covers aliases. |
| Parameter precedence | The wrapper prefers `n_estimators` over `num_boost_round`, while a remaining native `num_iterations` parameter overrides the round argument. The fixture requests 3, 1, and 2 respectively and records an effective limit of 2. This is native alias precedence, not mutation of the caller dictionary. Internal verbosity is forced to -1, patience is fixed at 10, and nonpositive requested rounds are reset to 10,000 with a warning by source inspection. |
| Predictions | Two-dimensional NumPy fit/predict, one-row DataFrames, canonical binary/multiclass labels, probabilities, explicit raw margins, and leaf IDs work. Unfitted predict/raw/save calls reject use. A missing column that changes width is rejected by LightGBM. F2 covers wrong names/order at unchanged width. The wrapper is not a sklearn estimator: `fit` returning `None` is not counted as a defect. |
| Categories | Native pandas categories and Boolean columns match an explicit categorical native reference. F4 and F5 describe the advertised-type and disable-category exceptions. Class labels follow the native numeric objective contract; no arbitrary string-label encoder or fitted `classes_` mapping is supplied. |
| Initial scores | Weighted regression, binary, and Poisson initial-score fits match native raw contributions. Adding the external baseline on the raw scale and applying the objective's inverse link reproduces the recorded validation loss. `predict_raw()` is a passthrough: it defaults to native transformed output. Request `raw_score=True` before adding a baseline; for binary use the logistic link and for Poisson the exponential link. F8 limits multiclass inputs; F9/F10 cover state. |
| Ordinary persistence | Regression, binary, multiclass, and categorical models reproduce predictions and raw margins exactly in a fresh interpreter. Feature names and best iteration survive. These are same-environment, trusted-joblib round trips; they do not establish cross-version compatibility. |
| Files and errors | `save` creates missing directories and writes a curve PNG when present. Existing filenames are silently overwritten; default names include only objective and date, so same-day saves can collide. An omitted directory returns a relative path despite the “full path” docstring. Missing loads and a file used as an output directory raise without destroying the sentinel file. All exercised files are under `/tmp`. |
| Permutation benchmark | A fixed 75/25 split, real weighted trees, and independent sklearn calls agree on all ten permutation repeats for regression and classification. Training and scoring weights are forwarded, feature labels follow sorted importance, and unweighted scoring differs. The helper fits the supplied estimator in place. `task` changes stratification; scoring uses the estimator's default score (weighted R² or accuracy in these fixtures), not an automatically selected actuarial loss. |
| LightForest and detection | Small weighted CPU regression/classification fits work. The factories request LightGBM RF mode, row subsampling 0.632, and feature fractions 1/3 or 1/sqrt(p). `gbm_flavour` recognizes native and sklearn LightGBM objects; `is_lightgbm` recognizes sklearn class names only. These heuristic detectors are not general estimator validation. No speed, superiority, or equivalence-to-sklearn-RF claim follows from the checks. |

The [LightGBM prediction API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict) distinguishes raw scores, leaves, and optional feature validation. The benchmark's default-score and weight interpretation follows [sklearn's permutation importance API](https://scikit-learn.org/1.6/modules/generated/sklearn.inspection.permutation_importance.html). The evidence verifies these particular operations; it does not measure generalization benefit or calibration.

**Findings**

P1 denotes silent wrong training/prediction semantics or failure of an ordinary supported workflow. P2 denotes narrower input, state, comparison, or documentation failures. Finding IDs match the executable markers. Remedies below are suggestions, not implemented fixes.

1. **Stratification silently discards group isolation.** F1; P1; small effort. Source: [gbm.py](../src/arfs/gbm.py), lines 775–787.

   Passing both `stratified=True` and `groups` selects the stratified branch and never supplies groups to the splitter. In the 200-row binary fixture, training and validation share groups despite `fit` promising isolation. Reproduce with `test_stratification_and_groups_do_not_leak`.

   Smallest remedy: reject the unsupported combination explicitly, or give group isolation documented precedence. Do not silently promise both. A combined stratified-group design would require a separate policy and is not needed to fix this unsafe ambiguity. Otherwise validation can include observations related to training rows.

2. **Named prediction columns can silently change meaning.** F2; P1; small effort. Source: [gbm.py](../src/arfs/gbm.py), lines 344–362 and 415–429.

   Both prediction methods pass named DataFrames to LightGBM without checking the fitted schema. Reversing the same three columns changes predictions; replacing a required feature name with `unrelated` at the same width is accepted. Only width mismatch fails automatically. Reproduce with `test_reordered_features_preserve_prediction_or_raise`, `test_replacement_feature_rejected`, and the passing native-guard check.

   Smallest remedy: validate named inputs against the fitted feature order in both methods, using native feature validation where appropriate, or deliberately reorder after validating a unique matching name set. Keep array inputs explicitly positional. Without this, a plausible prediction can use the wrong variable in every split.

3. **Multiclass objective aliases return probabilities when labels are requested.** F3; P1; small effort. Source: [gbm.py](../src/arfs/gbm.py), lines 359–386.

   `softmax`, `ova`, and `ovr` train successfully, but substring matching fails to recognize them as multiclass. Default `predict(X)` returns a `(200, 3)` probability array instead of 200 class indices; the corresponding native argmax is the independent reference. Reproduce with `test_multiclass_aliases_return_labels`.

   Smallest remedy: normalize supported objective aliases before prediction dispatch and retain the original training parameters separately. These are documented [LightGBM objective aliases](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#objective), not speculative backends. Otherwise an equivalent objective spelling changes the API's output meaning and shape.

4. **Automatic categorical detection advertises types that cannot train.** F4; P1; small effort for an honest support boundary. Source: [gbm.py](../src/arfs/gbm.py), lines 77–84, 218–243, and 586–611.

   Object strings, explicit pandas strings, datetimes, and timedeltas are identified as categorical, but their values remain in dtypes rejected by LightGBM. All four fixtures fail during `fit`; ordinary pandas categorical and Boolean inputs pass. Reproduce with `test_advertised_auto_categorical_input` and `test_native_categories_and_boolean_work`.

   Smallest remedy: state and validate the actually supported dtypes, requiring explicit conversion where necessary. If automatic conversion is retained as a promise, fit and preserve its mapping for prediction. Merely passing column names does not encode strings or time values. This is separate from unit 03's string-cardinality bug: this wrapper also includes the dtype helper's `unk` group.

5. **`cat_feat=None` does not disable categorical treatment.** F5; P2; small effort. Source: [gbm.py](../src/arfs/gbm.py), lines 82, 250–251, and 597–610.

   `None` is documented as no categorical features. The helper changes every falsey categorical selection to `"auto"`, so a pandas categorical column is still handled categorically. The native model dump contains its category codes where a numeric feature would have an empty categorical-value list. Reproduce with `test_none_disables_categorical_features`.

   Smallest remedy: distinguish automatic detection from an explicit empty categorical selection when constructing native datasets. Otherwise the documented modelling choice is ignored.

6. **Nonfinite responses become successful training input.** F6; P1; small effort. Source: [gbm.py](../src/arfs/gbm.py), lines 255–257 and 600–611.

   A regression response containing NaN, positive infinity, or negative infinity fits without rejection. A separate real reference shows that the NaN case gives exactly the same predictions as replacing that response with zero. Reproduce with `test_invalid_targets_rejected` and `test_missing_target_matches_fabricated_zero`.

   Smallest remedy: validate finite responses before splitting or constructing the native datasets; require an explicit policy if rows are to be removed. Do not silently invent outcomes. Unit 1B's missing-label finding concerned explicit `fillna(0)` in selectors; this is a separately exposed wrapper path relying on permissive backend conversion.

7. **Invalid sample weights are accepted.** F7; P1; small effort. Source: [gbm.py](../src/arfs/gbm.py), lines 259–264 and 614–618.

   Only weight length is checked. Negative, NaN, infinite, and entirely zero weights all reach a completed regression fit in the fixtures. Reproduce with `test_invalid_weights_rejected`. Successful ordinary weighted references do not validate these inputs.

   Smallest remedy: require one-dimensional, finite, nonnegative weights with positive mass, including usable mass in each evaluation split. Zero weights for individual observations remain valid. Otherwise an invalid weighting population is reported as a fitted model with no actionable input error.

8. **Multiclass initial-score matrices are unsupported without a stated boundary.** F8; P2; small effort to document/reject clearly, medium effort to support. Source: [gbm.py](../src/arfs/gbm.py), lines 196–199, 266–271, and 814–816.

   Native training accepts a `(200, 3)` initial-score matrix for the three-class reference. The wrapper first converts it to a one-dimensional Series and fails before training. Reproduce with `test_multiclass_initial_score_matrix`.

   Smallest remedy: explicitly restrict the advertised initial-score contract to supported tasks, or preserve native multiclass score matrices, validate their shape, and slice their rows with the split. The current general classification/initial-score description does not disclose the limit.

9. **Refits retain obsolete fitted state.** F9; P1 for the prediction guard; small effort. Source: [gbm.py](../src/arfs/gbm.py), lines 266–267 and 295–307.

   After a fit with initial scores, a successful fit without them leaves `is_init_score=True`; ordinary prediction still raises. Turning off validation-data and curve output before a later fit leaves the previous data and figure attached to the new model. Reproduce with `test_refit_resets_initial_score_flag` and `test_refit_resets_optional_outputs`.

   Smallest remedy: derive all fitted state from the successful current fit, clearing absent outputs and setting the initial-score flag in both directions. Otherwise callers can be blocked from predicting or can inspect/save a curve belonging to a different model.

10. **Persistence loses the wrapper's prediction contract and category metadata.** F10; P1; small-to-medium effort. Source: [gbm.py](../src/arfs/gbm.py), lines 481–483 and 517–540.

    Only the native Booster is saved. A model fitted with a constant baseline of 10 rejects ordinary prediction before save, but a fresh wrapper loads it with `is_init_score=False` and returns residual-model predictions as if they were complete predictions. Raw native contributions round-trip correctly; the wrapper contract does not. Loading an ordinary model into a previously offset-fitted wrapper leaves the opposite stale guard. Categorical metadata also changes from the feature name `['category']` to category levels `[['a', 'b']]`, even though native categorical predictions remain correct.

    Reproduce with `test_reload_preserves_initial_score_contract`, `test_load_into_previously_fitted_wrapper`, and `test_reload_preserves_category_names`; [reload.py](evidence/04-training-and-benchmarks/reload.py) performs fresh-process checks. Smallest remedy: save/load the required wrapper metadata with the Booster and reset loaded state coherently. An external baseline is still needed for new rows; do not imply that per-row training offsets can supply it. Any decision about older Booster-only files needs an explicit metadata policy.

11. **The permutation benchmark learns its category vocabulary from held-out rows.** F11; P2; small effort. Source: [benchmark.py](../src/arfs/benchmark.py), lines 59 and 68–76.

    Encoding precedes the train/test split. A category placed only in the held-out 25% is included in the fitted encoder's vocabulary. The independent train-only encoder has only `train` in its vocabulary and transforms the unseen category to NaN. Reproduce with `test_benchmark_preprocessing_uses_training_rows_only`.

    Smallest remedy: split first, fit the encoder on training rows, and transform each partition with that encoder. This demonstrates feature-distribution leakage and different unknown-category handling, not target leakage or a measured amount of optimistic scoring.

12. **Benchmark input assumptions exceed its documented contract.** F12; P2; small effort. Source: [benchmark.py](../src/arfs/benchmark.py), lines 31, 59, and 100–103; [preprocessing.py](../src/arfs/preprocessing.py), lines 161–164.

    A documented NumPy input fails at `.columns` in the encoder. A DataFrame with integer feature names gets through fitting/scoring and fails at substring-based highlighting. Reproduce with `test_benchmark_accepts_documented_array` and `test_benchmark_integer_feature_names`; plotting is isolated so F13 cannot hide these failures.

    Smallest remedy: either validate and document DataFrame-only input or normalize arrays with stable generated names, and stringify names only for display/highlighting. Preserve original names for model inputs and importance alignment.

13. **The standalone benchmark cannot plot on the newer dependency snapshot.** F13; P1; small effort. Source: [benchmark.py](../src/arfs/benchmark.py), lines 93–96.

    Matplotlib 3.11.2 rejects `boxplot(labels=...)` after fitting and scoring complete. The same end-to-end figure check passes on 3.9.4 with a deprecation warning. Reproduce with `test_benchmark_real_plot`. The project permits both versions.

    Smallest remedy: use `tick_labels`, available at the project's Matplotlib minimum. Matplotlib documents the [removal of `labels` in 3.11](https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.11.0.html#boxplot-tick-labels). `vert` is deprecated there, but its removal is not the observed failure and should not be conflated with this finding.

14. **GrootCV comparisons are labelled as methods and estimators that never run.** F14; P1; small effort. Source: [benchmark.py](../src/arfs/benchmark.py), lines 113–116 and 125–145; [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 2073–2088 and 2306–2324.

    The documented accepted selector list includes GrootCV. The helper assigns `estimator` and `importance`, but GrootCV never reads those attributes. In the frozen real-fit fixture, a supplied `DecisionTreeRegressor` is never fitted; six native LightGBM Boosters are fitted across the three labelled importance modes. All importance calculations follow GrootCV's SHAP path. Source tracing confirms the same dispatch in the newer environment, whose execution stops earlier at unit 0's shadow-array blocker.

    Reproduce with `test_compare_grootcv_configuration_is_real`. Smallest remedy: reject GrootCV in this cross-estimator/method helper and correct its accepted-selector documentation. Otherwise differences due to random shadows can be presented as differences between methods that were not actually used.

15. **Comparison runs reuse stale selector state after changing estimators.** F15; P2; small-to-medium effort. Source: [benchmark.py](../src/arfs/benchmark.py), lines 136–142; [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 293–294, 451, and 810–816.

    Models are cloned, but the selector is reused and its constructor-derived backend flags are not refreshed. A Leshy created with a random forest continues to report `is_lgb=False` after the helper switches to LightGBM, whereas a fresh equivalent Leshy reports True. Real fits also replace the integer seed with an advancing RandomState shared by later comparison cases. Reproduce with `test_compare_reinitializes_selector_for_each_model`; all three importance paths really execute in both environments.

    Smallest remedy: refresh estimator-dependent state at fit time and use a fresh selector configuration for each comparison with a stated seed policy. Cloning and then merely assigning a new estimator still leaves constructor-derived flags wrong. These fixtures establish stale dispatch/reused randomness, not a measured ranking bias or an untested CatBoost runtime failure.

**Inherited findings and interpretation**

- [Unit 0](00-baseline.md): the `pkg_resources`/setuptools clean-import workaround and newer-pandas shadow-array failure remain open. The latter blocks the newer GrootCV comparison. No dependency was installed during this review.
- [Unit 1B](01b-all-relevant.md), F3: `compare_varimp` delegates its permutation cases to Leshy/BoostAGroota, so their missing validation-weight forwarding still applies. `sklearn_pimp_bench` has a separate, correctly weighted call. Signed importance and categorical-shadow findings remain selector issues; no repeat calibration screen was run.
- [Unit 2](02-preprocessing.md), F1/F4: TreeDiscretizer's prediction-based bins and parameter drift remain open. They are not counted again as wrapper defects. Its wrapper caller supplies encoded univariate input and no groups or initial scores.
- [Unit 03](03-selector-contracts.md), F3/F11: VariableImportance's identical repeated split and multiclass NumPy-target postprocessing remain open. Its caller uses `return_valid_features=True`, no initial scores/groups, and direct native contributions; F1, F3's label conversion, and persistence findings should not be indiscriminately attributed to it. F4/F6/F7 can affect shared training inputs.

Notebook source was inspected without executing notebooks. Classification cells 32, 35–36, and 43 use the benchmark helpers; regression cells 34, 36, and 44 do likewise. Regression cell 34 calls `sklearn_pimp_bench`, but its imports omit that name; carry that clean-kernel issue into item 06's workflow review. The non-normal-loss/weight notebook uses GrootCV/Leshy/BoostAGroota pipelines and an external GLM, not direct GradientBoosting initial-score prediction or persistence. It does not establish those contracts. `compare_varimp` prints feature selections and importance plots; it is not a common held-out predictive-performance comparison across models or importance types. The “faster and better” source comment has no experimental support from this review.

**Reproduction and persistence boundary**

From the repository root, using the same saved environments:

```sh
python reviews/evidence/04-training-and-benchmarks/run.py \
  frozen /tmp/arfs-review-03f67d0/locked-env/bin/python
python reviews/evidence/04-training-and-benchmarks/run.py \
  current /tmp/arfs-review-03f67d0/wheel-env/bin/python
```

The runner verifies installed source identity, records environment pins, limits native threads, uses Agg with a temporary Matplotlib directory, then runs strict and unmasked checks. Rerun logs/manifests go to a new persistent temporary directory, printed by the runner; an optional final argument chooses an output directory outside the checkout. Saved evidence is not overwritten. Internally it invokes pytest with `-q -rx --tb=short -c /dev/null --rootdir=<evidence> -p no:cacheprovider`, adding `--runxfail` for the failure logs. Native model artifacts and child-process input files are created under pytest temporary directories in `/tmp`; only fixtures generated by these checks are loaded.

Joblib loading is a **trusted-input boundary**: it can execute code while reconstructing a Python object, as documented by [joblib.load](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html). No unknown artifact was loaded. Source inspection also shows no atomic replacement, no artifact-type validation before assigning `self.model`, and model saving before curve saving; invalid artifacts or later I/O errors need deliberate handling during remediation. Interrupted writes, disk-full behavior, hostile paths, and corrupt/foreign artifacts were not exercised. Silent overwriting is recorded as current behavior, not treated as a demonstrated promise violation.

**Completion and remaining limits**

Item 04 is reviewed, with 15 findings open and no library fixes. Each planned surface has executable evidence or a stated boundary. The next item is [05 — sampling, distances, and parallel execution](05-sampling-and-parallel.todo.md). Carry the inherited parallel-shadow blocker forward without recounting it as a new finding.

Not checked: GPU, distributed training, optional CatBoost/XGBoost/fasttreeshap, DART-specific early stopping, ranking objectives/query groups, arbitrary label encoding, sparse inputs, temporal validation, large-data performance, broad statistical calibration, full notebook execution, the complete package suite, fresh installation, a Python/platform matrix, or cross-version model loading. The review neither ranks model quality nor establishes release readiness.
