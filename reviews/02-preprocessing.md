What this change does: adds a review of **preprocessing and transformer behavior**, with executable counterexamples and reference checks. The code encodes categories, learns supervised tree bins, converts intervals to numeric values, and builds formula-based model inputs. No library behavior was changed.

**Scope assumption:** units 0, 1A, 1B, and 1C are already reviewed. Their handoffs do not preserve the original remaining review list. This report uses **unit 2, preprocessing** as the next working scope; its title and numbering are not a recovered or user-confirmed item from that missing list.

Reviewed commit: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11. Scope covers `preprocessing.py`, its training and dtype helpers, direct source callers, API documentation, notebook source, and existing tests. Assumed use is CPU tabular modelling with pandas, categorical and numeric predictors, optional weights, and sklearn pipelines. Library source, existing tests, dependencies, lockfile, CI, and previous reports were preserved.

**Result:** 15 findings. The most consequential are incorrect tree partitions, a generated-column collision that changes the training data, a pandas compatibility failure in midpoint conversion, and Patsy dropping rows despite an explicit request to reject missing values.

**Evidence and environment**

Both environments are unit 0's existing installed wheels. Runs executed outside the checkout from `/tmp/arfs-review-preprocessing`; fresh hashes verify that installed `preprocessing.py`, `gbm.py`, and `utils.py` match this checkout. No packages were installed and no ARFS methods were patched. Both environments retain unit 0's `setuptools==80.9.0` import workaround; these checks do not override its clean-install blocker.

| Environment | Python | pandas | NumPy | sklearn | LightGBM | Patsy |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen `locked-env` | 3.12.7 | 2.2.3 | 2.0.2 | 1.6.1 | 4.6.0 | 1.0.1 |
| Newer snapshot `wheel-env` | 3.12.7 | 3.0.6 | 2.5.4 | 1.9.1 | 4.7.0 | 1.0.3 |

Exact identities and dependency pins: [frozen environment](evidence/02-preprocessing/frozen-environment.json), [frozen requirements](evidence/02-preprocessing/frozen-environment.requirements.txt), [newer environment](evidence/02-preprocessing/current-environment.json), [newer requirements](evidence/02-preprocessing/current-environment.requirements.txt). “Newer” refers to the previously resolved snapshot, not a fresh resolution during this review.

The same [40 checks](evidence/02-preprocessing/checks.py) run in both environments:

| Environment | With strict expected-failure markers | Markers disabled |
| --- | --- | --- |
| Frozen | 13 passed, 27 xfailed; exit 0 | 27 failed, 13 passed; exit 1 |
| Newer | 9 passed, 31 xfailed; exit 0 | 31 failed, 9 passed; exit 1 |

Logs: [frozen checks](evidence/02-preprocessing/frozen-checks.txt), [frozen failures](evidence/02-preprocessing/frozen-failures.txt), [newer checks](evidence/02-preprocessing/current-checks.txt), [newer failures](evidence/02-preprocessing/current-failures.txt). Expected failures document defects; they do not make the library a passing release candidate.

Passing checks include ordinary ordinal encoding, unknown/missing categories, numeric passthrough, dtype filtering, ordinal cloning and inverse values, weighted tree predictions against a native LightGBM reference, categorical missing/unknown handling, input/index preservation, hand-calculated interval midpoints, and Patsy retaining its training mean during later centering. The weighted reference uses the same fixed training split and independently builds one native tree; an unweighted comparison confirms the fixture is sensitive to weights. It does not establish general statistical validity or out-of-sample benefit.

Three Patsy error checks save the exception message and assert outside the exception handler. Otherwise Python 3.12.7's traceback suggestion code crashes while pytest formats Patsy's evaluation namespace. This changes only the evidence harness, not the underlying call or library.

**Must fix**

All findings have concrete reproductions. P1 denotes wrong transformations or failure of an ordinary advertised workflow; P2 denotes a narrower option or repeated-use defect. Effort estimates describe possible repairs, not authorization to implement them. `F1`–`F15` in the checks map to the numbers below.

1. **Tree bins do not preserve the fitted tree's partition.** P1; medium effort. Locations: [preprocessing.py](../src/arfs/preprocessing.py), lines 609, 627–642, and 680–694; [gbm.py](../src/arfs/gbm.py), lines 366–379.

   **What this is:** supervised discretization should turn the learned numerical tree regions into reusable intervals.

   **Problem:** grouping by predictions merges separate leaves with equal scores; classification first converts probabilities into class labels. A six-leaf regression tree becomes five bins, and a six-leaf binary tree becomes two; the resulting bins mix observations from different leaves. Boundaries also use observed maxima rather than fitted thresholds: `120` and `120.25` reach the same tree leaf but receive different intervals.

   **Fix:** derive numeric intervals from the fitted split thresholds and use actual leaf membership for grouping and missing-value routing. The existing `predict_raw(..., pred_leaf=True)` path can expose leaf IDs; [LightGBM documents this output](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict). Merely switching to probabilities still leaves repeated scores and incorrect interval boundaries.

   **If we skip it:** downstream GLMs, MRMR, and feature selectors receive a different partition from the one trained on the response.

2. **A real predictor named `x_g` is overwritten while processing `x`.** P1; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 536–540 and 609–628.

   **What this is:** each univariate fit stores predictions in a temporary column of the shared training frame.

   **Problem:** for input columns `x` and `x_g`, fitting `x` replaces `x_g` before its own fit. In the fixture, the second predictor's fitted values differ from fitting that predictor alone by up to `2.3091`; its cut points are learned on predictions instead of its original values. The caller's frame is copied, so the corruption is internal to training.

   **Fix:** keep predictions in a separate Series or a temporary frame containing only the current predictor. Do not store generated names in the shared input copy.

   **If we skip it:** changing predictor names or order can silently change what data the discretizer learns from.

3. **Midpoint conversion fails with the allowed pandas 3 dependency.** P1; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 793–794 and 850.

   **What this is:** both midpoint APIs replace interval-valued columns with floating-point values; the Lasso notebook uses the estimator after tree discretization.

   **Problem:** assignment through `X.loc[:, c]` attempts to retain the interval dtype. On pandas 3.0.6, `(0, 2]` and `(2, 4]` raise `TypeError` instead of becoming `1` and `3`. Both public APIs and a real tree-to-midpoint pipeline fail; pandas 2.2.3 passes while warning about incompatible dtype assignment.

   **Fix:** replace the whole column with `X[c] = ...` in both entry points, preserving the copied frame and row index. Remove the redundant second cast if the helper already returns the required numeric dtype.

   **If we skip it:** an advertised preprocessing path breaks before the downstream model or selector runs.

4. **Tree training parameters can disagree with the estimator's displayed settings.** P1; small-to-medium effort. Locations: [preprocessing.py](../src/arfs/preprocessing.py), lines 481–498 and 600–605; [gbm.py](../src/arfs/gbm.py), lines 657–660.

   **What this is:** `n_bins`, `task`, and `boost_params` determine the univariate tree fitted per feature.

   **Problem:** derived parameters are written only in the constructor. Setting `n_bins=2` later still fits six leaves; setting `task="classification"` still fits an RMSE objective. A supplied `n_estimators=3` bypasses the forced one-round setting and produces two retained trees in the fixture. Construction also modifies the caller's parameter dictionary.

   **Fix:** retain constructor inputs unchanged and derive a copied training dictionary at fit time. Resolve conflicting tree-count and leaf-count aliases so the advertised single-tree/bin constraints have one meaning.

   **If we skip it:** parameter searches and reused dictionaries can silently fit a different model from the requested one.

5. **Raw categorical transformation skips the encoder used in training.** P1; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 664–680.

   **What this is:** `raw=True` requests the tree's numeric outputs rather than human-readable groups.

   **Problem:** categorical training uses ordinal codes, but raw transformation sends original strings directly to LightGBM. A frame containing `a`, `b`, `c`, and `d` fits successfully and then raises a dtype `ValueError` in `fit_transform`.

   **Fix:** apply the fitted encoder and missing-value treatment before either output branch, then choose raw values or labels.

   **If we skip it:** raw mode cannot process the categorical inputs supported by normal mode.

6. **Fitting replaces constructor options with learned state and breaks reuse.** P2; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 524–531, 766–773, and 977–979.

   **What this is:** an estimator can be fitted again, using its configured feature-selection rule or automatic formula.

   **Problem:** `TreeDiscretizer(bin_features="all")` keeps the first dataset's column names and fails when refitted from `x` to `z`. `IntervalToMidpoint(cols="all")` stores a pandas Index in `cols` and raises an ambiguous-truth-value error on a second fit with the same two-column frame. `PatsyTransformer(formula=None)` retains the inferred first formula and fails on the new `z` frame.

   **Fix:** preserve constructor parameters and store resolved columns/formulas in separate fitted attributes. Reset per-fit tree dictionaries and mappings when rebuilding the discretizer.

   **If we skip it:** reusing an estimator can fail or retain decisions from the previous dataset.

7. **Input normalization differs between fitting and transformation.** P2; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 544–551, 662–667, 968–970, and 1008–1011.

   **What this is:** tree fit/transform documents arrays, and Patsy documents named dict-like input while also accepting arrays during fit.

   **Problem:** tree `fit_transform(ndarray, y)` fits and then crashes when indexing the array with generated column names. Patsy can fit an array but cannot transform it afterward; a named dictionary `{"x": [1, 2]}` with formula `"x"` fails during fit because its key is replaced by `pred_0`.

   **Fix:** use each transformer's same input-preparation rule in fit and transform. Preserve dictionary keys and validate array width against fitted feature names.

   **If we skip it:** accepted or documented inputs fail later in the same workflow.

8. **Ordinal encoder metadata excludes columns that it returns.** P1; small-to-medium effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 159–163 and 180–189.

   **What this is:** the wrapper encodes categorical columns and passes through numeric columns, while inheriting sklearn's feature metadata methods.

   **Problem:** fitting the parent on only the categorical subset overwrites the full input names. For columns `cat` and `num`, transformation returns both but `get_feature_names_out()` returns only `cat`. Calling `.set_output(transform="pandas").fit_transform(...)` raises a column-length mismatch in both environments.

   **Fix:** maintain a consistent full input/output schema and keep the parent's categorical-subset validation separate. Override the relevant metadata/output path or use the existing native encoder as a contained component; simply overwriting the parent's names would break its subset checks.

   **If we skip it:** feature names cannot reliably describe the transformed data, and normal pandas output integration fails.

9. **Tree missing-value handling fails on supported boundary cases.** P2; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 585–589, 638–642, and 673–677.

   **What this is:** the discretizer advertises handling missing numeric and categorical values.

   **Problem:** a legitimate category named `missing_added` raises during category insertion. An entirely missing numeric column has no observed value to choose as the imputation representative, so fitting raises after training its tree. Ordinary partial missingness passed a separate categorical check.

   **Fix:** make the missing-category representation collision-safe and define an explicit all-missing-column result before searching for an observed representative. Keep fit and transform consistent.

   **If we skip it:** legitimate category names or all-missing columns stop an otherwise supported workflow.

10. **Patsy silently drops rows despite `NA_action="raise"`.** P1; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 983–988 and 1008–1011.

    **What this is:** the missing-value policy tells formula construction whether to reject a missing row or omit it.

    **Problem:** fit forwards the policy, but transform omits it. After fitting clean data with `NA_action="raise"`, transforming two rows with one missing predictor silently returns one row. Both dataframe and ndarray output are affected. The default omission behavior comes from [Patsy's documented `dmatrix` API](https://patsy.readthedocs.io/en/latest/API-reference.html#patsy.dmatrix).

    **Fix:** pass `NA_action=self.NA_action` in both transformation branches. Keep explicit drop behavior documented: sklearn transformers cannot also drop corresponding target or weight rows.

    **If we skip it:** predictions can have fewer rows than their inputs even when the user requested rejection, making positional joins unsafe.

11. **`add_intercept=False` is ignored by PatsyTransformer.** P2; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 981–984 and 1014–1022.

    **What this is:** the option is meant to remove the constant column when a downstream model supplies its own intercept.

    **Problem:** formula `"x"` still returns `Intercept` and `x` with `add_intercept=False`. The existing `_drop_intercept` call is commented out. The documentation also describes a false default, while the constructor uses true.

    **Fix:** apply the existing helper to the resolved formula and align the documented default with the chosen contract. Keep that fitted formula separate from the constructor parameter.

    **If we skip it:** model inputs can contain an unwanted constant or duplicate the downstream intercept.

12. **A single midpoint column name is treated as a set of characters.** P2; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 772–773 and 848.

    **What this is:** both midpoint APIs accept one column name or a list of names.

    **Problem:** `cols="interval"` leaves that interval column unchanged because the code intersects its columns with `set("interval")`. The call succeeds without doing the requested conversion in both environments.

    **Fix:** normalize a scalar name other than `"all"` into a one-element list before selecting columns. Apply the same rule in both APIs.

    **If we skip it:** a documented option silently sends unconverted intervals downstream.

**Should fix**

13. **Ordinal inverse transformation changes the caller's encoded frame.** P2; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 233–236.

    **What this is:** inverse transformation returns readable category labels from saved codes.

    **Problem:** unlike forward transformation, it writes directly into the supplied dataframe. After inspecting decoded values, the caller's saved model inputs now contain labels instead of codes.

    **Fix:** copy the frame before replacing the categorical columns, as forward transformation already does.

    **If we skip it:** inspecting decoded data can unexpectedly alter the inputs reused for later model calls.

14. **Changing `output_dtype` has no effect on ordinal output.** P2; small effort. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 119–132 and 162.

    **What this is:** `output_dtype` selects the numeric type of encoded columns.

    **Problem:** `.set_params(output_dtype=np.float32)` still returns float64 because the parent's `dtype` was set only at construction.

    **Fix:** synchronize the effective parent dtype from the current parameter before fitting, alongside the metadata repair.

    **If we skip it:** parameter updates do not give the requested output type or memory use.

15. **The legacy `cat_var` encoder loses missingness and fails on pandas 3 strings.** P2; small repair or removal after choosing the public API policy. Location: [preprocessing.py](../src/arfs/preprocessing.py), lines 372–382 and 389–390.

    **What this is:** this separate public helper returns codes and inverse dictionaries; no active repository caller was found.

    **Problem:** on frozen dependencies, `['a', None, 'b']` becomes `[0, 0, 1]`, so the inverse dictionary maps the missing observation to `a`. On pandas 3.0.6, even ordinary inferred string columns fail earlier because dtype-changing `.loc` assignment is rejected.

    **Fix:** if retained, keep a distinct missing code and replace whole columns when changing dtype. Since the project already uses `OrdinalEncoderPandas`, assess removing this duplicate helper rather than maintaining another encoding policy; absence of local callers does not prove absence of external users.

    **If we skip it:** direct callers either lose the distinction between missing and observed values or cannot encode strings.

**Coverage boundaries and handoff**

Source tracing covered every preprocessing definition, including `dtype_column_selector`, `cat_var`, midpoint helpers, and `_drop_intercept`; shared calls into `GradientBoosting`, `create_dtype_dict`, and `concat_or_group`; and encoder callers in `benchmark.py`, `unsupervised.py`, and `variable_importance.py`. The existing tests contain no direct preprocessing workflow assertions. Notebook source was inspected for preprocessing, basic selection, MRMR, Lasso, and ordinal encoding examples; notebooks were not executed.

Repair 1, 2, 3, and 10 first, then coordinate the remaining estimator input/state repairs. Copy the relevant counterexamples into ordinary tests when implementing fixes. Existing unit 0–1C findings remain open. No assertion here establishes that supervised discretization improves generalization or replaces group/fused regularization; those claims need separate methodology and held-out evidence. The familiar rule still applies: fit supervised preprocessing only on each training fold.

This working unit is **reviewed, with remediation outstanding**. The original remaining review list is still unavailable. A useful proposed next scope is **threshold selectors and selector API behavior** (`base.py`, `unsupervised.py`, `variable_importance.py`, and summary integration); this is a proposal, not a confirmed original item. No later review unit was started. The single-distribution working assumption is unchanged.

**Reproduction**

Run from the repository root. Each loop iteration uses an existing unit 0 environment, records its identity, runs the checks, and then exposes the same failures without their markers:

```sh
repo="$PWD"
evidence="$repo/reviews/evidence/02-preprocessing"
audit_dir=$(mktemp -d /tmp/arfs-preprocessing.XXXXXX)
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$audit_dir/mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$audit_dir"

for environment in locked-env wheel-env; do
  review_python="/tmp/arfs-review-03f67d0/$environment/bin/python"
  "$review_python" "$evidence/environment.py" "$repo" "$audit_dir/$environment.json"
  "$review_python" -m pytest "$evidence/checks.py" \
    -q -rx --tb=short -c /dev/null --rootdir="$evidence" -p no:cacheprovider
  "$review_python" -m pytest "$evidence/checks.py" \
    -q --runxfail --tb=short -c /dev/null --rootdir="$evidence" -p no:cacheprovider
done
```

The unmasked runs intentionally exit 1; do not use `set -e` across the diagnostic sequence. If those environments no longer exist, follow [unit 0's archive/build instructions](00-baseline.md) and install the audited wheel with the respective dependency pins, including its historical import workaround. The published package or a new unconstrained resolution is not the identical review target.

Verdict: fix 1, 2, 3, and 10 first.

Not checked: full notebooks, the full package suite, full sklearn estimator checks, fresh-process serialization, additional Python/platform versions, GPU paths, large-data performance, exhaustive missing/dtype combinations, and statistical benefit on held-out datasets. Only this report and its evidence were added; all demonstrated library defects remain unfixed.
