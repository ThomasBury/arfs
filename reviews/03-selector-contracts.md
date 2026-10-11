What this change does: adds the review of **selector contracts and remaining selectors**, with executable counterexamples and independent calculation checks. These selectors filter DataFrame columns using missingness, uniqueness, cardinality, association, or model importance, and report pipeline decisions. No library behavior was changed.

Reviewed commit: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11. Scope is item 03: `base.py`, `unsupervised.py`, `variable_importance.py`, `summary.py`, public exports, their direct helpers/callers, the threshold tests, and relevant notebook source. Expected use is CPU tabular modelling with pandas, categorical/numeric predictors, optional observation weights, and sklearn pipelines.

**Result:** 14 new findings, plus the previously reported missing public export. The most consequential are dropping the feature needed to reach the importance cutoff, retaining all features when importance is zero, duplicate names defeating selection, failure after a pipeline leaves one feature, and two independent blockers in pipeline summaries. Findings remain open; this is completion of the investigation, not remediation or release approval.

**Evidence and environment**

The [73 checks](evidence/03-selector-contracts/checks.py) run against unit 0's installed wheels, outside the checkout in `/tmp/arfs-review-selectors`. Fresh hashes verify ten installed modules against the reviewed source, including every selector in scope and its association, preprocessing, training, dtype, and parallel helpers. Both historical environments retain `setuptools==80.9.0`, unit 0's import workaround. Their clean-install blocker remains open.

| Snapshot | Python | pandas | NumPy | sklearn | LightGBM | Strict expected failures | Markers disabled |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen `locked-env` | 3.12.7 | 2.2.3 | 2.0.2 | 1.6.1 | 4.6.0 | 32 passed, 41 xfailed; exit 0 | 41 failed, 32 passed; exit 1 |
| Newer `wheel-env` | 3.12.7 | 3.0.6 | 2.5.4 | 1.9.1 | 4.7.0 | 32 passed, 41 xfailed; exit 0 | 41 failed, 32 passed; exit 1 |

“Newer” is the saved dependency snapshot, not a fresh resolution. Identities, module paths/hashes, and pins: [frozen environment](evidence/03-selector-contracts/frozen-environment.json), [frozen requirements](evidence/03-selector-contracts/frozen-environment.requirements.txt), [newer environment](evidence/03-selector-contracts/current-environment.json), [newer requirements](evidence/03-selector-contracts/current-environment.requirements.txt). Logs: [frozen checks](evidence/03-selector-contracts/frozen-checks.txt), [frozen failures](evidence/03-selector-contracts/frozen-failures.txt), [newer checks](evidence/03-selector-contracts/current-checks.txt), [newer failures](evidence/03-selector-contracts/current-failures.txt).

Neither historical environment includes Jinja2, so summary checks first fail on that missing dependency. To inspect beyond it, only the summary check was repeated with an isolated `/tmp/arfs-review-selectors/styling-deps` overlay containing `Jinja2==3.1.6` and `MarkupSafe==3.0.3`, installed offline from cache. The historical environments and project `.venv` were not changed.

| Styling overlay run | Strict markers | Markers disabled |
| --- | --- | --- |
| Frozen | 1 passed, 72 deselected; exit 0 | 1 passed, 72 deselected; exit 0 |
| Newer | 1 xfailed, 72 deselected; exit 0 | 1 failed, 72 deselected; exit 1 |

Overlay identities/pins: [frozen](evidence/03-selector-contracts/frozen-styling-environment.json), [frozen requirements](evidence/03-selector-contracts/frozen-styling-environment.requirements.txt), [newer](evidence/03-selector-contracts/current-styling-environment.json), [newer requirements](evidence/03-selector-contracts/current-styling-environment.requirements.txt). Logs: [frozen checks](evidence/03-selector-contracts/frozen-styling-checks.txt), [frozen unmasked](evidence/03-selector-contracts/frozen-styling-failures.txt), [newer checks](evidence/03-selector-contracts/current-styling-checks.txt), [newer failure](evidence/03-selector-contracts/current-styling-failures.txt).

Expected-failure markers are strict: an unexpected pass fails the evidence run. The unmasked logs show the real defects. Controlled arithmetic tests replace only the training boundary with declared contribution arrays; real-fit checks use installed LightGBM, with observation-only subclasses where needed. No ARFS files were patched. Nonfinite contribution tests establish a missing guard, not that ordinary native LightGBM produced those values.

**Confirmed behavior and contract decisions**

| Surface | Evidence and interpretation |
| --- | --- |
| Missing values | Numeric NaN and positive/negative infinity count as missing. Nullable integers work. The implementation retains a column only when its missing fraction is **strictly less** than the threshold; equality is rejected. Threshold 0 rejects even complete columns; threshold 1 rejects fully missing columns. |
| Unique values | Missing values are excluded from the count. Retention requires **more** unique values than the threshold. Constant/all-missing columns are rejected at the default 1. Notebook cell 17 explicitly says values equal to the threshold are rejected, resolving unit 0's misleading threshold-2 test. |
| Cardinality | Object/category/bool counts behave as hand calculated; numeric columns receive zero and ordinarily pass. Retention requires count **strictly less** than the threshold. All-missing categorical columns have count zero. Explicit nullable strings are the exception in F4. |
| Equality wording | Missing/cardinality docstrings say “larger”/“more” values are removed, whereas the base class mentions inclusive removal and the code removes equality. This is a genuine documentation conflict, not evidence to silently change both operators. Choose and document inclusive versus exclusive retention before remediation. Unique-count wording also needs cleanup, but its notebook example resolves the intended rule. |
| Standard selector metadata | With distinct names, the tested output columns, support masks, support indices, and feature names agree; selected DataFrame dtypes and row indexes are preserved. Fresh/refitted threshold and importance selectors use updated columns and valid updated parameters. F5, F6, and F12 identify exceptions. |
| Schema/input boundary | These fits require pandas DataFrames and reject arrays. Reordered columns transform back to fitted selection order. A missing selected column raises `KeyError`. Selection is name based and does not validate a complete unchanged input schema; strict schema rejection versus permissive projection remains an API decision. Array-like transform docstrings do not reflect the implementation. |
| Collinearity | Absolute values correctly catch an exact negative duplicate. A hand matrix with unequal directional associations removes its uniquely most connected feature and preserves a disconnected one. An association exactly at the cutoff is retained. Ties produce a valid retained set, but process-dependent choice is F9. |
| Weights | A real pipeline forwards weights into collinearity and importance. Weighted collinearity rejects one of a pair where the unweighted fit retains both. A separate native weighted LightGBM fit reproduces the selector's mean absolute contributions; an unweighted fit differs, proving that fixture detects lost weights. The three simple threshold statistics ignore `sample_weight`; the evidence records this without inventing a weighted unique/cardinality policy. Their inherited weight documentation needs clarification. |
| Importance arithmetic | Signed contributions become absolute values; bias columns are excluded; averaging two controlled iterations gives the hand reference. Three-class block aggregation also matches its reference. Negative SHAP contributions are valid and are not negative feature importance. Native contribution output includes an extra expected-value column, as documented by [LightGBM](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict). |
| Tasks/categories | Small regression, binary, and multiclass fits work with consistent objectives and Series targets. Categorical predictors work with internal encoding and with native categorical input when encoding is disabled. Refits preserve caller parameters/input. `task` does not select the training objective; the public API explicitly requires a consistent `lgb_kwargs` objective. Multiclass arrays fail in F11. |
| Summary decisions | With the styling dependency available on pandas 2.2, a real missing → unique → ordinal encoder → unique pipeline produces the correct original-feature rows, each rejection, and subsequent not-applicable cells, even when the final selection is empty. HTML includes all three decision colors. This is a name-preserving transformer case; generic renamed/expanded feature lineage was not established. |

The controlled weighted reference verifies training/aggregation arithmetic and alignment, not out-of-sample selection benefit. Global importance currently averages validation observations equally even when training uses weights; whether that summary should instead describe a weighted population is an unresolved policy question, not a demonstrated arithmetic defect.

**Must fix**

P1 means wrong selection or failure of an ordinary advertised workflow. P2 means a narrower contract, reproducibility, or error-path defect. All have concrete reproductions; `F1`–`F14` correspond to the strict markers in the checks. Suggested remedies were not implemented.

1. **The cumulative cutoff discards the feature needed to reach it.** F1; P1; small effort. Location: [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), lines 162–170.

   **What this is:** ranked positive importance should retain the leading features needed to reach the requested cumulative share.

   **Problem:** the selector drops the first feature whose cumulative importance meets or exceeds the threshold, as well as everything below it. Contributions `0.6, 0.3, 0.1` select none at `0.5`, only the first feature at `0.9`, and only the first two at `1`. A real fit with one useful feature and two constants assigns all importance to that feature, then returns no columns at the default `0.99`.

   **Fix:** retain the smallest positive-importance prefix reaching the threshold, including the crossing feature; handle zero totals separately. Do not fix only equality: changing `>=` to `>` still loses features that cross the cutoff.

   **If we skip it:** useful predictors disappear and downstream models can receive an empty frame. The public threshold prose also needs to state the retained-share rule clearly.

2. **All-zero importance keeps every predictor.** F2; P1; small effort. Location: [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), lines 439–445 and 162–170.

   **What this is:** importance is divided by its total before choosing columns.

   **Problem:** a real constant-response fit produces zero contributions for all three predictors. Division by zero produces NaN cumulative values; every comparison is false, so all predictors are retained despite the advertised zero-importance removal. Controlled NaN/infinite contribution arrays also reach selection without an error.

   **Fix:** validate finite importance, explicitly return an empty selection or a documented error for a zero total, and exclude zero-importance predictors independently of cumulative rounding.

   **If we skip it:** failed or uninformative importance calculations look like successful selection of everything.

3. **Repeated importance fits provide no variation to average.** F3; P2; small-to-medium effort. Locations: [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), lines 356–372; [gbm.py](../src/arfs/gbm.py), lines 761–789.

   **What this is:** `n_iterations` is advertised as reducing variance through repeated fits.

   **Problem:** all iterations use the same split seed, input, and model parameters. Three real iterations produce identical validation indexes and serialized fitted models in both environments. The default ten iterations therefore repeat identical work under these deterministic settings.

   **Fix:** the smallest honest behavior is one fit with revised documentation. If repeated resampling is required, vary a reproducible split/model seed and check that the iterations differ; merely repeating the current loop cannot reduce variance.

   **If we skip it:** users pay for repeated fitting without the promised stability gain. This finding is selector-specific; the general split design remains item 04's scope.

4. **Explicit string dtype bypasses cardinality removal.** F4; P1; small effort. Locations: [unsupervised.py](../src/arfs/feature_selection/unsupervised.py), lines 182–187; [utils.py](../src/arfs/utils.py), line 154.

   **What this is:** categorical columns are classified by dtype before unique levels are counted.

   **Problem:** `pd.Series(['a','b','c'], dtype='string')` receives cardinality zero and survives threshold 2 in both snapshots. The same labels as object/category correctly receive three. Ordinary inferred strings still pass the intended check in pandas 3.0.6 through its current object-selection compatibility behavior; this review does not claim those fail today.

   **Fix:** include explicit string dtypes in the shared categorical classification and check its association/training callers. Preserve the treatment of numeric and time columns.

   **If we skip it:** high-cardinality string predictors survive solely because of their pandas representation.

5. **Duplicate names defeat the selected mask.** F5; P1; small effort. Location: [base.py](../src/arfs/feature_selection/base.py), lines 113–114 and 137–139; parallel name-based transform paths in [unsupervised.py](../src/arfs/feature_selection/unsupervised.py), lines 381–384, and [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), lines 202–204.

   **What this is:** selection is learned positionally but transformed by column label.

   **Problem:** for two columns both named `a`, one constant and one varying, uniqueness reports support `[False, True]` but returns both columns. Cardinality reports two selected positions but returns four columns because each repeated selected label selects both originals. Missing-value selection errors earlier on the same duplicate frame.

   **Fix:** reject duplicate feature names at the selector input boundary, or consistently preserve positional identity. Rejecting them is the smaller safe contract for the current name-based API.

   **If we skip it:** rejected data reappears and feature-name/support metadata no longer describes the output.

6. **A labelled custom statistic selects the wrong column when reordered.** F6; P2; small effort. Location: [base.py](../src/arfs/feature_selection/base.py), lines 99–113.

   **What this is:** the base callback contract says the statistic Series is indexed by feature name.

   **Problem:** a callback returning `frame.nunique().sort_values(ascending=False)` for `constant, varying` produces labels `varying, constant`. The base class ignores those labels and uses the values positionally, selecting the constant instead of the varying predictor at threshold 1.

   **Fix:** validate the callback's index and align its Series to the fitted input columns before constructing the mask. Keep missing/extra/duplicate statistic labels as explicit errors.

   **If we skip it:** custom subclasses silently select the wrong features. Built-in callbacks happen to preserve input order, so this is a narrower failure than F5.

7. **Invalid parameters can silently select everything or nothing.** F7; P2; small effort. Locations: [base.py](../src/arfs/feature_selection/base.py), lines 94–111; [unsupervised.py](../src/arfs/feature_selection/unsupervised.py), lines 318–322 and 351–366; [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), lines 119–120 and 352–357.

   **What this is:** thresholds, method names, and iteration counts determine selection.

   **Problem:** out-of-range/NaN missing thresholds and zero unique/cardinality thresholds are accepted. `set_params(threshold=1.1)` bypasses constructor checks in collinearity and importance; a misspelled collinearity method also fits. Zero or negative importance iterations do no training and retain every feature after NaN normalization.

   **Fix:** validate finite thresholds and documented ranges, supported methods, and positive integer iteration counts at fit time, including after `set_params`. Settle the equality wording separately from rejecting clearly invalid values.

   **If we skip it:** parameter searches and configuration mistakes yield plausible selector results instead of actionable errors. sklearn likewise recommends validation where parameters are used so updates cannot bypass it; see its [estimator development guidance](https://scikit-learn.org/stable/developers/develop.html#instantiation).

8. **Collinearity crashes when earlier steps leave fewer than two features.** F8; P1; small effort. Locations: [unsupervised.py](../src/arfs/feature_selection/unsupervised.py), lines 356–366; [association.py](../src/arfs/association.py), lines 1258–1260 and 2087.

   **What this is:** pairwise association has no pairs for a one-column or empty frame.

   **Problem:** a real uniqueness → collinearity pipeline leaving one varying column, or zero columns, calls the pivot conversion on a completely empty association frame. Both cases raise `KeyError` for missing `row`/`col` fields.

   **Fix:** handle zero/one candidate features explicitly before matrix conversion and populate support and an appropriately labelled empty/zero association matrix.

   **If we skip it:** otherwise valid earlier selections break the next stage even though there is no collinearity to remove.

9. **Collinearity ties depend on the process hash seed.** F9; P2; small effort. Location: [unsupervised.py](../src/arfs/feature_selection/unsupervised.py), lines 433–442.

   **What this is:** equally redundant predictors need a stable tie-break rule.

   **Problem:** the candidates pass through a Python set before sorting equal totals. The same matrix drops `a` with `PYTHONHASHSEED=0` or `1`, and `b` with seeds `2` or `3`, in both environments. Both retained sets satisfy the association threshold; the defect is reproducibility, not failure to remove the redundant pair.

   **Fix:** preserve original feature order when constructing candidates and use it to break equal totals.

   **If we skip it:** restarting the process can change the selected predictor without any data or parameter change.

10. **Pipeline summaries fail with pandas 3 even after installing their styling dependency.** F10; P1; small effort. Location: [summary.py](../src/arfs/feature_selection/summary.py), line 69.

    **What this is:** the final summary step applies colors to a pandas Styler.

    **Problem:** the real pipeline summary succeeds on pandas 2.2.3 with the overlay, but pandas 3.0.6 raises `AttributeError` on `Styler.applymap`. The [pandas 3 release notes](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html#removal-of-prior-version-deprecations-changes) confirm that this method was removed.

    **Fix:** use `Styler.map`, already available in the supported pandas 2.2 baseline. No compatibility wrapper is needed.

    **If we skip it:** the advertised summary workflow remains broken on an allowed dependency version after F14 is addressed.

11. **Multiclass selection crashes for NumPy targets after fitting.** F11; P2; small effort. Location: [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), line 409.

    **What this is:** multiclass contribution blocks are reshaped using a count of target classes.

    **Problem:** the wrapper successfully trains from a NumPy target, then calls `y.fillna(0)` on the original array and raises `AttributeError`. The same three-class data as a Series succeeds. Replacing missing labels with class zero also lacks an explicit target policy, but missing-target semantics were not separately benchmarked here.

    **Fix:** obtain the number of class blocks from the fitted model/contribution shape, or normalize and validate targets once before use; do not call a pandas-only method on accepted array-like targets.

    **If we skip it:** ordinary array targets waste the training run and cannot complete multiclass selection.

12. **Fitted-state and feature-count contracts are incomplete.** F12; P2; small effort. Locations: [base.py](../src/arfs/feature_selection/base.py), lines 94–95 and 137–139; [unsupervised.py](../src/arfs/feature_selection/unsupervised.py), lines 344–347 and 381–384; [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), lines 142–145 and 202–204.

    **What this is:** public sklearn-style selectors expose training feature counts and distinguish unfitted use.

    **Problem:** all five selector classes document `n_features_in_` but never set it. All five `transform` methods raise a raw `AttributeError` before fit, while `get_support` and `get_feature_names_out` correctly raise `NotFittedError`.

    **Fix:** populate the feature count during successful fit and guard transform with `check_is_fitted` for the actual learned selection. Preserve the currently correct DataFrame output.

    **If we skip it:** metadata consumers and standard fitted-state error handling behave differently depending on which method they call. sklearn's [universal attributes contract](https://scikit-learn.org/stable/developers/develop.html#universal-attributes) also requires the count for tabular estimators.

13. **A missing optional explanation backend produces the wrong error.** F13; P2; trivial effort. Location: [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), lines 380–385.

    **What this is:** `fastshap=True`, also shown in the basic-selection notebook, imports `fasttreeshap`.

    **Problem:** the import handler constructs an `ImportError` but does not raise it. With that backend unavailable, the next line raises `UnboundLocalError` for `FastTreeExplainer`; a controlled check isolates this import path without requiring optional backend installation.

    **Fix:** raise the existing descriptive import error, ideally before expensive training.

    **If we skip it:** users receive an internal variable error instead of the actionable missing-dependency explanation. Backend numerical equivalence remains untested.

14. **The exported summary function requires an undeclared dependency.** F14; P1; small effort. Locations: [summary.py](../src/arfs/feature_selection/summary.py), line 68; runtime dependency list in [pyproject.toml](../pyproject.toml).

    **What this is:** `make_fs_summary` always returns a styled table, which imports Jinja2.

    **Problem:** both recorded installed-wheel environments satisfy the declared runtime dependencies yet raise `ImportError: Missing optional dependency 'Jinja2'` on this function. pandas lists Jinja2 as an optional [output-formatting dependency](https://pandas.pydata.org/docs/getting_started/install.html#optional-dependencies), so depending on pandas alone does not supply it.

    **Fix:** declare the dependency needed by the existing unconditional public return type. If styling is intentionally optional, make that API/install contract explicit instead; documentation-only environments should not be relied on to install runtime requirements accidentally.

    **If we skip it:** a normal runtime installation cannot use an exported, documented function even on the frozen pandas version.

**Inherited findings and boundaries**

- [Unit 0](00-baseline.md), F4: `BaseThresholdSelector` is still in `__all__` without being imported. The all-exports check reproduces it; this is not counted as a new finding. Unit 0's broken synthetic column and threshold assertions were not edited. This review resolves uniqueness equality from the notebook, but leaves the missing/cardinality documentation conflict explicit.
- [Unit 1A](01a-statistical-foundations.md), especially F5–F10: constant categorical associations, weight validation/scaling, zero-weight categories, and missing-row handling remain shared upstream risks for collinearity. The new elimination checks isolate graph/selection behavior; they do not establish correctness for those already-failing statistical inputs. Unlike MRMR's known negative-correlation issue, this collinearity loop takes absolute associations correctly.
- [Unit 2](02-preprocessing.md), especially F8: ordinal metadata still excludes passed-through numeric columns. The summary can nevertheless work with a name-preserving encoder between selectors because it does not ask that encoder for output names. This does not establish that full-pipeline `get_feature_names_out` is repaired. Tree discretization, midpoint conversion, and inverse/input defects remain open.
- `method='correlation'` encoding and plot code were traced but not given separate exhaustive runtime checks. Plot rendering, generic feature-renaming/expansion lineage, and full sklearn estimator compliance remain outside this evidence. `_compute_varimp_lgb`'s default fast backend is private; only the public option's missing-backend failure was exercised.

**Reproduction**

Run from the checkout to define paths, then execute outside it. These commands intentionally produce failures in the unmasked runs; do not use a shell setting that aborts before preserving those results.

```sh
review_repo="$PWD"
review_evidence="$review_repo/reviews/evidence/03-selector-contracts"
review_run=/tmp/arfs-review-selectors
mkdir -p "$review_run"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$review_run"

for review_mode in locked wheel; do
  review_python="/tmp/arfs-review-03f67d0/$review_mode-env/bin/python"
  export MPLCONFIGDIR="$review_run/mpl-$review_mode"
  "$review_python" "$review_evidence/environment.py" "$review_repo" \
    "$review_run/$review_mode-environment.json"
  "$review_python" -m pytest "$review_evidence/checks.py" -q -rx --tb=short \
    -c /dev/null --rootdir="$review_evidence" -p no:cacheprovider
  "$review_python" -m pytest "$review_evidence/checks.py" -q --runxfail --tb=short \
    -c /dev/null --rootdir="$review_evidence" -p no:cacheprovider
done
```

Both ordinary environments should yield 32 passed/41 xfailed, then 41 failed/32 passed. The [environment recorder](evidence/03-selector-contracts/environment.py) verifies installed code before testing; if the saved environments are gone, use unit 0's recreation instructions and the recorded requirements, not the published PyPI package silently.

For the separate styling check, create an overlay without modifying either environment. The original install used `--offline` with cached wheels; omit that flag only if deliberately fetching those exact pins.

```sh
review_overlay="$review_run/styling-deps"
uv pip install --offline \
  --python /tmp/arfs-review-03f67d0/locked-env/bin/python \
  --target "$review_overlay" 'Jinja2==3.1.6' 'MarkupSafe==3.0.3'
for review_mode in locked wheel; do
  review_python="/tmp/arfs-review-03f67d0/$review_mode-env/bin/python"
  PYTHONPATH="$review_overlay" "$review_python" -m pytest "$review_evidence/checks.py" \
    -k summary -q -rx --tb=short -c /dev/null --rootdir="$review_evidence" -p no:cacheprovider
  PYTHONPATH="$review_overlay" "$review_python" -m pytest "$review_evidence/checks.py" \
    -k summary -q --runxfail --tb=short -c /dev/null --rootdir="$review_evidence" -p no:cacheprovider
done
```

The overlay summary passes in the frozen environment and fails on removed `applymap` in the newer one. Tests that replace a backend use pytest-scoped monkeypatches only; these restore automatically and never modify installed source.

**Handoff**

All five item-03 review questions are accounted for above. Review artifacts are the only changes, together with the plan/queue status update. No fixes, dependency declarations, lockfile updates, existing-test changes, commits, or publication were performed; previous reports remain historical evidence.

Next queued item: [04 — GBM training, prediction, persistence, and benchmarks](04-training-and-benchmarks.todo.md). Carry F3's repeated fixed split and F11's selector-local target handling into source tracing without counting them twice. Do not infer that prediction, group splitting, initial scores, persistence, benchmark scoring, or general training validation passed from this review's narrow weighted reference.

Verdict: fix F1, F2, F4, F5, F8, F10, and F14 before relying on these workflows; the existing import/export and upstream statistical findings also remain open.

Not checked: full package suite or notebooks, docs build, GPU/optional backend numerical equivalence, held-out selection validity, large-data speed/memory, full Python/platform/version matrix, generic transformed-feature lineage, and persistence/security. All demonstrated library defects remain unfixed.
