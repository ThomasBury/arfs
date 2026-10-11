What this review covers: unit **1B, all-relevant selectors**: Leshy, BoostAGroota, GrootCV, and their shared importance, shadow, weighting, and categorical-encoding paths. The preceding handoff names 1B but does not define its subject; this report uses the all-relevant selectors as the working scope, not as a recovered copy of the original review plan.

Reviewed commit: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11. Only this report and its evidence were added. Library code, existing tests, prior reports, dependencies, the lockfile, and CI were preserved.

**Result:** 11 findings supported by executable cases. Selection can bypass Leshy's tentative-feature policy, reverse signed permutation importances, lose categorical semantics, and overwrite the caller's data. GrootCV also fails on multiclass objective aliases. The small statistical screen recovers strong signals but does not support a claim that noise is reliably excluded.

**Evidence and environment**

This unit reused the installed wheel in `/tmp/arfs-review-03f67d0/locked-env`: Python 3.12.7, LightGBM 4.6.0, SHAP 0.47.1, pandas 2.2.3, NumPy 2.0.2, SciPy 1.13.1, and scikit-learn 1.6.1. `setuptools==80.9.0` remains unit 0's explicit import workaround. No package was installed or patched during this review. Hashes confirm that the installed `allrelevant.py` and `utils.py` match this checkout; see [environment and source hashes](evidence/01b-all-relevant/environment.json) and [exact dependency pins](evidence/01b-all-relevant/requirements.txt).

The [24 diagnostic checks](evidence/01b-all-relevant/checks.py) produce **8 passes and 16 strict expected failures**, covering the 11 findings. With expected-failure markers disabled, the result is **16 failures and 8 passes**. See the [normal run](evidence/01b-all-relevant/checks.txt) and [unmasked diagnostics](evidence/01b-all-relevant/failures.txt). Expected failures identify defects; they are not a green package gate.

Passing references cover Benjamini–Hochberg arithmetic against statsmodels, both Leshy binomial-correction branches against SciPy/statsmodels calculations, ordinary shadow value/index/dtype preservation on frozen pandas, positional fold/weight alignment, multiclass SHAP aggregation with explicit `softmax`, a real explicit-softmax fit, and rejection of constant numeric inputs by Leshy. They do not establish calibration of the complete selection procedures.

Some cases substitute fixed model importances or record training arguments to isolate selection arithmetic and metadata propagation. Those cases are explicitly marked by `monkeypatch` in the evidence; they are not end-to-end model results. Input mutation, weighted permutation scoring, missing responses, multiclass failures, and the separate screen use real fitted estimators.

**Findings**

P1 means a supported workflow can select incorrectly, corrupt its caller's input, or fail. P2 marks a narrower boundary or an ambiguous documented contract. Confidence is high in the observed behavior; finding 9 distinguishes that from uncertainty about the intended formula. Effort estimates describe possible remedies, not authorization to implement them.

1. **Leshy can select every tentative feature despite `keep_weak=False`.** P1; high confidence; small effort. Locations: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 982–991 and 1058–1064.

   `_calculate_support` correctly excludes tentative features when `keep_weak=False`. The subsequent ranking method sets the entire support mask to true if every feature is either confirmed or tentatively retained. Its meaning of “selected” includes the tentative set even when the user excludes it.

   A two-iteration fit with three fixed real importances above all shadows returns `n_features_=0`, all three `support_weak_` values true, and all three `support_` values true. No feature passed the significance test, yet `transform` keeps them all.

   Remedy: let support calculation own the support mask; ranking must not override the tentative-feature policy. Check confirmed-only, tentative-only, mixed, and empty selections. Otherwise short or unresolved runs can silently promote weak evidence into a final selection.

2. **Leshy's tentative decision discards its first real iteration.** P2; high confidence; small effort. Locations: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 930, 1046, and 1122–1125.

   The active loop starts with an empty importance history, but postprocessing still skips its first row under the old assumption that it contains placeholder zeros. Rejected-feature ranking has the same obsolete slice.

   With real importances `[9, 4]` and shadow thresholds `[5, 5]`, the intended median comparison is `6.5 > 5`. The implementation uses only `4` and discards the tentative feature. This counterexample directly tests the tentative helper; the ranking slice is established by tracing the same history array.

   Remedy: remove the obsolete first-row exclusion from both consumers and test the actual recorded-history layout. Otherwise the first observation counts toward hit tests but not the final tentative decision or rejected-feature ranking.

3. **Permutation importance ignores validation sample weights.** P1; high confidence; small effort. Location: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 1349–1355; callers at 565 and 1784.

   `_split_fit_estimator` returns correctly aligned validation weights, but `_get_perm_imp` drops them when calling sklearn. Both Leshy and BoostAGroota therefore train a weighted model and judge its features with an unweighted score.

   The included real LightGBM fixture gives importances `[1.552487, 0.013375, -0.027775]`, exactly matching the unweighted reference. Passing the returned validation weights gives `[2.280213, 0.105371, -0.014110]`. The installed sklearn API supports weights specifically for scoring; see its [version 1.6 documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.inspection.permutation_importance.html).

   Remedy: forward `w_tt` as `sample_weight` at the shared permutation call and retain the reference test. Otherwise weighted selection optimizes a different validation population than the one supplied by the user.

4. **BoostAGroota reverses signed importances when their total is negative.** P1; high confidence; small effort after defining zero-total behavior. Location: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 1795–1817.

   Every importance vector is divided by its signed sum. Permutation importances can legitimately be negative, so this operation can reverse their order. A zero total also leaves an undefined normalization.

   Fixed real importances `[-0.1, -0.5]` and shadows `[-0.2, -0.2]` are normalized to `[0.1, 0.5, 0.2, 0.2]`. At `cutoff=1`, the selector keeps only the worst real feature, whose original importance was `-0.5`.

   Remedy: handle signed permutation scores explicitly; any normalization must preserve their ordering and have a defined zero-total result. Do not treat a negative score as positive evidence. Otherwise a noisy or poorly predictive model can cause the importance ranking to be inverted.

5. **BoostAGroota and GrootCV overwrite categorical values in the caller's dataframe.** P1; high confidence; small effort. Locations: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 1874–1880 and 2070–2071; [utils.py](../src/arfs/utils.py), lines 215–226.

   The shared encoder assigns integer codes directly into its argument. BoostAGroota creates a copy but then passes the original `X` to the encoder; GrootCV also passes the original. The selected-column transform does not restore those values.

   In real fits, a categorical `cat` column containing `no`/`yes` becomes an `int64` column of codes in the input object itself. Reusing that training frame or comparing it with untouched validation data can therefore change the downstream model's inputs.

   Remedy: encode an owned copy at the selector boundary and preserve the public input. Trace all five encoder call sites before changing the helper's mutation contract. Otherwise calling feature selection changes data that other pipeline steps may still use.

6. **Categorical predictors and their shadows do not retain matching model semantics.** P1; high confidence; medium effort. Locations: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 549–554, 1787–1789, 1880, 1910–1911, 2071, and 2286–2314.

   BoostAGroota passes categorical positions from the original frame without adding the corresponding shadow positions. After a reduction round it redetects types from already encoded integers and loses the remaining categorical positions altogether. GrootCV encodes first, then redetects types, so it supplies an empty categorical list even on its first fold. Leshy's permutation also converts numeric-valued pandas categories into ordinary numeric shadow columns.

   Argument-recording checks observe BoostAGroota metadata `[0]` and then `None`, where the original and its shadow require `[0, 2]` and then `[0, 1]`. GrootCV supplies `[]` on both folds of a categorical fixture. Leshy preserves the real column as categorical but produces numeric shadows from categories `10, 20, 30`.

   Remedy: preserve the categorical identities across encoding and feature removal, and give each shadow the same categorical role as its original. The checks prove the mismatch, not its population false-selection rate. Otherwise nominal predictors are treated as ordered numbers or compared against controls with different splitting rules.

7. **GrootCV's multiclass objective aliases crash during SHAP aggregation.** P1; high confidence; small effort. Locations: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 2319–2321, 2418–2423, and 2600–2605.

   Parameter setup resolves multiclass classification to `softmax`, but the importance function receives the original objective argument. Unless that argument was literally `softmax`, it leaves the output/class axis intact and then tries to sort vectors as scalar feature scores.

   Real three-class fits with `objective="multiclass"` and automatic promotion from `objective="binary"` both raise the ambiguous-array-truth-value `ValueError`. The same fixture completes with explicit `softmax`. SHAP documents the [additional output axis](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) for multi-output models; the independent aggregation reference checks that axis directly.

   Remedy: use the resolved objective/output shape consistently for training and aggregation. Otherwise equivalent supported multiclass objective spellings do not give equivalent working fits.

8. **GrootCV does not implement its documented maximum of shadow medians.** P1; high confidence; small effort once the intended rule is confirmed. Locations: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 1943–1944 and 2332–2341; [Methods overview](../docs/Methods%20overview.rst), lines 69–74.

   A column called `Med` stores the mean, not the median. The threshold then averages each fold's maximum shadow importance and also the maximum of that extra mean column. This is a third formula, rather than either the maximum shadow median or a pure average of fold maxima.

   In the three-fold reference, a real feature has values `[0.57, 0.57, 0.16]`; the shadows have `[0.35, 0.05, 0.70]` and `[0.05, 0.35, 0.05]`. The documented rule retains it: real median `0.57` exceeds maximum shadow median `0.35`. The implementation rejects it: real mean `0.433333` is below its threshold `0.441667`.

   Remedy: choose and implement one explicit aggregation rule using fold columns only, then align its names, prose, and hand-calculated tests. Otherwise the published method and the actual selected features disagree.

9. **BoostAGroota's shadow threshold contradicts its API description.** P2; high confidence in the mismatch; small effort after deciding the contract. Locations: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 1439–1440 and 1811–1815; [Methods overview](../docs/Methods%20overview.rst), lines 25–34.

   The parameter description says maximum shadow importance divided by `cutoff`. The code instead takes each shadow's maximum across iterations and averages those values. With one iteration and normalized real scores `[0.3, 0.1]` and shadow scores `[0.5, 0.1]`, it uses `0.3`, not `0.5`, and retains the `0.3` feature.

   The prose describing the original BoostARoota algorithm elsewhere uses an average of shadows, so the documentation itself does not settle which adaptation ARFS intends. The executable case asserts the local API's maximum rule; it does not prove that a maximum is universally preferable.

   Remedy: settle the intended rule, document its exact axes and order of operations, and pin it with a reference example. Otherwise `cutoff=1` does not mean the advertised comparison with the strongest shadow.

10. **Generated shadow names can collide with real feature names.** P2; high confidence; small effort. Location: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 1695–1705; shared callers at 1780 and 2303–2304.

    A unique input with a feature named `ShadowVar1` or `ShadowVar2` collides with generated controls. Reindexing by the resulting duplicate labels expands a two-feature input to six columns instead of four. Real/shadow membership can no longer be determined safely from those labels.

    Remedy: create collision-free labels and track the generated names exactly; do not reserve an undocumented feature namespace. Otherwise an ordinary user-supplied name can break model fitting or real-versus-shadow bookkeeping in BoostAGroota and GrootCV.

11. **Leshy silently replaces missing response values with zero.** P1; high confidence; small effort. Location: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), lines 446–449.

    Before validation, `y.fillna(0)` turns unknown outcomes into observed zeros. The real regression fixture with one missing target completes successfully instead of reporting invalid training labels. For classification this also risks creating or enlarging class zero.

    Remedy: reject missing responses, or apply an explicitly supported row-removal policy that also filters predictors and weights. Otherwise selection uses invented response values without informing the caller.

**Small statistical screen**

The [screen script](evidence/01b-all-relevant/screen.py) ran **80 selector fits**: five seeds, regression/binary classification, signal/null responses, and four selector configurations. Every dataset has eight independent standard-normal predictors, 300 training rows, and 1,200 independent test rows. In signal cases, regression uses `3*x0 + 2*x1 + N(0,1)`; classification thresholds that response at zero. Null responses are independent noise. Thus signal cases have exactly two relevant predictors and six irrelevant predictors; null cases have eight irrelevant predictors.

All importances in this screen use SHAP. Leshy uses 20 iterations, 30 trees, and its default percentile 90. BoostAGroota uses three iterations and at most three rounds, with both `cutoff=1` and its default cutoff 4. GrootCV uses three folds and one repeat. These deliberately small settings are **not complete factory-default runs**. Model settings and seeds are explicit in the script; NumPy/Python global seeds are reset before every selector fit because BoostAGroota/GrootCV shadows use global NumPy randomness. CPU/thread limits are recorded.

Every configuration recovered both planted signals in all five seeds of both tasks. Noise selection was much less reassuring:

| Configuration | Signal: mean extra features, regression / classification | Null: runs selecting anything, regression / classification | Null: mean features selected, regression / classification |
| --- | --- | --- | --- |
| Leshy | 0.8 / 0.4 | 3/5 / 4/5 | 0.8 / 1.0 |
| BoostAGroota, cutoff 1 | 0.6 / 0.8 | 4/5 / 4/5 | 1.8 / 1.4 |
| BoostAGroota, cutoff 4 | 3.2 / 4.0 | 5/5 / 5/5 | 7.2 / 7.2 |
| GrootCV | 0.4 / 1.2 | 2/5 / 3/5 | 1.2 / 0.8 |

For predictive context, fresh models fitted on the selected columns were compared on untouched test observations with the same model using all columns, the two largest absolute training-only univariate correlations, the known signal columns, and a constant dummy. Empty selections use the dummy. The oracle columns are used only as a comparison, never to tune selection.

In signal regression, mean test R² was 0.8726–0.8753 across selector configurations, versus 0.8726 for all columns and 0.8762 for both the univariate and oracle baselines. Signal classification accuracy was 0.8903–0.8913, versus 0.8890 for all columns and 0.8917 for univariate/oracle. The task is intentionally easy: these results demonstrate recovery of strong effects, without establishing an advantage over the simpler baseline.

On null regression, selector/model mean test R² ranged from -0.0914 to -0.0380, versus -0.0041 for the dummy. Null classification accuracy was 0.5062–0.5168, versus 0.5157 for the dummy. Finite-test fluctuations around chance are expected; these are observations from five seeds, not significance tests or general performance estimates. Full feature lists, scores, and baselines are in [screen.json](evidence/01b-all-relevant/screen.json), with [progress](evidence/01b-all-relevant/screen-progress.txt) and [model logs](evidence/01b-all-relevant/screen-log.txt).

The null cases show why the existing “non-empty intersection with expected features” assertions are inadequate. They do **not** establish a formal false-discovery rate, family-wise error rate, or a universal ranking of these methods. Leshy's binomial arithmetic can be correct while its assumed null hit probability and repeated-testing calibration remain unvalidated. BoostAGroota and GrootCV expose empirical cutoffs rather than a fitted significance guarantee. GrootCV computes SHAP values on each fold's **training** observations; validation observations drive early stopping, so its internal importances should not be described as held-out explanations.

**Reproduction**

From the repository root, reuse the verified frozen wheel and run outside the checkout:

```sh
repo="$PWD"
review_python=/tmp/arfs-review-03f67d0/locked-env/bin/python
audit_dir=$(mktemp -d /tmp/arfs-all-relevant-checks.XXXXXX)
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$audit_dir/mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONHASHSEED=0
cd "$audit_dir"
"$review_python" -m pytest \
  "$repo/reviews/evidence/01b-all-relevant/checks.py" \
  -q -rx --tb=short -c /dev/null \
  --rootdir="$repo/reviews/evidence/01b-all-relevant" -p no:cacheprovider
# Expected: 8 passed, 16 xfailed; exit 0.
# Add --runxfail: 16 failed, 8 passed; exit 1.

"$review_python" "$repo/reviews/evidence/01b-all-relevant/screen.py" \
  "$audit_dir/screen" > "$audit_dir/screen-progress.txt" 2>&1
# Expected: 80 recorded selector fits; exit 0.
```

If the environment has gone, follow [unit 0's archive/build instructions](00-baseline.md), install that audited wheel and this unit's requirements in a new Python 3.12.7 environment, and verify module paths. The pins include the import workaround deliberately. Current dependency resolution or the published PyPI package is not an equivalent target.

**Scope and handoff**

Source tracing covered all three fit paths, shadow construction, decision/support/ranking calculations, shared native/SHAP/permutation helpers, categorical encoding and all its callers, fold/weight splitting, LightGBM parameter setup, SHAP output aggregation, and reduction thresholds. Existing selector tests, exports, and methods prose were inspected. The evidence remains outside the normal test suite and does not repair its weak assertions or baseline failures.

Prioritize caller-data preservation and categorical metadata together, then weighted/signed permutation scoring, Leshy support/history decisions, and GrootCV objective/aggregation consistency. Resolve the documented threshold contracts explicitly before changing selected-feature behavior. Move the relevant reference cases into normal tests during remediation. Unit 0's clean-import and pandas shadow-write blockers and unit 1A's association/MRMR findings remain open.

Unit **1B is reviewed**, with remediation outstanding. The next staged label is **1C**; its original detailed scope is still absent from the local review files. Weighted GLM/Lasso selection is a reasonable proposed next subject, not a confirmed item recovered from the original plan. Units 1C–7 remain unreviewed; the single-distribution working assumption is unchanged.

Not checked: large-scale statistical calibration, weak/correlated/interacting signals, high-cardinality stress, weighted or categorical repeated-seed screens, temporal/grouped validation, the complete factory-default configurations, current-dependency compatibility, CatBoost/XGBoost/fasttreeshap/GPU paths, full estimator/serialization contracts, notebook execution, the full package suite, or a Python/platform matrix. No library fix or release-readiness claim is made.
