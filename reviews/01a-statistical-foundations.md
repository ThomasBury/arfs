What this review covers: unit **1A, statistical foundations** of ARFS. It checks weighted associations, predictor relevance, missing values, and the scores used by minimum redundancy–maximum relevance (MRMR) selection. It assumes CPU use on pandas frames with numeric, categorical, and boolean predictors and optional sample weights.

Reviewed commit: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11. Only this report and its evidence were added. Library code, existing tests, dependencies, the lockfile, CI, and the unit 0 report were preserved.

**Result:** 13 reproducible findings. Default MRMR can discard valid predictors, choose a constant or an exact negative duplicate, and crash on a small mixed frame. Passing ordinary association calculations does not establish correct selection.

**Evidence and environment**

The review reused unit 0's disposable installed-wheel environment at `/tmp/arfs-review-03f67d0/locked-env`. Python is 3.12.7; NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1, and scikit-learn 1.6.1 are the frozen runtime versions. Its existing `setuptools==80.9.0` installation is the unit 0 import workaround, not a source repair or a clean-install success. Nothing was installed during this unit.

SHA-256 checks verified that the installed `association.py`, `parallel.py`, `utils.py`, `feature_selection/mrmr.py`, and `feature_selection/unsupervised.py` match the checkout byte for byte. See [environment and hashes](evidence/01a-statistical-foundations/environment.json) and [exact installed dependency pins](evidence/01a-statistical-foundations/requirements.txt).

The [42 runnable checks](evidence/01a-statistical-foundations/checks.py) produce **17 passes and 25 expected failures**, covering the 13 findings below. Strict expected-failure markers identify known defects; they do not make those behaviors acceptable. Running with those markers disabled produces **25 failures and 17 passes**. The [normal run](evidence/01a-statistical-foundations/checks.txt) and [unmasked failure diagnostics](evidence/01a-statistical-foundations/failures.txt) preserve the outcomes and warnings. This is a diagnostic suite, not a green package gate.

Passing checks cover unweighted Pearson and Spearman against SciPy; weighted Pearson against integer row replication and rescaling; a hand-calculated weighted rank example; directional and weighted Theil's U; bias-corrected Cramer's V; correlation ratio and frequency-weighted ANOVA; mixed matrix/series agreement with one and two workers; ordinary complete-case masking with aligned weights and an unchanged input; and the two-worker normalized Pearson path. Five independent seeds select a strong planted numeric signal. A separate noise-only case still selects the requested two features, consistent with MRMR requiring the user to choose the number of features; this is not a false-discovery-control claim.

The weighted rank check uses the convention documented by [wCorr's authors](https://american-institutes-for-research.github.io/wCorr/articles/wCorrFormulas.html): weight below the observation plus an averaged contribution from tied observations. It does not assume that weighted Spearman must equal integer row replication. The linear-F reference follows [scikit-learn's documented Pearson-based calculation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html), checked against the installed version. The grouped reference uses between-group and within-group sums of squares and their degrees of freedom, as described by [NIST](https://www.itl.nist.gov/div898/handbook/prc/section4/prc433.htm).

**Must fix**

All findings have high confidence from the included cases. Priority P1 means a supported statistical workflow gives a wrong result or fails; P2 marks a narrower input or magnitude boundary. Effort estimates describe the smallest remedy, not authorization to implement it. Changes to scoring will intentionally change selected features.

1. **Relevance normalization erases predictors and reverses the redundancy penalty.** P1; medium effort. Locations: [association.py](../src/arfs/association.py), lines 1701–1707 and 1945–1951; [mrmr.py](../src/arfs/feature_selection/mrmr.py), lines 176–184, 209, and 314.

   What this is: regression and classification separately center and divide the numeric and categorical relevance scores by their sample standard deviation. MRMR then divides relevance by redundancy.

   Problem: with one numeric and one categorical predictor, both standardized scores are NaN and both MRMR tasks raise `IndexError`. With two numeric predictors and one categorical predictor, the numeric scores become approximately `0.7071` and `-0.7071`, while the categorical predictor disappears; equal perfect predictors also empty the selection. A negative relevance divided by a larger redundancy moves closer to zero, rewarding greater redundancy instead of penalizing it.

   Fix: remove the separate z-score centering from the quotient score. Keep relevance nonnegative, define the intended common scale for mixed types, and handle equal/perfect scores and an empty candidate set explicitly; replacing every NaN with zero alone does not correct the scoring rule.

   If skipped: ordinary mixed tables can crash, lose an entire predictor type, or rank redundant features ahead of less redundant ones.

2. **An exact negative duplicate receives almost no redundancy penalty.** P1; small effort. Locations: [mrmr.py](../src/arfs/feature_selection/mrmr.py), lines 137–140 and 288–305; [association.py](../src/arfs/association.py), lines 1174–1176.

   What this is: the default redundancy function rescales signed associations to the current minimum and maximum before MRMR takes their absolute value.

   Problem: associations `[1, 0.8, -1]` become `[1, 0.9, 0]`. With fixed positive relevance scores `[3, 1, 2]`, MRMR selects `a` and its exact negative copy `neg`, rather than `a` and `b`; the counterexample isolates the default redundancy path from finding 1.

   Fix: disable min-max normalization in the default MRMR redundancy function and use absolute association magnitudes directly. Keep the existing small denominator floor for genuinely near-zero associations.

   If skipped: changing a feature's sign can turn complete redundancy into the strongest apparent reason to select it.

3. **Numeric regression relevance is not the documented linear F statistic.** P1; small effort after the weight contract in finding 7. Location: [association.py](../src/arfs/association.py), lines 1625–1633.

   What this is: `f_cont_regression_parallel` promises a univariate linear regression statistic, but calls `wcorr_series` without overriding its Spearman default. It also uses the original target length after dropping incomplete rows.

   Problem: for `x=[0,1,2,3,4]` and `y=x²`, the linear F statistic is `34.285714`; ARFS returns the largest float, approximately `1.798e308`. In another case, adding one row that is subsequently dropped changes the result from `12.789474` to `17.052632` despite identical retained observations.

   Fix: explicitly request Pearson correlation and compute degrees of freedom from the retained observations under the chosen weight convention. Preserve the documented treatment of constant and perfectly correlated inputs.

   If skipped: relevance can imply a perfect linear relationship where none exists, and missing rows inflate scores.

4. **Constant categorical predictors can outrank informative predictors.** P1; small effort. Locations: [association.py](../src/arfs/association.py), lines 217–218 and 1871–1889; [mrmr.py](../src/arfs/feature_selection/mrmr.py), lines 176–179.

   What this is: Theil's U measures the fraction of uncertainty removed from its first argument. Its constant-input convention returns one, which the classification relevance wrapper converts into a maximum finite score.

   Problem: an eight-row constant predictor scores approximately `1.798e308`, while an informative binary predictor scores `2.585851`. Default classification MRMR selects the constant first in the recorded fixture; the helper's warning that constant columns were removed is false because it only collects their names.

   Fix: assign zero relevance to predictors constant on the retained positive-weight observations, or exclude them before relevance scoring. Preserve a separately documented entropy convention in the association API rather than treating that convention as predictive information.

   If skipped: a feature that carries no information about the outcome can consume the first selection slot.

5. **A constant categorical column crashes association matrices.** P1; small effort. Location: [association.py](../src/arfs/association.py), lines 217–218 and 232–235; caller: [parallel.py](../src/arfs/parallel.py), lines 176–181 and 45–47.

   What this is: matrix construction requests `weighted_theils_u(..., as_frame=True)` and concatenates the returned frames. `CollinearityThreshold.fit` uses that matrix path at [unsupervised.py](../src/arfs/feature_selection/unsupervised.py), lines 356–366.

   Problem: the zero-entropy early return emits a float even when a frame was requested. A frame containing one constant string column and one varying string column raises `TypeError: cannot concatenate object of type '<class 'float'>'`.

   Fix: route the constant case through the existing output-formatting branch. Separately settle constant-column selection semantics under finding 4.

   If skipped: ordinary association and collinearity workflows fail before they can process constant categorical inputs.

6. **Normalized weights break only the single-worker Pearson matrix.** P1; small effort. Location: [association.py](../src/arfs/association.py), lines 1041–1052; dispatch at 986–1001.

   What this is: the optimized matrix path divides covariance and variance by `sum(weights)-1`, even though that common factor cancels in correlation.

   Problem: with four weights equal to `0.25`, `wcorr_matrix(..., method="pearson", n_jobs=1)` returns an empty frame after NaNs are dropped. Two workers and the pairwise function return the valid correlation `0.748331` on the same data.

   Fix: compute the correlation from centered weighted cross-products and sums of squares without the unnecessary sample-covariance correction.

   If skipped: changing worker count or expressing weights as proportions changes a valid result into no result.

7. **Relevance branches disagree on what a sample weight means.** P1; medium effort because the public statistical contract needs a decision. Locations: [association.py](../src/arfs/association.py), lines 1457–1464, 1629, and 1869–1876.

   What this is: grouped ANOVA treats total weight as a replicated sample count. Numeric regression and the categorical classification score instead use the number of stored rows.

   Problem: grouped observations `[1,2]` and `[4,6]` with weights `[1,2,1,2]` correctly reproduce frequency-weighted ANOVA `F=24.2`; dividing those positive weights by six gives the impossible negative statistic `-6.05`. Conversely, doubling every numeric weight gives `3.555556`, while physically repeating those rows gives `10.666667`.

   Fix: state whether these relevance functions accept frequency counts, relative observation weights, or another specified interpretation, then make degrees of freedom and input validation consistent with that choice. Reject unsupported totals instead of returning negative F values; automatically normalizing every weight vector would destroy legitimate count information.

   If skipped: the same advertised `sample_weight` parameter has incompatible meanings across predictor types, so mixed relevance has no stable statistical interpretation.

8. **Invalid weights produce plausible numbers, while valid lists crash.** P1; small effort. Locations: [association.py](../src/arfs/association.py), lines 1992–1999 and 2018; direct correlation helpers at 747–802.

   What this is: the shared frame preprocessor checks finiteness and converts pandas Series, but does not enforce one-dimensional length, nonnegative entries, or positive retained mass, and leaves Python lists unchanged.

   Problem: all-zero weights produce a series of zero associations instead of an error; negative weights are accepted. Direct Pearson correlation for `x=[0,1,2,3]`, `y=[0,2,1,3]`, and weights `[1,1,1,-0.9]` returns `-1.31818`, outside the valid range; a documented array-like list of four positive weights raises `TypeError` during masking.

   Fix: convert weights to an array once at the input boundary; validate shape, length, finiteness, nonnegativity, and positive mass after row filtering. Apply the same contract to directly callable pairwise statistics.

   If skipped: malformed weighting can silently look like independence or a meaningful association, while ordinary list input remains unusable.

9. **Zero-weight categories poison otherwise valid grouped statistics.** P1; small effort. Locations: [association.py](../src/arfs/association.py), lines 543–554, 609–613, 1451–1460, and 388–389.

   What this is: category discovery keeps levels with no positive weight, then divides by their weight or passes empty margins to a contingency-table calculation.

   Problem: append a new category with weight zero to `[1,2]` versus `[4,6]`. Correlation ratio changes from `0.911322` to NaN, ANOVA changes from `9.8` to NaN, and the categorical Cramer's V example raises an error about a zero expected frequency; `_compute_series` can subsequently replace those NaNs with misleading zeros.

   Fix: discard zero-mass groups and contingency margins before counting groups or dividing by group weights. Retain explicit handling for a dataset with no usable group or no positive weight.

   If skipped: an observation intended to contribute nothing can erase or invalidate the association of every usable observation.

10. **The missing-value policy can report independence without any complete observations.** P1; small effort. Locations: [association.py](../src/arfs/association.py), lines 2014–2018 and 1294; [parallel.py](../src/arfs/parallel.py), lines 126–139.

    What this is: the default policy drops any row containing a missing value. A nested series helper also hard-codes that policy even when the public call requests `handle_na=None`.

    Problem: when every row has a missing value, `if na_mask.any()` skips filtering entirely; complementary missingness in two columns then yields zero associations. A custom callable supplied with `handle_na=None` receives only three complete rows from a four-row input, including no missing target values.

    Fix: apply the complete-case mask unconditionally and reject an empty usable sample. Carry the requested policy through nested calls; distinguish an undefined statistic from a demonstrated zero association.

    If skipped: “no usable evidence” becomes “no relationship”, and custom statistics cannot rely on the requested sample.

11. **Large offsets destroy grouped variance calculations.** P2; small effort. Locations: [association.py](../src/arfs/association.py), lines 547–555 and 1443–1456.

    What this is: correlation ratio and ANOVA subtract large uncentered sums of squares. Nearly equal large floating-point values lose the small difference that represents actual variation.

    Problem: adding `1e10` to every value of the weighted four-row example changes correlation ratio from `0.926367` to zero and ANOVA from `24.2` to NaN. The observations remain distinct and the statistical problem is unchanged.

    Fix: center observations before accumulating sums of squares, preserving the existing weights and group structure.

    If skipped: predictors measured around a large baseline can be judged unrelated solely because of their units or origin.

12. **A feature named `target` silently disappears from MRMR.** P1; small effort. Locations: [mrmr.py](../src/arfs/feature_selection/mrmr.py), line 167; [association.py](../src/arfs/association.py), lines 1545–1555, 1621–1623, 1790–1797, and 1863–1865.

    What this is: relevance functions temporarily add the response to the predictor frame using the response's name. MRMR always names that response `target`.

    Problem: with predictors `target`, `other`, and `third`, requesting three features returns only `other` and `third`. Renaming the original feature to `signal` restores it and puts it first in relevance; the temporary assignment overwrites its copied data before scoring.

    Fix: use a collision-free temporary label in every response-augmentation path, or pass the response separately using the existing parallel helper. Do not reserve an undocumented user feature name.

    If skipped: renaming an input column changes which information is even eligible for selection.

13. **Boolean-only association series take the wrong statistical branch.** P2; small effort. Locations: [association.py](../src/arfs/association.py), lines 1143–1155 and 1317–1324; [utils.py](../src/arfs/utils.py), lines 153–163.

    What this is: the public series dispatcher uses `is_numeric_dtype`, which regards booleans as numeric, while the shared dtype dictionary treats them as categorical and excludes them from its numeric list.

    Problem: a two-column boolean frame succeeds through `association_matrix` but `association_series` raises `AttributeError` on `None.sort_values`. The private numeric branch finds no numeric predictors after the public branch has chosen it.

    Fix: reuse the shared dtype classification in the public dispatcher so both entry points choose the categorical statistic for boolean predictors.

    If skipped: the same legitimate binary data works or fails depending on which association entry point is called.

**Reproduction**

To reuse the verified unit 0 environment from the repository root:

```sh
repo="$PWD"
review_python=/tmp/arfs-review-03f67d0/locked-env/bin/python
audit_dir=$(mktemp -d /tmp/arfs-statistical-checks.XXXXXX)
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$audit_dir/mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYTHONHASHSEED=0
cd "$audit_dir"
"$review_python" -m pytest \
  "$repo/reviews/evidence/01a-statistical-foundations/checks.py" \
  -q -rx --tb=short -c /dev/null \
  --rootdir="$repo/reviews/evidence/01a-statistical-foundations" \
  -p no:cacheprovider
# Expected: 17 passed, 25 xfailed; exit 0.
# Add --runxfail to expose all 25 defects as failures; expected exit 1.
```

If the disposable environment no longer exists, recreate the audited wheel using [unit 0's archive/build instructions](00-baseline.md), make a fresh Python 3.12.7 environment, and install that wheel together with this unit's `requirements.txt`. Those pins intentionally include the historical setuptools import workaround. Run from outside the checkout and verify `arfs.__file__`; neither a clean current dependency resolution nor the published PyPI package is an equivalent test target.

**Scope and handoff**

Source tracing covered the association primitives, dtype dispatch, matrix/series wrappers, parallel helpers, every relevance wrapper, MRMR's scoring loop, and their collinearity-selector caller. Existing tests, API prose, the MRMR/association notebook sources, and the unit 0 evidence were inspected. The repo has no existing direct calculation assertions for this statistical surface; the new checks are review evidence only, outside the established test suite.

The fixes should reach the shared statistical functions, both relevance tasks, default MRMR scoring, matrix/series agreement, and the affected collinearity caller. Move the relevant reference cases into normal regression tests when implementing fixes; then validate mixed-type selection again. Repair the two normalization rules first, alongside constant relevance and the regression statistic, before interpreting larger MRMR benchmarks. Keep the weight-meaning decision explicit rather than hiding it in normalization. Unit 0's import, packaging, shadow-generation, and baseline-test findings remain open.

Unresolved statistical contract: categorical classification currently maps directional Theil's U through `U²/(1-U²) * (n-2)`. The arithmetic is implemented, but this review establishes no F-distribution calibration for that mapping or comparability with numeric ANOVA. Treat it as a ranking heuristic until a justified mixed-type relevance definition is chosen; a finite value alone is not a significance result.

Unit **1A is reviewed**, with remediation outstanding. The next staged review is **1B**; its detailed scope is not defined in the local unit 0 handoff and was not invented here. Units 1B–7 remain unreviewed; the single-distribution working assumption is unchanged.

Verdict: fix the reported statistical failures before treating default weighted MRMR as reliable.

Not checked: all-relevant selector significance tests, repeated-seed false-selection rates, held-out model comparisons, full notebook execution, estimator cloning/serialization contracts, current dependency compatibility, a Python/platform matrix, performance at scale, and GPU/optional backends. No library fix or broader release-readiness claim was made.
