# R19 — Actual fitted tree partitions

**R19 complete on 2026-10-14**, based on clean `main` at `132e0e0`.
Local phase commit: `6af7415`. R19 runs alone. No R20 implementation or publication is included.

## Changes and decisions

Numerical bins now follow the fitted single tree's exact split thresholds and
leaf IDs. A walk of its numerical <= split structure assigns each leaf its
right-closed interval; native pred_leaf output selects the interval for each row.
The sorted endpoints in bin_upper_bound_dic are fitted thresholds bounded by
negative/positive infinity, without observed-max or display-precision rounding.
Separate leaves remain separate even when their response values are identical.
num_bins_as_category retains the same intervals as ordered pandas categories.

Missing values use the native tree route, including zero_as_missing behavior
when requested. Missing/unknown categories use native NaN encoding, not a literal
sentinel label. An observed category named missing_added is ordinary data.
All-missing numerical input yields one (-inf, inf] bin; all-missing categorical
input yields its single native leaf group. Raw outputs remain finite native
response predictions on these fixtures. No observed-value representative is
invented. An unbounded one-bin interval does not promise a finite numeric
midpoint; the midpoint helper's existing infinite-endpoint behavior remains.

Categorical fit and both transform branches use the R18 OrdinalEncoderPandas
component, including nullable strings/booleans and unknown values. Grouping uses
actual leaf IDs; labels include the leaf ID before the existing short readable
category text so truncated text cannot merge distinct groups. No generated
prediction column is written into the shared frame, so x/x_g and column order
cannot change another feature's fit. Inputs, row indices and unselected columns
remain owned/preserved. Categorical groups now return the documented category dtype.

Raw mode returns native response-scale values: regression predictions or binary
probabilities, not classification labels or logits. Plain binary groups still
represent all tree leaves, irrespective of which class a leaf would predict.
The shared GradientBoosting prediction API is unchanged; R19 uses its existing
predict_raw entry point for native values and leaf IDs.

Constructor boost_params is retained unchanged (including None). Fit validates
current task, n_bins and dictionary type, derives a copy in boost_params_, removes
conflicting documented objective/iteration/leaf aliases, and sets current rmse or
binary objective, one round and the maximum n_bins leaves. n_bins must be an
integer >=2, excluding booleans. Constant/all-missing input can have one leaf.
Fitted dictionaries are rebuilt on fit. Other LightGBM options remain forwarded;
numerical interval construction requires numerical univariate splits.

The old heuristic tree_imputer attribute is removed; leaf_interval_dic_ exposes
actual numerical leaf intervals. cat_bin_dict keys are now leaf IDs rather than
prediction values; ordinal_encoder_dic holds fitted pandas encoders. Refit previously saved discretizers to build the new fitted maps; old fitted
dictionaries are not migrated. There are no compatibility aliases for these
changed fitted-state contracts. The API docs
and Unreleased changelog record them. Existing notebook code/saved outputs are
not rewritten or claimed as rerun.

Training still uses GradientBoosting's fixed 80/20 training/validation split;
independent native weighted references reproduce that training subset. Binning
is supervised preprocessing and must fit inside training folds. It does not
implement group/fused penalties or establish held-out improvement/calibration.

Caller tracing covered TreeDiscretizer's public/internal methods, GradientBoosting
fit/predict/predict_raw and its split/parameter handling, R18 ordinal fit/transform/
inverse, shared dtype/grouping helpers, maintained pandas/observation/dependency
callers, API autodoc and preprocessing/Lasso/MRMR notebook source. No training
wrapper change, new helper framework, dependency or backend is added.
LightGBM's [Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)
and [parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) were
checked through Context7 for dump/leaf semantics, missing routes and aliases.
Installed 4.6.0 alias metadata was checked read-only; production uses documented
names without importing private alias machinery.

## Validation

Exact commands, identities and check limits are retained under
[evidence/r19-tree-partitions/commands.txt](evidence/r19-tree-partitions/commands.txt).
Final fresh installed-wheel suites pass: **py39-locked 922; py312-locked 922;
py312-newer 906**. All **14** adapted historical tree checks pass unmasked in
both Python 3.12 cells. The first 58-case wheel matrix passed 920/920/904;
its separate interim logs are retained alongside the later column-label failure
and the final 60-case suite results.
The 60 maintained cases report **49 failed, 11 passed** on the verified R31 wheel;
preprocessing source bytes match `132e0e0`. The original selected historical
checks reproduce **11 failures and 3 passing controls** unmasked.

The historical module imports cat_var, intentionally removed by R31. A disposable
copy removes exactly that import line, with the diff and both SHA256 hashes
retained. No assertion or test body changes. Both before/after historical runs
use that same copy and --runxfail; original reports/scripts/logs remain intact.
The two R20-specific tree refit/array checks are excluded from this phase.

The final source run with tree/pandas, ordinal, categorical identity, dependency
and observation-boundary callers passes **338 tests**. Newer source tree/pandas
checks pass **65 tests**; the frozen source has more Copy-on-Write combinations.
Tests cover actual native partitions and thresholds/adjacent floating values,
equal-valued categorical leaves via real Booster leaf-output edits, truncated
group labels, x/x_g and column-order equivalence, independent weighted native
regression/binary fits, all 11 iteration aliases, conflicting leaf/objective
aliases and changed current parameters, missing/unknown/zero routes, all-missing
numeric and categorical types, raw categories and binary probabilities, integer
parameter rejection, cloning on the same schema, category output, input/index
ownership, unselected passthrough and the maintained midpoint pipeline.
An additional caller probe caught an unnamed categorical-column regression in
the replacement grouping code. Explicit to_frame(name=col) preserves the label;
None/integer-label controls now run in the maintained suite. The interim wheel
reports one failure/one pass on these two controls; final source checks pass both.
Interim full-suite and historical results remain retained separately.

Locked cells use Python 3.9.25 / 3.12.7, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, statsmodels 0.14.4, LightGBM 4.6.0 and pytest 8.3.5. Newer uses
Python 3.12.7, NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, statsmodels 0.15.0,
LightGBM 4.7.0 and pytest 9.1.1. Per-cell JSON lists full versions, module paths,
hashes, exact commands and exit codes. Copied tests match source bytes; all 12
installed module hashes and 19 source/sdist/wheel payloads match. All environments
install outside the checkout, without setuptools. Dependency checks, lock, Ruff
0.17.0 (397 total, zero new), gate unit tests (2 passed), test/evidence lint/format,
local handoff links and whitespace pass. Baseline allowances are unchanged.
Existing model, convergence, tag, precision and plotting warnings remain visible.
Hosted CI, full docs/notebooks, full sklearn compliance, optional numerical
backends, other platforms, performance, statistical calibration and cross-process
artifacts are not checked. Passing R19 does not establish release readiness.

## Resume

Stop after R19. Next group: **R20 alone**, with R02/R18/R19 complete.
Read [02 F6/F7/F10–F12](02-preprocessing.md) and TreeDiscretizer, IntervalToMidpoint,
transform_interval_to_midpoint and PatsyTransformer. The remaining tree issues
are constructor bin_features still becoming resolved column names, refitting a
different schema, and ndarray normalization differing between fit/transform.
R19's boost-parameter ownership/dictionary reset is not a claim that R20's broader
state/input contract is settled. Midpoint scalar names/refits and Patsy formula,
NA_action/intercept/row alignment also remain R20. R20–R30 and R32–R33 remain
pending; R31 is already complete. No version bump or release is authorized.
