# R20 — Reusable transformer inputs and Patsy rows

**R20 complete on 2026-10-14**, based on clean `main` at `e030fde`.
Local phase commit: `cff3c59`. R20 runs alone, with R02/R18/R19 complete.

## Changes and decisions

TreeDiscretizer retains its constructor bin_features rule/list across fits.
bin_features_ and cat_features_ hold resolved fitted columns; n_features_in_
records full input width. Each fit resolves the current schema and filtering
rule and rebuilds its existing tree/encoder/interval dictionaries. Refitting
from x/cat to z/kind and cloning fitted estimators use the configured rule,
without retaining the previous schema. There is no cat_features compatibility
alias or saved-state migration; refit previously saved transformers.

One input-preparation method now serves tree fit and transform. Arrays get
pred_0, pred_1, etc. at fit; later arrays take the fitted names positionally
and must have the same width, including arrays queried after a named-frame fit.
Named frames must contain the full fitted schema in the same order, including
passthrough columns. Duplicate feature names are rejected at fit. Transforms
return copied DataFrames; frame indices survive, while unnamed arrays get a
fresh RangeIndex. Existing tree partition, weighting, category/missing-route,
raw response and fixed training split contracts remain those recorded in R19.

IntervalToMidpoint preserves constructor cols and resolves cols_ and
columns_to_transform_ at each fit. Both midpoint entry points treat a scalar
string other than all as one column name, accept name sequences, and convert
selected interval columns once in input order. Absent/non-interval names remain
ignored. Inputs and indices are preserved; the estimator now raises a fitted
state error before transform. Infinite endpoint behavior is unchanged.

PatsyTransformer preserves constructor formula; formula_ holds the resolved
fit-time formula and design_ retains native Patsy DesignInfo, categorical levels
and stateful transforms. None discovers a fresh formula on every fit; a named
Series target is excluded when present in the predictors. Named dictionaries
retain their keys. Arrays receive generated fit names and must have the full
fitted width on transform; named data retains native formula-variable lookup,
including reordered keys and unused columns. All accepted forms become owned
DataFrames before constructing the design. No target conversion or constructor
formula mutation occurs.

The existing add_intercept=True default is retained and documented accurately.
True allows native formula intercept rules, including explicit 0 + x exclusion;
False removes INTERCEPT from a new ModelDesc without changing a caller-supplied
ModelDesc. No default flip or new policy is introduced. The existing evaluation
capture behavior is retained; explicit EvalEnvironment controls work through
both public fit paths and subsequent dictionary transforms. Implicit callable
lookup through sklearn's fit_transform wrapping frame remains outside this repair.

NA_action is passed during both fitting and transformation, for dataframe and
ndarray returns. Native raise and drop strings and NAAction objects are honored.
Explicit drop retains the surviving DataFrame row indices but removes predictor
rows only: this transformer cannot drop or align y/sample weights. Use raise in
pipelines needing one output row per input; callers opting into drop must align
other data themselves. Ndarray output carries no row-index information.

API autodoc reads the updated docstrings. The Unreleased changelog records the
schema/state/input changes and row-dropping limit. Caller tracing covered every
affected method/helper, maintained tree/pandas/ordinal/category/dependency and
observation-boundary tests, direct source references and preprocessing/Lasso/MRMR
notebook source. Existing notebook outputs and historical review evidence are
preserved. No new dependency, backend, compatibility shim or helper framework.
Patsy's [formula documentation](https://patsy.readthedocs.io/en/latest/formulas.html)
and [developer guidance](https://patsy.readthedocs.io/en/latest/library-developers.html)
were checked through Context7 for ModelDesc/INTERCEPT and fitted-design reuse.

## Validation

Exact commands and check boundaries are retained in
[evidence/r20-transformer-inputs/commands.txt](evidence/r20-transformer-inputs/commands.txt).
Fresh installed-wheel complete suites pass **975 / 975 / 959** in
py39-locked / py312-locked / py312-newer. All **16** selected historical checks
pass unmasked in both Python 3.12 cells. Affected source/caller checks pass
**391** tests. The maintained R20 file adds **53** cases; against the verified
R19 wheel on Python 3.9, **36 fail and 17 pass**. All 12 old installed-module
hashes match base e030fde; the selected original historical failures reproduce
**11 failures and five controls** before repair on Python 3.12 locked.

The historical module imports cat_var, removed in R31. The same disposable copy
before/after removes exactly that import line; the diff and both hashes are
retained. Assertions/test bodies and historical scripts/reports/logs are unchanged.
Before-fix maintained runs on Python 3.12 encounter its inherited chained-Patsy-
NameError traceback formatting bug, even with --tb=no. The aborted logs are
retained separately; Python 3.9 provides the complete before-fix run. An interim
source control incorrectly expected a frame's custom index to survive an unnamed
array and assumed implicit callable lookup through fit_transform; those test
assumptions were corrected to fresh RangeIndex and explicit EvalEnvironment.
No interpreter or production evaluation-capture patch is included.

Locked cells use Python 3.9.25 / 3.12.7, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, statsmodels 0.14.4, LightGBM 4.6.0, Patsy 1.0.1 and pytest 8.3.5.
Newer uses Python 3.12.7, NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1,
statsmodels 0.15.0, LightGBM 4.7.0, Patsy 1.0.3 and pytest 9.1.1. Per-cell
JSON records full versions, command arguments, cwd, exit codes, module paths
and hashes. All copied tests and 12 installed source-module hashes match;
all 19 source/sdist/wheel payloads agree. The fresh environments install outside
the checkout without setuptools. Dependency checks and uv lock --check pass.

Ruff 0.17.0's baseline gate passes (396 total, zero new); the resolved
preprocessing C414 allowance is removed. Gate unit tests (two passed), new-test/
evidence lint and format, local handoff links and whitespace checks pass.
Existing model, convergence, tag, precision and plotting warnings remain visible.
Hosted CI, full docs/notebooks, full sklearn compliance, optional numerical
backends, other platforms, performance, statistical calibration, implicit
fit_transform evaluation-scope repair and cross-process artifacts are not checked.
Passing R20 does not establish release readiness.

## Resume

Stop after R20. Next group: **R21 alone**, with R05/R06 complete.
Read [04 F1–F5/F8/F9](04-training-and-benchmarks.md), GradientBoosting and its
split, prediction, categorical/offset and refit callers. Settle incompatible
splitting options, prediction schema/objective shapes, supported categorical
input and multiclass offsets before the corresponding repair. R22 persistence
remains dependent on R21. R21–R30 and R32–R33 remain pending; R31 is complete.
No publication, version bump or release is included.
