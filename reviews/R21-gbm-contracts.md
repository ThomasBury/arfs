# R21 — GBM prediction and split contracts

**R21 complete on 2026-10-14**, based on clean `main` at `d51c567`.
Local phase commit: `6204a3f`. R21 runs alone, with R05/R06 complete.

## Changes and decisions

GradientBoosting rejects `groups` together with `stratified=True`. The shared
split helper enforces this boundary, so no caller silently loses group isolation.
Group, stratified and ordinary splits separately retain the existing fixed
80/20 split and seed 42. Features, responses, weights and initial scores remain
aligned by row position. There is no combined stratified-group algorithm or
full-data refit after early stopping.

Both prediction methods require DataFrames to contain the full fitted names in
the fitted order; reordered, replaced, missing, extra and duplicate names raise
ValueError before native prediction. Names are compared as strings, matching
LightGBM's name conversion. Ambiguous fit names, including integer/string names
that collide after conversion, are rejected. Arrays remain positional and must
be two-dimensional with the fitted width. One-row inputs, native raw margins,
leaf IDs and feature contributions retain their native shapes.

feature_names_in_ retains the original stringified input names. Named prediction
inputs get native model names on a new frame, so LightGBM's replacement of spaces
with underscores does not break prediction on the original fitted schema.
validate_features=False cannot bypass the wrapper's schema check. Loading an
ordinary Booster refreshes this new schema field from its native feature names;
that one-line caller repair prevents a newly introduced stale-schema failure.
Original pre-normalization names still require R22's wrapper metadata persistence.

Classification label conversion recognizes multiclass, softmax, multiclassova,
multiclass_ova, ova and ovr. Probability requests preserve native output: binary
returns a positive-class vector, multiclass returns a row-by-class matrix, and
one-versus-all scores are not renormalized. Labels remain native numeric class
indices; no arbitrary string-label encoder or custom-objective link is inferred.
Regression and callable-objective predictions retain native behavior.

Automatic categorical selection covers pandas category and Boolean columns.
Explicit names also allow numeric category codes, including integer feature
names. None and an empty list disable categorical splitting through an explicit
empty native selection; pandas categories then use native numeric codes, with
native missing/unknown handling. Object/string, time and other unsupported feature
dtypes raise an actionable error: convert object/string columns to pandas category
and engineer numeric time/other features before fitting. No new encoder or learned
vocabulary is introduced. Category-selection lists and caller inputs are preserved.

Initial scores must be finite and have shape (n_samples,) for single-output
models or (n_samples, num_class) for multiclass. Both num_class and its native
num_classes alias are recognized. Multiclass arrays and DataFrames preserve
class columns in native numeric class order and split rows positionally;
DataFrame indices are not joined. Flat multiclass scores, single-output column
matrices, wrong widths/lengths and nonfinite scores fail clearly. Predictions on
new rows still require an external baseline: use predict_raw(raw_score=True),
add that baseline on the raw scale, then apply the objective's inverse link or
native output transformation. Training offsets cannot supply new-row baselines.

Successful refits replace the initial-score guard, category names, original input
schema, model parameters, validation features and learning curve. Disabled outputs
become None. Validation/native-training failures preserve the last successfully
fitted contract. Clearing an old curve reference does not close a caller-owned
figure. fit still returns None; this wrapper is not a sklearn estimator.

Caller tracing covered both prediction methods, the shared trainer/split helper,
TreeDiscretizer, VariableImportance, maintained input/category/dependency checks,
benchmark/native controls, save/load and notebook source. API autodoc reads the
updated docstrings; the Unreleased changelog records the intentional contracts.
Historical review reports, scripts and logs are unchanged. No dependency, backend,
compatibility shim or following phase was added. Native behavior was checked
through Context7 against LightGBM's [Dataset implementation](https://lightgbm.readthedocs.io/en/stable/_modules/lightgbm/basic.html)
and [objective parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

## Validation

Exact commands and check boundaries are in
[evidence/r21-gbm-contracts/commands.txt](evidence/r21-gbm-contracts/commands.txt).
Fresh installed-wheel full suites pass **1031 / 1031 / 1015** in
py39-locked / py312-locked / py312-newer. All **39** selected unchanged historical
checks pass unmasked in both Python 3.12 cells. Group-conflict and unsupported-dtype
rejection policies are verified by maintained tests because the old historical
cases expected successful fitting. The old missing-feature case specifies the
backend exception type; the maintained suite verifies the new wrapper ValueError.

The maintained R21 file adds **56** cases. On the verified R20 wheel, **45 fail
and 11 pass**; all 12 installed source-module hashes match base d51c567 and there
is no setuptools workaround. The original R21 counterexamples reproduce **15
failures** before repair. Affected source/caller checks pass **389** tests before
the final ordinary-load schema control; complete installed runs include that case.
Independent weighted native models verify categories, aliases, row-varying offsets,
raw contributions and both softmax/one-versus-all validation log losses.

Locked cells use Python 3.9.25 / 3.12.7, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, LightGBM 4.6.0 and pytest 8.3.5. Newer uses Python 3.12.7,
NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, LightGBM 4.7.0 and pytest 9.1.1.
Per-cell JSON records full versions, exact commands/cwd/exit codes, installed
paths and hashes. All copied tests and 12 installed source-module hashes match
final source; all 19 source/sdist/wheel payloads agree. Dependency checks and
uv lock --check pass. Ruff 0.17.0's baseline gate passes (**395 total, zero new**);
one resolved GBM BLE001 allowance is removed. Gate unit tests (**2 passed**),
changed-test/evidence lint and format, handoff links and whitespace checks pass.

Existing warnings remain visible. Hosted CI, docs/notebook execution, full sklearn
compliance, optional backends, GPU/distributed training, ranking/query datasets,
other platforms, performance, statistical calibration and cross-process artifact
repairs are not checked. Passing R21 does not establish release readiness.

## Resume

Stop after R21. Next group: **R22 alone**, with R21 complete.
Read [04 F10](04-training-and-benchmarks.md), GradientBoosting.save/load and
[evidence/04-training-and-benchmarks/reload.py](evidence/04-training-and-benchmarks/reload.py).
Persist the required wrapper metadata, including offset guard, categorical names
and original fitted schema; reset loaded state coherently. Decide explicit
handling of legacy Booster-only files without guessing an unknown offset contract.
Verify trusted-input/type/error boundaries, fresh-process ordinary/category/offset
round trips, required external baselines and safe write failures. R22–R30 and
R32–R33 remain pending; R31 is complete. No publication, version bump or release.
