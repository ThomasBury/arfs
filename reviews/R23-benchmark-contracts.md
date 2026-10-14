# R23 — Comparisons that run what they label

**R23 complete on 2026-10-14**, based on clean `main` at `9a6a8b7`.
Local phase commit: ff8faa8.
R23 runs alone, with R07/R10–R12/R18/R21 complete.

## Changes and decisions

sklearn_pimp_bench now splits the original predictors, targets and optional row
weights before fitting OrdinalEncoderPandas. The fitted training vocabulary
transforms both splits; a category present only in the holdout receives the
encoder's default NaN unknown code. This preserves the existing encoder policy:
estimators must support missing values if unknown or missing categories occur.
No holdout-derived vocabulary or imputation is added. Classification keeps its
stratified split; training and validation weights keep their positional alignment
and validation scoring remains weighted.

The documented two-dimensional ndarray input is retained, using a DataFrame with
stable x0, x1, ... column names. DataFrame input keeps its original names and
order, including integer names. Unsupported input types/dimensions raise TypeError
before training. Existing encoder schema/duplicate-name checks remain in force.
Highlighting reuses highlight_tick's rendered tick text, so display conversion
does not alter model inputs or importance alignment. The benchmark still fits
the supplied standalone model and returns its figure.

compare_varimp accepts **Leshy and BoostAGroota only**. GrootCV raises TypeError
before any training, including when the model list is empty; its internal LightGBM
and importance path cannot represent these estimator/method comparisons. Fit
GrootCV directly for its own algorithm. Other selector types also fail clearly.
The helper prints selections and plots for SHAP, permutation and native importance;
it does not measure common held-out predictive performance or rank model quality.

Each model/method pair constructs a fresh selector from copied top-level constructor
parameters, with the cloned comparison estimator supplied to the constructor.
Neither the supplied selector nor the supplied models are fitted or modified.
A fitted template contributes its current constructor parameters, not fitted
selection/history. Integer seeds are reused for every pair. RandomState instances
are copied at their current state, avoiding cross-case advancement and mutation
of the template RNG. A previously fitted Leshy may already have converted its
integer seed into an advanced RandomState; that current state is the template.
None keeps the usual global-RNG behavior. BoostAGroota's global shadow RNG is
unchanged and is not reset by this helper; fresh configurations do not promise
identical random shadows or calibrated comparison results.

Leshy.fit also refreshes is_cat/is_lgb from its current estimator. This repairs
the shared stale-backend cause for direct set_params/refits and comparison callers.
Constructor-time flags remain available for inspection. No selector RNG API,
importance rule, shadow algorithm, backend or dependency is added.

Caller tracing covered both benchmark helpers, OrdinalEncoderPandas, all relevant
selector constructors/fit/importance helpers, backend seed dispatch, maintained
benchmark/weight/selector tests, and every direct source and notebook reference.
Existing notebook comparison calls use Leshy or BoostAGroota. Their separate
clean-kernel setup issues remain with R32; historical review evidence is unchanged.
API docstrings and the Unreleased changelog record the input, supported-selector,
mutation and seed contracts. Context7's scikit-learn clone/get_params documentation
was consulted; the installed clone implementation and real tests verify dictionary
parameter cloning and independent RandomState copies in the supported snapshots.

## Validation

Exact commands and boundaries are in
[evidence/r23-benchmark-contracts/commands.txt](evidence/r23-benchmark-contracts/commands.txt).
Fresh installed-wheel full suites pass **1088 / 1088 / 1072** in
py39-locked / py312-locked / py312-newer. All **five** selected unchanged historical
benchmark checks pass unmasked in both Python 3.12 cells. Original held-out-vocabulary,
array, integer-name and stale-selector cases reproduce **four failures** on the
verified R22 wheel. The original GrootCV check expects a supplied estimator to run;
its acceptance policy is superseded by explicit rejection tests rather than edited
historical assertions.

The maintained suite adds **20** cases. Identical finalized tests on the R22 wheel
report **19 failures and one passing control**; all 12 installed source-module
hashes match base 9a6a8b7, without setuptools. Affected source/callers pass **316**
tests. New real fits exercise both tasks, both selectors, random forest and
LightGBM, all three importance methods, training/validation weights, unknown
categories, ndarray/integer/string labels, unchanged caller data/objects,
integer/instance seed templates and direct backend changes. Plot input labels
are rendered and checked; unchanged selector plotting is outside this phase.

Per-cell JSON records versions, exact commands/cwd/exit codes, installed paths
and hashes. All copied tests and 12 installed source-module hashes match final
source; all 19 source/sdist/wheel payloads agree. Dependency checks, uv lock --check,
Ruff baseline (**395 total, zero new**), gate unit tests (**two passed**), changed-test/
evidence lint and format, local links and whitespace checks pass. Existing
warnings remain visible.

Hosted CI, docs/notebook execution, optional CatBoost/XGBoost/fasttreeshap, other
platforms, performance and statistical calibration were not checked. Unknown
categories require a missing-value-capable estimator; global RNG use is unchanged.
Passing R23 does not establish release readiness.

## Resume

Stop after R23. Next group: **R24 alone**, with R06 complete.
Read [05 F1–F5](05-sampling-and-parallel.md), gower_matrix and its sampling/nearest-
neighbor callers. Resolve pairwise missing/no-comparable-pair behavior, numeric
ranges, cross/self matrices, input masks and feature weights with independent
scalar references. R24–R30 and R32–R33 remain pending; R31 is complete.
No publication, version bump or release.
