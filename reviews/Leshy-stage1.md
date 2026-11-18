# Leshy — implementation cleanup

This change simplifies Leshy without changing its shadow comparison or statistical
selection procedure. The baseline is the working-tree source captured before this
change, including the pending GrootCV implementation work. The Leshy class drops
from 947 to 544 lines, including docstrings.

## Changes and migration

- Replace Leshy's `importance="fastshap"` with `importance="shap"`. SHAP, signed
  permutation importance (`"pimp"`) and native importance remain supported.
  BoostAGroota and VariableImportance keep their optional FastTreeSHAP paths.
- Call `fit` instead of the removed `select_features` orchestration method.
  Constructor defaults, fitted result names, DataFrame transforms and plots retain
  their contracts. `n_features_in_` records input width and `n_iter_` records
  completed iterations, including early completion.
- Collect history vectors in a list and assemble the matrix once. Hit counting no
  longer changes those vectors. Close the progress bar on success and failure.
- Use statsmodels' existing Benjamini–Hochberg implementation rather than its
  copied implementation. The binomial probabilities, correction branches,
  percentile comparisons and tentative-feature policy remain unchanged.
- Use native LightGBM SHAP contributions in the shared Leshy/BoostAGroota helper;
  retain TreeExplainer for other compatible estimators. The existing shared
  reducer validates shape/finiteness and drops every native output's bias column.
  Binary/list and multiclass aggregation conventions remain unchanged.
- Validate current parameters during `fit`, including changes through
  `set_params`. Reject invalid iteration/tree counts, unsupported importance
  modes, nonfinite percentile/alpha values and duplicate input names. Selection
  operations require a completed fit and raise `NotFittedError` otherwise; failed
  refitting invalidates the previous selection.
- Correct automatic tree sizing: unlimited depth uses the historical depth-10
  fallback, and the resulting count is positive. Previously, default LightGBM
  settings produced negative tree counts. Refit and recheck downstream results
  when using `n_estimators="auto"`.

Learner and input ownership, shadow multiplicity and naming, permutation order,
RNG draws, active-feature inclusion, fixed validation splitting, equal-row SHAP
averages, signed permutation scores, thresholds, support and ranking rules are
unchanged. No dependency, export, version or new tuning parameter is added.
Maintained notebook calls are migrated and outputs from changed cells are cleared.
Historical review evidence and the pending GrootCV implementation are preserved.

## Verification

The maintained regression tests cover current-parameter validation after cloning
and `set_params`, unfitted operations, positive automatic tree counts, early
completion and iteration reporting, missing history values, both binomial
correction branches, and native/TreeExplainer parity for weighted categorical
regression, binary, multiclass and LightGBM random-forest fits. Existing suites
cover selection masks/rankings, signed permutation scores, shadow dtypes and names,
RNG replay, input/learner ownership, plots and shared importance callers.

Full source suite: **2177 passed, 609 warnings in 452.59s (0:07:32)**.

| Python / dependencies | Installed-wheel suite |
| --- | --- |
| 3.9 / locked | 2177 passed |
| 3.9 / minimum | 2177 passed |
| 3.10 / locked | 2177 passed |
| 3.11 / locked | 2175 passed; 2 timeouts; 8 persistence cases passed on retry |
| 3.12 / locked | 2177 passed |
| 3.12 / newer | 2161 passed |

The initial Python 3.11 run hit the existing 60-second subprocess timeout in two
GBM persistence tests (offset binary and softmax). All eight fresh-process cases
passed on targeted retry (55.13 seconds total). These subprocesses import the
unchanged GBM module, without Leshy. No test timeout or unrelated code was changed;
the manifest preserves the initial failures and retry.

Each cell installs a wheel built through an sdist, imports it outside the checkout
with isolated Python, checks package ownership/version and the absence of setuptools,
and runs the maintained tests. Source/sdist/wheel Python modules and copied tests
match. Ruff, the unchanged lockfile check, `git diff --check` and the standard
Sphinx `-b html -W --keep-going` build pass. Existing model/tag/numerical warnings
remain visible. [Validation manifest](evidence/leshy-stage1/validation.json).

## Measurements

| Case | Rows × features | Baseline fit (s) | Revised fit (s) | Baseline / revised peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| Tall LightGBM regression | 3000 × 25 | 2.017 | 1.857 | 285.2 / 284.6 |
| Wide LightGBM regression | 500 × 301 | 4.714 | 4.554 | 303.5 / 302.9 |
| LightGBM multiclass | 1000 × 25 | 3.162 | 2.852 | 280.7 / 280.1 |
| Random forest | 500 × 13 | 3.579 | 3.548 | 273.8 / 273.3 |
| LightGBM permutation | 200 × 7 | 0.867 | 0.887 | 277.2 / 276.5 |

These are median full-fit wall times. Every case matches the baseline exactly
for histories, thresholds, cumulative hits, decisions, learner seeds, masks,
rankings and selected names. All three fits replay exactly within each snapshot.
Timing changes are workload-dependent; the table does not establish a universal
speedup. [Raw benchmark evidence](evidence/leshy-stage1/benchmark.json) records
timings, memory, settings, versions, source hashes and complete traces. The table
uses a complete repeat after the test suites finished. The evidence also retains the
initial measurements taken during concurrent suite execution.

## Reproduction

The [benchmark runner](../tests/leshy_benchmark.py) runs against either source
snapshot through `PYTHONPATH`. Each case performs three single-worker fits with
fixed seed 42 and verifies exact replay and input/estimator ownership. It records
full importance histories, shadow thresholds, cumulative hits, decisions, learner
seeds, masks, rankings and selected names alongside timings, source hashes and
package versions. Compare score arrays with `rtol=1e-12, atol=1e-14`, and require
exact agreement for discrete state and selected names.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 MPLBACKEND=Agg \
PYTHONPATH=src python tests/leshy_benchmark.py wide /tmp/leshy-wide.json
```

Cases are `tall`, `wide`, `multiclass`, `forest` and `permutation`. All include one
categorical predictor and observation weights. Models use 40 trees and depth 3;
LightGBM uses seven leaves and minimum leaf size 15. The first four allow 20
selection iterations; permutation uses four iterations to bound its repeated
scoring cost. These workloads are not factory-default performance measurements.
Peak process RSS includes imports and all three fits; it does not isolate history
or contribution-array storage. Three fits on one machine cannot establish a
universal speedup.

Hosted CI, full notebooks, GPU execution, optional FastTreeSHAP and CatBoost/XGBoost
execution, and statistical recalibration are outside this verification. This is
an implementation cleanup, not evidence of improved selection quality or
calibrated false-discovery control.
