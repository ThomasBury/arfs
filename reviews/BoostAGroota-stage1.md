# BoostAGroota — implementation cleanup

This phase simplifies the implementation while preserving the selection rule.
The baseline is the working-tree source captured before this change, including
pending Leshy and GrootCV work. The BoostAGroota class and its removed helpers
shrink from 554 to 324 lines, including docstrings. Leshy, GrootCV and the retained
shared importance helpers are unchanged; both BoostAGroota and GrootCV use the
revised shadow constructor.

## Changes and migration

- Replace `importance="fastshap"` with `importance="shap"`. Supported modes are
  SHAP, signed permutation importance (`"pimp"`) and native importance. LightGBM
  already uses native SHAP contributions through the shared helper; other
  compatible estimators use TreeExplainer. VariableImportance keeps its separate
  optional FastTreeSHAP path.
- Collect each round's normalized vectors in an array and construct the public
  final-round history once. Remove the mixed-type transpose round trip and the
  single-use outer orchestration helper. Constructor defaults and existing
  result attributes retain their contracts.
- Build shadow frames in one construction instead of copying and assigning each
  shuffled column. Preserve categorical/nullable dtypes, missing values, index,
  column order, collision-free names, input ownership and RNG draws.
- Validate current parameters during `fit`, including after `set_params`. Supply
  a compatible tree estimator. Positive integer `iters` and `max_rounds` are
  required; invalid round limits now raise an error instead of being coerced.
  Reject duplicate column names, empty frames, unsupported importance modes and
  malformed importance vectors before aggregation. Existing target/weight and
  zero/nonfinite-importance checks remain.
- Require a successful fit for support, transform, feature-name and plot
  operations. Failed refitting invalidates the previous selection; sklearn fitted-state
  checks use successful selection as their marker, even after a first training
  failure has created runtime attributes. `n_rounds_`
  counts completed reduction rounds and `n_features_in_` records input width.
  Construction stores options; warnings and validation run during fitting.

Per-iteration normalization, maximum shadow mean divided by cutoff, positive
mean requirement, threshold equality, stopping boundaries, first-round stable
ranking, final-round diagnostics/plots, categorical roles, fixed validation
split, equal-row SHAP averages and learner seeds are unchanged. The caller's
learner and input data remain untouched. No dependency, export, version or tuning
parameter is added. Maintained notebook calls are migrated and affected outputs
are cleared; historical review evidence is preserved.

## Measurements

The [benchmark runner](../tests/boostagroota_benchmark.py) runs each snapshot in a
fresh Linux process, with three single-worker fits, fixed seed 42, observation
weights and one categorical predictor per workload. Models use 40 trees, depth
three and (for LightGBM) seven leaves with minimum leaf size 15. Selection uses
cutoff one, at most three rounds and ten iterations per round (three for
permutation). These are bounded workloads, not factory-default measurements.

| Case | Rows × features | Baseline fit (s) | Revised fit (s) | Baseline / revised peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| Tall LightGBM regression | 3000 × 25 | 1.713 | 1.687 | 285.3 / 284.0 |
| Wide LightGBM regression | 500 × 301 | 4.951 | 4.258 | 301.0 / 300.8 |
| LightGBM multiclass | 1000 × 25 | 2.599 | 2.634 | 282.6 / 280.2 |
| Random forest | 500 × 13 | 3.206 | 3.232 | 274.1 / 274.5 |
| LightGBM permutation | 200 × 7 | 0.982 | 0.919 | 279.3 / 278.0 |

These are median full-fit times. Wide regression was about 14% faster; the
other cases ranged from about 1% slower to 7% faster in this run. Peak process RSS
includes imports and trace collection across all three fits. Three repeats on
one machine do not establish a universal speedup or lower memory use.

All cases replay exactly within each snapshot. Across snapshots, every round's
scores, means and threshold agree within `rtol=1e-12, atol=1e-14`; the largest
observed score difference is `5.55e-17`. Candidate identities, selected names,
stopping decisions, history labels, support and rankings agree exactly.
[Raw benchmark evidence](evidence/boostagroota-stage1/benchmark.json) records
settings, versions, source hashes, timings, memory and full traces.

The [Leshy reproduction steps](Leshy-stage1.md#reproduction) restore this
benchmark's baseline in `$snapshots/leshy`. Extract the revised source from
`d8aa054` with `git archive d8aa054 src`. Put the desired source directory on
`PYTHONPATH`:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 MPLBACKEND=Agg \
PYTHONPATH="$snapshots/leshy/src" python tests/boostagroota_benchmark.py wide /tmp/boost-wide.json
```

Cases are `tall`, `wide`, `multiclass`, `forest` and `permutation`.

## Verification

Full source suite: **2265 passed, 728 warnings in 341.73s (0:05:41)**.

| Python / dependencies | Installed-wheel suite |
| --- | --- |
| 3.9 / locked | 2,265 passed |
| 3.9 / minimum | 2,265 passed |
| 3.10 / locked | 2,265 passed |
| 3.11 / locked | 2,265 passed |
| 3.12 / locked | 2,265 passed |
| 3.12 / newer | 2,249 passed |

The wheel was built through an sdist; all Python modules match current source.
Each matrix cell imports the wheel outside the checkout with `-I`, verifies
package/version ownership and the absence of setuptools, and passes dependency
consistency checks. Copied tests and documentation match the current checkout.
[Validation manifest](evidence/boostagroota-stage1/validation.json).

Coverage includes parameter changes through cloning and `set_params`, invalid
frames and importance shapes, successful-fit checks after first-fit and refit
failures, exact stopping boundaries, completed-round reporting, and legacy shadow
frame/RNG parity with nullable, categorical and timezone-aware columns. Existing
suites cover weighted/signed scores, zero/nonfinite values, categorical roles,
RNG replay, ownership, support/rankings, plots, quiet modes, real shared SHAP
backends, GrootCV and VariableImportance. Maintained notebook sources are checked
for removed selector modes.

Ruff 0.17.0, the unchanged lockfile check, `git diff --check` and the strict
Sphinx `-W --keep-going -b html` build pass. Notebook execution remains disabled
by the existing documentation configuration. Existing model/tag/numerical
warnings remain visible.

An initial editable-environment source run passed 2,253 cases and failed eight
GBM child-process package-location assertions. The final full source run uses a
clean interpreter with the matching wheel installed: the parent loads current
source through `PYTHONPATH`, while isolated children load that wheel. No
persistence checks were skipped or weakened. Intermediate validation was stopped
and restarted after adding the first-training-failure fitted-state regression.


Statistical recalibration, full notebooks, GPU execution, hosted CI, and optional
CatBoost/XGBoost/FastTreeSHAP execution are outside this phase. This cleanup does
not claim better selection quality or calibrated false-discovery control.
