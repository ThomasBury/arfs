# GrootCV — Stage 1

Stage 1 simplifies the implementation while retaining training-row explanations
and the median-based shadow threshold. Stage 2's selection-quality experiment
remains separate. Baseline: `ba5acb326dd1efc5ebc096c8d0f9864e25e288bb`.

## Changes and migration

- Use native LightGBM SHAP contributions and separate training from explanation.
  Remove GrootCV's `fastshap` constructor argument from existing calls. Leshy and
  BoostAGroota still had FastTreeSHAP in the measured snapshot. In 5.0.0, use
  `importance="shap"` for those selectors. VariableImportance keeps its optional
  backend.
- Reduce contributions through one shared helper, checking shape and finiteness
  and removing the bias column from every native output block. Scores stay in
  input-column order; VariableImportance uses the same reducer.
- Collect normalized fold vectors and construct the importance table once. A
  single final outer merge preserves historical row order and mixed-type feature
  labels. Public table columns, selected-feature order, support, rankings and
  plots retain their contracts.
- Honor boosting-round parameters and aliases using LightGBM's precedence.
  No supplied limit still means a 10,000-round ceiling with 20-round early
  stopping where supported. DART disables early stopping. Set an explicit round
  limit for it. Previously ignored limits can change selections, so refit and
  check downstream results.

The aggregation rule, categorical roles, fold and shadow RNG policies,
equal-row explanation averages and zero/nonfinite handling are unchanged.
Floating-point reduction order can differ; the benchmark comparison below
observed score differences of at most `1.67e-16`.

Affected notebook sources use the new constructor. Obsolete backend comparisons
and outputs from changed cells are removed; the historical comparison-notebook
filename remains valid for existing links. Other historical review evidence is
unchanged.

## Measurements

The [benchmark runner](../tests/grootcv_benchmark.py) compares the baseline and
revised source in separate Linux processes. Each case uses three timed fits,
one CPU thread, fixed seed 42, observation weights, one categorical predictor,
and identical learner settings without a boosting-round cap. Learners use
seven leaves, depth three and minimum leaf size 15; these are bounded workloads,
not factory-default learner benchmarks. The 25-fold case uses the default
five-fold/five-repeat schedule.

| Case | Rows × features | Folds × repeats | Baseline fit (s) | Revised fit (s) | Baseline / revised peak RSS (MiB) |
| --- | --- | --- | ---: | ---: | ---: |
| Tall regression | 3,000 × 25 | 3 × 2 | 1.774 | 1.672 | 303.7 / 286.5 |
| Wide regression | 500 × 301 | 3 × 2 | 4.803 | 4.656 | 303.9 / 299.3 |
| Multiclass | 1,500 × 25 | 3 × 2 | 6.189 | 5.786 | 293.8 / 286.9 |
| Default fold schedule | 800 × 13 | 5 × 5 | 2.836 | 2.865 | 282.2 / 282.2 |

These are median full-fit wall times. Three cases were about 3–7% faster; the
25-fold case was about 1% slower. Explanation time fell in every case, but
training dominates much of the work. Three repetitions on one machine do not
establish a universal speedup. Peak process RSS includes imports and all three
fits, rather than isolating contribution-array storage.

Every case reproduced its scores exactly across refits within each snapshot.
Cross-snapshot tables matched with `rtol=1e-12, atol=1e-14`, including feature
labels and row order; selected features matched exactly. The compact
[evidence JSON](evidence/grootcv-stage1/benchmark.json) records raw timings,
medians, versions, source hashes, settings, memory and comparison results.
Aggregation/bookkeeping time is the residual after timed preparation, training
and explanation; it is not an isolated aggregation-kernel measurement.

The [Leshy reproduction steps](Leshy-stage1.md#reproduction) restore the revised
GrootCV source in `$snapshots/grootcv`. From the repository root, extract the
baseline into a separate directory and run both snapshots:

```bash
mkdir "$snapshots/baseline"
git archive ba5acb3 src | tar -x -C "$snapshots/baseline"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 MPLBACKEND=Agg
PYTHONPATH="$snapshots/baseline/src" python tests/grootcv_benchmark.py wide "$snapshots/grootcv-wide-baseline.json"
PYTHONPATH="$snapshots/grootcv/src" python tests/grootcv_benchmark.py wide "$snapshots/grootcv-wide-revised.json"
```

## Verification

| Python / dependencies | Installed-wheel suite |
| --- | --- |
| 3.9 / locked | 2,110 passed |
| 3.9 / minimum | 2,110 passed |
| 3.10 / locked | 2,110 passed |
| 3.11 / locked | 2,110 passed |
| 3.12 / locked | 2,110 passed |
| 3.12 / newer | 2,094 passed |

[Validation manifest](evidence/grootcv-stage1/validation.json).

The wheel was built through an sdist. Its Python modules match current source;
each test cell imports the installed package outside the checkout with `-I`,
checks package/version ownership and the absence of setuptools, and uses the
existing locked/minimum/newer CI requirement sets. The new checks compare native
and TreeExplainer fold scores against independent legacy aggregation for
regression, binary, multiclass and random-forest models with categories and
weights. They also cover bias blocks, malformed shapes, real early stopping,
boosting limits, alias precedence and parameter ownership. Existing suites cover
zero/nonfinite results, custom folds, RNG replay, input ownership, rankings,
plots, quiet modes and the shared VariableImportance path. Notebook constructor
calls are checked for removed arguments.

Ruff, the unchanged lockfile check, `git diff --check`, and the project's
Sphinx `-b html -W --keep-going` build pass. Notebook execution is disabled by
the existing Sphinx configuration. An additional exploratory `-n` cross-reference
lint failed with 1,236 unresolved API references outside the established build
check; it was not used as the documentation acceptance gate.

Full notebooks, GPU execution, hosted CI and statistical calibration were not
checked. Stage 2 remains pending; this change makes no selection-quality or
false-discovery-control claim.
