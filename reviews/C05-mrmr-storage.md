# C05 — MRMR storage and built-in reductions

Implemented on `perf/c05-mrmr-storage-reducers`, based on `9383bdd` on updated
`main`. This phase changes MRMR storage and recognized denominator reductions;
it keeps the association metrics and selection rules.

## Behavior and migration

The former redundancy table allocated every available predictor against every
other predictor. The table now starts with no columns. Each selection step after
the first adds the preceding selected predictor, initialized to the existing
0.001 floor, then fills the same candidate/domain pairs as before. For p available
predictors and k selections, the table retains p × max(k − 1, 0) float64 values,
plus pandas index overhead. Selecting every predictor still has quadratic cost.
The final selected predictor needs no column because no subsequent step uses it.

Code inspecting `selector.redundancy` must accept the smaller table. Its columns
follow selection order. The selected order, `redundancy_`, `relevance_`, `ranking_`
and support mask retain their meanings. Refitting resets the table.

String `"mean"`/`"max"` and NumPy `mean`/`max` use pandas vectorized row reductions.
Other callable reducers retain `DataFrame.apply` and receive labeled Series in
selected order. Frequency-weight relevance, relative-weight associations, shared
row sampling, absolute values, NaN/zero floors, exact-one penalties, domain
restrictions and stable input-order ties are unchanged. In particular, when the
last selected feature has no remaining domain peers, the existing unit-denominator
fallback is preserved rather than changing the selection policy.

## Regression checks

`tests/test_mrmr_storage.py` compares small cases with an independent square-table,
row-wise reference. It covers mean/max/custom reducers, asymmetric custom
callbacks with reversed labels, missing/negative/zero/perfect associations,
unavailable relevance, ties, domain restrictions, empty/single/full selections,
refit cleanup, weighted regression/classification, row sampling, input preservation
and support masks. Frequency-weight relevance also matches integer row
replication; weighted Spearman redundancy does not promise that equivalence.
A separate check rejects row-wise `apply` for recognized reducers.

The first run against the old implementation gave **46 failures, 10 passes**:
the table-shape checks and the four vectorization checks reproduced the missing
behavior. Test expectations were then corrected for empty column-index metadata
and the existing weighted-Spearman replication limit. All **56 C05 cases** and
the existing ranking/sampling suites pass (**123 tests** total).

## Benchmark protocol

The existing [manual benchmark](../tests/mrmr_performance.py) is unchanged. It uses
fresh worker processes, one native thread, one worker, the same verified Adult
training file and seeded synthetic inputs, with separate retained-table byte
accounting and Linux peak RSS. Adult has one warm-up and five timed fits per
metric; wide cases have one fit per input. Profiles run separately from timings.
No test suites or documentation builds ran concurrently with measurement.

The original [benchmark report](mrmr-performance.md) remains historical evidence.
Its original source precedes C01–C04, so before/after runtime and RSS values are
cross-session observations, not an isolated causal speed comparison. No predictive
quality claim or speedup target is made. The unchanged conservative preflight still
models a full square and skips both 20,000-feature cases; those skips do not show
that the new table exceeds the budget. The six originally completed wide cases
are the storage comparison. Runtime RSS guards remain enabled.

Reproduction from the checkout root, after extracting `adult.data` from the UCI ZIP:

```sh
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0
export MPLCONFIGDIR=/tmp/arfs-c05/mpl
/tmp/arfs-association/py312-locked/bin/python tests/mrmr_performance.py --self-check
/tmp/arfs-association/py312-locked/bin/python tests/mrmr_performance.py /tmp/arfs-c05/benchmark-final --adult-data /tmp/arfs-c05/adult.data
/tmp/arfs-association/py312-locked/bin/python -m pytest tests -q --tb=short -p no:cacheprovider
/tmp/arfs-association/py312-newer/bin/python -m pytest tests/test_mrmr_storage.py tests/test_mrmr_ranking.py tests/test_mrmr_sampling.py -q --tb=short -p no:cacheprovider
/tmp/arfs-association/docs-locked/bin/python -m sphinx -b html -W --keep-going docs /tmp/arfs-c05/html
uv tool run --from ruff==0.17.0 ruff check .
uv lock --check
git diff --check
```

Use a new benchmark output directory when rerunning; verify disposable environments
before reuse. Raw results, hashes, profiles and logs stay under `/tmp/arfs-c05/`
and are not committed. The initial benchmark attempt found that the cached file
named `adult.data` was a ZIP archive; that incomplete attempt is excluded. The
final run uses the extracted training file with SHA-256
`5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`.

The initial provenance hash for `mrmr.py` precedes a docstring-only clarification
that empty selections retain no columns. `/tmp/arfs-c05/mrmr-measured.py` and
`docstring-clarification.diff` reconcile that exact text change against the
initial C05 commit, `d5df337`; its runtime code and the runner matched the
measurements. The Codex review follow-up below adds periodic consolidation.
The measured 10/20-feature selections do not reach that new branch; the wide
benchmark was not repeated after this follow-up.

## Measured results — 2026-10-19

Historical → C05 values below use the same input hashes and selected orders.

| Input | Features | Retained table (MiB) | Peak RSS (GiB) | Fit (s) |
| --- | ---: | ---: | ---: | ---: |
| Numeric | 1,000 | 7.71 → 0.23 | 0.404 → 0.376 | 20.13 → 19.94 |
| 10% categorical | 1,000 | 7.71 → 0.23 | 0.395 → 0.366 | 18.56 → 18.47 |
| Numeric | 5,000 | 191.13 → 1.12 | 1.009 → 0.802 | 138.53 → 120.00 |
| 10% categorical | 5,000 | 191.13 → 1.12 | 0.944 → 0.755 | 144.63 → 105.65 |
| Numeric | 10,000 | 763.73 → 2.24 | 2.098 → 1.331 | 242.86 → 297.80 |
| 10% categorical | 10,000 | 763.73 → 2.24 | 2.004 → 1.237 | 223.10 → 282.34 |

All six wide cases completed below the RSS guard. At 10,000 features, the table
retains 19 columns instead of 10,000: about **2.24 MiB versus 763.73 MiB**
(including its row index). The data values alone require 1,520,000 bytes.
This is a table-storage reduction, not an equivalent reduction in whole-process
memory. Retained-table accounting includes the row index and excludes the column
index and other selector state, matching the original report.

Adult retains 2,385 table bytes, versus 2,945 historically. Its five-fit results:

| Redundancy | C05 fits (s) | Historical → C05 median (s) | Historical → C05 peak RSS (MiB) |
| --- | --- | ---: | ---: |
| Theil’s U | 0.833547, 0.940316, 0.887252, 0.820321, 0.811418 | 0.601090 → 0.833547 | 310.45 → 290.27 |
| Cramér’s V | 1.122004, 1.108649, 1.093314, 1.075750, 1.102120 | 0.842417 → 1.102120 | 310.81 → 291.20 |

The separate warmed profiles took 1.258 s (Theil) and 1.433 s (Cramér).
Do not compare these profiled times with unprofiled fits. All ten completed
workers reproduced the historical input/target hashes and selected order.
Adult and the 10,000-feature timings are slower than historical measurements;
the 5,000-feature cases are faster. These observations do not establish an
overall runtime improvement or isolate the effect of this change.

Measured environment: CPython 3.12.7, NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1,
scikit-learn 1.6.1, LightGBM 4.6.0 and SHAP 0.47.1 on Linux x86-64,
Intel i5-7300U, using the locked Python 3.12 validation snapshot. The historical
report used the same hardware and core versions, but an earlier source tree.
Raw final results are `/tmp/arfs-c05/benchmark-final/results.json`.

## Final validation and limits

- Full locked Python 3.12 source suite: **2,042 passed**, 564 warnings, 259.22 s.
- Newer Python 3.12 MRMR storage/ranking/sampling suites: **123 passed**,
  46 warnings, 32.96 s (NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
  pytest 9.1.1).
- Locked Python 3.9.25 MRMR suites: **123 passed**, 66 warnings, 29.89 s
  (NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, pytest 8.3.5).
- Strict HTML build, benchmark self-check, Ruff 0.17.0 lint, new-test formatting,
  `uv lock --check`, whitespace and local report-reference checks pass.
  Documentation builds keep notebook execution off.

The test/build logs are `/tmp/arfs-c05/{full-tests,newer-tests,py39-tests,docs}.log`.
No new installed-wheel/full Python matrix, real optional-backend runs, full
notebook execution, predictive-quality study or isolated runtime comparison was
performed. No dependencies, package version, release or historical reports were
changed. C06 remains pending; this implementation stops after C05.

## Codex review follow-up — 2026-10-19

[Codex found](https://github.com/ThomasBury/arfs/pull/75#discussion_r4312289793)
that inserting more than about 100 selected columns fragmented the pandas table,
causing repeated `PerformanceWarning`s and extra reduction/slicing overhead.
MRMR now consolidates the existing selected columns with a native deep copy every
64 selection steps. This bounds fragmentation without private pandas APIs,
preallocating a square, or copying the growing table after every selection.
Copies temporarily add to process memory; table storage remains O(p k).

Three new tests select 130 of 140 features with mean, max and a custom reducer,
crossing two consolidation points. They treat pandas performance warnings as
errors and compare every retained value and ranking with the dense reference.
All three failed with `PerformanceWarning` before the fix. C05 now has 59 cases.
The existing wide benchmark selects only 20 features and does not exercise the
consolidation branch, so its measurements remain historical evidence for the
initial implementation. Validation of this follow-up is recorded in PR #75.

All 126 MRMR storage/ranking/sampling checks pass after the fix on locked
Python 3.12 (69 warnings, 42.45 s), newer Python 3.12 (46 warnings, 45.12 s)
and locked Python 3.9 (66 warnings, 43.22 s). Ruff and whitespace checks pass.
The fresh full-source and hosted-wheel results are tracked in PR #75.
