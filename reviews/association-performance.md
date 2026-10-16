# Association correctness and performance handoff

Complete: both correctness bugs and the four association kernels are implemented.
All six final installed-wheel suites, both fresh strict documentation builds,
Ruff, lock and whitespace checks pass. On the frozen workloads, final serial
medians are 1.7–174.3× faster than old serial and 2.6–295.2× faster than warmed
two-process execution, with labels/order preserved and maximum serial absolute
numerical difference 7.22e-16. Full figures and limits follow below.

Scope: weighted ranks, nullable numeric input, prepared Pearson/Spearman,
observed-joint Theil's U and grouped correlation ratio, with native numeric
execution and optional joblib categorical batches. Public custom callbacks keep
the existing path. No new dependencies or statistical defaults.

## Checkpoint 1: baseline and regressions (complete)

Starting commit: `937917749f4272658533c952170c4412a5141080`; clean working tree.
Baseline distribution: `/tmp/arfs-association/baseline/`; verbose logs and timing
records: `reviews/evidence/association-performance/` (locally ignored).
Manual workload and fixed seed: `tests/association_performance.py`; 2,000 rows,
24 numeric columns, 10 categorical columns with 40 levels, and a 10+10 mixed
frame. The earlier timing workload was not present in the supplied plan; these
bounded dimensions are frozen for this comparison. Five measured repetitions
follow a warm-up for each serial/two-process case; native threads are fixed to 1.

Caller inventory: direct `wm/wcov/wcorr/wrank/wspearman/weighted_corr`, entropy/U,
ratio, specialized matrix/series, mixed `_callable_association_*` dispatch,
`f_cont_regression_parallel`, `f_cat_classification_parallel`, MRMR and
CollinearityThreshold. Exports/signatures stay unchanged. Tests must preserve
positional weights, whole-frame row policy, callback pass-through, original
labels, constant/empty outputs, stable ties, directional and block ordering.

Baseline wheel built with `uv build --out-dir /tmp/arfs-association/baseline`.
Installed into a separate environment using the existing locked requirements.
Both new regressions fail on that installed wheel (`baseline-regressions.txt`):
positive tied rank 1.5 instead of 2; nullable matrix raises a NumPy sqrt TypeError.
Baseline timings completed in `baseline.json` and `baseline-timing.txt`.

Commands (native thread variables set to 1, LOKY_MAX_CPU_COUNT=2):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2
/tmp/arfs-association/base-env/bin/python -I -m pytest tests/test_association_kernels.py -q --tb=short
/tmp/arfs-association/base-env/bin/python -I tests/association_performance.py reviews/evidence/association-performance/baseline.json
```

## Checkpoint 2: correctness and numeric kernels (complete)

Zero-weight observations no longer enter tie counts; zero-only groups retain
preceding mass and rank output length is unchanged. Float conversion handles
nullable numeric data. Centered products share a single square-root denominator,
which preserves exact perfect-duplicate results in the existing regression and
MRMR checks. An initial split-denominator implementation failed 9 such checks;
`focused-initial.txt` retains that failure. The corrected focused numeric run
passes **137 tests** in `focused-numeric.txt`.

```sh
PYTHONPATH=src /tmp/arfs-association/base-env/bin/python -m pytest tests/test_association_kernels.py tests/test_statistical_arithmetic.py tests/test_mrmr_ranking.py -q --tb=short -p no:cacheprovider
```

## Checkpoint 3: categorical kernels and hybrid dispatch (complete)

Category codes/marginals are prepared once per evaluated set. Joint entropy uses
observed pairs only and shares work for both U directions, then restores original
permutation order before stable sorting. Ratio uses grouped weighted means and
direct residual squares after shifting away numeric offsets. Existing joblib
chunking returns one frame/Series per batch; custom wrappers and callbacks retain
the old dispatcher. Method-only Pearson/Spearman partials use numeric kernels,
including the partial object's existing combination/permutation behavior.

Expanded regressions passed **473 tests** (`focused-expanded.txt`). Final edge
checks additionally cover series worker validation and exact large-integer
Spearman order before float conversion. Process dispatch sends prepared arrays
and labels, without sending the original observations a second time. Ruff and
`git diff --check` pass.

```sh
PYTHONPATH=src /tmp/arfs-association/base-env/bin/python -m pytest tests/test_association_kernels.py tests/test_statistical_arithmetic.py tests/test_parallel_contracts.py tests/test_observation_boundaries.py tests/test_category_identity.py tests/test_mrmr_ranking.py -q --tb=short -p no:cacheprovider
uv tool run --from ruff==0.17.0 ruff check .
git diff --check
```

## Checkpoint 4: final validation (complete)

The first candidate validation was stopped during its first cell to preserve an
additional direct-API compatibility case found during review: Spearman still
accepts comparable nonnumeric Series. Ranking now precedes float conversion for
both those labels and large integers. That partial run was superseded; every
final cell below uses the same rebuilt wheel.

All final gates below are complete. The temporary runner restart initially
rejected an existing venv; `uv venv --clear` recreates only this task's temporary environments.

Final validation refresh: the earlier candidate passed all six suites (1,559
tests in five cells, 1,543 with newer pandas) and both strict documentation builds. A separate realistic-size duplicate probe
then exposed reduction-order rounding that weakened exact unit correlations
(`duplicate-regressions.txt`, four failing regressions). The repair derives
matrix normalization from the same Gram product and preserves exact identical
or opposite centered vectors in series calculations, with no statistical
tolerance. Consistent per-column weighted-mean reductions preserve exact
opposites;
`negative-duplicate-centering-failure.txt` retains that additional failing check.
The final focused arithmetic/selector run passed **213 tests**
(`focused-duplicates.txt`). These earlier wheel results do not substitute for
the changed numeric module's final validation. The final wheel was rebuilt,
then all six cells and both docs builds were refreshed. Independent runtime cells
ran in groups of three; benchmarks ran after all heavy work stopped.

## Final validation results

- `/tmp/arfs-association/baseline/arfs-4.0.0.tar.gz` SHA-256: `be8ae1c6e1efe6725ad772bf6f444eefdabfb04a0e278bfcd52429fcbda6f225`.
- `/tmp/arfs-association/baseline/arfs-4.0.0-py3-none-any.whl` SHA-256: `ab28473ce688854c0bf0bd583da838d2141fcd27c4662ed8669ec531c2e57f02`.
- `/tmp/arfs-association/final/arfs-4.0.0.tar.gz` SHA-256: `29e3af9692c1ce97cb96c98ed77e24caaa9c2eca4c7ed86e186354b22481be64`.
- `/tmp/arfs-association/final/arfs-4.0.0-py3-none-any.whl` SHA-256: `fe69d16e3ad0d384fe38527973463deace5600838aa82f14b3c4c7e22c85cf10`.

- `py39-locked`: 1563 passed, 478 warnings in 289.04s (0:04:49).
- `py39-minimum`: 1563 passed, 478 warnings in 291.88s (0:04:51).
- `py310-locked`: 1563 passed, 478 warnings in 297.83s (0:04:57).
- `py311-locked`: 1563 passed, 478 warnings in 272.74s (0:04:32).
- `py312-locked`: 1563 passed, 487 warnings in 280.26s (0:04:40).
- `py312-newer`: 1547 passed, 253 warnings in 294.38s (0:04:54).

The newer pandas 3 cell has 16 fewer pre-existing parameter combinations because
Copy-on-Write is mandatory there; those cases are not skipped tests.

All six fresh environments installed the final wheel outside the checkout and
passed `uv pip check`; import paths and runtime module SHA-256 agree with the
wheel and source. All Python/data payloads match source, sdist and wheel. The
strict Sphinx HTML builds pass with `-W --keep-going` in fresh locked and newer
Python 3.12 documentation environments. Both documentation imports match the
final module hash and their runtime dependency snapshots (`docs-*-environment.json`). Ruff 0.17.0, whitespace and lock checks
pass. Exact commands, environment package versions, source/artifact hashes and
full logs are in the local evidence directory. The runner is retained at
`/tmp/arfs-association/validate.py`.

Commands (thread limits described above apply to test and benchmark processes):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2
uv lock --check
uv export --locked --group test --no-emit-project --output-file /tmp/arfs-association/locked-requirements.txt
uv build --out-dir /tmp/arfs-association/final
/tmp/arfs-association/base-env/bin/python /tmp/arfs-association/validate.py
/tmp/arfs-association/py312-locked/bin/python -I tests/association_performance.py reviews/evidence/association-performance/final.json
/tmp/arfs-association/py312-locked/bin/python /tmp/arfs-association/summarize.py
```

## Frozen-wheel timing comparison

Both runs use Python 3.12.7; arfs 4.0.0, numpy 2.0.2, pandas 2.2.3, scipy 1.13.1, joblib 1.4.2.
Each workload has its own warm-up followed by five measured repetitions, fixed
seed 20261016, one native thread and no concurrent heavy checks. Baseline and
final are separate processes/environments; the process pool is warmed before
two-worker measurements. Numeric `n_jobs=2` on the final wheel uses NumPy, while
categorical `n_jobs=2` continues to use joblib. Numbers are milliseconds; ranges
are min–max. Full baseline/final individual repetitions are
retained in `baseline.json` and `final.json`.

| Workload | Old serial median (range) | Old two-process median (range) | New serial median (range) | Speedup vs old serial / two-process |
| --- | ---: | ---: | ---: | ---: |
| pearson-matrix | 8.15 (7.55–8.62) | 1402.10 (1345.37–1458.84) | 4.75 (4.33–5.62) | 1.7× / 295.2× |
| spearman-matrix | 2918.27 (2818.15–2989.81) | 1715.08 (1639.17–1762.05) | 16.74 (15.76–17.49) | 174.3× / 102.4× |
| pearson-series | 200.73 (192.92–236.28) | 118.26 (105.84–152.77) | 4.92 (4.72–5.69) | 40.8× / 24.1× |
| spearman-series | 244.60 (233.69–252.18) | 149.05 (147.46–202.80) | 16.88 (16.10–21.05) | 14.5× / 8.8× |
| theil-matrix | 405.57 (385.44–465.09) | 257.52 (247.46–285.12) | 25.16 (23.69–25.64) | 16.1× / 10.2× |
| theil-series | 46.24 (42.63–49.75) | 47.43 (37.86–48.78) | 18.41 (18.15–23.68) | 2.5× / 2.6× |
| ratio-matrix | 3330.97 (3233.43–3433.28) | 1653.57 (1593.44–1848.16) | 21.65 (18.91–23.91) | 153.9× / 76.4× |
| ratio-series | 340.26 (314.24–364.93) | 204.79 (178.70–305.96) | 19.00 (18.64–22.47) | 17.9× / 10.8× |

Every final serial median beats both baseline medians on these frozen workloads.
All output labels/order agree; values agree at rtol=1e-10, atol=1e-12 across both
worker counts. The maximum serial absolute difference is 7.22e-16. These
workloads use positive fractional weights, so the deliberate zero-weight rank
correction is tested separately rather than treated as an agreement failure.
Timing assertions live only in this manual comparison, outside CI.

No remaining implementation or validation blocker. Refit/recheck selections
that used zero-weight tied rows. Full association matrices still require
quadratic output/work in the number of columns; categorical joblib overhead can
outweigh its benefit on small sets. No new dependencies, Cramér's V optimization,
MRMR storage redesign, automatic sampling or statistical-default changes.
