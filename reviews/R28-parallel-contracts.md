# R28 — Parallel labels and bounded useful work

**R28 complete on 2026-10-14**, based on clean `main` commit
`81dc6c9226efb696292b35eb415b6e87a3a41a5c`.
Branch: `fix/r28-parallel-contracts`. Local phase commit: `2181790`.
R28 runs alone, with R06/R08 complete.

## Changes and decisions

Matrix chunks split pair positions and retain the original Python labels;
NumPy never converts the label objects. Mixed integer/string names, ordinary
tuple indexes and named MultiIndex columns work in both worker modes. Column
chunks continue to use positional pandas slicing. Observation weights and input
frames, including duplicate row indexes, retain their existing contracts.

Both parallel helpers use joblib's native effective_n_jobs for the active
backend. Negative integers retain joblib semantics: -1 uses available CPUs,
-2 all but one, and more negative counts bottom out at one. Native CPU
availability includes affinity, quotas and LOKY_MAX_CPU_COUNT for the default
process backend. Positive requests retain native semantics, then the shared
helper caps workers to the number of pairs/columns. Every dispatched chunk is
nonempty; one unit of work runs directly, without starting Parallel. No new
execution framework, dependency, backend selection or thread configuration is
added. Counts must be nonzero integers; Boolean, fractional, string and None
counts raise ValueError even for empty helper work.

Matrix results sort by val descending with stable ties in pair/callback order,
and receive a fresh RangeIndex. Series results sort descending with stable
callback/column-order ties while retaining their label index. The two helpers
now apply the same ordering in serial and parallel modes. Generic mixed
association matrices retain their numeric/mixed/categorical block order;
each block uses the shared stable ordering. Subsequent association/relevance
series sorts also remain stable.

Empty pair work or an empty list returned by matrix callbacks produces row/col
object columns and a float val column. Empty column work produces a float Series.
Empty work invokes no callback or worker; callback exceptions on actual work
propagate. wcorr_matrix with one numeric column now returns the empty schema
for Pearson and Spearman. Generic association_matrix shares that schema when
there are no eligible pairs. Specialized wrappers keep None when no eligible
dtype group exists; their existing policy is unchanged.

Serial Pearson retains its native matrix calculation, but its tidy output now
matches the pairwise path: both directions of distinct pairs, no diagonal rows,
stable descending order and original labels. The old stack conversion had a
different row layout and expanded MultiIndex labels. Undefined constant-column
correlations remain NaN in both matrix paths, matching direct weighted_corr;
the existing series wrapper separately fills undefined scores with zero. The
public weighted_correlation_1cpu utility retains its square-matrix API.

Caller tracing covered all uses of parallel_matrix_entries and parallel_df,
their callback factories, association/relevance wrappers, MRMR/threshold tests,
API autodoc and notebook sources. Duplicate multiprocessing.cpu_count rewrites
were removed from every affected wrapper so they no longer override native
negative-count/quota behavior. The repaired parallel-module import allowance
was removed from the Ruff debt baseline. Historical reports and evidence stay
unchanged; new evidence remains local under the user's ignored
reviews/evidence/r28-parallel-contracts/ rule.

## Validation

All **67** maintained R28 regressions pass in every installed-wheel cell. The
identical finalized tests against the unchanged pre-R28 wheel produce **62
failures and five passing controls**. They cover label types, exact result
order/index, independent pairwise correlations, weights, input preservation,
constant/tied Pearson pairs, empty work/results, invalid counts, callback errors,
real two-process execution and threading/sequential backends. Dispatch probes
observe nonempty chunks and the useful-worker cap; native negative-count and
CPU-limit checks establish the worker semantics separately.

Affected source checks pass **508 tests in each Python 3.12 dependency snapshot**.
The finalized installed-wheel matrix includes all affected callers:

| Cell | Full installed suite | Selected unchanged historical checks |
| --- | --- | --- |
| Python 3.9.25 locked | 1372 passed | Not rerun in this cell |
| Python 3.12.7 locked | 1372 passed | 12 passed unmasked |
| Python 3.12.7 newer | 1356 passed | 12 passed unmasked |

The selected historical checks retain their original F13 failure markers and
run with --runxfail. Historical assertions encoding the old inconsistent order,
empty errors and unsupported -2 behavior are preserved without being reused as
current expectations. New maintained tests record the settled contracts.

Locked: NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1, sklearn 1.6.1, joblib 1.4.2.
Newer: NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, sklearn 1.9.1, joblib 1.6.0.
Exact package identities are recorded in the environment JSON for each cell.
Fresh venvs install the rebuilt wheel outside the checkout, without setuptools.
All 13 installed module hashes match source, copied test
bytes match the maintained suite, and all 19 source/sdist/wheel payloads agree.
Dependency checks, lock validation, the exported package pins/hashes, Ruff
(**330 total, zero new**), two gate unit tests, focused lint/format and whitespace
checks pass. association.py retains exactly its four inherited formatting
differences; before/after formatter diffs prove they are unchanged.

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
  PYTHONPATH=src /tmp/arfs-r25-r27-review/py312-locked/bin/python -m pytest \
  tests/test_parallel_contracts.py tests/test_category_identity.py \
  tests/test_observation_boundaries.py tests/test_statistical_arithmetic.py \
  tests/test_mrmr_ranking.py tests/test_threshold_contracts.py \
  -q --tb=short -p no:cacheprovider
uv build --python /tmp/arfs-r25-r27-review/py312-locked/bin/python --out-dir /tmp/arfs-r28-final/dist
/tmp/arfs-r25-r27-review/py312-locked/bin/python /tmp/arfs-r28-final/check_payloads.py
/tmp/arfs-r25-r27-review/py312-locked/bin/python /tmp/arfs-r28-final/validate.py py39-locked
/tmp/arfs-r25-r27-review/py312-locked/bin/python /tmp/arfs-r28-final/validate.py py312-locked
/tmp/arfs-r25-r27-review/py312-locked/bin/python /tmp/arfs-r28-final/validate.py py312-newer
uv lock --check --python /tmp/arfs-r25-r27-review/py312-locked/bin/python
uv export --locked --extra test --no-emit-project --output-file /tmp/arfs-r28/export-check.txt
uv tool run --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
uv tool run --from ruff==0.17.0 ruff check src/arfs/parallel.py tests/test_parallel_contracts.py
uv tool run --from ruff==0.17.0 ruff format --check src/arfs/parallel.py tests/test_parallel_contracts.py
/tmp/arfs-r25-r27-review/py312-locked/bin/python -m unittest discover -s .github/scripts -p 'test_*.py'
git diff --check
```

The affected-source command also passed with py312-newer. The runner and logs
under ignored evidence record exact argv, cwd, exit codes, hashes and versions.
Frozen requirements come from the checked lock; the newer snapshot uses
.github/requirements/runtime-newer-py312.txt. A regenerated frozen export differs
only in its generated command-header output path. Context7's official joblib
documentation and the installed native implementation were consulted for worker
semantics; quota/affinity behavior is delegated to joblib, with its environment
CPU cap exercised locally.

## Limits and resume

Hosted CI, full docs/notebooks, optional backends, other platforms, interrupted
workers and large-data throughput/memory were not checked. Stable sorting and
nonempty chunks establish deterministic output and bounded useful dispatch,
not a performance or fault-recovery guarantee. Serial Pearson's tidy row layout
changes intentionally; downstream callers must not rely on the old diagonal /
triangle layout or worker-dependent matrix index/order.

Stop after the local R28 phase and handoff commits. Next is **R29 alone**, with
R03/R10/R17 complete: fix plots to follow the data, actual selection masks,
terminal thresholds and plot limits, and bypass singleton clustering. No R29
implementation, push, PR, version bump or release belongs to this request.
