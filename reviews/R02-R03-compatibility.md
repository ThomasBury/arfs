# R02 and R03 dependency compatibility

**R02 and R03 are complete on 2026-10-11. Next phase: R04, pending.**
Source changes are committed as `36f1ee6` (R02) and `65f894a` (R03) on
`fix/r02-r03-dependency-compatibility`, based on `a4b7ff6`. Historical review
scripts and logs remain unchanged.

## R02 — Owned pandas column writes

Shadow generation now permutes positions and assigns the resulting extension
array to an owned column. This preserves categorical/nullable dtypes, missing
values, the index, and caller input without mutating a read-only pandas view.
Both midpoint APIs replace the whole interval column with the helper's numeric
result. The estimator's redundant float conversion was removed.

The new [pandas-write regressions](../tests/test_pandas_writes.py) have no expected-failure markers. Before
the repair it recorded one failure with frozen dependencies (Copy-on-Write on)
and four with newer dependencies (shadow generation, both midpoint APIs, and
the tree-to-midpoint pipeline). After the repair all nine frozen cases and all
five newer cases pass. pandas 2.2 runs with Copy-on-Write off/on; pandas 3 runs
with its mandatory Copy-on-Write behavior. All six existing BoostAGroota/GrootCV
tests pass in each snapshot.

Scope excludes shadow-label collisions and selector category-role/encoding
changes (R07), midpoint refit/scalar-name handling (R20), and tree partition
correctness (R19). The pipeline check respects the existing finite-bound rule
for one-sided infinite intervals; it does not claim correct tree partitions.

## R03 — Supported dependency APIs

The replacements are native APIs already available at the declared minima:

- sklearn's `ensure_all_finite` replaces `force_all_finite` in the shared GLM
  search, preserving the existing finiteness setting.
- `Styler.map` replaces `Styler.applymap` in the real summary pipeline.
- Matplotlib's `tick_labels` replaces `labels` in the real permutation plot.
- Clustering imports `pdist` from `scipy.spatial.distance` instead of reaching
  into `scipy.cluster.hierarchy` for it.
- The weighted ANOVA constant check uses scalar comparisons. Its arithmetic,
  warning condition/message, length-one output, and NaN/infinity behavior are
  preserved. Nonconstant unit/integer-weight fixtures match independent SciPy
  references; this does not settle general weight semantics or numerical
  stability (R08). GLM scoring/CV defects remain R13.

The new [dependency API regressions](../tests/test_dependency_apis.py) promote
the existing real GLM fit, summary HTML, permutation plot, association plot,
and serial/parallel relevance checks without expected-failure markers. They
also cover scalar constant/nonconstant ANOVA and its existing output shape.
Before repair all twelve tests passed frozen with deprecation warnings and
all twelve failed newer. After repair all twelve pass in each snapshot.

Context7 checked the official pandas, sklearn, Matplotlib, SciPy, and NumPy
documentation. No dependency minima, lock, runtime CI, public signature, or
statistical policy changed. The existing Ruff gate allowance was lowered by one
for the removed diagnostic. `boxplot(vert=False)` remains supported in both snapshots; its
newer deprecation is retained because `orientation` is absent at the declared
Matplotlib minimum.

## Validation and limits

Initial source checks use Linux CPython 3.12.7 and the verified disposable R01 environments at
`/tmp/arfs-r01.7gLmPw/{frozen,newer}-env`. Imports are explicitly directed to
this checkout with `PYTHONPATH=/home/bsatom/Documents/arfs/src`; these source
checks are not new installed-wheel checks. Recreate environments using the R01
requirements snapshots if these temporary paths disappear.

For each snapshot, the command prefix is:

```sh
env PYTHONPATH=/home/bsatom/Documents/arfs/src MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/arfs-r02-r03/mpl-$snapshot \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /tmp/arfs-r01.7gLmPw/$snapshot-env/bin/python -m pytest
```

R02 command suffixes:

```sh
tests/test_pandas_writes.py -q --tb=short -p no:cacheprovider
tests/test_allrelevant.py -k 'boostagroota or grootcv' -q --tb=short -p no:cacheprovider
```

R03 uses the same prefix, with:

```sh
tests/test_dependency_apis.py -q --tb=short -p no:cacheprovider
```

The final wheel was built from the changed checkout and installed into two new
environments. All installed ARFS payloads match the wheel, and the wheel payloads
match the source. Runtime setuptools is absent and dependency consistency passes
in both. The final test copies ran outside the checkout in isolated Python mode.

| Check | Frozen | Newer |
| --- | --- | --- |
| pandas / NumPy | 2.2.3 / 2.0.2 | 3.0.6 / 2.5.4 |
| sklearn / SciPy / Matplotlib | 1.6.1 / 1.13.1 / 3.9.4 | 1.9.1 / 1.18.1 / 3.11.2 |
| R02 regressions | 9 passed | 5 passed |
| R03 regressions | 12 passed | 12 passed |
| Existing BoostAGroota/GrootCV callers | 6 passed | 6 passed |
| Full installed-wheel suite | 56 passed, 2 inherited failures | 52 passed, 2 inherited failures |

Both inherited failures are unchanged R04 cases:
`test_identify_single_unique_classification` and
`test_identify_high_cardinality_classification`. The first has the wrong
threshold/comparison; the second expects the missing synthetic `nice_guys`
column. They were neither patched nor marked as expected failures. All 22
R01 installation regressions and all twelve existing all-relevant tests pass.

Installed-wheel reproduction commands (set `repo` to the checkout and
`snapshot` to `frozen` or `newer`):

```sh
uv build --wheel --out-dir /tmp/arfs-r02-r03/dist
uv venv --python /tmp/arfs-r01.7gLmPw/$snapshot-env/bin/python \
  /tmp/arfs-r02-r03/$snapshot-env
uv pip install --offline --python /tmp/arfs-r02-r03/$snapshot-env/bin/python \
  Jinja2==3.1.6 MarkupSafe==3.0.3
uv pip install --offline --cache-dir /tmp/arfs-review-03f67d0/cache \
  --python /tmp/arfs-r02-r03/$snapshot-env/bin/python \
  -r "$repo/reviews/evidence/r01-installed-package/$snapshot-requirements.txt" \
  /tmp/arfs-r02-r03/dist/arfs-3.0.0-py3-none-any.whl
uv pip check --python /tmp/arfs-r02-r03/$snapshot-env/bin/python
```

The styling packages use the default cache; the historical runtime pins use
the earlier review's cache. If those caches are gone, omit `--offline` and the
historical `--cache-dir`, retaining the exact pins. Copy `tests/` to
`/tmp/arfs-r02-r03/run/tests`, then from `/tmp/arfs-r02-r03/run` execute:

```sh
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arfs-r02-r03/mpl-installed-$snapshot \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /tmp/arfs-r02-r03/$snapshot-env/bin/python -I -m pytest \
  tests -q --tb=short -p no:cacheprovider
```

The two new test files pass Ruff check and format check. Touched source files
introduce no new diagnostic codes/counts; removing the ANOVA warning formatting
also removes one inherited `UP031` finding. No unrelated lint migration was
performed. `git diff --check` passes for code and prose; raw tool logs may retain
their original trailing whitespace.

Before/after, caller, build, installed-suite, requirements, and environment
evidence is in [the result ledger](evidence/r02-r03-compatibility/validation.json)
and its adjacent files. Resume with R04's findings in review 00 and the two
existing test failures above. R04's lock and runtime CI work remains untouched.

Other Python versions/platforms, full notebooks/docs, and optional plotting
backends are outside these phases. Statistical and baseline release blockers
remain assigned to later phases.

## Delivery and release decision

The source repairs have separate Conventional Commits; this handoff and the
combined installed-wheel evidence are a separate documentation commit. The
repository Ruff baseline gate passes with 431 existing diagnostics and zero new
ones; both gate regression tests pass.

Keep version `3.0.0` and defer a new GitHub release. R04 must establish the green
baseline and runtime CI, and the remaining release-blocking correctness phases
must satisfy R33's release evidence. Passing this compatibility batch alone
does not establish release readiness. No version tag or release was created.
