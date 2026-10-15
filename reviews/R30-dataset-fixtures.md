# R30 — Honest local dataset fixtures

**R30 complete on 2026-10-15**, based on R29 handoff commit `56f4f6c`.
Branch: `fix/r30-dataset-fixtures`. Local implementation commit: `db2f7f7`.
R30 runs alone, with R04 complete.

## Changes and decisions

Both synthetic generators use one local RandomState seeded with 42 for their
features, targets, missing-value positions, categorical noise and weights.
Weights are drawn last, preserving the previous seeded feature/target
construction for even sizes. Repeated calls now reproduce every generated value
independently of the caller's NumPy state, and leave its global stream untouched.
The fixed seed remains internal; there is no new option or dependency.

Regression's repeated high-cardinality column concatenates ranges of lengths
ceil(size/2) and floor(size/2), so odd sizes retain the final row. Even sizes keep
their original repeated-level order. Empty/single-row inputs work through the
same construction. Frames, NumPy targets/weights and categorical dtypes retain
their existing return contracts; the regression return-type docstring is corrected.

Titanic and cancer augmentation also use their existing local seeded RNG for
every random draw. Titanic's family_size counts relatives rather than including
the passenger; is_alone is therefore one only for zero relatives. The loader's
download, imputation and returned schema remain the same. Random augmented values
and weights that previously depended on global state now have fixed local draws.

Caller tracing covered both generators, both augmented loaders, load_data,
all maintained tests, API/notebook references and historical R30 evidence.
The affected code is confined to utils.py, with regressions, API docstrings,
changelog and this handoff. Boston/housing resources, selector algorithms,
dependencies and notebook workflows are unchanged. Historical reports and scripts
remain intact; new evidence is local and ignored under
reviews/evidence/r30-dataset-fixtures/.

## Validation

The finalized **25** regression cases produce **13 failures and 12 passing
controls** on the unchanged R29 installed wheel. Affected locked source checks
pass **68 tests**, including the existing classification schema/signal screen,
real selector callers and installed-resource tests. All 25 maintained regressions
pass in each final installed-wheel cell.

| Cell | Full installed suite | Selected unchanged historical checks |
| --- | --- | --- |
| Python 3.9.25 locked | 1450 passed | Not rerun in this cell |
| Python 3.12.7 locked | 1450 passed | 3 passed unmasked |
| Python 3.12.7 newer | 1434 passed | 3 passed unmasked |

The historical RNG-isolation/odd-size checks run unchanged with --runxfail. The
old historical repeatability check deliberately expects variable weights and the
pre-R04 classification schema, so it remains historical evidence; the maintained
suite asserts the repaired contract instead. Titanic's historical dataset ledger
also retains its original counterexample assertion.

A direct old/new installed-wheel comparison confirms exact feature/target
equality for both generators at sizes 0/2/10/1000: eight comparisons. Weight
repeatability is independently asserted by the maintained tests.

Fresh environments install the rebuilt wheel outside the checkout, without
setuptools. All 13 installed module hashes match source, all copied test bytes
match the maintained suite and all 19 source/sdist/wheel payloads agree. Exact
versions, hashes, argv, working directories and exit codes remain in local
evidence. Locked and newer dependencies use the same snapshots as R29.

Dependency checks, lock validation, exported pins/hashes, Ruff (**330 total,
zero new**), two gate unit tests, new-test formatting and whitespace pass.
utils.py retains only its two inherited formatter differences; before/after
diffs prove they are unchanged.

Regressions assert exact caller-stream preservation at two seeds, repeatability
of all generated features/targets/weights across seeds and consecutive calls,
exact sizes 0/1/2/3/10/11/1001, categorical schemas/repeated levels, constant and
missing columns, strong positive/negative signals and a bounded noise correlation.
The network-free four-row Titanic fixture covers zero, one and two relatives,
including both possible sources of a single relative. These construction checks
are not evidence of calibrated selection or held-out predictive performance.

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
  PYTHONPATH=src /tmp/arfs-r29/py312-locked/bin/python -m pytest \
  tests/test_dataset_fixtures.py tests/test_runtime_baseline.py \
  tests/test_installed_package.py tests/test_allrelevant.py tests/test_featselect.py \
  -q --tb=short -p no:cacheprovider
uv build --python /tmp/arfs-r29/py312-locked/bin/python --out-dir /tmp/arfs-r30/dist
/tmp/arfs-r29/py312-locked/bin/python /tmp/arfs-r30/check_payloads.py
/tmp/arfs-r29/py312-locked/bin/python /tmp/arfs-r30/validate.py py39-locked
/tmp/arfs-r29/py312-locked/bin/python /tmp/arfs-r30/validate.py py312-locked
/tmp/arfs-r29/py312-locked/bin/python /tmp/arfs-r30/validate.py py312-newer
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/arfs-r30/py312-locked/bin/python -I /tmp/arfs-r30/check_construction.py
uv lock --check --python /tmp/arfs-r29/py312-locked/bin/python
uv export --locked --extra test --no-emit-project --output-file /tmp/arfs-r30/export-check.txt
uv tool run --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
uv tool run --from ruff==0.17.0 ruff check tests/test_dataset_fixtures.py
uv tool run --from ruff==0.17.0 ruff format --check tests/test_dataset_fixtures.py
/tmp/arfs-r29/py312-locked/bin/python /tmp/arfs-r30/check_format.py
/tmp/arfs-r29/py312-locked/bin/python -m unittest discover -s .github/scripts -p 'test_*.py'
git diff --check
```

## Limits and resume

No live OpenML call, complete notebooks/docs build, hosted CI, optional backends,
other platforms or large-data checks were run. Titanic is still a downloaded
benchmark with full-frame imputation, not an example of held-out preprocessing.
Synthetic weights and signal/noise checks do not establish selection calibration.

Stop after the local R30 implementation and handoff. Next is **R32 alone**, with
R01–R31 complete: repair and execute the documentation workflows and settle their
optional dependency/claim boundaries. R32 and R33 remain pending. No push, PR,
version bump or release is included.
