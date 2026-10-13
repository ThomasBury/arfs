# R04 — Reproducible baseline and runtime gate

**R04 complete on 2026-10-11. Next group: R05 alone.**
Based on `170b3e95fee635e8243ed8cdc740ecb80c4738ad` on `main`; changes are
uncommitted. The pre-existing edits to `reviews/README.md` and the batching
instructions in `reviews/REMEDIATION.md` were preserved. No commit, push,
version bump, or release was performed.

## Changes and decisions

The constant-removal test now uses threshold 1 and an array-aware comparison.
New boundary cases record the existing strict comparison: features with unique
counts equal to the threshold are excluded. Selector semantics and their wider
documentation review remain R16.

The classification generator now assigns its existing random `nice_guys`
labels to a string-valued column, matching the regression generator's intended
schema. Its new fixture check covers the exact schema, category cardinality,
constant/missing columns, aligned target/weights, and strong versus noise
correlations. This intentionally adds one predictor to generated classification
frames and advances their random stream. Global RNG ownership, regression odd
sizes, and other dataset repairs remain R30.

The Leshy regression test uses the regression generator. Four Leshy test names
now describe their actual LightForest checks rather than a commented-out Boruta
comparison. New independent seeded numerical screens exercise Leshy,
BoostAGroota, and GrootCV. Each must recover the signal and select at most two
of eight independent noise predictors. A deliberately select-all replacement
fails all three cases. This is a small runtime regression screen, not evidence
of calibrated error rates, weighted/mixed superiority, or held-out performance;
those assessments remain R33.

Python support remains `>=3.9,<3.13`; runtime evidence covers Linux x86-64
Python 3.9.25 with the lock and Python 3.12.7 with locked and recorded newer
dependencies. Python 3.10/3.11 and other platforms are not certified by this
phase. The newer pins are explicitly Python-3.12-only and are a tested snapshot,
not a promise to support every future allowed resolution.

`uv lock` refreshed the baseline without `--upgrade`. All 115 dependency
package versions and archive hashes remain unchanged; Jinja2 is now a runtime
dependency in the lock. uv 0.12.15 records the project's dynamic version instead
of the stale 2.4.0 entry, adds artifact upload timestamps and lock revision 3,
and updates metadata/markers. Built sdist and wheel metadata report 3.0.0.
The [package comparison](evidence/r04-runtime-baseline/lock-packages.json)
records the only changed package identity. No dependency minima changed.

The [installed-wheel workflow](../.github/workflows/runtime.yml) checks the
lock, builds the wheel through its sdist, installs it in a fresh environment,
checks dependency consistency/version/import provenance and absence of the
setuptools workaround, then copies and runs the whole established suite outside
the checkout. Its three cells match the evidence above. Locked dependencies
come from `uv export --locked`; the newer cell uses the maintained
[snapshot](../.github/requirements/runtime-newer-py312.txt). Imports, public
exports, bundled datasets, styling and resource lifetime are checked by the
existing installed-package tests. Context7 verified uv's locking/export and
CI commands.

The actual oldest-interpreter run exposed a Python 3.9 test-fixture defect:
fresh `ZipPath` wrappers copied a caller-owned archive's shared handle, and a
discarded wrapper closed it before subsequent loads. Keeping one root through
the loads fixes the fixture without changing library resource access or
weakening assertions. The failure and final checks are retained separately.

## Validation and reproduction

Evidence is in [r04-runtime-baseline](evidence/r04-runtime-baseline/).
The old baseline reproduced two test failures; the new fixture assertion also
failed on the installed R03 wheel. The stale lock failed `uv lock --check`.
The new baseline has no expected-failure markers.

| Installed wheel cell | Full suite | Final zip-fixture check |
| --- | --- | --- |
| Python 3.9.25, locked | 64 passed, 56 warnings | Included in final full suite |
| Python 3.12.7, locked | 64 passed, 59 warnings | 1 passed after fixture repair |
| Python 3.12.7, newer | 60 passed, 104 warnings | 1 passed after fixture repair |

The pandas 2.2 cells include Copy-on-Write off/on; pandas 3 runs its mandatory
behavior, hence four fewer cases. Warnings include existing sklearn tags,
SHAP output changes, GLM convergence and plotting warnings. No statistical or
deprecation debt was silently suppressed. `uv pip check` succeeds in every
cell, setuptools is absent, and ARFS imports from each environment's
`site-packages` with version 3.0.0. Full environment versions are recorded in
the three `*-environment.json` files. The wheel's 19 code/data payloads match
the current source; its SHA-256 is recorded in `wheel.json`.

Commands from the checkout (use fresh disposable directories if needed):

```sh
repo="$PWD"
audit_dir=$(mktemp -d /tmp/arfs-r04-repeat.XXXXXX)
export UV_CACHE_DIR="$audit_dir/cache"
uv lock --check --python 3.12
uv export --locked --extra test --no-emit-project \
  --output-file "$audit_dir/locked-requirements.txt"
uv build --python 3.12 --out-dir "$audit_dir/dist"
mkdir -p "$audit_dir/run"
cp -r tests "$audit_dir/run/tests"

# Repeat with cell=py39-locked, python_version=3.9 and locked requirements;
# or cell=py312-newer, python_version=3.12 and the newer snapshot below.
cell=py312-locked
python_version=3.12
requirements="$audit_dir/locked-requirements.txt"
# Newer only: requirements="$repo/.github/requirements/runtime-newer-py312.txt"
uv venv --python "$python_version" "$audit_dir/$cell"
uv pip install --python "$audit_dir/$cell/bin/python" -r "$requirements" \
  "$audit_dir/dist/arfs-3.0.0-py3-none-any.whl"
uv pip check --python "$audit_dir/$cell/bin/python"
cd "$audit_dir/run"
env MPLBACKEND=Agg MPLCONFIGDIR="$audit_dir/mpl-$cell" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONHASHSEED=0 \
  "$audit_dir/$cell/bin/python" -I -m pytest tests \
  -q --tb=short -p no:cacheprovider
```

Local evidence used `/tmp/arfs-r04`, uv 0.12.15, Python 3.12.7 and a
disposable Python 3.9.25 installation. The final Python 3.9 rerun used the copied
absolute test path in isolated Python mode; both Python 3.12 full runs and
their final zip checks used `/tmp/arfs-r04/run` as their working directory.

The deliberate select-all probe replaced only the three classes' `fit` methods
in a disposable process with an implementation assigning every input name to
`selected_features_`, then ran:

```sh
python -I -m pytest tests/test_runtime_baseline.py -k recovers_signal \
  -q --tb=short -p no:cacheprovider
```

Its process called `pytest.main` and asserted `TESTS_FAILED`: all three cases
failed specifically at the noise-selection bound. The real implementations
pass unmasked in each cell.

The existing Ruff gate has 429 diagnostics and zero new ones. The two repaired
allowances (`utils.py`'s unused `nice_guys` and `test_featselect.py`'s previously
unused NumPy import) were removed/lowered. Gate unit tests pass (2 tests).
Run `python .github/scripts/check_ruff_baseline.py` with Ruff 0.17.0 on PATH and
`python -m unittest discover -s .github/scripts -p 'test_*.py'`.
The new baseline tests pass Ruff check/format check; `git diff --check` passes.

Hosted GitHub Actions has not run yet. The corresponding build/install/runtime
steps were exercised locally. Docs/notebooks, optional backends, performance,
full sklearn compliance and broad statistical validation were not run.

## Resume

Run **R05 alone** next, with R01 already complete. Read its findings
`1A F8–F10`, `1B F11`, `04 F6/F7` and trace shared validation/statistics/training
callers. Its remaining decisions concern boundary validation and retained/split
weight mass; weight interpretation remains explicit in R08/R15/R25. R06 follows
in a separate session. R05–R33 remain pending; a green baseline is not release
readiness.
