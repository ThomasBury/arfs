# R01 — Usable installed package

**Complete on 2026-10-11. Next phase: R02, pending.** Only R01 was implemented.
Implementation commit: `6b675fd5d366894f829f677c7fa1c66c95fac4d8`, on
`fix/r01-installed-package`, based on checkpoint
`0270aca405f523e87b979b0292a6ccd9d74e5acf`. The session guidance in
README/REMEDIATION was preserved. PR delivery was subsequently authorized;
no package release or next repair phase is included.

## Changes and decisions

- Replaced all three `pkg_resources` resource lookups in
  [utils.py](../src/arfs/utils.py) with Python 3.9's `importlib.resources.files`.
  Joblib and pandas read binary streams inside their context managers; the
  description uses UTF-8. No consumer requires an extracted temporary path.
- Corrected the package-data key to `arfs.dataset.descr`, so `housing.rst` ships
  in both the source distribution and wheel.
- Bound the existing `BaseThresholdSelector` in the public feature-selection
  namespace; all twelve declared exports resolve.
- Kept `make_fs_summary`'s unconditional `Styler` return, documented that return,
  and declared `Jinja2>=3.1.6`. Version 3.1.6 works in both tested snapshots.
- Added [installation regression tests](../tests/test_installed_package.py).
  They cover sixteen module imports, exports, local datasets, the declared
  styling dependency, and actual zip-backed resource streams after closure.

Housing's `filename` remains a usable string path in normal unpacked installs.
For zip imports it identifies the archive resource, rather than a temporary file
that disappears after loading. This distinction is documented in the loader.

## Validation

Checks ran outside the checkout against an installed wheel, using CPython 3.12.7
on Linux. A clean Git archive plus the R01 files supplied the source snapshot;
the wheel was built through its source distribution with setuptools 84.0.0.
All three bundled resources are present in both archives. Installed payloads
match the final wheel byte for byte; hashes are in
[distributions.json](evidence/r01-installed-package/distributions.json).

| Check | Frozen snapshot | Newer snapshot |
| --- | --- | --- |
| R01 regression tests | 22 passed | 22 passed |
| Real summary pipeline, unmasked | Passed, including rendered HTML | Reaches styling, then fails on removed `Styler.applymap` (R03) |
| Original fifteen-test suite | 13 passed, 2 inherited failures (R04) | 7 passed, 8 inherited failures (R02/R04) |
| Installed dependency consistency | Passed | Passed |
| Runtime setuptools / pkg_resources | Absent | Absent |

The original clean wheel fails twenty of the new checks and passes two. With the
historical setuptools workaround, it passes eighteen and fails four, exposing
the missing export, housing description, and undeclared styling dependency.
Those diagnostic runs do not count as successful installations.

Frozen pins came from the review's original frozen runtime/test export, excluding
the later setuptools diagnostic packages. Newer pins are the saved review
snapshot, not a fresh resolution. Both snapshots installed Jinja2 3.1.6 and
MarkupSafe 3.0.3. A third fresh environment installed the wheel without
preinstalling styling dependencies: its resolver supplied Jinja2, the styling
check passed, dependency consistency passed, and setuptools remained absent.

The new test file passes Ruff. The three touched source files retain the same
thirteen diagnostic codes/counts as the checkpoint. Whitespace checks exclude
raw evidence logs, which are preserved verbatim. Summary results are in
[validation.json](evidence/r01-installed-package/validation.json); requirements,
environment identities, build output, and unmasked test logs are alongside it.

## Reproduction and resume

Use a clean source snapshot containing the R01 changes, not the published 3.0.0
archives. Keep its build products and environments outside the checkout. In the
completed run, `r01_run` was `/tmp/arfs-r01.7gLmPw`; that directory is disposable.

```sh
uv build --sdist "$r01_run/source" --out-dir "$r01_run/dist"
uv build --wheel "$r01_run/dist/arfs-3.0.0.tar.gz" --out-dir "$r01_run/dist"
uv venv --python 3.12 "$r01_run/check-env"
uv pip install --python "$r01_run/check-env/bin/python" \
  -r "$repo/reviews/evidence/r01-installed-package/frozen-requirements.txt" \
  "$r01_run/dist/arfs-3.0.0-py3-none-any.whl"
uv pip check --python "$r01_run/check-env/bin/python"
```

Set `repo` to the checkout, copy its `tests/` into `$r01_run/run/`, then run there.
For the newer snapshot, use `newer-requirements.txt` instead of the frozen file.
The recorded matrix additionally pinned Jinja2 3.1.6 and MarkupSafe 3.0.3.

```sh
cd "$r01_run/run"
export MPLBACKEND=Agg MPLCONFIGDIR="$r01_run/mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$r01_run/check-env/bin/python" -I -m pytest tests/test_installed_package.py \
  -q --tb=short -p no:cacheprovider
"$r01_run/check-env/bin/python" -I -m pytest \
  "$repo/reviews/evidence/03-selector-contracts/checks.py" \
  -k summary_real_pipeline --runxfail -q --tb=short -c /dev/null -p no:cacheprovider
"$r01_run/check-env/bin/python" -I -m pytest \
  tests/test_featselect.py tests/test_allrelevant.py -q --tb=short -p no:cacheprovider
```

Resume with [R02 in the remediation plan](REMEDIATION.md), the shadow-write
finding in [review 00](00-baseline.md), and midpoint assignment in
[review 02](02-preprocessing.md). Recheck the working tree and relevant callers.
R03 owns the newer summary API failure; R04 owns the two baseline test defects,
stale lock, dependency support choices, and runtime CI. No repair for those
phases was included here.

Not checked: other Python versions/platforms, full notebooks/docs, optional
backends, statistical validity, or performance. Version remains 3.0.0 for local
development; these repaired archives were not published. R01 completion does
not establish overall release readiness.
