# R06 — Category identity and dtype agreement

**R06 complete on 2026-10-11. Next group: R07 alone.**
Based on `170b3e95fee635e8243ed8cdc740ecb80c4738ad` on `main`; changes are
uncommitted. Pre-existing R04/R05 work and review/batching edits were preserved.
No commit, push, version bump or release was performed.

## Changes and decisions

The shared dtype classifier and direct correlation ratio now use one private
pandas-aware categorical rule: object, category, string and boolean dtypes are
nominal. This covers nullable strings/booleans, pandas 3 inferred strings and
numeric-valued categories. Association series dispatch uses the same dtype
groups as matrix dispatch and its workers, fixing the numerical routing of
booleans. Cardinality counting and existing dtype/encoder consumers inherit the
rule. Numeric, time, interval and unknown group construction is unchanged,
including pandas' existing inclusion of timedelta in the numeric group.

`get_pandas_cat_codes` uses pandas factorization of original values rather than
stringification and sorting. Integer `1` and string `'1'` stay distinct; literal
`'nan'`/`'None'` stay distinct from missing values. Nonmissing codes follow first
occurrence, using pandas equality semantics; values pandas regards as equal
(for example `1` and `1.0`) share a code. All missing sentinels (`None`, NaN,
`pd.NA`, NaT) receive **-1**, preserving missingness for the existing model
callers. Unused categorical levels do not create observed codes.

The helper still mutates the supplied frame in place, returns the same frame
and converted names/positions, and retains no fitted vocabulary for new data.
These limits are recorded in its API docstring. Numeric and interval columns,
row index and passthrough dtypes are preserved. Codes can differ from the old
sorted string codes; callers must not depend on the old category numbers.

Correlation ratio and categorical ANOVA group original values with `pd.unique`
instead of sorting with NumPy. This allows mixed integer/string levels without
merging them or raising during sorting. R05 still rejects missing observations
at direct statistic boundaries; R06 does not impute them or change statistical
weight arithmetic.

Scope tracing covered every `create_dtype_dict` consumer in association,
cardinality/collinearity, preprocessing, GBM and all-relevant selection, and
every shared category encoder call in importance fitting, BoostAGroota and
GrootCV. The [50 maintained cases](../tests/test_category_identity.py) include
real native, SHAP, permutation, one-round BoostAGroota and two-fold GrootCV
fits with assertions on the actual encoded input. They also cover all nominal
dtype forms, numeric-valued categories, missing/unused levels, time/unknown
encoding, passthrough schema, and one/two-worker series/matrix agreement.
Series self-associations remain present; matrix comparisons cover off-diagonal
values by labels, without imposing the result-order policy reserved for R28.

R07 still owns selector input ownership, categorical roles after feature
reduction, shadow roles and labels. R18/R19/R21/R24/R31 still own their separate
schema, native model, Gower extension-input and legacy-helper contracts. No new
public helper, dependency, preprocessing API or compatibility layer was added.
Three obsolete association Ruff allowances were removed.

## Validation and reproduction

Evidence is in [r06-category-identity](evidence/r06-category-identity/).
The named historical failures reproduce against the installed R05 wheel:
locked **5 failed, 2 passed**, newer **7 failed** (212 deselected in each).
The final maintained R06 cases on the old newer wheel record **38 failed,
12 passed**. Historical scripts/logs were preserved; maintained tests have no
expected-failure markers.
The previously blocked R05 inferred-string zero-weight-category references also
pass on the newer wheel (3 passed, 39 deselected).

| Fresh installed R06 wheel | Full maintained suite | Named historical R06 checks and controls |
| --- | --- | --- |
| Python 3.9.25, locked; pandas 2.2.3 / NumPy 2.0.2 | 286 passed, 93 warnings | Not repeated on 3.9 |
| Python 3.12.7, locked; pandas 2.2.3 / NumPy 2.0.2 | 286 passed, 96 warnings | 10 passed, 209 deselected |
| Python 3.12.7, newer; pandas 3.0.6 / NumPy 2.5.4 | 282 passed, 85 warnings | 10 passed, 209 deselected |

Runs use isolated Python outside the checkout with actual two-worker cases.
The four-case count difference comes from the existing pandas Copy-on-Write
off/on checks. All three environments were newly created, match R05's recorded
dependency versions, pass `uv pip check`, and have no setuptools workaround.
Environment JSON records exact dependencies, installed import paths and module
hashes. The final test files match the copied suite. The sdist-then-wheel build
passes; all 19 packaged source/data payloads match the checkout, recorded with
the wheel hash in [wheel.json](evidence/r06-category-identity/wheel.json).

Exact local commands are recorded in
[commands.txt](evidence/r06-category-identity/commands.txt). The core commands
from the checkout are:

```sh
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv lock --check \
  --python /tmp/arfs-r05/py312-locked/bin/python
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv export --locked --extra test \
  --no-emit-project --output-file /tmp/arfs-r06/locked-requirements.txt
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv build \
  --python /tmp/arfs-r05/py312-locked/bin/python --out-dir /tmp/arfs-r06/dist

# Repeat for Python 3.9 locked and Python 3.12 newer; newer uses
# .github/requirements/runtime-newer-py312.txt instead of the locked export.
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv venv \
  --python /tmp/arfs-r05/py312-locked/bin/python /tmp/arfs-r06/py312-locked
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv pip install \
  --python /tmp/arfs-r06/py312-locked/bin/python \
  -r /tmp/arfs-r06/locked-requirements.txt \
  /tmp/arfs-r06/dist/arfs-3.0.0-py3-none-any.whl
mkdir -p /tmp/arfs-r06/run
cp -r tests /tmp/arfs-r06/run/
cd /tmp/arfs-r06/run
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arfs-r06/mpl-py312-locked \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0 \
  /tmp/arfs-r06/py312-locked/bin/python -I -m pytest tests \
  -q --tb=short -p no:cacheprovider
```

Use fresh disposable paths and verify/recreate the interpreters when resuming;
the `/tmp/arfs-r05` paths above were verified before use. Validation used uv
0.12.15. `uv lock --check`, Ruff 0.17.0's baseline gate (425 existing diagnostics,
zero new), gate unit tests (2 passed), new-test lint/format checks and
`git diff --check` pass. Existing model/tag/precision warnings remain visible.

Hosted CI, docs/notebooks, optional backends (including fasttreeshap and Arrow
string storage), performance, full sklearn compliance and statistical
calibration were not checked. Green R06 does not imply release readiness.

## Resume

Run **R07 alone** next, with R02/R06 complete. Read `1B F5/F6/F10` in
[the all-relevant report](01b-all-relevant.md) and its evidence. Trace ownership
and categorical metadata through Leshy, BoostAGroota and GrootCV, including
successive reductions, shadows and fold inputs. R07–R33 remain pending; do not
continue into another group from this session. No additional R06 decision is
unresolved.
