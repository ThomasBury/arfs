# R07 — Faithful shadow controls

**R07 complete on 2026-10-12. Next group: R08 alone.**
Based on `170b3e95fee635e8243ed8cdc740ecb80c4738ad` on `main`; changes are
uncommitted. Pre-existing R04–R06 changes and review/batching edits were
preserved. No commit, push, version bump or release was performed.

## Changes and decisions

All source changes are in `allrelevant.py`. BoostAGroota encodes its existing
owned copy rather than the caller's frame. GrootCV's reducer now owns copying,
encoding and the original category names together, before fold splitting.
Shared native/SHAP/permutation importance fitting also encodes owned frames.
The public selectors preserve the caller's values, dtypes and index; transform
returns the selected columns of the raw input.

BoostAGroota keeps the categorical names returned by the encoder and updates
their positions after each reduction without trying to infer categories from
integer codes. Its importance calls receive both original categorical positions
and the corresponding shadow positions. When no categorical feature survives,
the explicit list is empty. GrootCV passes original and matching shadow names
to every training/validation fold. Encoding occurs once before splitting, so
both datasets use the same code identities; this does not introduce a fitted
preprocessing vocabulary or claim fold-local preprocessing (R23/R33).

Leshy shuffles extension arrays by row position rather than letting pandas
re-infer the dtype from permuted values. Numeric-valued categories, category
ordering/unused levels, nullable strings/booleans and missing values survive
shadow replication and shuffling. The existing minimum-shadow rule and RNG
sources are unchanged. R06's pandas-equality/first-occurrence encoding and
missing code **-1** remain in effect. Categorical originals and controls now
have the same nominal role, so selected sets can change from the defective
implementation even with the same seed.

One small private naming helper serves both shadow generators. Existing names
remain unchanged when free; collisions append underscores until the candidate
is distinct from every original and already generated shadow. No input-name
namespace is reserved. Shared shadow generation returns originals first and
shadows second, without prefix-based reordering. GrootCV ranking and plotting
select real columns using fitted original identities; they do not guess from
prefixes or merge row order. Genuine input columns such as `ShadowVar1` remain
real features. The model's own feature-name restrictions and duplicate-input
schema policies are not expanded here.

Caller tracing covered all encoding, shadow, reduction and fold paths in the
three selectors, shared importance helpers, GrootCV ranking/plotting, and
their maintained/historical tests. The [new maintained regressions](../tests/test_shadow_contracts.py)
cover real fits and raw transforms under pandas Copy-on-Write off/on, owned
shared importance inputs, preserved shadow values/dtypes/index, repeated name
collisions, numeric labels at the generator boundary, and categorical removal
across three forced BoostAGroota rounds. Real LightGBM fit wrappers check
original/shadow categorical positions. Real GrootCV folds check train/validation
codes, distributions and native Booster categorical metadata; the plot executes
and its displayed labels contain exactly the original features. Argument-based
round tests use fixed importances to isolate metadata, not to claim selection
calibration. Five obsolete lint allowances were removed.

R10–R12 still own Leshy support/history, weighted signed permutation importance
and aggregation/objective decisions. Their arithmetic was not changed. No new
dependency, public option, export, model backend or compatibility layer was added.
No R07 policy decision remains unresolved.

## Validation and reproduction

Evidence is in [r07-shadow-contracts](evidence/r07-shadow-contracts/).
The named historical R07 failures reproduce against the installed R06 wheel in
both 3.12 snapshots: **7 failed, 17 deselected**. The final maintained R07 cases
on the old newer wheel record **26 failed, 4 passed**. Historical scripts and
logs remain intact; maintained regressions have no expected-failure markers.

| Fresh installed R07 wheel | Full maintained suite | Historical R07 cases and shadow/fold controls |
| --- | --- | --- |
| Python 3.9.25, locked; pandas 2.2.3 / NumPy 2.0.2 | 328 passed, 94 warnings | Not repeated on 3.9 |
| Python 3.12.7, locked; pandas 2.2.3 / NumPy 2.0.2 | 328 passed, 97 warnings | 9 passed, 15 deselected |
| Python 3.12.7, newer; pandas 3.0.6 / NumPy 2.5.4 | 312 passed, 87 warnings | 9 passed, 15 deselected |

R07 contributes 42 cases on pandas 2.2 and 30 on pandas 3: twelve real-fit
combinations run with Copy-on-Write both off/on in 2.2 and only its mandatory
mode in 3.0. Four additional count differences come from prior checks.
Each environment was newly created; exact dependencies match its R06 snapshot,
`uv pip check` passes, and setuptools is absent. Environment JSON records
installed paths, dependency versions and source hashes. The R06 installed
source hash also matches the saved pre-edit source used for reproduction.

The sdist-then-wheel build passes, and all 19 packaged source/data payloads match
the checkout; wheel/source hashes are in
[wheel.json](evidence/r07-shadow-contracts/wheel.json). Tests run outside the
checkout with isolated Python and two-worker cases enabled. Final copied test
files match the maintained suite. Commands are recorded in
[commands.txt](evidence/r07-shadow-contracts/commands.txt); disposable environments
must be verified/recreated on resume. The existing R06 interpreter paths were
verified before use.

```sh
# From the checkout; repeat installation for py39-locked and py312-newer.
# The newer cell uses .github/requirements/runtime-newer-py312.txt.
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv lock --check \
  --python /tmp/arfs-r06/py312-locked/bin/python
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv export --locked --extra test \
  --no-emit-project --output-file /tmp/arfs-r07/locked-requirements.txt
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv build \
  --python /tmp/arfs-r06/py312-locked/bin/python --out-dir /tmp/arfs-r07/dist
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv venv \
  --python /tmp/arfs-r06/py312-locked/bin/python /tmp/arfs-r07/py312-locked
env UV_CACHE_DIR=/tmp/arfs-r04/cache uv pip install \
  --python /tmp/arfs-r07/py312-locked/bin/python \
  -r /tmp/arfs-r07/locked-requirements.txt \
  /tmp/arfs-r07/dist/arfs-3.0.0-py3-none-any.whl
mkdir -p /tmp/arfs-r07/run
cp -r tests /tmp/arfs-r07/run/
cd /tmp/arfs-r07/run
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arfs-r07/mpl-py312-locked \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0 \
  /tmp/arfs-r07/py312-locked/bin/python -I -m pytest tests \
  -q --tb=short -p no:cacheprovider
```

Validation used uv 0.12.15. `uv lock --check`, Ruff 0.17.0's baseline gate
(420 existing diagnostics, zero new), gate unit tests (2 passed), new-test
lint/format and `git diff --check` pass. Existing model/tag/precision warnings
remain; the newer GrootCV plot also exposes pandas' Matplotlib `vert`
deprecation warning. No dependency workaround was added.

Hosted CI, complete docs/notebooks, optional model/explanation backends, Arrow
string storage, performance, full sklearn compliance and statistical calibration
were not checked. Plot labels/rendering were checked programmatically, without
a separate visual image review. Green R07 does not imply release readiness.

## Resume

Run **R08 alone** next, with R05/R06 complete and R03 available for newer NumPy.
Read `1A F3–F7/F11/F12` in [the statistical report](01a-statistical-foundations.md)
and its evidence, then trace association/relevance consumers. R08 must settle
weight meaning and degrees of freedom, constant/output behavior, stable weighted
arithmetic and collision-free response augmentation explicitly. Those decisions
remain for R08; this phase does not choose them. R08–R33 remain pending; do not
continue into another group from this session.
