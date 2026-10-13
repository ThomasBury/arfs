# R05 — Valid observations at boundaries

**R05 complete on 2026-10-11. Next group: R06 alone.**
Based on `170b3e95fee635e8243ed8cdc740ecb80c4738ad` on `main`; changes are
uncommitted. Pre-existing R04 changes and review/batching edits were preserved.
No commit, push, version bump or release was performed.

## Changes and decisions

The duplicate `validate_sample_weight` converter is replaced by one shared
validator. Lists, arrays and Series become positional float arrays. Weights must
be one-dimensional, match the observation count, be finite and nonnegative, and
have positive mass. Individual zeros remain valid. Values are not normalized:
frequency versus relative/exposure weight semantics remain R08/R15/R25.

The shared target validator rejects empty, multidimensional, misaligned,
missing and infinite responses. Categorical labels remain supported. Leshy no
longer imputes missing outcomes as zero. All three all-relevant selectors, shared
importance helpers, GBM training/splitting, GLM fitting/scoring/search/CV, MRMR,
and the permutation benchmark use the appropriate boundary checks. GBM supplies
those checks to VariableImportance and tree-discretizer training callers. Rows
are aligned by position, with no index join or response imputation.

Association frame preprocessing now converts weights before masking, always
applies the complete-case mask, and rejects empty or zero-mass retained samples.
The dispatcher and worker no longer impose a second missing-value policy.
`handle_na=None` passes original rows and aligned weights to custom callbacks,
including with two workers. Direct built-in pairwise statistics require complete
one-dimensional observations. Empty worker chunks retain a Series result; worker
allocation/label policies remain R28.

Grouped correlation ratio/ANOVA validate all supplied observations and weights,
then omit zero-mass groups and zero-weight terms. Cramer's V omits zero-margin
rows/columns before chi-square evaluation. ANOVA requires at least two
positive-mass groups; Cramer's V requires at least two positive-mass categories
per variable. Undefined inputs raise rather than inventing an association.
Existing constant-response arithmetic and degrees of freedom remain R08.

Every train/evaluation split requires valid observations and positive weight
mass. This includes the GBM holdout, shared SHAP/permutation holdouts, GrootCV
folds and GLM CV folds. Validation reads caller data without mutating it; the
existing categorical encoding/mutation issues assigned to R06/R07 remain open.
Direct covariance and contingency tables now also use positional Series values.

The maintained [regressions](../tests/test_observation_boundaries.py) cover these
failures without expected-failure markers, including independent weighted
benchmark references through the real plot path. Zero-weight category fixtures
use explicit object dtype to isolate R05 from R06's inferred-string defect.
The repaired duplicate-converter Ruff allowance was removed.

## Validation and reproduction

Evidence is in [r05-observation-boundaries](evidence/r05-observation-boundaries/).
The original named historical failures reproduce on the installed R04 wheel:
16 failed, 121 deselected with `--runxfail`. The final promoted suite on that
wheel records 106 failed, 66 passed. Those logs are historical failure evidence,
not a claim that every passing control must fail before repair.

| Final installed wheel cell | Full maintained suite |
| --- | --- |
| Python 3.9.25, locked | 236 passed, 76 warnings |
| Python 3.12.7, locked | 236 passed, 79 warnings |
| Python 3.12.7, newer | 232 passed, 125 warnings |

Each cell runs outside the checkout in isolated Python mode, with two-worker
cases enabled. The suite includes 172 observation-boundary/reference cases,
installed-package imports/exports/resources, and the existing selection checks.
Pandas 2.2 runs Copy-on-Write off/on; pandas 3 has mandatory Copy-on-Write,
which accounts for four fewer cases. Environment JSON records Python, all
dependency versions, import paths and
module hashes. `uv pip check` passes in all cells; setuptools is absent.
The final wheel's source payload matches the checkout; its hash is recorded in
[wheel.json](evidence/r05-observation-boundaries/wheel.json).

Locked numerical/historical checks record 26 passed, 111 deselected. The newer raw
historical checks retain one known R06 failure: the old zero-category fixture
infers pandas 3's string dtype, which `correlation_ratio` does not yet recognize.
The explicit-object maintained case passes in all cells. The first reference
runs also retain two obsolete benchmark-mock failures (`labels` versus R03's
`tick_labels`); the new maintained references use actual plotting and verify
both weighted training and validation importances against sklearn.
First wheel failures are retained separately: removing worker-side filtering
initially changed empty-chunk result inference, now covered by the existing
one/two-worker relevance checks. No historical scripts were edited.

Commands from the checkout, using fresh disposable paths for a repeat:

```sh
repo="$PWD"
audit_dir=$(mktemp -d /tmp/arfs-r05-repeat.XXXXXX)
export UV_CACHE_DIR="$audit_dir/cache"
uv lock --check --python 3.12
uv export --locked --extra test --no-emit-project \
  --output-file "$audit_dir/locked-requirements.txt"
uv build --python 3.12 --out-dir "$audit_dir/dist"
mkdir -p "$audit_dir/run"
cp -r tests "$audit_dir/run/tests"

# Repeat for py39-locked / Python 3.9;
# for py312-newer, use .github/requirements/runtime-newer-py312.txt.
cell=py312-locked
python_version=3.12
requirements="$audit_dir/locked-requirements.txt"
uv venv --python "$python_version" "$audit_dir/$cell"
uv pip install --python "$audit_dir/$cell/bin/python" -r "$requirements" \
  "$audit_dir/dist/arfs-3.0.0-py3-none-any.whl"
uv pip check --python "$audit_dir/$cell/bin/python"
cd "$audit_dir/run"
env MPLBACKEND=Agg MPLCONFIGDIR="$audit_dir/mpl-$cell" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0 \
  "$audit_dir/$cell/bin/python" -I -m pytest tests \
  -q --tb=short -p no:cacheprovider
"$audit_dir/$cell/bin/python" -I \
  "$repo/reviews/evidence/r05-observation-boundaries/environment.py"
```

Local evidence used uv 0.12.15 and `/tmp/arfs-r05`, with dependency pins and
interpreters verified against R04 before reuse. Each runtime cell was newly
created; final wheel revisions were explicitly reinstalled after the worker fix.
The sdist-then-wheel build uses native package resources without the old
setuptools workaround. Dependency versions did not change in R05.

The exact historical-check selectors and commands are recorded in
[commands.txt](evidence/r05-observation-boundaries/commands.txt).
Ruff 0.17.0 reports 428 existing diagnostics and zero new ones; its gate unit
tests pass (2 tests). New tests and the evidence recorder pass Ruff check/format
check, and `git diff --check` passes. An initial check with Ruff 0.16.10 reported
an unrelated date rule absent from the pinned gate; validation uses CI's 0.17.0.
Warnings from SHAP, sklearn tags, GLM convergence and plotting remain visible.

Hosted CI, docs/notebooks, optional backends, performance, full sklearn
compliance and broad statistical calibration were not checked. A green R05 is
not release readiness; weight arithmetic/ranking and other pending defects
remain assigned to later phases.

## Resume

Run **R06 alone** next. Its R01/R05 prerequisites are complete. Read `1A F13`,
`03 F4`, `05 C1`, `06 F1` and trace pandas dtype/category encoding callers.
Use this handoff's boundary contract when updating category identity and dtype
agreement. R06–R33 remain pending; do not carry this session into R06.
