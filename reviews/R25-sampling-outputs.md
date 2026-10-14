# R25 — Sampling returns the intended data

**R25 complete on 2026-10-14**, based on clean `main` at `34641a2`.
This is the first phase of the requested sequential R25 → R26 batch.
Branch: `fix/r25-r26-remediation`. Local phase commit: `b352069`.

## Changes and decisions

Gower sampling rejects observation weights explicitly, even when the row count
happens to equal the feature count. Weighted cluster representatives have no
established public contract, so this batch does not introduce one. Direct
`gower_matrix(weight=...)` remains a separate feature-weight API. Isolation
sampling forwards validated positional observation weights to the native forest
fit. Zero-weight rows are omitted from fitting but all original rows are scored.
Subset draws, score evaluation and KS screening remain unweighted; no
weighted representativeness claim follows from fit-weight forwarding.

Gower numeric imputation now uses each column's mean. Nullable integers can have
fractional imputed means, so the scoring/aggregation copy uses floating numeric
columns. Categorical imputation uses the first column mode; representative-mode
ties retain the first encountered value within the cluster. Entirely missing
columns are rejected rather than inventing a value. Gower still requires at
least two rows and an integer cluster count between 1 and the input row count.

Cluster IDs are an external positional grouper, with the existing `clus_N`
index named `label`. An original feature named `label` is retained, and the
output has exactly the original feature names and order. These are synthetic
means/modes, so original row indexes and numeric dtypes are not promised.
Isolation instead returns a copy of the original selected rows, preserving
values, duplicate indexes, column order and dtypes. The existing string/category
encoding is retained only in its scoring copy; no nominal-scoring policy is added.
Unknown methods raise `NotImplementedError`; invalid feature schemas and weights
fail explicitly. No dependency or public parameter is added.

Caller tracing covered sample, get_most_common, gower_matrix, isolation_forest,
isof_find_sample, get_5_percent_splits, existing tests, notebook source and API
autodoc. Two repaired sampling Ruff allowances are retired. Historical review
reports and evidence are unchanged. New diagnostic evidence is local and ignored
under the user's `reviews/evidence/` rule; maintained regression tests and this
handoff remain reviewable in Git.

## Delivery review follow-up — 2026-10-14

Codex's [PR #61 finding](https://github.com/ThomasBury/arfs/pull/61#discussion_r4270994641)
showed that a zero-weight category could change ordinal-code spacing before the
forest omitted that row. The delivery repair learns the existing sorted string
category vocabulary from positive-weight rows only. Native Index.get_indexer
keeps known codes and gives unseen score-only categories -1, avoiding the newer
pandas deprecation for constructing categoricals with unknown values. Original
selected values/dtypes, unweighted draws and KS population are retained.

Four new regressions fail on the previous wheel and pass after this repair:
object/category/string vocabularies, known versus unseen score-only categories,
input/output ownership, and exact native positive-row score equality with the
zero-weight row physically removed. All 217 affected sampling checks pass in
both dependency snapshots. The original phase validation below is historical;
post-review installed-wheel/hosted results are recorded in the current delivery
entry in [REMEDIATION.md](REMEDIATION.md) and on the PR.

## Validation

The 50 maintained R25 checks on the old R24 installed wheel produce **43 failures
and seven passing controls**. Initial output-only cases produced 14 failures.
Affected source checks pass **189 tests**, using:

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  /tmp/arfs-r24/py312-locked/bin/python -m pytest \
  tests/test_sampling_outputs.py tests/test_gower_distances.py \
  tests/test_category_identity.py -q --tb=short -p no:cacheprovider
```

Installed-wheel full-suite results: **1227 / 1227 / 1211 passed** in
py39-locked / py312-locked / py312-newer. All **11** selected unchanged historical
R25 checks pass unmasked in both Python 3.12 cells. The old accept-Gower-weights
and known-broken-schema controls are superseded by the explicit policy above.
The initial newer run found zero-weight native sampling failure (1210 passed,
one failed); omitting zero-weight fit rows fixes it, and all three final suites
were rerun on the rebuilt wheel.
The matrix uses Linux Python 3.9.25 locked and 3.12.7 locked/newer.
Locked: NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1, sklearn 1.6.1.
Newer: NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, sklearn 1.9.1.
Fresh venvs install the newly built wheel outside the checkout, without
setuptools; all 13 installed source-module hashes and copied test bytes match.
All 19 source/sdist/wheel payloads agree. Dependency checks, `uv lock --check`,
Ruff (**331 total, zero new**, with ignored evidence excluded), two gate unit
tests, changed-code lint/format and whitespace checks pass.

Build and matrix commands:

```sh
uv export --locked --extra test --no-emit-project --output-file /tmp/arfs-r25-final/locked-requirements.txt
uv build --python /tmp/arfs-r24/py312-locked/bin/python --out-dir /tmp/arfs-r25-final/dist
/tmp/arfs-r24/py312-locked/bin/python /tmp/arfs-r25-final/check_payloads.py
/tmp/arfs-r24/py312-locked/bin/python /tmp/arfs-r25-final/validate.py py39-locked
/tmp/arfs-r24/py312-locked/bin/python /tmp/arfs-r25-final/validate.py py312-locked
/tmp/arfs-r24/py312-locked/bin/python /tmp/arfs-r25-final/validate.py py312-newer
uv lock --check --python /tmp/arfs-r24/py312-locked/bin/python
uv tool run --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
/tmp/arfs-r24/py312-locked/bin/python -m unittest discover -s .github/scripts -p 'test_*.py'
```

The runner records exact argv/cwd/exit codes and environment identities under
ignored `reviews/evidence/r25-sampling-outputs/`. Context7's pandas 2.2.3 and
native IsolationForest documentation were consulted; behavior is exercised in
both dependency snapshots. Native fit weights do not change the sampler's
unweighted draws or comparison population.

Hosted CI, notebook/docs execution, optional backends, other platforms, large
matrix performance and sampling calibration are not checked. Gower storage
remains dense and quadratic. At the R25 phase boundary, R26's retry/tiny-input repair was still required
before isolation sampling was bounded; it is now complete in [R26](R26-isolation-attempts.md). This phase does not establish release readiness.

## Resume

R25 and [R26](R26-isolation-attempts.md) are now complete in separate sequential
local phase commits. R26 records the chosen tiny-input, size and exhaustion
policies. Stop after this requested batch; the next group is **R27 alone**, with
R24 complete. R28–R30, R32 and R33 also remain pending; R31 is complete.
No push, PR, version bump or release is part of this request.
