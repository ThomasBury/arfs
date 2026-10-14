# R27 — Query-aware nearest neighbors

**R27 complete on 2026-10-14**, based on clean local R25/R26
handoff commit `fcb811e`. Branch: `fix/r27-nearest-neighbors`.
R27 runs alone, with R24 complete. Local phase commit: `45c3f98`.

## Changes and decisions

`gower_topn` now requires an explicit one-row query. The existing default None
remains in the signature so omission raises a clear ValueError; no guessed
query or compatibility behavior is introduced. A DataFrame/ndarray must have
exactly one row; a Series is normalized to one named reference row. All named
feature schemas, numeric/missing inputs, category masks and feature weights
retain the shared R24 validation, including for a zero-neighbor request.
Feature weights and masks describe the scored columns after removing a key.

With a key, both inputs must be named, the key must exist in each, and the query
identifier must be nonmissing. Every candidate matching that identifier is
excluded; missing candidate identifiers are ineligible. The key is not a scored
feature. The remaining finite-distance rows are ranked once, then pandas
`duplicated(keep="first")` retains the nearest row per ID. Equal distances retain
candidate order, so the first original row resolves tied duplicate IDs.
Pandas membership/duplicate semantics preserve mixed and tuple identifiers
without sorting/coercing them. Without a key, finite-distance rows remain
eligible even when a candidate is the query itself.

n is a nonnegative integer (excluding Boolean counts). Zero returns typed empty
index/value arrays. Asking for more than the finite, eligible, unique-key count
raises ValueError; results are not silently clipped. Undefined NaN and infinite
distances are excluded, never converted to a plausible maximum distance.
`smallest_indices` shares the finite-value and stable-tie policy, preserves
flattened positions and numeric value dtypes, and rejects non-real/nonnumeric
values clearly. Inputs and duplicate DataFrame indexes are preserved.

The lookup calls shared `gower_matrix(candidates, query)` once and retains its
N-by-1 result as a distance vector, avoiding an N-by-N candidate self-matrix.
Candidate-first orientation retains candidate dtype roles for heterogeneous
Series queries and numeric-valued categorical features. Gower's existing pooled
ranges include all candidates and the query before key filtering. Distance
output storage is O(N); shared feature preparation still needs O(NP) space.
Stable ranking is O(N log N), followed by one key-deduplication pass. The recorded
shortcut keeps one shared Gower row calculation per candidate; vectorize only
after a measured throughput need. No new distance backend, approximation,
public parameter or dependency is added.

Caller tracing covered both functions, their only source calls, the R24
nearest-neighbor caller check, all maintained tests, historical review scripts,
API autodoc and notebook source. No source/notebook callers outside the sampling
module were found. Historical reports and evidence are unchanged; new logs remain
local and ignored under the user's `reviews/evidence/` rule. Maintained tests,
API prose, changelog and this handoff are reviewable in Git.

## Validation

All **45** maintained R27 checks pass. The identical finalized tests on the R26
installed wheel produce **40 failures and five passing controls**. Selected
unchanged historical checks reproduce their key/query/count failures separately.
The old optional-query and nonfinite-substitution controls are superseded by the
explicit required-query/finite-only policies above and are preserved unchanged.

Affected source checks pass **213 tests in each dependency snapshot**:

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  /tmp/arfs-r26/py312-locked/bin/python -m pytest \
  tests/test_nearest_neighbors.py tests/test_gower_distances.py \
  tests/test_sampling_outputs.py tests/test_isolation_sampling.py \
  -q --tb=short -p no:cacheprovider
```

The same command using `/tmp/arfs-r26/py312-newer/bin/python` also passes 213.
Cases cover changed actual queries, key exclusion, nearest duplicate rows,
insufficient unique candidates, mixed/tuple/nullable IDs, stable ties, integer
feature names, Series/array queries, feature weights/masks/schema validation,
missing/undefined queries, empty candidates, zero/invalid counts, finite-only
helper ranking and a spy asserting one N-by-1 matrix call with the real query.

Installed-wheel full-suite results: **1301 / 1301 / 1285 passed** in
py39-locked / py312-locked / py312-newer. All **six** selected unchanged
historical key/query/count checks pass unmasked in both Python 3.12 cells;
the old wheel produces four failures/two passing controls for those checks.
The matrix is Linux Python 3.9.25 locked and 3.12.7 locked/newer.
Locked: NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1, sklearn 1.6.1.
Newer: NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, sklearn 1.9.1.

```sh
uv build --python /tmp/arfs-r26/py312-locked/bin/python --out-dir /tmp/arfs-r27/dist
/tmp/arfs-r26/py312-locked/bin/python /tmp/arfs-r27/check_payloads.py
/tmp/arfs-r26/py312-locked/bin/python /tmp/arfs-r27/validate.py py39-locked
/tmp/arfs-r26/py312-locked/bin/python /tmp/arfs-r27/validate.py py312-locked
/tmp/arfs-r26/py312-locked/bin/python /tmp/arfs-r27/validate.py py312-newer
uv lock --check --python /tmp/arfs-r26/py312-locked/bin/python
uv tool run --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
uv tool run --from ruff==0.17.0 ruff check src/arfs/sampling.py tests/test_nearest_neighbors.py
uv tool run --from ruff==0.17.0 ruff format --check src/arfs/sampling.py tests/test_nearest_neighbors.py
/tmp/arfs-r26/py312-locked/bin/python -m unittest discover -s .github/scripts -p 'test_*.py'
git diff --check
```

The runner reuses R26's locked export or checked-in newer requirements. Fresh
venvs install the rebuilt wheel outside the checkout without setuptools;
13 installed source-module hashes and copied test bytes are verified. All 19
source/sdist/wheel payloads agree. Exact argv/cwd/exit codes, package identities
and logs are local under ignored `reviews/evidence/r27-nearest-neighbors/`.
Dependency/lock checks, Ruff (331 total, zero new), gate unit tests (two passed),
changed lint/format and whitespace pass. Context7's pandas 2.2.3 membership and
first-duplicate documentation was consulted; mixed/nullable identifier behavior
is exercised in both dependency snapshots.

Hosted CI, full docs/notebooks, optional backends, other platforms, large-data
throughput and release/calibration evidence are not checked. Stable sorting and
candidate-first row calculations are deliberately retained; linear distance
storage alone does not establish large-data throughput. An insufficient finite
neighbor population now raises instead of selecting undefined matches.

## Resume

Stop after the local R27 phase commit. Next is **R28 alone**, with R06/R08
complete. Trace parallel_matrix_entries/parallel_df and their association callers,
preserving original labels and defining worker/order/empty-result contracts.
R29–R30, R32 and R33 also remain pending; R31 is complete.
No push, PR, version bump or release is part of this request.
