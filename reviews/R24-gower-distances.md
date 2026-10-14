# R24 — Valid Gower distances

**R24 complete on 2026-10-14**, based on clean `main` at `4e50f38`.
Local phase commit: 5325b7f.
R24 runs alone, with R06 complete.

## Changes and decisions

The user chose **NaN for pairs with no comparable positive-weight features**,
including the diagonal of an entirely missing row. Missing numeric and nominal
features are omitted separately for each pair, with the remaining feature weights
renormalized. Observed constant numeric features still contribute zero and retain
their denominator weight. Missing columns no longer dilute available information.
The pairwise weighted mean and undefined-pair rule follow the
[cluster package's Gower documentation](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/daisy.html).
This implements numeric and nominal dissimilarity; ordinal and asymmetric-binary
scoring are not added.

Numeric ranges retain the existing **pooled X/Y observed range** policy, now stated
in the API. Numeric columns use float64 max-minus-min spans and centered scaling;
negative/zero maxima and nearby large values no longer corrupt distances.
Infinite numeric values and ranges exceeding float64 capacity fail clearly.
Constant and entirely missing columns avoid reduction/division warnings. Values
are still represented in float64: arbitrary integer/decimal precision is not
promised. Feature weights are validated as a positional vector of finite,
nonnegative real values with positive mass. They are not observation weights.
Zero weights omit contributions. Scaling weights within each available pair
avoids overflow and preserves comparability when only a tiny-weight feature is
observed; common positive rescaling leaves the weighted mean unchanged.

Only an actual self-comparison (omitted Y or the same input object) uses triangular
calculation and mirroring. Every cross-comparison computes all pairs, including
equal-sized frames and arrays. The returned dense float64 matrix has shape
(n_x, n_y). Empty row axes are supported; zero feature columns are rejected.
Float64 doubles the output's bytes per entry relative to the old float32 result.
Self-distance storage remains quadratic, with row-sized feature scratch arrays;
no full pair-by-pair-by-feature tensor or new execution backend is introduced.

Inputs accept two-dimensional ndarrays and DataFrames, plus a named reference
Series normalized to one row. Named schemas must match in order, column names
must be unique and widths must agree. Sparse/unsupported/non-matrix input fails
before dtype/schema processing. Arrays get soft per-column pandas inference with
missing sentinels normalized for inference; DataFrame dtypes retain their declared
roles. Automatic categories reuse R06's pandas-aware dtype helper for object,
string, category and Boolean columns. Nullable numeric and categorical inputs
work without changing the caller's values, indexes or dtypes. Nominal equality
uses original values, so 1 and '1' remain different and literal 'nan' stays observed.
Numeric conversion failures raise TypeError instead of silently coercing labels.

Explicit category specifications accept string DataFrame names, a complete Boolean
mask, or zero-based integer positions in lists/tuples/one-dimensional arrays.
An empty sequence means all columns are numeric. Unknown names, out-of-range
positions, wrong-length Boolean masks and mixed/unsupported masks fail clearly.
Integer entries always denote positions, even on integer-named frames. The sample
caller now passes a Boolean mask derived from its chosen nonnumeric column names,
so integer feature labels are not mistaken for positions. Its imputation, output,
observation-weight and retry defects remain assigned to R25/R26. gower_topn uses
the shared repaired distance; its key/query/count/nonfinite-selection defects
remain with R27, including its handling of undefined distances.

Caller tracing covered gower_matrix, its only private row-helper caller, sample,
gower_topn, dtype helpers, all source/tests/notebook references and API autodoc.
Historical review scripts, reports and logs are unchanged. The Unreleased changelog
and function docstring record the policy and API boundaries. Context7's pandas
infer_objects documentation was consulted for noncoercive column inference;
nullable and missing-sentinel behavior is verified in both dependency snapshots.
No dependency, public parameter, sampling algorithm or nearest-neighbor policy
is added. Ruff allowances removed with the replaced code are retired; the
remaining sampling allowances belong to pending sampling repairs.

## Validation

Exact commands and boundaries are in
[evidence/r24-gower-distances/commands.txt](evidence/r24-gower-distances/commands.txt).
Fresh installed-wheel full suites pass **1177 / 1177 / 1161** in
py39-locked / py312-locked / py312-newer. All **28** selected unchanged historical
R24 checks pass unmasked in both Python 3.12 cells. On the verified R23 locked
wheel, those checks reproduce **17 failures and 11 passing controls**.

The maintained suite adds **89** cases. Identical finalized tests on the R23 wheel
report **75 failures and 14 passing controls**. All 13 installed source-module
hashes, including sampling.py, match base 4e50f38, without setuptools. Affected
source/dtype callers pass **139 tests**. Hand-calculated and independent scalar
references cover numeric translation/scale, weighted mixtures, cross/transposed
matrices, pairwise missingness/undefined pairs/constants, category identity,
arrays/Series/nullable dtypes, masks, schemas, sparse/shape errors, feature-weight
scaling/extremes, numeric overflow and affected sampling/neighbor callers.

Per-cell JSON records versions, exact commands/cwd/exit codes, installed paths
and hashes. All copied tests and 13 installed source-module hashes match final
source; all 19 source/sdist/wheel payloads agree. Dependency checks and uv lock
--check pass. Ruff baseline (**390 total, zero new**), gate unit tests (**two
passed**), changed-test/evidence lint and format, local links and whitespace
checks pass. Existing warnings remain visible.

Hosted CI, docs/notebook execution, optional backends, other platforms, large
matrix performance and statistical calibration were not checked. Undefined
pairs return NaN and require deliberate handling by downstream consumers;
sampling/nearest-neighbor repairs and final release evidence remain pending.
Passing R24 does not establish release readiness.

## Resume

Stop after R24. Next recorded group: **R25 → R26**, conditional on resolving
observation-weight/representative, tiny-input and exhaustion policies. R05 and
R24 prerequisites are complete. Read [05 F6–F10](05-sampling-and-parallel.md),
sample, isolation_forest, isof_find_sample and get_5_percent_splits. Complete and
commit R25 before starting R26; stop at that boundary if its scope or choices
remain unresolved. R25–R30 and R32–R33 remain pending; R31 is complete.
No publication, version bump or release.
