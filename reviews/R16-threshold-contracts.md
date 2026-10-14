# R16 — Threshold selector identity and state

**R16 complete on 2026-10-13. Next group: R17 alone.**
Based on clean `main` at `985b0a3`. R16 runs alone; the local implementation
commit and completed validation are recorded in [REMEDIATION.md](REMEDIATION.md).
No publication or version bump is part of this group.

## Changes and decisions

**Names and output: retain named projection, reject duplicates.** The base,
collinearity and variable-importance selectors reject duplicate column names at
fit and transform. Transform selects learned names in fitted order, accepts
reordered/extra columns and omitted rejected columns, and raises KeyError for
missing selected columns. It preserves rows, index and dtypes, including an
empty selection or a transform input with no rows. Fit requires observations;
zero-feature unsupervised pipeline stages remain valid. Input arrays remain
unsupported by these DataFrame selectors. Successful refits replace feature
names, counts and support. The shared guard changes no encoding or input data.

**Statistics: align labels before making the mask.** A base callback must return
one finite real numeric statistic per input feature in a Series. Missing, extra
or duplicate statistic labels, missing/nonfinite values and nonnumeric/complex
statistics fail clearly. Reordered labels are aligned to input columns; named
Series produce the same `statistic` summary column as unnamed Series. Summary
ties retain input order. Built-in missingness, uniqueness and cardinality
statistics remain unweighted; their inherited weight documentation now says so.

**Parameters and equality: preserve the existing strict rules.** Validation runs
at fit, including after `set_params`. Base cutoffs must be finite real numbers;
its callback must be callable and its direction boolean. Missingness and
collinearity cutoffs are in [0, 1]. Unique/cardinality cutoffs are integers >= 1;
fractional counts and booleans are rejected. Collinearity validates its current
method at fit, rather than only at construction. Equality is unchanged:

| Selector | Retain rule | Equality |
| --- | --- | --- |
| MissingValueThreshold | missing fraction < threshold | reject |
| UniqueValuesThreshold | nonmissing unique count > threshold | reject |
| CardinalityThreshold | categorical nonmissing count < threshold | reject |
| CollinearityThreshold | remove pairs with absolute association > threshold | retain |

Missing threshold zero still rejects complete columns; one rejects wholly
missing columns. Numeric cardinality is zero. Collinearity threshold one keeps
perfectly associated pairs. No statistical threshold rule is invented here.
API docstrings and the Unreleased changelog describe the repaired contract.

**Collinearity: no pairs is valid, and ties follow input order.** An empty pair
result becomes a zero matrix labelled with the input columns. Boolean support
retains all zero/one candidate features. The existing association input path
still validates observations, missing-row retention and observation weights;
the no-pair path does not bypass those boundaries. Correlation mode skips the
unneeded encoder for fewer than two candidates. Full matrices are reindexed to
fitted order before elimination; equally connected candidates drop the first
in that order, independent of Python's process hash seed. Directional totals,
absolute association and strict cutoff behavior remain unchanged.

All five concrete selectors expose `n_features_in_` after successful fit.
Transform and support access check the actual learned support and raise
NotFittedError before fit, including when unrelated metadata exists.
VariableImportance changes only its duplicate-name boundary, metadata and
fitted guards. Its crossing-feature rule, zero/nonfinite contributions,
iteration behavior, multiclass arrays and optional-backend error remain R17.

Caller tracing covered base callbacks, the three derived threshold selectors,
collinearity encoding/association/pivot/elimination, VariableImportance
fit/support/transform, public exports, pipeline summaries, established tests,
historical finding checks and notebook source callers. No export, dependency,
backend, compatibility layer or notebook execution is added. Historical review
scripts and logs are preserved. Constructor validation follows the
[sklearn estimator guidance](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/developers/develop.rst),
checked through Context7; sklearn 1.6.1 and 1.9.1 run in the validation matrix.

## Validation

The maintained regression suite contains **84 R16 cases**. Against the verified
installed R15 wheel, it reports **73 failed, 11 passed**. The initial 79-case
probe is also preserved (**71 failed, 8 passed**). Before-provenance checks all
three affected installed selector modules against `985b0a3`.
The final source suite with affected category/observation/API callers passes
**321 tests**. Its initial run exposed an empty-mask dtype error; both the
failure and successful correction remain recorded. Reference cases cover
strict equality, reordered/invalid labelled callbacks, integer/tuple callback
names, duplicate fit/transform names, updated parameters, named projection,
clone/refit/schema changes, empty pipeline stages under both methods, weight
errors without pairs, directional associations, real LightGBM metadata, and
four fresh hash-seed processes.

Installed-wheel results are recorded in the per-cell command manifests and
logs under [the evidence directory](evidence/r16-threshold-contracts/commands.txt).
Full suites pass: **py39-locked 772 passed; py312-locked 772 passed;
py312-newer 756 passed**. All **29** selected historical checks and controls
pass unmasked in each Python 3.12 snapshot.
The wheel is installed into fresh environments outside the checkout without
setuptools; copied tests match source bytes. Eleven installed module hashes
match source, including all three repaired selectors. All 19 declared
source/sdist/wheel payloads agree. Exact orchestration, environment checks and
payload checks are retained alongside the logs.

The locked cells use Python **3.9.25** / **3.12.7**, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, statsmodels 0.14.4 and pytest 8.3.5. The newer Python 3.12.7 cell
uses NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, statsmodels 0.15.0 and pytest
9.1.1. Per-cell JSON lists complete versions, paths, hashes, commands and exit
codes. Ruff 0.17.0 reports **414 total, zero new** diagnostics; three repaired
allowances are removed. New test/evidence lint and format, Ruff gate unit tests,
lock consistency, local handoff links and whitespace checks pass; the gate unit
suite has **two passing tests**. Exact commands are recorded separately. Existing model,
convergence, tag, precision and plotting warnings remain visible.

Hosted CI, full docs/notebooks, full sklearn compliance, other platforms,
optional backends, performance and statistical calibration are not checked.
Passing R16 checks does not establish release readiness.

## Resume

Stop after R16. Next group is **R17 alone**, with R03/R05/R06 complete. Read
[03 F1–F3/F11/F13](03-selector-contracts.md), `variable_importance.py`, its
training/contribution callers and the controlled/real importance evidence.
Settle the retained cumulative prefix, zero-total selection, weighted explanation
population and whether repeated identical fits should remain. R17–R33 remain
pending; the earliest later pair remains R18 → R31, with the public legacy-helper
decision required before that group.
