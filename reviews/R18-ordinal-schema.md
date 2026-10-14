# R18 — Ordinal encoder full schema

**R18 complete on 2026-10-13**, based on clean `main` at `37cf6b2`.
Local phase commit: `7df5daa`, completed before R31 started.

## Changes and decisions

OrdinalEncoderPandas owns one native OrdinalEncoder for the categorical subset.
The wrapper keeps the full input names and feature count, including passthrough
columns; inherited get_feature_names_out and sklearn pandas set_output now agree
with the returned DataFrame. categories_ remains the native category vocabulary
in categorical_features_ order. Numeric-only and zero-feature frames remain valid
passthroughs, including inverse transformation and refit after a categorical fit.

Fit requires a DataFrame with unique column names. Transform and inverse require
the same full names in the fitted order; reordered, missing, extra, replaced or
duplicate columns fail clearly. This deliberately replaces permissive projection
for this encoder. The threshold selectors' separate named-projection contract is
unchanged. Integer labels remain intact in metadata and pandas output.

The native encoder is constructed at fit from current parameters, so dtype,
unknown-value and missing-value changes apply on refit and cloning. Encoded columns
use output_dtype; passthrough values, dtypes, index and caller ownership are
preserved. Inverse returns a copy and passes a writable owned NumPy array to sklearn, whose
missing-value decoding writes into the codes. Decoded category-column dtypes are
inferred by pandas; this is not an original-category-dtype restoration contract. Defaults now include nullable string columns;
nullable string/boolean missing values are converted to native np.nan only in the
owned categorical input. Original labels are not stringified. Native category
sorting and its homogeneous-type restriction remain; this phase does not add a
second category-mapping policy.

Distinct unknown codes decode to None. When unknown and fitted-missing codes both
use the default np.nan, native inverse decoding treats them as missing; the tests
record this ambiguity instead of claiming that inverse can recover it.

Caller tracing covered public fit/transform/inverse and inherited names/output,
benchmark.py, CollinearityThreshold, VariableImportance, pipeline summaries,
maintained tests, notebook source, module exports and the shared dtype selector.
No internal caller needs a signature change. The Unreleased changelog and encoder
API docs describe the fitted schema, dtype and inverse ownership.
Sklearn's [set_output developer contract](https://scikit-learn.org/stable/developers/develop.html#developer-api-for-set-output)
was checked through Context7, together with its OrdinalEncoder names and missing
value behavior; installed sklearn 1.6.1 source confirms inherited names use the
wrapper's n_features_in_ and feature_names_in_.

## Validation

Exact commands and per-cell manifests are retained under
[evidence/r18-ordinal-schema](evidence/r18-ordinal-schema/commands.txt).
The first repaired wheel passed both locked suites (856 tests each), but the newer
cell exposed three native inverse failures (837 passes): pandas 3 supplied a
read-only array that sklearn modifies when restoring missing codes. Interim logs
remain intact. The owned-array repair passes all 35 maintained encoder cases and
7 historical controls/defects in the newer source snapshot. Final installed-wheel full suites pass: **py39-locked 856; py312-locked 856;
py312-newer 840**. All 7 original historical ordinal checks pass unmasked in
both Python 3.12 snapshots. No historical adaptation is required.
The maintained suite has 35 cases: the installed R17 wheel reports 32 failures
and 3 passes. Old-wheel preprocessing source bytes match `37cf6b2` in
before-provenance.json. All four original named defects reproduce unmasked, with
3 historical passing controls. Historical scripts and logs remain unchanged.
The source suite with dependency, importance and threshold callers passes 180 tests.

Locked cells use Python 3.9.25 / 3.12.7, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, statsmodels 0.14.4 and pytest 8.3.5. Newer uses Python 3.12.7,
NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, statsmodels 0.15.0 and pytest 9.1.1.
Complete versions, isolated module paths/hashes and commands are in per-cell JSON.
Copied tests match source bytes; all 12 installed modules and 19 source/sdist/wheel
payloads match. Dependency checks, lock consistency, Ruff 0.17.0 (403 total,
zero new), gate unit tests (2 passed), test/evidence lint/format, local handoff
links and whitespace pass. Four repaired Ruff allowances are removed, none added.
Existing model, convergence, tag, precision and plotting warnings remain visible.
Hosted CI, full docs/notebooks, full sklearn compliance, optional backend numerical
equivalence, other platforms, performance and statistical calibration are skipped.
Passing this phase does not establish release readiness.

## Resume

R18 was completed and committed before R31. The user explicitly chose to remove both
public legacy helpers, cat_var and validate_pandas_input. R31 must trace their
exports/docs and remove the commented benchmark call, record migration advice,
and test the deliberate removals and the supported encoder alternative.
Stop after R31; the following group is R19 alone, with R05/R06/R18 complete.
No publication, dependency change, version bump or release is authorized.
