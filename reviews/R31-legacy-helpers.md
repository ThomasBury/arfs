# R31 — Remove legacy helpers deliberately

**R31 complete on 2026-10-14**, on `main` after the complete R18 commit
`7df5daa`. Its separate local implementation commit is recorded in REMEDIATION.md. The user explicitly chose **remove both public legacy
helpers**. This is an intentional breaking API decision, with no aliases,
deprecation wrappers or compatibility shims.

## Changes and migration

Remove `arfs.preprocessing.cat_var` and `arfs.utils.validate_pandas_input`.
Neither has an active internal source, maintained-test or notebook caller. Remove
the sole dead benchmark comment and the tqdm import used only by cat_var.
Public modules have no __all__ entry or root re-export for either helper, and
Sphinx documents these modules through autodoc rather than explicit helper lists.
The Unreleased changelog records both removals and migration advice. Historical
review reports and diagnostic scripts remain unchanged.

Use OrdinalEncoderPandas for reusable categorical encoding and inverse decoding:

```python
from arfs.preprocessing import OrdinalEncoderPandas

encoder = OrdinalEncoderPandas(
    exclude_cols=["id"],
    output_dtype="int64",
    unknown_value=-1,
    encoded_missing_value=-2,
)
encoded = encoder.fit_transform(X)
decoded = encoder.inverse_transform(encoded)
```

The encoder returns a DataFrame, not cat_var's four-item tuple. Use
categorical_features_ and categories_ for the fitted categorical names and
vocabularies, and inverse_transform for decoding. exclude_cols replaces col_excl;
return_pandas_categorical replaces return_cat. Default encoding covers object,
category, nullable/inferred strings and booleans, with other columns passed
through. It does not preserve the old all-nonnumeric/date-stringification policy
or guarantee identical legacy codes. Keep the fitted encoder for later transforms
and supply the same full column schema in order. Missing values now remain
separate from observed categories. Default NaN unknown/missing codes cannot be
distinguished on inverse; the example chooses distinct integer sentinels.

For array extraction, use DataFrame.to_numpy() / Series.to_numpy() and keep an
existing NumPy array directly. validate_pandas_input's permissive .values access
and documented array mismatch are removed; there is no replacement validator.
Callers must retain their own input-type/shape validation where needed.

Caller tracing covers all repository source, maintained tests, notebook code and
text, module exports, README/changelog and generated-docs sources. No active
caller is rewritten, and no unrelated unused helper is deleted.

## Validation

Exact commands and per-cell manifests are retained in
[evidence/r31-legacy-helpers/commands.txt](evidence/r31-legacy-helpers/commands.txt).
Fresh installed-wheel full suites pass: **py39-locked 862; py312-locked 862;
py312-newer 846**.
The six maintained cases fail on the verified R18 wheel with **2 failed, 4 passed**:
the two deliberate-removal assertions fail; the four supported encoder migration
references already pass. Before module hashes match `7df5daa`.
Original historical F15/F10 diagnostics reproduce unmasked on R18: cat_var has
1 failure / 1 passing roundtrip control; the declared NumPy validator case fails.
The source suite with ordinal, category identity and dependency callers passes
103 tests. New removal checks assert absent public attributes/exports; migration
checks cover object/category/nullable/newer string inputs, distinct missing codes,
integer encoding, metadata, inverse values and exact passthrough/input ownership.

Historical preprocessing checks import the intentionally removed helper at module
load, so they are not recollected against the R31 wheel. Their original scripts
remain historical evidence; no fake helper or shim is injected. The maintained
ordinal and removal suites cover the supported API and run unmasked. R18 already
records all seven original ordinal checks on its wheel.

Locked cells use Python 3.9.25 / 3.12.7, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, statsmodels 0.14.4 and pytest 8.3.5. Newer uses Python 3.12.7,
NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, statsmodels 0.15.0 and pytest 9.1.1.
Complete versions, isolated module paths/hashes and commands are in per-cell JSON.
Copied tests match source bytes; all 12 installed modules and 19 source/sdist/wheel
payloads match. Dependency checks, lock consistency, Ruff 0.17.0 (397 total,
zero new), gate unit tests (2 passed), test/evidence lint/format, handoff links and
whitespace pass. Six repaired Ruff allowances are removed, none added. Existing
model, convergence, tag, precision and plotting warnings remain visible.
Hosted CI, full docs/notebooks, full sklearn compliance, optional backend numerical
equivalence, other platforms, performance and statistical calibration are skipped.
Passing the batch does not establish release readiness.

## Resume

Stop after this R18 -> R31 batch. Next group: **R19 alone**, with R05/R06/R18
complete. Read [02 F1/F2/F4/F5/F9](02-preprocessing.md) and TreeDiscretizer fit/transform, fitted tree
thresholds, missing values and copied fit-time parameters. R19-R30 and R32-R33
remain pending. No publication, version bump or release is authorized.
