# R22 — Persist the GBM wrapper contract

**R22 complete on 2026-10-14**, based on clean `main` at `4d0e240`.
Local phase commit: `4bd3681`.
R22 runs alone, with R21 complete.

## Changes and decisions

GradientBoosting.save now writes a versioned joblib dictionary containing the
format identifier, format version 1, native Booster, is_init_score, cat_feat and
feature_names_in_. The fitted input names preserve their original string form,
including spaces that LightGBM normalizes internally. Category names retain
original labels, including integer names and valid repeated category selections.
Native category vocabularies, model parameters and best iteration remain in the
Booster. Per-row training offsets, validation features and figures are excluded
from the wrapper metadata; the optional curve remains a separate PNG.

load requires the versioned wrapper artifact. **Legacy Booster-only files are
rejected with TypeError** because their missing offset contract cannot be inferred
safely. Refit with the intended configuration and save again; there is no migration
flag or compatibility shim. Direct joblib.load on a new saved file now returns a
dictionary rather than a Booster; use GradientBoosting.load for its prediction
contract. File extension, default objective/date names, named overwrite behavior
and relative paths when save_path is omitted are retained.

Before replacing fitted state, load validates the artifact identifier/version,
required fields, Booster/parameter types, Boolean offset guard, original schema
against native feature names and category names against that schema. Invalid,
missing and corrupt files preserve the previous model and all fitted metadata.
Deserialization errors propagate. Metadata validation happens after unpickling:
joblib can execute code, so only trusted inputs are allowed. This is not a safe
format for unknown files or a guarantee of cross-version compatibility.

Successful loads replace model, model_params, is_init_score, cat_feat and the
original fitted schema. Validation features and learning_curve become None;
training rows are not reconstructed. Clearing a curve reference does not close
a caller-owned figure. Constructor options stay with the receiving wrapper for
its next fit; the artifact restores fitted prediction behavior, not a training
configuration or a resumable dataset.

Offset models keep their prediction guard in a fresh process and when loaded
into a previously fitted wrapper. New rows require their own external baseline:
use predict_raw(raw_score=True), add that baseline on the raw scale, then apply
the objective's inverse link/output transformation. The model cannot infer a
new-row baseline from training offsets. Ordinary regression/classification,
multiclass aliases and native categorical missing/unknown handling round-trip.

Save validates a non-None curve's Figure type before touching outputs. It stages
joblib serialization and PNG rendering in a private temporary directory on the
destination filesystem, flushes and fsyncs the model file, then replaces the
model and optional PNG. Serialization, file-sync, rendering and model-replacement
failures preserve existing target bytes and clean staging files. The model's
replacement is the commit point. If PNG replacement fails afterward, the new
model remains saved/loadable, the old PNG remains, the error propagates and the
figure stays open. This two-file limit is documented and marked with a shortcut
comment; no rollback or joint transaction is implied. Saving without a curve
leaves any existing PNG untouched. Process interruption, directory durability,
concurrent writers and power-loss recovery are outside this repair's claims.

Caller tracing covered save/load, both prediction methods, fitted/refit metadata,
TreeDiscretizer and VariableImportance, maintained input tests and every direct
source/test/documentation/notebook persistence reference. Existing review 04
scripts and reports are preserved. API autodoc uses the updated method docstrings;
the Unreleased changelog records the artifact format/migration and write boundary.
No new dependency, backend or general persistence framework is added.
Joblib's [persistence documentation](https://joblib.readthedocs.io/en/stable/persistence.html)
was checked through Context7 for file streams, trusted loading and compatibility.

## Validation

Exact commands and boundaries are in
[evidence/r22-gbm-persistence/commands.txt](evidence/r22-gbm-persistence/commands.txt).
Fresh installed-wheel full suites pass **1068 / 1068 / 1052** in
py39-locked / py312-locked / py312-newer. All **11** selected unchanged historical
persistence/unfitted checks pass unmasked in both Python 3.12 cells, including
fresh-process ordinary/category models, offset guards and loading into an already
fitted wrapper. The three original R22 counterexamples reproduce **3 failures**
on the verified R21 wheel before repair; original checks.py/reload.py are unchanged.

The maintained suite adds **37** cases. The same finalized tests on the R21 wheel
report **33 failures and 4 passing controls**; all 12 installed source-module hashes
match base 4d0e240, without setuptools. Affected source/callers pass **194** tests,
with eight fresh-process cases deferred to installed runs. Those eight cases cover
ordinary and offset regression, binary, softmax and categorical models on new
prediction rows, including unknown/missing categories and integer/original names.
Malformed/legacy/missing/corrupt artifacts, both loaded offset states, refits and
six injected write-failure paths have explicit maintained checks.

Locked cells use Python 3.9.25 / 3.12.7, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, LightGBM 4.6.0, joblib 1.4.2 and pytest 8.3.5. Newer uses
Python 3.12.7, NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, LightGBM 4.7.0,
joblib 1.6.0 and pytest 9.1.1. Per-cell JSON records full versions, exact
commands/cwd/exit codes, installed paths and hashes. All copied tests and 12
installed source-module hashes match final source; all 19 source/sdist/wheel
payloads agree. Dependency checks and uv lock --check pass. Ruff 0.17.0's baseline
gate passes (**395 total, zero new**, unchanged allowances). Gate unit tests
(**2 passed**), changed-test/evidence lint and format, local links and whitespace
checks pass. Existing warnings remain visible.

Hosted CI, docs/notebook execution, optional numerical backends, other platforms,
cross-version loading, hostile artifacts, concurrent writers, interruption/power-
loss recovery, performance and statistical calibration were not checked.
Passing R22 does not establish release readiness.

## Resume

Stop after R22. Next group: **R23 alone**, with R07/R10–R12/R18/R21 complete.
Read [04 F11/F12/F14/F15](04-training-and-benchmarks.md), sklearn_pimp_bench,
compare_varimp and their encoder/selector/model callers. Fit preprocessing on
training rows, settle array/name support, reject unsupported GrootCV comparisons
and construct fresh configurations for each actual comparison. R23–R30 and
R32–R33 remain pending; R31 is complete. No publication, version bump or release.
