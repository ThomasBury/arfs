# R17 — Truthful cumulative importance selection

**R17 complete on 2026-10-13. Next group: R18 → R31.**
Based on clean `main` at `1e88c54`. R17 runs alone; the local implementation
commit and completed validation are recorded in [REMEDIATION.md](REMEDIATION.md).
No publication or version bump is part of this group.

## Changes and decisions

The user chose **one fit, removing `n_iterations`**, and **equal averaging over
validation rows**. VariableImportance and its private computation function no
longer accept `n_iterations`; remove that keyword from callers. There is no
ignored parameter, compatibility shim or new resampling mode. The existing
wrapper retains its fixed validation split and early stopping. `verbose` now
uses the wrapper's native training-metric logging frequency; nonpositive values
disable those logs. One fit makes no resampling-variability claim. Mutable
default model dictionaries are replaced by None, resolving to the same default
rmse/zero-as-missing parameters during use; caller dictionaries remain intact.

**Selection keeps the crossing feature.** Predictors are ranked by descending
positive importance, with input-order ties. Selection retains the smallest
prefix reaching the requested share, including the feature crossing the cutoff.
Zero-importance predictors are excluded independently of cumulative rounding.
Threshold zero selects nothing; threshold one retains every positive feature,
including tiny positive values after the cumulative sum has rounded to one.
An all-zero total returns finite zero normalized/cumulative shares and an empty
selection. Current cutoffs must be finite real numbers in [0, 1], with booleans
rejected; validation runs at fit, including after `set_params`.

**Contribution axes follow the model output.** Native LightGBM output is checked
as row-by-output blocks, each containing the feature contributions and one bias
column. Bias columns are excluded and the blocks are converted to the common
row/feature/output layout. No target imputation, `fillna` call or class-frequency
guess is used. Array and Series multiclass targets therefore follow the same
path. Optional TreeExplainer lists and arrays use their explicit output axis.
Both paths reuse the existing `_compute_importance` helper without changing it:
mean absolute contributions over validation rows, summed across returned
outputs. In particular, optional multi-output list scores now sum outputs
instead of averaging them; raw score scale changes, while that constant output
factor cancels on normalization. Invalid native layouts and nonfinite
contributions/totals raise ValueError. A missing fasttreeshap import raises an
actionable ImportError before encoding or training, without a silent fallback.

**Training weights and explanation weights are distinct.** Observation weights
still enter training and validation loss. Global importance describes the
observed validation rows with equal row mass, rather than a weight-adjusted
population. The maintained weighted pipeline reference reproduces the native
weighted LightGBM fit and proves that the equal-row and weighted-row explanation
means differ on its fixture. No new exposure/frequency interpretation is assigned.

API docstrings, the Unreleased changelog and one explanatory notebook paragraph
record these decisions. No live notebook code passes `n_iterations`; one maintained
R16 test is updated, and its controlled summary gains the now-required importance
column. The basic notebook's code, saved outputs, metadata and all other cells
are preserved. No dependency, export, backend or wrapper split change is added.
Plotting repairs remain R29; this phase does not claim cutoff-one/empty-selection
plot correctness or a full notebook run.

Caller tracing covered public fit/support/transform/plot, private native/optional
contribution computation, the reused all-relevant aggregation helper, training
wrapper split/weight handling, encoder use, public exports, pipeline summaries,
maintained tests and notebook source callers. LightGBM's
[prediction documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
was checked through Context7 for native contribution shapes and bias columns.
No source change is made to the shared aggregation helper or training wrapper.

## Validation

The maintained R17 suite contains **49 cases**. It reports **36 failed, 13 passed**
on the verified installed R16 wheel; the earlier 47-case probe is preserved
(**34 failed, 13 passed**). Original historical checks reproduce **11 failures
and 13 passing controls**, unmasked. Before-provenance verifies the installed
selector source against `1e88c54`.

The final source suite with R16 selector, observation/weight, all-relevant
aggregation and dependency-API callers passes **354 tests**. The initial source
run has **352 passes**; the final run adds cutoff-one rounding and stable-tie/refit
references. Checks cover `[.6,.3,.1]` and `[6,3,1]` hand cutoffs including equality,
zero shares/totals/features, tiny positive importance, finite totals and invalid
contributions, threshold updates before training, signed absolute means, native
bias exclusion, native/list/array multi-output axes, invalid native shapes,
one fit/retired parameter rejection, missing backend before encoding/training,
real single-signal and constant-response fits, binary/multiclass array/Series
parity, and a real weighted categorical pipeline against native LightGBM.

Installed-wheel commands, versions and results are recorded under
[the evidence directory](evidence/r17-importance-selection/commands.txt).
Full suites pass: **py39-locked 821 passed; py312-locked 821 passed;
py312-newer 805 passed**. All **24** adapted historical checks/controls pass
unmasked in each Python 3.12 snapshot.
Fresh environments install outside the checkout with isolated Python and no
setuptools. Copied tests match source bytes; eleven installed module hashes
match source. All 19 declared source/sdist/wheel payloads agree.
Exact validation/environment/payload orchestrators are retained with the logs.

The historical script still passes the retired parameter. A disposable copy
removes exactly two `n_iterations=1` arguments, and an explicit diff and both
hashes record that adaptation. Historical scripts/logs remain unchanged.
Assertions and remaining code are preserved; obsolete repeated-fit and
iteration-average expectations are excluded. Maintained tests assert the new
one-fit/removed-parameter contract instead. The adapted selected assertions run
unmasked on both installed Python 3.12 snapshots.

Locked cells use Python **3.9.25** / **3.12.7**, NumPy 2.0.2, pandas 2.2.3,
sklearn 1.6.1, statsmodels 0.14.4 and pytest 8.3.5. The newer Python 3.12.7 cell
uses NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1, statsmodels 0.15.0 and pytest 9.1.1.
Per-cell JSON lists complete versions, module paths/hashes, exact commands and
exit codes. Ruff 0.17.0 reports **407 total, zero new** diagnostics; seven repaired
allowances are removed and none added. New/affected test and evidence lint/format,
gate unit tests (**two passed**), lock consistency, notebook preservation, links
and whitespace pass. Exact commands are recorded separately. Existing model,
convergence, tag, precision and plotting
warnings remain visible.

Hosted CI, full docs/notebooks, full sklearn compliance, other platforms,
installed optional-backend numerical equivalence, performance and statistical
calibration are not checked. Optional list/array fixtures test aggregation and
the missing-import boundary; they do not certify fasttreeshap execution.
Passing R17 checks does not establish release readiness.

## Resume

Stop after R17. The next group is **R18 → R31**, with R06 complete. Start with
[02 F8/F13/F14](02-preprocessing.md) and `OrdinalEncoderPandas`, its passthrough
schema, inverse ownership and changed output dtype. Complete and commit R18
before R31. Before R31, explicitly decide whether to retain or remove the public
`cat_var` and `validate_pandas_input` helpers; trace their callers/exports/docs
against [02 F15](02-preprocessing.md) and [06 F10](06-utilities-and-workflows.md).
Stop at the phase boundary if that decision or wider scope remains unresolved.
R18–R33 remain pending.
