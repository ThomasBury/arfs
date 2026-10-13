# R11 — Weighted signed permutation importance

**R11 complete on 2026-10-12. Next group: R12 alone.**
Based on R10 local commit `acf99fc` on `main`. Pre-existing uncommitted R04–R09
work is preserved. This phase has a separate local repair commit; its identity
is recorded in [REMEDIATION.md](REMEDIATION.md). No push, release or version bump
was performed. Both repair commits rely on earlier work still in the checkout.

## Changes and decisions

The shared `_get_perm_imp` passes the aligned validation weights returned by
`_split_fit_estimator` to sklearn's permutation scorer. Weighted training and
validation now use the supplied population; the split, estimator's default
scoring, five repeats and random seed stay as before. Unweighted calls pass None.
Both Leshy and BoostAGroota use this shared path. API support was verified through
Context7 against the [official sklearn API](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html),
which documents sample weights for scoring since version 0.24; installed 1.6.1
and 1.9.1 execute the real references.

The user explicitly chose **absolute-total normalization** for permutation
scores. Each BoostAGroota iteration divides the signed vector by its sum of
absolute values, preserving signs and order. A cancelling signed total remains
well defined; an all-zero vector stays zero. For example [-1, 1, 0, 0] becomes
[-.5, .5, 0, 0]. Only features with a strictly positive mean normalized permutation
score can pass the existing shadow comparison. Negative and zero means provide
no positive permutation evidence, even when the shadow threshold is negative.
All-zero permutation runs produce a fitted empty selection and zero-column
transform. Native/SHAP normalization is unchanged; R12 owns the separate
fold/shadow aggregation choice. Leshy retains raw signed scores and its existing
shadow/binomial decision rule; this phase does not change that rule or claim
calibrated false-discovery control.

Caller tracing covered shared split/fit/weights, permutation scoring, Leshy
augmented original/shadow calls and hit/history consumers, BoostAGroota iteration
normalization/threshold/reduction/public support and transform, existing tests,
and selector API docstrings. Only `allrelevant.py`, the maintained regression file
and remediation handoffs/evidence change. No dependency, option, export or
compatibility layer was added.

## Validation

Evidence is in [r11-permutation-importance](evidence/r11-permutation-importance/).
Both named historical F3/F4 checks fail unmasked against the installed R10 wheel
(2 failed). The 16 maintained regressions fail 12 times on that wheel, with four
passing controls. All 16 pass against the repaired source without xfail markers.
They cover real weighted/unweighted sklearn regression and classification,
public Leshy/BoostAGroota validation weight/target alignment and input ownership,
negative-total, cancelling, all-zero, mixed, positive and zero-real fixed vectors,
public empty/signed masks/transforms and separate per-iteration scales.
Fixed-vector tests isolate arithmetic; real LightGBM/sklearn tests verify scoring.

Fresh installed R11 wheel full maintained suites: **py39-locked 496 passed;
py312-locked 496 passed; py312-newer 480 passed**. Historical F4 plus FDR,
binomial and constant-input controls pass unmasked in both 3.12 cells
(**5 passed** each). The historical F3 script intentionally asserts equality
with the old unweighted output before asserting the weighted reference. It is
preserved unchanged; the promoted maintained test drops that diagnostic
old-behavior assertion and compares directly with sklearn's weighted result.

The three dependency snapshots match R10: Python 3.9.25 and 3.12.7; locked
pandas 2.2.3 / NumPy 2.0.2 / sklearn 1.6.1; newer pandas 3.0.6 / NumPy 2.5.4 /
sklearn 1.9.1. Complete versions and commands are recorded per cell. Tests run
outside the checkout with isolated Python on freshly installed wheels without
setuptools. All eight recorded installed module hashes match source, all 19
source/sdist/wheel payloads agree, copied tests match source, and twelve unrelated
tracked diff sections remain unchanged. Historical scripts and logs are preserved.

`uv lock --check`, Ruff 0.17.0 baseline gate (zero new diagnostics), gate unit
tests (2 passed), new-test lint/format and `git diff --check` pass. Commands are
in [commands.txt](evidence/r11-permutation-importance/commands.txt) and per-cell
command JSON. Existing model/tag/precision warnings remain visible; new fixed
zero-score tests also run with divide/invalid floating errors raised.

Hosted CI, full docs/notebooks, optional backends, performance, visual plot
review and statistical calibration were not checked. These two green phases
are not a release-readiness claim.

## Resume

The authorized R10 → R11 batch is complete; stop here. Next is **R12 alone**,
with R07 complete. Read [1B F7–F9](01b-all-relevant.md), `_set_lgb_parameters`,
`_compute_importance`, `_reduce_vars_lgb_cv` and `_reduce_vars_sklearn`.
Resolve GrootCV's resolved objective/output axis, then explicitly choose the
fold/shadow statistic for GrootCV and BoostAGroota and align API/method prose.
R12–R33 remain pending. Calibration remains separate from aggregation correctness.
