# R12 — Explicit all-relevant aggregation

**R12 complete on 2026-10-13. Next group: R13 alone.**
Based on R11 commit `f68e482` on `main`. Earlier uncommitted R04–R09 work
is preserved. R12 has a separate local source/test/docs/evidence commit;
its identity is recorded in [REMEDIATION.md](REMEDIATION.md). Shared queue and
remediation edits remain uncommitted with earlier updates. No push, release
or version bump was performed. The commit depends on earlier checkout work.

## Changes and decisions

The user explicitly chose these aggregation rules, retaining equality at the
threshold and R11's strictly positive mean requirement for permutation scores:

- **GrootCV:** normalize each fold's real/shadow importance vector, take each
  feature's median across all folds and repeats, then take the maximum shadow
  median divided by `cutoff`. Compare real medians with that threshold.
- **BoostAGroota:** normalize each iteration's real/shadow importance vector,
  take each feature's mean across iterations, then take the maximum shadow
  mean divided by `cutoff`. Compare real means with that threshold in each
  reduction round. R11's signed absolute-total permutation normalization stays
  in effect; SHAP/native normalization is unchanged.

Both calculations use only `fscore` columns, excluding feature labels and
summary columns. GrootCV's `Med` now contains the median; absolute ranking and
plot order use the same statistic. Public table/attribute names remain the same.
The API docstrings and [method overview](../docs/Methods%20overview.rst) describe
the exact axes, order of operations and equality policy. These repairs can
change selected features, as the historical hand examples demonstrate.

GrootCV passes its resolved training objective into importance aggregation.
The SHAP reducer follows the returned layout: `(rows, features)` or
`(rows, features, outputs)`, or a list of per-output matrices. It takes absolute
values before summing outputs, then averages training rows, yielding one scalar
per feature. It checks the row/feature axes and keeps every feature column;
TreeExplainer's expected values are separate. Backend/version guesses and the
incorrect bias-column deletions are removed from this reducer. Synthetic list
cases exercise its reduction without claiming fasttreeshap runtime support.
The [official TreeExplainer API](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
documents these layouts; Context7 retrieval was inconclusive for this detail.

Objective resolution copies `lgbm_params`, honors its objective when the explicit
argument is None, and retains explicit-argument precedence. Recognized
multiclass classification objectives resolve to softmax with `num_class`, even
for two-class targets; binary is promoted for more than two target classes.
The existing softmax treatment of one-vs-all spellings for multiclass targets
remains explicit in the API/method prose. Random-forest feature fraction now
uses the resolved objective. The supported spellings and their native meanings
were checked through Context7 against [LightGBM's parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
This phase does not add a separate one-vs-all selector algorithm or redefine
classification labels.

Caller tracing covered GrootCV parameter resolution, real fold training and
SHAP extraction, fold normalization/merging, selection/support/transform,
absolute ranking and plot order; BoostAGroota iteration normalization,
shadow thresholds, reduction rounds and public outputs; shared SHAP comments,
existing R07/R11 tests, selector exports and method prose. No dependency, option,
export or compatibility layer was added. Historical evidence is unchanged.

## Validation

Evidence is in [r12-all-relevant-aggregation](evidence/r12-all-relevant-aggregation/).
The installed R11 wheel fails all four named historical counterexamples, with
two passing controls. The final 37 maintained cases fail **35** times on that
wheel, with two passing single-output controls. All pass on repaired source;
the source run including R07/R11 caller regressions is **95 passed**.

The maintained cases cover real binary and three-class fits for binary,
multiclass, softmax and one-vs-all spellings, objective fallback/override and
parameter/input ownership, single/multiple/list output axes, malformed shapes,
fold/repeat medians, cutoff scaling, maximum shadow means for SHAP/native/
permutation scores, one-iteration strongest-shadow behavior, equality,
selected output schemas and median ranking/plot order. Fixed scores isolate
arithmetic; real LightGBM/SHAP fits exercise objective/output handling.

Fresh installed R12 wheel full maintained suites, outside the checkout with
isolated Python and without setuptools: **py39-locked 533 passed;
py312-locked 533 passed; py312-newer 517 passed**. Historical R12 cases plus
SHAP/FDR/binomial/constant-input controls pass unmasked in both 3.12 cells
(**10 passed** each). All per-cell command exit codes, dependency/provenance
checks and copied-test byte comparisons pass. All eight installed module hashes
match source; all 19 declared source/sdist/wheel payloads agree. Twelve unrelated
tracked diff sections remain byte-for-byte unchanged.

Verified runtimes are Python 3.9.25 and 3.12.7. Locked dependencies include
NumPy 2.0.2, pandas 2.2.3, sklearn 1.6.1, SHAP 0.47.1 and LightGBM 4.6.0.
The newer 3.12 snapshot uses NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1,
SHAP 0.53.0 and LightGBM 4.7.0. Complete versions and commands are in the
per-cell evidence JSON and [commands.txt](evidence/r12-all-relevant-aggregation/commands.txt).
`uv lock --check`, Ruff 0.17.0's baseline gate (417 total, zero new diagnostics),
gate unit tests (2 passed), new-test lint/format and `git diff --check` pass.
Existing model/tag/precision warnings remain visible.

Hosted CI, full docs/notebooks, optional backend execution, performance,
visual image inspection and statistical calibration were not checked.
Programmatic plot-label/order checks pass. SHAP explanations still use each
fold's training rows; validation drives early stopping. Correct aggregation
does not establish calibrated false-discovery control or release readiness.

## Resume

The authorized R12 group is complete; stop here. Next is **R13 alone**, with
R03/R05 complete. Read [1C F2–F4/F9/F12](01c-weighted-glm-lasso.md), GLM
response-scale deviance/scoring, `weighted_cross_val_score` and `grid_search_cv`
in `lasso.py`, their callers and historical Poisson signal counterexample.
R13 must expose a higher-is-better sklearn score while preserving BIC
minimization, pass stratification targets, slice documented arrays and clone
each fold. R14's design/intercept state and R15's likelihood/refit policy
remain separate. R13–R33 are pending; no R12 decision remains unresolved.
