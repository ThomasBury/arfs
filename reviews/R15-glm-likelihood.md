# R15 — Honest likelihood and refit contract

**R15 complete on 2026-10-13. Next group: R16 alone.**
Based on `2577289` on `main`, with a clean initial checkout. R15 has a separate local
implementation commit; its identity is recorded in [REMEDIATION.md](REMEDIATION.md).
No publication or version bump is part of this group.

## Changes and decisions

**Zero-weight policy: omit those rows.** Response, weight and input-name validation
remains in place; missing/nonfinite responses are rejected even on zero-weight
rows, and a wholly zero-weight input still fails the positive-mass boundary.
The shared omission helper removes aligned feature/response/weight rows
positionally. EnetGLM fitting and scoring, internal grid search, and weighted CV
with a direct EnetGLM argument use it. Omission precedes constant detection,
regularized objective scaling, likelihood observation counts and GLM fold
construction, so results match physically removing the rows first. Positive
weights remain unchanged; observation count is the retained row count rather
than weight mass. Prediction/transform still produce outputs for supplied rows.
Generic CV estimators retain their existing weight semantics and per-fold
positive-mass guards. This phase does not infer a zero-row preprocessing policy
for arbitrary pipelines wrapped around a GLM.

**Invalid candidate policy: fail immediately.** A NaN or positive/negative infinite
BIC or mean validation score raises a ValueError naming the criterion and alpha.
Neither a wholly invalid nor a partly invalid grid returns a model by silently
discarding candidates. Backend fit exceptions propagate; no fallback, retry,
criterion substitution or extra estimator mode is added. Convergence warnings
remain visible; finite criteria alone do not prove convergence or calibration.

Independent likelihood checks exposed a backend refit bookkeeping defect:
statsmodels writes the active-column count into `df_model`, then its information
criteria add one to both parameter count and inferred observation count. R15
computes **wrapper AIC/BIC** from the returned log-likelihood, actual retained
observation count and rank of the selected design, including a selected
intercept. Empty selection has rank zero; redundant columns use their rank,
rather than the number of displayed nonzero coefficients. As in the native
GLM default convention, these criteria count mean parameters without dispersion.
AIC is `-2*llf + 2*k`; BIC is `-2*llf + log(n)*k`. No backend result object is
patched. Raw backend summary/results keep their original bookkeeping and their
information criteria can differ from the corrected `aic_`/`bic_` attributes.
These fixes can change candidate rankings and selected features.

The estimator uses **regularized variable selection followed by an unpenalized
refit** of surviving columns. `alpha` is penalty strength; `L1_wt` is the L1
fraction used during selection. Returned coefficients/predictions do not retain
ridge shrinkage. The near-ridge workaround claim is removed; unsupported pure
ridge and invalid mixtures now raise a clear fit-time `L1_wt` error. The default
mixture and existing tiny positive fitting penalty for `alpha=0` remain unchanged.
Ordinary summary p-values use the selection data and do not account for variable
selection. Fixed-scale variance-weight likelihoods have a quasi-likelihood
interpretation; finite BIC does not establish calibrated model selection.
The Poisson rate/exposure equivalence is stated narrowly, rather than as an
arbitrary GLM/weight equivalence.

Caller tracing covered observation validation, GLM design/fit/score/statistical
attributes/summary, weighted CV and private fold scoring, internal penalty
search, selector fit/support/transform, R05/R13/R14/API tests, method prose,
exports and Lasso notebook cells. The older zero-mass-fold test is updated to
assert omission before GLM splitting; a new generic-CV test retains its old
boundary. API and method prose describe the decisions. Two notebook explanatory
sentences correct the intercept significance and refit claims; code, saved
outputs and all other cells remain byte-equivalent as JSON objects to `2577289`.
No dependency, option, export, compatibility layer or notebook execution is added.
Historical evidence is preserved.

Contracts were checked through Context7 against statsmodels'
[regularized fit/refit](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_regularized.html),
[variance-weight likelihood caveat](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html)
and [information-criterion parameter convention](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.info_criteria.html).
The installed elastic-net refit and GLM criterion source were traced to explain
the extra parameter/row. Tests use independent Gaussian/Poisson likelihood
formulas rather than accepting the backend's refit criterion as a reference.

## Validation

Evidence is in [r15-glm-likelihood](evidence/r15-glm-likelihood/). The 52 maintained
cases produce **41 failures and 11 passes** on the unchanged installed R14
wheel, whose GLM hash is verified against `2577289` in
[before-provenance.json](evidence/r15-glm-likelihood/before-provenance.json).
Original likelihood/near-ridge diagnostics reproduce **two failures and two
passing controls** unmasked. All maintained cases pass after repair; source
plus affected R05/R13/R14/API callers is **339 passed**. The initial independent
BIC/reference and old-fold-policy failures are retained in `source.txt`.

Checks cover NaN and both infinities under BIC/deviance, wholly and partly invalid
grids, propagation of a real degenerate Gaussian fit error; invalid/pure-ridge
mixtures; tiny-L1 unpenalized weighted-least-squares behavior; independent
Gaussian/Poisson likelihood and AIC/BIC formulas across penalties, including
sparse/null selections; rank-deficient design counting; zero-row omission
equivalence for Gaussian/Poisson/binomial/Gamma, arrays/dataframes and custom
indexes; selector BIC/deviance tuning and direct GLM CV omission before folds,
with one/two workers; active-row constant detection; all-zero rejection,
invalid-response preservation and generic-CV fold-mass rejection.
Fixed-score tests isolate search failure handling; real fits verify weighting,
likelihood, refit and active-design behavior.

Fresh installed-wheel full suites, outside the checkout with isolated Python
and without setuptools: **py39-locked 688 passed; py312-locked 688 passed;
py312-newer 672 passed**. Twelve selected historical checks and controls pass
unmasked in each 3.12 cell. The old near-ridge assertion remains historical;
the repaired documented contract is independently verified by maintained tests.
All dependency/provenance checks and command exit codes pass. Eight installed
module hashes match current source; copied tests match source bytes. All
**19** declared source/sdist/wheel payloads agree.

Verified Python versions are **3.9.25** and **3.12.7**. Both locked cells use
NumPy 2.0.2, pandas 2.2.3, sklearn 1.6.1, statsmodels 0.14.4 and pytest 8.3.5.
The newer 3.12 cell uses NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1,
statsmodels 0.15.0 and pytest 9.1.1. Per-cell JSON records complete versions,
module paths/hashes, exact commands, working directories and exit codes;
[commands.txt](evidence/r15-glm-likelihood/commands.txt) records shared commands.
`uv lock --check`, Ruff 0.17.0's baseline gate (**417 total, zero new**),
gate unit tests (**two passed**), new/affected-test lint, new-test format,
notebook preservation, local handoff links and `git diff --check` pass.
Changed-file Ruff diagnostics were inspected; no allowance was added.
Existing convergence, estimator-tag, model, precision and plotting warnings
remain visible.

Hosted CI, full docs/notebooks, full sklearn estimator compliance, other
platforms, optional backends, performance and statistical calibration are not
checked. Backend summary/result bookkeeping is not repaired; use the wrapper's
corrected criterion attributes. A finite likelihood/criterion and ordinary
post-selection summary do not establish valid selection-adjusted inference.
R15 checks do not establish release readiness.

## Resume

Stop after this completed R15 group. Next is **R16 alone**, with R06/R08 complete.
Read [03 F5–F9/F12](03-selector-contracts.md), then inspect `BaseThresholdSelector`
in `base.py`, threshold selectors in `unsupervised.py`, their callback labels,
parameter validation, empty/one-feature collinearity, tie order and fitted
metadata. Settle equality wording explicitly, preserve legitimate named
projection if retained, and test duplicate/reordered labels, changed parameters,
empty pipeline stages and NotFittedError. R16–R33 remain pending.
