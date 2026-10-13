# R13 — Correct GLM scoring and independent CV

**R13 complete on 2026-10-13. Next group: R14 alone.**
Based on `64b0ce7` on `main`, with a clean initial checkout. R13 has a separate
local source/test/docs/evidence commit; its identity is recorded in
[REMEDIATION.md](REMEDIATION.md). No publication or version bump was performed.

## Changes and decisions

`EnetGLM.score` now returns **negative total response-scale deviance**: prediction
already supplies the response mean, so no second inverse link is applied.
Higher scores mean better predictions, as required by sklearn searches. Direct
score callers must account for the changed sign. Variance weights and unit scale
are retained; losses are not normalized by row count or weight mass.

The internal `score="deviance"` search maximizes the mean negative fold score,
equivalently minimizing mean total validation deviance across five folds.
`score="bic"` still minimizes training BIC. The public docstrings and method
prose use the accepted `"deviance"` spelling instead of the unsupported
`"mean_cv"` spelling; no alias is added.

Weighted CV uses sklearn's native `check_cv` with the task supplied explicitly:
classifiers and binomial GLMs with binary/multiclass discrete targets use
stratified folds. Regression, including constant and two-valued numeric
responses, and fractional binomial responses use KFold. The target is passed
to `split`; folds retain the existing deterministic, unshuffled order. This
split rule is not validation that arbitrary multiclass values form a valid
binomial response. Each fold fits a fresh clone, leaving the caller's fitted or
unfitted state intact. Dataframes use positional `iloc`; NumPy arrays use integer
row positions. Target and weight alignment remains positional.

Caller tracing covered `EnetGLM.predict/score`, `_fit_and_score`, weighted CV,
`grid_search_cv`, `LassoFeatureSelection.fit`, selector exports, boundary/API
tests, the method overview, and the Lasso notebook. No export, dependency,
configuration or notebook change is needed for R13. R14's intercept/schema,
fit return and parameter-derived objective repair, and R15's zero-weight
likelihood/finite-criterion/refit policy remain separate. Historical evidence
and unrelated source are preserved.

The library contracts were checked through Context7 against sklearn's
[scoring convention](https://scikit-learn.org/stable/modules/model_evaluation.html),
[native CV selection](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html),
[cloning API](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
and statsmodels' [variance-weight deviance](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Family.deviance.html).
Real predictions are checked against backend response means and independent
loss formulas, rather than treating a library call as an independent reference.

## Validation

Evidence is in [r13-glm-scoring](evidence/r13-glm-scoring/). The final 33 maintained
cases produce **29 failures and four passes** on the unchanged installed R12
wheel. All pass after repair, without expected-failure markers. The source run
including R05 boundaries and real dependency/API callers is **217 passed**.

The maintained checks cover independent Gaussian, Poisson, binomial and Gamma
loss formulas, weighted/unweighted scores and weight rescaling; external real
`GridSearchCV`; dataframe/array and custom-index positional parity; independently
computed weighted least-squares folds; binary selector fits against stratified
backend references; task/target split behavior; fitted and unfitted stateful
estimators with one/two workers and optional weights; direct BIC minimization;
and the historical seeded Poisson signal counterexample.

That Poisson case now chooses **alpha=1**, retaining its signal coefficient,
with independent mean fold deviance **31.7024**. The former alpha=10 null model
has deviance **236.7310**. All seven candidate losses and returned coefficients
are recorded in [poisson-case.json](evidence/r13-glm-scoring/poisson-case.json).
This is a regression counterexample, not a broad selection/calibration claim.

Fresh installed-wheel full suites, outside the checkout with isolated Python
and without setuptools: **py39-locked 566 passed; py312-locked 566 passed;
py312-newer 550 passed**. Fourteen selected historical R13 checks and controls
pass unmasked in each 3.12 cell. Old positive-score assertions remain historical;
maintained tests replace them with the explicit negative-score contract.
All dependency/provenance checks and command exit codes pass. Eight installed
module hashes match current source; copied tests match source bytes. All
**19** declared source/sdist/wheel payloads agree.

Verified Python versions are **3.9.25** and **3.12.7**. Both locked cells use
NumPy 2.0.2, pandas 2.2.3, sklearn 1.6.1, statsmodels 0.14.4 and pytest 8.3.5.
The newer 3.12 cell uses NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1,
statsmodels 0.15.0 and pytest 9.1.1. Per-cell JSON records complete versions,
module paths/hashes, exact commands, working directories and exit codes;
[commands.txt](evidence/r13-glm-scoring/commands.txt) records shared commands.
`uv lock --check`, Ruff 0.17.0's baseline gate (**417 total, zero new**),
gate unit tests (**two passed**), new-test lint/format and `git diff --check`
pass. Changed-file Ruff diagnostics were inspected; no allowance was added.
Existing backend convergence, estimator-tag, model, precision and plotting
warnings remain visible.

Hosted CI, full docs/notebooks, full sklearn estimator compliance, other
platforms, optional backends, performance and statistical calibration were not
checked. Remaining R14/R15 defects still affect ordinary GLM workflows; green
R13 tests do not establish release readiness.

## Resume

Stop after this R13 group. Next is **R14 alone**, with R13 complete. Read
[1C F5/F7/F8/F10](01c-weighted-glm-lasso.md), then inspect `EnetGLM.fit/predict`,
`LassoFeatureSelection.fit/transform`, constant-column handling, fitted metadata,
`set_params` and `make_fs_summary`. Settle one intercept/input-schema contract,
construct family/link from current parameters at fit, return self and validate
single-row, reordered/missing/extra/duplicate-name and array inputs. R15's
likelihood/refit decision remains separate. R14–R33 are pending; no R13 policy
choice remains unresolved.
