# R14 — Stable GLM design and parameters

**R14 complete on 2026-10-13. Next group: R15 alone.**
Based on `fb942fc` on `main`, with a clean initial checkout. R14 has a separate local
implementation commit; its identity is recorded in [REMEDIATION.md](REMEDIATION.md).
No publication or version bump is part of this group.

## Changes and decisions

Fit owns the design used by `EnetGLM.predict` and
`LassoFeatureSelection.transform`. Statsmodels detects a nonzero training
constant once. Existing constants retain their original names and values;
no duplicate intercept is added. **`fit_intercept=False` disables only automatic
addition** and preserves every supplied column, including columns named `const`,
`Intercept` or `intercept`. A zero-valued constant remains an input and does not
replace a nonzero intercept.

An automatically added column is called `Intercept`, or `Intercept_1`,
`Intercept_2`, etc. if necessary to avoid input-name collisions. Inference
recreates that fitted column with ones without inspecting batch values, so
single-row prediction/transformation uses the training design. This also works
when ordinary varying predictors have names such as `const` or `Intercept`.

Named inputs align to fitted input names and order. Missing, extra, replacement
and duplicate columns are rejected; all training column names must be unique.
Arrays require two dimensions, fitted input width and positional training order.
They receive `pred_0`, `pred_1`, etc. at fit. Array inference after named fitting
uses fitted positions. Dataframe indexes and original integer, mixed and tuple
labels are preserved. Callers' inputs are not modified.

**Input metadata describes original inputs:** `feature_names_in_` and
`n_features_in_` exclude a generated intercept; selector `support_` has one
boolean per original input. `design_feature_names_` records actual coefficient
columns, and selector `design_support_` records their nonzero coefficients.
`selected_features_`, transformed columns and `get_feature_names_out` continue
to include a generated intercept if selected. It remains penalized as before.
A generated intercept is therefore an output feature, without being an input
feature in pipeline summaries. Existing code that treated input metadata as the
coefficient design must use the design attributes instead. `get_feature_names_out`
accepts and validates the optional original input names for sklearn pipeline
feature-name propagation and pandas output.

`EnetGLM.fit` now resolves family/link from current parameters, stores the fitted
objective and returns `self`. Updating parameters takes effect on refit; it does
not alter an already fitted prediction design/objective. The existing tiny
positive fitting penalty for `alpha=0` is local and no longer overwrites the
constructor parameter. Successful refits replace schema/intercept metadata.
Pre-fit prediction, transformation and feature-name queries raise
`NotFittedError`. Target/weight alignment remains positional, including direct
GLM fitting; backend input and fit errors remain visible.

Caller tracing covered family/link mapping, GLM fit/predict/score/coefficient
metadata, weighted CV and grid search, selector fit/transform/output names,
exports, `make_fs_summary`, R05/R13/API tests, method prose and Lasso notebook
cells. The shared input normalization and fitted-design method serve all affected
callers. No dependency, option, export, summary implementation or notebook change
is needed. The now-unused module-level constant-dropping helper is retained;
removing public helpers is a separate API decision. Historical evidence is intact.

The contracts were checked through Context7 against statsmodels'
[constant handling](https://www.statsmodels.org/stable/generated/statsmodels.tools.tools.add_constant.html)
and sklearn's [estimator parameter and fitted-input conventions](https://scikit-learn.org/stable/developers/develop.html).
The class retains its established synthetic array names and original pandas
labels; this phase does not claim full sklearn estimator compliance.

## Validation

Evidence is in [r14-glm-design](evidence/r14-glm-design/). The final 70 maintained
cases produce **67 failures and three passes** on the unchanged installed R13
wheel. Its GLM source hash is verified against `fb942fc` in
[before-provenance.json](evidence/r14-glm-design/before-provenance.json).
The separate original R14 diagnostics reproduce **nine failures and eight
passing controls** unmasked. All 70 maintained cases pass after repair;
repaired source plus affected R05/R13/API callers is **287 passed**.

Checks cover independent weighted least squares; single-row/batch equality;
arrays/dataframes and positional array inference; supplied constants under both
intercept modes; generated-name collisions and zero constants; input ownership;
reordered/missing/extra/replacement/duplicate and wrong-width/dimension inputs;
input/coefficient/support/output metadata agreement; original integer/mixed/tuple
labels; `set_params` versus fresh construction; successful schema/parameter
refits; alpha parameter preservation; pre-fit errors; and a real scaled pandas
pipeline with feature-name propagation and rendered `make_fs_summary` HTML.

Fresh installed-wheel full suites, outside the checkout with isolated Python
and without setuptools: **py39-locked 636 passed; py312-locked 636 passed;
py312-newer 620 passed**. Seventeen original R14 checks and controls pass
unmasked in each 3.12 cell. All dependency/provenance checks and command exit
codes pass. Eight installed module hashes match current source; copied tests
match source bytes. All **19** declared source/sdist/wheel payloads agree.

Verified Python versions are **3.9.25** and **3.12.7**. Both locked cells use
NumPy 2.0.2, pandas 2.2.3, sklearn 1.6.1, statsmodels 0.14.4 and pytest 8.3.5.
The newer 3.12 cell uses NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1,
statsmodels 0.15.0 and pytest 9.1.1. Per-cell JSON records complete versions,
module paths/hashes, exact commands, working directories and exit codes;
[commands.txt](evidence/r14-glm-design/commands.txt) records shared commands.
`uv lock --check`, Ruff 0.17.0's baseline gate (**417 total, zero new**),
gate unit tests (**two passed**), new-test lint/format, local handoff links and
`git diff --check` pass. Changed-file Ruff diagnostics were inspected; no
allowance was added. Existing backend convergence, estimator-tag, model,
precision and plotting warnings remain visible.

Hosted CI, full docs/notebooks, full sklearn estimator compliance, other
platforms, optional backends, performance and statistical calibration are not
checked. R15's zero-weight likelihood, all-nonfinite criterion and refit/near-ridge
claims remain unresolved. R14 checks do not establish release readiness.

## Resume

Stop after this completed R14 group. Next is **R15 alone**, with R13/R14 complete.
Read [1C F6/F11](01c-weighted-glm-lasso.md), the likelihood/refit findings and
`EnetGLM.fit`, BIC/statistical attributes and `grid_search_cv` candidate handling.
Decide whether zero-weight likelihood rows are omitted or clearly rejected,
reject all-invalid candidate criteria, describe regularized variable selection
followed by unpenalized refitting, and remove the near-ridge promise. Preserve
R13's scoring and R14's fitted design. R15–R33 remain pending.
