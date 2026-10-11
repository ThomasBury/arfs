What this change does: adds the review and executable evidence for unit **1C, weighted GLM/Lasso feature selection**. The reviewed code fits weighted generalized linear models, selects a penalty by BIC or cross-validated deviance, and keeps predictors with nonzero coefficients; no library behavior was changed.

Reviewed commit: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11. Scope is `EnetGLM`, family/link mapping, weighted cross-validation, the penalty search, `LassoFeatureSelection`, intercept handling, exports, summary integration, existing tests, and notebook callers. The user explicitly supplied this unit's subject. The assumed use is CPU-based tabular modelling, including exposure-weighted Poisson rates, binary responses, arrays, dataframes, and sklearn pipelines.

**Result:** 12 findings. Current allowed dependencies prevent the selector from fitting. In the frozen environment, nonlinear scoring can select a much worse model, binary cross-validation fails, and intercept handling breaks ordinary prediction/transform inputs. Library code, existing tests, dependencies, lockfile, CI, and previous reports were preserved.

**Evidence and environment**

Both environments are unit 0's existing installed wheels, run from `/tmp/arfs-review-1c` outside the checkout. Their installed `lasso.py` files match the checkout byte for byte. No packages were installed or patched. Both retain unit 0's `setuptools==80.9.0` import workaround; these results do not supersede the clean-install failure.

| Environment | Python | NumPy | pandas | scikit-learn | statsmodels | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| `locked-env` | 3.12.7 | 2.0.2 | 2.2.3 | 1.6.1 | 0.14.4 | Statistical references, API diagnostics, selection screen |
| `wheel-env` | 3.12.7 | 2.5.4 | 3.0.6 | 1.9.1 | 0.15.0 | Reproduce the current-dependency entry-point failure |

Fresh versions, module paths, and source hashes are in [frozen-environment.json](evidence/01c-weighted-glm-lasso/frozen-environment.json) and [current-environment.json](evidence/01c-weighted-glm-lasso/current-environment.json). Their corresponding [frozen pins](evidence/01c-weighted-glm-lasso/frozen-environment.requirements.txt) and [current pins](evidence/01c-weighted-glm-lasso/current-environment.requirements.txt) exclude ARFS itself, which must be installed from the audited build.

The [33 frozen-environment checks](evidence/01c-weighted-glm-lasso/checks.py) produce **14 passes and 19 strict expected failures**. Disabling expected-failure markers produces **19 failures and 14 passes**: [normal output](evidence/01c-weighted-glm-lasso/checks.txt), [unmasked failures](evidence/01c-weighted-glm-lasso/failures.txt). The separate [current-dependency check](evidence/01c-weighted-glm-lasso/current_checks.py) produces **one strict expected failure**, also reproduced [without the marker](evidence/01c-weighted-glm-lasso/current-failures.txt). Expected failures document defects; they are not a passing release gate.

Passing references cover weighted least squares, Poisson rate/exposure equivalence, Poisson BIC differences between fixed subsets, positional fold/weight alignment with non-default indexes, the BIC search against direct statsmodels fits, NumPy BIC transformation, cloning, missing-target rejection, and response-scale predictions for all six advertised families. Gamma uses an explicit log link. These are limited reference cases, not validation of every family/link combination. All checks use real fits except finding 12's deliberately stateful test estimator, which isolates fold reuse. No ARFS methods are patched.

**Must fix**

Confidence is high for the reproduced behaviors below. P1 marks wrong selection/prediction or an ordinary supported workflow failure; P2 marks a narrower API or documentation defect. Effort estimates describe remedies, not permission to implement them.

1. **Allowed current scikit-learn versions prevent every selector fit.** P1; small effort. Location: [lasso.py](../src/arfs/feature_selection/lasso.py), line 449; declared dependency at [pyproject.toml](../pyproject.toml), line 26.

   **What this is:** every selector fit enters `grid_search_cv`, which validates the target with `check_array`.

   **Problem:** it passes `force_all_finite=False`. The current wheel's scikit-learn 1.9.1 rejects this removed keyword with `TypeError`, even for a finite, one-feature Gaussian dataset. Frozen 1.6.1 already emits its removal warning.

   **Fix:** use the supported `ensure_all_finite` argument; the declared minimum 1.6.1 already supports it. Decide separately whether missing responses should be rejected at this boundary instead of reaching statsmodels.

   **If we skip it:** `LassoFeatureSelection.fit` fails on an allowed installation before fitting any model, after unit 0's import blocker is bypassed.

2. **Nonlinear deviance applies the inverse link twice and can discard every real predictor.** P1; small effort. Locations: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 235–239, 274–277, and 472–477.

   **What this is:** deviance measures prediction error for the chosen model family and drives the cross-validated penalty search.

   **Problem:** `predict` already returns response means, but `score` transforms them again. In the seeded weighted Poisson case, the search chooses `alpha=10`, with all coefficients zero; independently calculated mean fold deviance is 236.7310, versus 31.7024 at `alpha=1`. Separate Poisson, binomial, and Gamma checks all show incorrect scores. The response-scale contract is confirmed by the installed source and [statsmodels prediction documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.predict.html).

   **Fix:** calculate deviance directly from `self.predict(X)`. Coordinate its public score sign with finding 9 and retain independent loss calculations in the regression tests.

   **If we skip it:** non-Gaussian cross-validation optimizes the wrong predictions; Gaussian identity-link tests cannot detect this error.

3. **Binary cross-validation never receives the target needed for stratification.** P1; small effort. Location: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 321–335.

   **What this is:** targets with two or fewer distinct values trigger `StratifiedKFold`, which tries to preserve class proportions across folds.

   **Problem:** the loop calls `splitter.split(X)` without `y`. A weighted binomial selector with `score="deviance"` raises `TypeError: StratifiedKFold.split() missing 1 required positional argument: 'y'` before any fit.

   **Fix:** pass `y` when generating splits. Also make the task rule explicit: two distinct numeric response values alone do not establish a classification task.

   **If we skip it:** the advertised cross-validated binary workflow remains unusable, independently of the wrong deviance formula.

4. **The public weighted-CV helpers reject their documented NumPy input.** P2; small effort. Location: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 378–381; caller at 472–474.

   **What this is:** `weighted_cross_val_score` and `grid_search_cv` accept array-like features and split rows positionally.

   **Problem:** `_fit_and_score` always uses `X.iloc`. Passing the same numeric data as an ndarray raises `AttributeError`; the dataframe version passes an independent weighted-CV reference. Direct `grid_search_cv(..., score="deviance")` inherits the failure. `LassoFeatureSelection` itself converts arrays first and avoids this particular defect.

   **Fix:** slice arrays by integer positions and dataframes with `iloc`, while retaining the working positional treatment of targets and weights.

   **If we skip it:** callers must discover an undocumented dataframe-only restriction, and the helper's BIC/deviance input behavior differs.

5. **Intercept construction changes between fit, prediction, and transformation.** P1; small-to-medium effort. Locations: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 180–186, 235–239, 588–609, and 628–638.

   **What this is:** the wrapper adds or removes constant columns, while the selector independently invents the associated feature names.

   **Problem:** `add_constant` detects constants anew in each batch. A single nonzero row therefore gets no added intercept: prediction raises a shape error and selector transformation raises `KeyError: ['Intercept'] not in index`. Fitting a frame with an existing `const` or differently named constant gives a support/name length mismatch. With `fit_intercept=False`, fit removes `const` but prediction retains it and also fails.

   **Fix:** use one explicit intercept/design-column rule across fit, predict, transform, and support metadata. Derive names from the actual fitted design; do not infer them separately or rediscover the intercept from prediction-batch values. Preserve already supplied constants without adding duplicates.

   **If we skip it:** ordinary single-record inference and prebuilt design matrices fail despite successful training on larger batches.

6. **Zero weights can make every Gaussian BIC infinite and silently choose the first penalty.** P1; small effort after choosing the zero-weight contract. Locations: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 188–203 and 468–477.

   **What this is:** sample weights become statsmodels variance weights; BIC ranks the resulting log-likelihoods.

   **Problem:** setting five weights to zero on a finite Gaussian fixture produces `bic_=inf`. All four candidates on the tested grid have infinite BIC, yet `np.argmin` chooses the first, `alpha=.001`, and returns it as the best model. Backend runtime and convergence warnings do not prevent that result.

   **Fix:** define zero-weight handling before fitting: omit those rows consistently, or reject them clearly for unsupported likelihood paths. Require finite candidate criteria and fail clearly when no valid comparison remains; do not silently discard arbitrary failed candidates without reporting it.

   **If we skip it:** a common weighting boundary returns a plausible feature set without having compared models meaningfully.

7. **Changing `family` or `link` can leave the fitted model using the old objective.** P1; small effort. Locations: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 114–122 and 188–192.

   **What this is:** sklearn's `set_params` updates constructor parameters, including during external parameter searches.

   **Problem:** the statsmodels family/link object is constructed only in `__init__`. `EnetGLM().set_params(family="poisson")` still fits a Gaussian model; changing `link="log"` similarly leaves the identity link. Constructing a fresh estimator with those parameters gives the requested objective instead.

   **Fix:** construct the objective from the current parameters when fitting. The internal search keeps family/link fixed, so this finding concerns direct updates and external searches, not every internal alpha search.

   **If we skip it:** displayed parameters can describe a different statistical model from the one actually fitted.

8. **Reordered dataframe columns silently change predictions.** P1; small effort alongside finding 5. Location: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 231–239.

   **What this is:** a dataframe provides predictor names as well as column positions.

   **Problem:** prediction hands its columns to statsmodels in the supplied order without checking the fitted schema. Changing `[a, b, c]` to `[c, b, a]` changes every prediction in the reference fixture, with a maximum absolute difference of 11.3163; no error identifies the mismatch.

   **Fix:** record the original input names and order, then either validate that order or align named inputs explicitly. Reject missing, extra, or ambiguous columns and keep array width checks consistent with the fitted design.

   **If we skip it:** correct values under correct column names can still be multiplied by the wrong coefficients.

9. **External sklearn searches reward larger prediction errors.** P1; small coordinated change. Locations: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 252–277 and 475–477.

   **What this is:** sklearn's default model search calls `estimator.score` and maximizes it; [its scoring contract](https://scikit-learn.org/1.6/modules/model_evaluation.html#scoring-parameter) represents losses with a negative sign.

   **Problem:** `EnetGLM.score` returns positive deviance. A real Gaussian `GridSearchCV` over `.001` and `1000` selects `1000`, the all-zero model, because its prediction error is larger. This case uses the identity link and therefore isolates the sign error from finding 2.

   **Fix:** expose a score with the sklearn direction and update the internal deviance search consistently. BIC must still be minimized; do not blindly change the shared `argmin` for both criteria.

   **If we skip it:** the wrapper's claimed sklearn integration can choose the worst candidate even after the nonlinear formula is repaired.

**Should fix**

10. **`EnetGLM.fit` returns `None` instead of the fitted estimator.** P2; small effort. Location: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 198–206.

    **What this is:** the method documents returning `self`, matching normal sklearn estimator use.

    **Problem:** the method ends after storing statistics without returning anything. `EnetGLM().fit(X, y).predict(X)` therefore fails even though fitting completed. The internal selector avoids this because it ignores the return value.

    **Fix:** return `self` after a successful fit and retain the direct assertion.

    **If we skip it:** documented chained use remains broken.

11. **The documented near-ridge workaround discards the shrinkage it promises.** P2; small documentation change, larger only if true elastic-net prediction is required. Locations: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 67–78, 104–111, and 195–197.

    **What this is:** the class recommends a tiny nonzero `L1_wt` as a substitute for ridge regression, which normally shrinks coefficients toward zero.

    **Problem:** every fit uses `refit=True`, which [statsmodels documents](https://www.statsmodels.org/v0.14.4/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_regularized.html) as an unpenalized refit of surviving variables. At `alpha=10, L1_wt=1e-9`, the main coefficient is 3.0949, versus 0.3240 from the same converged penalized fit without refitting; the returned coefficients are the weighted least-squares fit when all variables survive.

    **Fix:** describe this as regularized variable selection followed by an unpenalized fit and remove the near-ridge claim. Correct the class-level `alpha` description too: it is penalty strength, not the L1/L2 mixing parameter. Do not add a second estimator mode unless true shrinkage is required.

    **If we skip it:** users can believe they are using regularization against unstable coefficients while prediction uses an unpenalized model. Ordinary post-selection p-values also need an explicit interpretation limit; they are not a feature-selection significance guarantee.

12. **Cross-validation reuses estimator state across folds.** P2; small effort. Locations: [lasso.py](../src/arfs/feature_selection/lasso.py), lines 330–335 and 391–395.

    **What this is:** the public weighted-CV helper accepts sklearn estimators, including estimators that retain state during repeated fits.

    **Problem:** each fold receives the same estimator without cloning. With `n_jobs=1`, the stateful diagnostic reports fit counts `[1, 2, 3]` instead of `[1, 1, 1]`, and the caller's estimator is modified. The built-in `EnetGLM.fit` rebuilds its statsmodels model, so this check does not claim that its coefficients leak between folds; the generic helper cannot promise independent fits.

    **Fix:** clone the estimator for each fold before fitting, using the existing imported sklearn `clone`.

    **If we skip it:** results for state-retaining estimators depend on fold execution and backend behavior, and validation can change the caller's model.

**Selection screen**

The [screen script](evidence/01c-weighted-glm-lasso/screen.py) attempted **60 selector fits**: five seeds, three families, signal/null data, and both BIC/deviance criteria. Fifty completed; all ten binary deviance attempts failed as in finding 3. Each case uses 180 training rows, 1,000 independently generated test rows, six numeric predictors, and positive weights. Signal cases have two true predictors and four independent noise predictors; null cases have no predictor effect. The grid has seven penalties from `.001` to `1000` and uses one CV worker.

Gaussian noise variance varies inversely with its weight. Poisson counts are generated from exposure times the true rate, then divided by exposure. Binary observations have independent positive importance weights; no replication-count or calibrated BIC interpretation is assumed for those arbitrary weights.

For a fair comparison of selected subsets, each completed selection is followed by a weighted, unpenalized GLM with an intercept, fitted on training rows only. Baselines use all predictors, an intercept only (dummy), the known true subset (oracle), and the two best single-feature training GLMs (univariate). No test response guides selection or refitting. The artifact separately retains loss from `best_estimator_` itself, whose intercept may have been discarded. Lower held-out deviance per unit weight is better.

| Data | Criterion | Mean true predictors kept | Mean noise predictors kept | Selected-subset loss | All predictors | Dummy | Oracle / univariate top 2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian signal | BIC | 2.0 / 2 | 0.8 | 0.9389 | 0.9416 | 1.9111 | 0.9317 / 0.9317 |
| Gaussian signal | Deviance | 2.0 / 2 | 0.8 | 0.9389 | 0.9416 | 1.9111 | 0.9317 / 0.9317 |
| Poisson signal | BIC | 2.0 / 2 | 0.2 | 0.9712 | 0.9902 | 3.6542 | 0.9664 / 0.9664 |
| Poisson signal | Deviance | 0.0 / 2 | 0.0 | 3.6542 | 0.9902 | 3.6542 | 0.9664 / 0.9664 |
| Binomial signal | BIC | 1.4 / 2 | 0.0 | 1.0407 | 1.0097 | 1.3933 | 0.9812 / 0.9812 |
| Binomial signal | Deviance | Failed | Failed | Failed | 1.0097 | 1.3933 | 0.9812 / 0.9812 |

Poisson deviance selection removed both signals in every seed. Its returned wrapper model also removed the intercept, giving mean test deviance 5.9933, worse than the downstream intercept-only refit at 3.6542. This distinction matters: successful downstream refitting cannot recover predictors that selection discarded.

On null data, Gaussian BIC/deviance and binomial BIC kept no predictors in these five seeds. Poisson BIC kept noise in three of five seeds, averaging 1.2 selected predictors and loss 1.0397 versus dummy loss 1.0272. Poisson deviance kept none, but its failures on signal data preclude interpreting that as evidence of reliable noise control. Detailed subsets, losses, warnings, and baseline values are in [screen.json](evidence/01c-weighted-glm-lasso/screen.json), with [execution progress](evidence/01c-weighted-glm-lasso/screen-log.txt). Five seeds on these simple independent-predictor distributions do not establish false-discovery control, selection stability in correlated data, or general performance superiority.

**Statistical and API boundaries**

`sample_weight` is passed as `var_weights`, not replication-frequency weights. The positive-weight Poisson reference verifies rate/exposure coefficient, prediction, and deviance agreement. A separate fixed-subset comparison verifies equal BIC differences between the rate/weight and count/exposure formulations on the same observations. Thus this review does not label the notebook's Poisson exposure use invalid merely because absolute log-likelihoods differ. More general variance-weight likelihood claims require care: [statsmodels documents the quasi-likelihood limitation](https://www.statsmodels.org/v0.14.4/generated/statsmodels.genmod.generalized_linear_model.GLM.html).

Intercept selection is explicitly described by the notebook and was not treated as an accidental penalty bug. The API currently uses statsmodels' default Tweedie power and negative-binomial dispersion; the six-family prediction checks do not establish arbitrary dispersion/power support. No ordinary summary p-value, BIC minimum, or screen result is evidence of calibrated post-selection inference.

The tuning docstrings advertise `score="mean_cv"`, while validation accepts `"deviance"`; align that spelling when documenting the repaired scoring contract. The selector also puts a generated intercept into `feature_names_in_`, and `EnetGLM.n_features_in_` counts the augmented design. The intercept repair should distinguish actual inputs from generated columns and preserve `make_fs_summary` and notebook behavior. This review did not run the full sklearn estimator-check suite.

**Reproduction**

Run from the repository root, using the existing unit 0 environments:

```sh
repo="$PWD"
evidence="$repo/reviews/evidence/01c-weighted-glm-lasso"
audit_dir=$(mktemp -d /tmp/arfs-1c.XXXXXX)
frozen_python=/tmp/arfs-review-03f67d0/locked-env/bin/python
current_python=/tmp/arfs-review-03f67d0/wheel-env/bin/python
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$audit_dir/mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
cd "$audit_dir"

"$frozen_python" "$evidence/environment.py" "$repo" "$audit_dir/frozen.json"
"$current_python" "$evidence/environment.py" "$repo" "$audit_dir/current.json"
"$frozen_python" -m pytest "$evidence/checks.py" \
  -q -rx --tb=short -c /dev/null --rootdir="$evidence" -p no:cacheprovider
# Expected: 14 passed, 19 xfailed; exit 0.
# Add --runxfail: 19 failed, 14 passed; exit 1.
"$current_python" -m pytest "$evidence/current_checks.py" \
  -q -rx --tb=short -c /dev/null --rootdir="$evidence" -p no:cacheprovider
# Expected: 1 xfailed; add --runxfail for 1 failed.
"$frozen_python" "$evidence/screen.py" "$audit_dir/screen.json" \
  > "$audit_dir/screen-log.txt" 2>&1
# Expected: 60 attempts recorded, including 10 binary-CV failures; exit 0.
```

If those environments no longer exist, follow [unit 0's archive/build instructions](00-baseline.md), install the audited wheel with this unit's respective dependency pins in fresh Python 3.12.7 environments, and verify module hashes before running. The pins deliberately retain the historical import workaround. A new unconstrained resolution or the published PyPI package is not the identical review target.

**Handoff**

Repair compatibility and the score calculation/direction together, then binary splitting, intercept/schema consistency, and finite-criterion handling. Carry the relevant evidence into ordinary tests during remediation. The repository's existing tests contain no GLM/Lasso workflow assertions. Family/link updates, fit return, and per-fold cloning belong in that API repair; no extra dependency is needed.

Unit **1C is reviewed, with remediation outstanding**. Previous unit 0, 1A, and 1B findings remain open. This review stops here; later units were not reviewed, and their detailed scope is not invented from the missing original plan. The single-distribution working assumption remains unchanged.

Verdict: fix 1–9 first.

Not checked: full notebooks, full package suite, additional Python/platform versions, large or highly correlated designs, parallel backends, exhaustive invalid-input/family-link combinations, or formal statistical calibration. Review artifacts only were added; the demonstrated selection and runtime defects remain in the library.
