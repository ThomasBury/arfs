"""Unit 1C review evidence; strict xfails expose defects, not a release gate.

Run on unit 0's frozen wheel. current_checks.py targets its newer-dependency wheel.
No monkeypatches or repairs to ARFS are applied.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, KFold

from arfs.feature_selection.lasso import (
    EnetGLM, LassoFeatureSelection, grid_search_cv, weighted_cross_val_score,
)


def data():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    w = rng.uniform(.2, 2, len(X))
    y = 2 + 3 * X.a.to_numpy() + rng.normal(size=len(X))
    return X, y, w


def poisson_data():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    w = rng.uniform(.2, 2, len(X))
    y = rng.poisson(w * np.exp(1 + .7 * X.a.to_numpy())) / w
    return X, y, w


def test_weighted_gaussian_matches_weighted_least_squares():
    X, y, w = data()
    before = X.copy(deep=True)
    model = EnetGLM(alpha=.001, L1_wt=1)
    model.fit(X, y, w)
    design = sm.add_constant(X).to_numpy()
    beta = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    assert_allclose(model.coef_, beta, atol=1e-8)
    assert_allclose(model.predict(X), design @ beta, atol=1e-8)
    assert_allclose(model.score(X, y, w), np.sum(w * (y - design @ beta) ** 2))
    assert_frame_equal(X, before)


def test_poisson_rate_weights_match_count_exposure():
    X, rate, exposure = poisson_data()
    model = EnetGLM(family="poisson", alpha=.001, L1_wt=1)
    model.fit(X, rate, exposure)
    ref = sm.GLM(rate * exposure, sm.add_constant(X), exposure=exposure,
                 family=sm.families.Poisson()).fit()
    assert_allclose(model.coef_, ref.params, atol=1e-8)
    assert_allclose(model.predict(X) * exposure, ref.fittedvalues, atol=1e-8)
    assert_allclose(model.deviance_, ref.deviance, atol=1e-8)


def test_poisson_bic_differences_match_exposure_reference():
    # Absolute likelihoods differ by response-only terms; subset comparisons agree.
    X, rate, exposure = poisson_data()
    rate_bic, count_bic = [], []
    for names in [["a"], ["a", "b", "c"]]:
        design = sm.add_constant(X[names])
        rate_bic.append(sm.GLM(rate, design, var_weights=exposure,
                               family=sm.families.Poisson()).fit().bic_llf)
        count_bic.append(sm.GLM(rate * exposure, design, exposure=exposure,
                                family=sm.families.Poisson()).fit().bic_llf)
    assert_allclose(np.diff(rate_bic), np.diff(count_bic), atol=1e-8)


def test_gaussian_cv_weights_follow_positions_with_custom_index():
    X, y, w = data()
    X.index = np.arange(len(X)) + 1000
    y, w = pd.Series(y, index=X.index), pd.Series(w, index=X.index)
    actual = weighted_cross_val_score(EnetGLM(alpha=.001, L1_wt=1), X, y, w, n_jobs=1)
    reference = []
    for tr, te in KFold(5).split(X):
        fit = sm.WLS(y.iloc[tr], sm.add_constant(X.iloc[tr]), weights=w.iloc[tr]).fit()
        pred = fit.predict(sm.add_constant(X.iloc[te]))
        reference.append(np.sum(w.iloc[te] * (y.iloc[te] - pred) ** 2))
    assert_allclose(actual, reference, atol=1e-8)


def test_bic_grid_matches_direct_statsmodels_search():
    X, y, w = data()
    grid = np.logspace(-3, 3, 7)
    scores = [sm.GLM(y, sm.add_constant(X), var_weights=w).fit_regularized(
        alpha=a, L1_wt=1, refit=True).bic_llf for a in grid]
    selected = grid_search_cv(X, y, w, n_iterations=7)
    assert selected.alpha == grid[np.argmin(scores)]


def test_numpy_bic_selector_transform_and_names_agree():
    X, y, w = data()
    selector = LassoFeatureSelection(n_iterations=4).fit(X.to_numpy(), y, w)
    transformed = selector.transform(X.to_numpy())
    assert_array_equal(transformed.columns, selector.get_feature_names_out())
    assert transformed.shape[0] == len(X)


@pytest.mark.parametrize("family,link", [
    ("gaussian", None), ("poisson", None), ("binomial", None),
    ("gamma", "log"), ("negativebinomial", None), ("tweedie", None),
])
def test_supported_family_predict_matches_direct_response_mean(family, link):
    X, y, w = data()
    if family == "binomial":
        y = (y > 2).astype(float)
    elif family != "gaussian":
        y = np.exp(y / 6)
    model = EnetGLM(family=family, link=link, alpha=.001, L1_wt=1)
    model.fit(X, y, w)
    assert np.isfinite(model.coef_).all()
    assert_allclose(model.predict(X), model.result.fittedvalues)


@pytest.mark.xfail(strict=True, reason="F2: inverse link is applied twice")
@pytest.mark.parametrize("family", ["poisson", "binomial", "gamma"])
def test_score_equals_response_scale_deviance(family):
    X, y, w = poisson_data()
    if family == "binomial":
        y = (y > np.median(y)).astype(float)
    elif family == "gamma":
        y = y + .1
    model = EnetGLM(family=family, link="log" if family == "gamma" else None,
                    alpha=.001, L1_wt=1)
    model.fit(X, y, w)
    expected = model.objective.deviance(y, model.predict(X), var_weights=w)
    assert_allclose(model.score(X, y, w), expected)


@pytest.mark.xfail(strict=True, reason="F2: wrong Poisson deviance selects the null model")
def test_poisson_cv_chooses_minimum_actual_deviance():
    X, y, w = poisson_data()
    grid = np.logspace(-3, 3, 7)
    scores = []
    for a in grid:
        fold_scores = []
        for tr, te in KFold(5).split(X):
            fit = sm.GLM(y[tr], sm.add_constant(X.iloc[tr]), var_weights=w[tr],
                         family=sm.families.Poisson()).fit_regularized(
                             alpha=a, L1_wt=1, refit=True)
            mu = fit.predict(sm.add_constant(X.iloc[te]))
            fold_scores.append(sm.families.Poisson().deviance(y[te], mu, var_weights=w[te]))
        scores.append(np.mean(fold_scores))
    actual = grid_search_cv(X, y, w, 7, "poisson", score="deviance", n_jobs=1)
    assert actual.alpha == grid[np.argmin(scores)], {
        "selected_true_mean_deviance": scores[list(grid).index(actual.alpha)],
        "minimum_true_mean_deviance": min(scores),
        "selected_coefficients": actual.coef_.tolist(),
    }


@pytest.mark.xfail(strict=True, reason="F3: StratifiedKFold.split is called without y")
def test_binary_deviance_selector_runs():
    X, y, w = data()
    selector = LassoFeatureSelection(family="binomial", score="deviance", n_iterations=3, n_jobs=1)
    selector.fit(X, (y > 2).astype(int), w)
    assert np.isfinite(selector.best_estimator_.coef_).all()


@pytest.mark.xfail(strict=True, reason="F4: CV unconditionally uses DataFrame.iloc")
def test_array_cv_matches_dataframe_cv():
    X, y, w = data()
    estimator = EnetGLM(alpha=.001, L1_wt=1)
    expected = weighted_cross_val_score(estimator, X, y, w, n_jobs=1)
    assert_allclose(weighted_cross_val_score(estimator, X.to_numpy(), y, w, n_jobs=1), expected)


@pytest.mark.xfail(strict=True, reason="F5: add_constant detects constants separately in each batch")
def test_single_row_prediction_matches_batch():
    X, y, w = data()
    model = EnetGLM(alpha=.001, L1_wt=1)
    model.fit(X, y, w)
    assert_allclose(model.predict(X.iloc[[0]]), model.predict(X).iloc[[0]])


@pytest.mark.xfail(strict=True, reason="F5: transform omits the learned intercept for a single row")
def test_single_row_transform_matches_batch():
    X, y, w = data()
    selector = LassoFeatureSelection(n_iterations=3).fit(X, y, w)
    assert_frame_equal(selector.transform(X.iloc[[0]]), selector.transform(X).iloc[[0]])


@pytest.mark.xfail(strict=True, reason="F5: feature names assume an added intercept even when none is added")
@pytest.mark.parametrize("name", ["const", "constant"])
def test_existing_constant_preserves_design_and_support_alignment(name):
    X, y, w = data()
    X[name] = 1.
    selector = LassoFeatureSelection(n_iterations=3).fit(X, y, w)
    assert len(selector.support_) == len(selector.feature_names_in_)


@pytest.mark.xfail(strict=True, reason="F5: fit drops named constants but predict does not")
def test_no_intercept_predict_uses_same_columns_as_fit():
    X, y, w = data()
    X["const"] = 1.
    model = EnetGLM(fit_intercept=False)
    model.fit(X, y, w)
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.xfail(strict=True, reason="F6: zero Gaussian weights produce infinite BIC and arbitrary first-alpha selection")
def test_zero_weights_are_excluded_or_rejected_explicitly():
    X, y, w = data()
    w[:5] = 0
    # Either a clear early rejection or a valid fit is defensible; returning inf is not.
    try:
        selector = LassoFeatureSelection(n_iterations=4).fit(X, y, w)
    except ValueError as exc:
        assert "weight" in str(exc).lower()
        return
    assert np.isfinite(selector.best_estimator_.bic_)


@pytest.mark.xfail(strict=True, reason="F7: objective is constructed only in __init__")
@pytest.mark.parametrize("params", [{"family": "poisson"}, {"link": "log"}])
def test_set_params_rebuilds_model_objective(params):
    X, y, w = poisson_data()
    model = EnetGLM(alpha=.001, L1_wt=1).set_params(**params)
    fresh = EnetGLM(alpha=.001, L1_wt=1, **params)
    model.fit(X, y, w)
    fresh.fit(X, y, w)
    assert type(model.objective) is type(fresh.objective)
    assert type(model.objective.link) is type(fresh.objective.link)
    assert_allclose(model.predict(X), fresh.predict(X))


@pytest.mark.xfail(strict=True, reason="F8: dataframe feature names are ignored at prediction")
def test_reordered_prediction_columns_are_aligned_or_rejected():
    X, y, w = data()
    model = EnetGLM(alpha=.001, L1_wt=1)
    model.fit(X, y, w)
    try:
        actual = model.predict(X[["c", "b", "a"]])
    except ValueError:
        return
    assert_allclose(actual, model.predict(X))


@pytest.mark.xfail(strict=True, reason="F9: sklearn maximizes the positive deviance score")
def test_sklearn_search_prefers_better_gaussian_predictions():
    X, y, _ = data()
    search = GridSearchCV(EnetGLM(L1_wt=1), {"alpha": [.001, 1000]}, cv=3, error_score="raise").fit(X, y)
    assert search.best_params_["alpha"] == .001


@pytest.mark.xfail(strict=True, reason="F10: fit returns None")
def test_estimator_fit_returns_self():
    X, y, w = data()
    model = EnetGLM()
    assert model.fit(X, y, w) is model


@pytest.mark.xfail(strict=True, reason="F11: refit=True discards the advertised near-ridge shrinkage")
def test_documented_near_ridge_matches_penalized_fit():
    X, y, w = data()
    model = EnetGLM(alpha=10, L1_wt=1e-9)
    model.fit(X, y, w)
    penalized = sm.GLM(y, sm.add_constant(X), var_weights=w).fit_regularized(
        alpha=10, L1_wt=1e-9, refit=False)
    assert penalized.converged
    assert_allclose(model.coef_, penalized.params, atol=1e-5)


class StatefulEstimator(BaseEstimator):
    """A minimal estimator that reuses state, as warm-start estimators may do."""
    def fit(self, X, y, sample_weight=None):
        self.fits_ = getattr(self, "fits_", 0) + 1
        return self

    def score(self, X, y, sample_weight=None):
        return self.fits_


@pytest.mark.xfail(strict=True, reason="F12: CV shares the caller's estimator across folds")
def test_cv_folds_are_independent_and_leave_estimator_unfitted():
    X, y, w = data()
    estimator = StatefulEstimator()
    scores = weighted_cross_val_score(estimator, X, y, w, cv=3, n_jobs=1)
    assert_array_equal(scores, [1, 1, 1])
    assert not hasattr(estimator, "fits_")


def test_clone_preserves_constructor_parameters():
    model = EnetGLM(family="poisson", link="log", alpha=.2, L1_wt=1)
    assert clone(model).get_params() == model.get_params()


def test_missing_target_does_not_silently_fit():
    X, y, w = data()
    y[0] = np.nan
    with pytest.raises(ValueError):
        LassoFeatureSelection(n_iterations=3).fit(X, y, w)
