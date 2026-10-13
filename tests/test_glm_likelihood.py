"""R15: finite criteria and the unpenalized post-selection refit contract."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator

from arfs.feature_selection import lasso


def data():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    w = rng.uniform(0.2, 2, len(X))
    y = 2 + 3 * X.a.to_numpy() + rng.normal(size=len(X))
    return X, y, w


class FixedCriterion(BaseEstimator):
    """Isolate invalid search evidence; no failed candidate may be hidden."""

    def __init__(
        self, family="gaussian", link=None, L1_wt=1, fit_intercept=True, alpha=0
    ):
        self.family = family
        self.link = link
        self.L1_wt = L1_wt
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        return self


@pytest.mark.parametrize("criterion", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("score", ["bic", "deviance"])
@pytest.mark.parametrize("partial", [False, True])
def test_search_rejects_nonfinite_criteria_with_candidate_context(
    monkeypatch, criterion, score, partial
):
    X, y, w = data()
    seen = []

    def evidence(estimator):
        seen.append(estimator.alpha)
        return 1.0 if partial and estimator.alpha < 1 else criterion

    monkeypatch.setattr(lasso, "EnetGLM", FixedCriterion)
    monkeypatch.setattr(FixedCriterion, "bic_", property(evidence), raising=False)
    monkeypatch.setattr(
        lasso,
        "weighted_cross_val_score",
        lambda estimator, *a, **k: [evidence(estimator)] * 5,
    )
    with pytest.raises(ValueError, match=rf"{score}.*alpha=.*finite"):
        lasso.grid_search_cv(X, y, w, n_iterations=3, score=score, n_jobs=1)
    assert len(seen) == (2 if partial else 1)


def test_real_degenerate_gaussian_search_does_not_hide_backend_failure():
    X, _, w = data()
    with (
        np.errstate(divide="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="deviance.*nan"),
    ):
        lasso.grid_search_cv(X, np.zeros(len(X)), w, n_iterations=3)


@pytest.mark.parametrize("L1_wt", [0, -0.1, 1.1, np.nan, np.inf])
def test_unsupported_ridge_or_invalid_mixture_has_clear_fit_error(L1_wt):
    X, y, w = data()
    with pytest.raises(ValueError, match="L1_wt"):
        lasso.EnetGLM(L1_wt=L1_wt).fit(X, y, w)


def test_tiny_l1_returns_unpenalized_refit_without_claiming_ridge_shrinkage():
    X, y, w = data()
    model = lasso.EnetGLM(alpha=10, L1_wt=1e-9).fit(X, y, w)
    design = sm.add_constant(X)
    refit = sm.WLS(y, design, weights=w).fit()
    penalized = sm.GLM(y, design, var_weights=w).fit_regularized(
        alpha=10, L1_wt=1e-9, refit=False
    )
    assert_allclose(model.coef_, refit.params, atol=1e-8)
    assert abs(np.asarray(model.coef_)[1]) > 5 * abs(np.asarray(penalized.params)[1])
    assert_allclose(model.predict(X), refit.predict(design), atol=1e-8)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("alpha", [0.001, 1, 10])
def test_weighted_bic_matches_independent_likelihood_formula(family, alpha):
    from scipy.special import gammaln, xlogy

    X, y, w = data()
    if family == "poisson":
        rng = np.random.default_rng(105)
        y = rng.poisson(w * np.exp(1 + 0.7 * X.a.to_numpy())) / w
    model = lasso.EnetGLM(family=family, alpha=alpha, L1_wt=1).fit(X, y, w)
    mu = np.asarray(model.predict(X))
    if family == "gaussian":
        design = sm.add_constant(X).to_numpy()
        selected = np.asarray(model.coef_) != 0
        if selected.any():
            beta = np.linalg.lstsq(
                design[:, selected] * np.sqrt(w[:, None]),
                y * np.sqrt(w),
                rcond=None,
            )[0]
            assert_allclose(np.asarray(model.coef_)[selected], beta, atol=1e-8)
        variance = np.sum(w * (y - mu) ** 2) / len(y)
        llf = -0.5 * np.sum(
            np.log(2 * np.pi * variance / w) + w * (y - mu) ** 2 / variance
        )
    else:
        llf = np.sum(w * (xlogy(y, mu) - mu - gammaln(y + 1)))
    count = np.count_nonzero(model.coef_)
    expected = -2 * llf + np.log(len(y)) * count
    assert_allclose(model.bic_, expected, atol=1e-8)
    assert_allclose(model.aic_, -2 * llf + 2 * count, atol=1e-8)


@pytest.mark.parametrize(
    "family,link",
    [("gaussian", None), ("poisson", None), ("binomial", None), ("gamma", "log")],
)
@pytest.mark.parametrize("as_array", [False, True])
def test_zero_weight_fit_and_score_equal_physical_row_omission(family, link, as_array):
    X, y, w = data()
    X.index = np.arange(len(X)) + 1000
    if family == "binomial":
        y = (y > np.median(y)).astype(float)
    elif family != "gaussian":
        y = np.exp(y / 6)
    w[np.arange(len(w)) % 4 == 0] = 0
    keep = w > 0
    features = X.to_numpy() if as_array else X
    retained = features[keep] if as_array else features.iloc[keep]
    model = lasso.EnetGLM(family=family, link=link, alpha=0.001, L1_wt=1).fit(
        features, pd.Series(y, index=X.index), pd.Series(w, index=X.index)
    )
    reference = lasso.EnetGLM(family=family, link=link, alpha=0.001, L1_wt=1).fit(
        retained, y[keep], w[keep]
    )
    assert model.model.nobs == keep.sum()
    assert_allclose(model.coef_, reference.coef_, atol=1e-8)
    assert_allclose(model.predict(features), reference.predict(features), atol=1e-8)
    assert_allclose(model.bic_, reference.bic_, atol=1e-8)
    assert_allclose(
        model.score(features, y, w),
        reference.score(retained, y[keep], w[keep]),
        atol=1e-8,
    )


@pytest.mark.parametrize("score", ["bic", "deviance"])
@pytest.mark.parametrize("as_array", [False, True])
def test_zero_weight_selector_tuning_matches_omission_before_splitting(score, as_array):
    X, y, w = data()
    w[np.arange(len(w)) % 4 == 0] = 0
    keep = w > 0
    features = X.to_numpy() if as_array else X
    retained = features[keep] if as_array else features.iloc[keep]
    model = lasso.LassoFeatureSelection(n_iterations=4, score=score, n_jobs=1).fit(
        features, y, w
    )
    reference = lasso.LassoFeatureSelection(n_iterations=4, score=score, n_jobs=1).fit(
        retained, y[keep], w[keep]
    )
    assert model.best_estimator_.alpha == reference.best_estimator_.alpha
    assert_allclose(
        model.best_estimator_.coef_, reference.best_estimator_.coef_, atol=1e-8
    )
    assert model.best_estimator_.model.nobs == keep.sum()
    assert np.array_equal(
        model.get_feature_names_out(), reference.get_feature_names_out()
    )


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_direct_glm_cv_omits_zero_weight_rows_before_folds(family, n_jobs):
    X, y, w = data()
    if family == "binomial":
        y = (y > np.median(y)).astype(float)
    w[np.arange(len(w)) % 4 == 0] = 0
    keep = w > 0
    estimator = lasso.EnetGLM(family=family, alpha=0.001, L1_wt=1)
    actual = lasso.weighted_cross_val_score(estimator, X, y, w, cv=3, n_jobs=n_jobs)
    expected = lasso.weighted_cross_val_score(
        estimator, X.iloc[keep], y[keep], w[keep], cv=3, n_jobs=1
    )
    assert_allclose(actual, expected, atol=1e-8)
    assert not hasattr(estimator, "coef_")


def test_zero_weight_rows_do_not_determine_training_intercept_design():
    X, y, w = data()
    X["given_constant"] = 1.0
    X.loc[0, "given_constant"] = 2.0
    w[0] = 0
    model = lasso.EnetGLM(alpha=0.001, L1_wt=1).fit(X, y, w)
    assert model.added_intercept_ is None
    assert np.array_equal(model.design_feature_names_, X.columns)


@pytest.mark.parametrize("entry", ["fit", "score", "cv", "grid"])
def test_all_zero_weights_are_still_rejected(entry):
    X, y, w = data()
    estimator = lasso.EnetGLM(alpha=0.001, L1_wt=1)
    if entry == "score":
        estimator.fit(X, y, w)
    calls = {
        "fit": lambda: estimator.fit(X, y, np.zeros(len(X))),
        "score": lambda: estimator.score(X, y, np.zeros(len(X))),
        "cv": lambda: lasso.weighted_cross_val_score(
            estimator, X, y, np.zeros(len(X)), n_jobs=1
        ),
        "grid": lambda: lasso.grid_search_cv(X, y, np.zeros(len(X))),
    }
    with pytest.raises(ValueError, match="positive"):
        calls[entry]()


def test_information_criteria_use_selected_design_rank_for_redundant_columns():
    X, y, w = data()
    X["a_copy"] = X.a
    model = lasso.EnetGLM(alpha=0.001, L1_wt=1e-6).fit(X, y, w)
    design = sm.add_constant(X).to_numpy()
    active = np.asarray(model.coef_) != 0
    rank = np.linalg.matrix_rank(design[:, active])
    assert rank < active.sum()
    mu = np.asarray(model.predict(X))
    variance = np.sum(w * (y - mu) ** 2) / len(y)
    llf = -0.5 * np.sum(np.log(2 * np.pi * variance / w) + w * (y - mu) ** 2 / variance)
    assert_allclose(model.bic_, -2 * llf + np.log(len(y)) * rank, atol=1e-8)
    assert_allclose(model.aic_, -2 * llf + 2 * rank, atol=1e-8)


@pytest.mark.parametrize("entry", ["fit", "score", "cv", "grid"])
def test_zero_weight_does_not_hide_invalid_response_at_validation_boundary(entry):
    X, y, w = data()
    estimator = lasso.EnetGLM(alpha=0.001, L1_wt=1)
    if entry == "score":
        estimator.fit(X, y, w)
    y[0], w[0] = np.nan, 0
    calls = {
        "fit": lambda: estimator.fit(X, y, w),
        "score": lambda: estimator.score(X, y, w),
        "cv": lambda: lasso.weighted_cross_val_score(estimator, X, y, w, n_jobs=1),
        "grid": lambda: lasso.grid_search_cv(X, y, w),
    }
    with pytest.raises(ValueError, match="nonmissing"):
        calls[entry]()


def test_generic_cv_keeps_its_zero_mass_fold_guard():
    from sklearn.dummy import DummyRegressor

    X, y, w = data()
    w[:30] = 0
    with pytest.raises(ValueError, match="positive"):
        lasso.weighted_cross_val_score(DummyRegressor(), X, y, w, n_jobs=1)
