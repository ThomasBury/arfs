"""R13: response-scale losses, score direction and independent weighted folds."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.special import xlogy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from arfs.feature_selection.lasso import (
    EnetGLM,
    LassoFeatureSelection,
    grid_search_cv,
    weighted_cross_val_score,
)


def data(family="gaussian"):
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    w = rng.uniform(0.2, 2, len(X))
    if family == "gaussian":
        y = 2 + 3 * X.a.to_numpy() + rng.normal(size=len(X))
    else:
        y = rng.poisson(w * np.exp(1 + 0.7 * X.a.to_numpy())) / w
        if family == "binomial":
            y = (y > np.median(y)).astype(float)
        elif family == "gamma":
            y += 0.1
    return X, y, w


def deviance(family, y, mu, w):
    """Independent unit-scale formulas, including zero counts and binary endpoints."""
    if family == "gaussian":
        residual = (y - mu) ** 2
    elif family == "poisson":
        residual = 2 * (xlogy(y, y / mu) - y + mu)
    elif family == "binomial":
        residual = 2 * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu)))
    else:
        residual = 2 * ((y - mu) / mu - np.log(y / mu))
    return np.sum(w * residual)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial", "gamma"])
@pytest.mark.parametrize("weighted", [False, True])
def test_score_is_negative_independent_response_scale_deviance(family, weighted):
    X, y, w = data(family)
    model = EnetGLM(
        family=family, link="log" if family == "gamma" else None, alpha=0.001, L1_wt=1
    )
    model.fit(X, y, w)
    weights = w if weighted else None
    mu = np.asarray(model.predict(X))
    expected = deviance(family, y, mu, w if weighted else np.ones(len(X)))
    assert_allclose(model.score(X, y, weights), -expected, atol=1e-8)
    if weighted:
        assert_allclose(model.score(X, y, 3 * w), -3 * expected, atol=1e-8)


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_cv_matches_independent_weighted_least_squares_by_position(as_array, weighted):
    X, y, w = data()
    X.index = np.arange(len(X)) + 1000
    original = X.copy(deep=True)
    weights = w if weighted else np.ones(len(X))
    expected = []
    for train, test in KFold(5).split(X):
        design = sm.add_constant(X.iloc[train]).to_numpy()
        beta = np.linalg.lstsq(
            design * np.sqrt(weights[train, None]),
            y[train] * np.sqrt(weights[train]),
            rcond=None,
        )[0]
        mu = sm.add_constant(X.iloc[test]).to_numpy() @ beta
        expected.append(-np.sum(weights[test] * (y[test] - mu) ** 2))
    actual = weighted_cross_val_score(
        EnetGLM(alpha=0.001, L1_wt=1),
        X.to_numpy() if as_array else X,
        pd.Series(y, index=X.index),
        pd.Series(w, index=X.index) if weighted else None,
        n_jobs=1,
    )
    assert_allclose(actual, expected, atol=1e-8)
    assert_frame_equal(X, original)


class StatefulEstimator(BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.fits_ = getattr(self, "fits_", 0) + 1
        return self

    def score(self, X, y, sample_weight=None):
        return self.fits_


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("prefitted", [False, True])
def test_folds_clone_and_preserve_caller_state(n_jobs, weighted, prefitted):
    X, y, w = data()
    estimator = StatefulEstimator()
    if prefitted:
        estimator.fit(X, y)
    scores = weighted_cross_val_score(
        estimator, X, y, w if weighted else None, cv=3, n_jobs=n_jobs
    )
    assert_array_equal(scores, [1, 1, 1])
    assert vars(estimator) == ({"fits_": 1} if prefitted else {})


class FoldRows(BaseEstimator):
    def __init__(self, family="gaussian"):
        self.family = family

    def fit(self, X, y, sample_weight=None):
        return self

    def score(self, X, y, sample_weight=None):
        return np.asarray(X)[:, 0].tolist()


class ClassifierFoldRows(ClassifierMixin, FoldRows):
    score = FoldRows.score


@pytest.mark.parametrize(
    "estimator,y,stratified",
    [
        (FoldRows(), np.repeat([0, 1], 15), False),
        (FoldRows(), np.repeat([0.2, 0.8], 15), False),
        (FoldRows(), np.zeros(30), False),
        (FoldRows("binomial"), np.repeat([0, 1], 15), True),
        (FoldRows("binomial"), np.repeat([0.2, 0.8], 15), False),
        (ClassifierFoldRows(), np.repeat([0, 1, 2], 10), True),
    ],
)
def test_splitter_uses_task_and_target_instead_of_unique_value_count(
    estimator, y, stratified
):
    X = np.arange(30).reshape(-1, 1)
    splitter = StratifiedKFold(3) if stratified else KFold(3)
    expected = [test.tolist() for _, test in splitter.split(X, y)]
    assert weighted_cross_val_score(estimator, X, y, cv=3, n_jobs=1) == expected


@pytest.mark.parametrize("as_array", [False, True])
def test_binary_selector_and_grid_match_stratified_reference(as_array):
    X, y, w = data("binomial")
    features = X.to_numpy() if as_array else X
    grid = np.logspace(-3, 3, 3)
    expected = []
    for alpha in grid:
        losses = []
        for train, test in StratifiedKFold(5).split(X, y):
            fit = sm.GLM(
                y[train],
                sm.add_constant(X.iloc[train]),
                var_weights=w[train],
                family=sm.families.Binomial(),
            ).fit_regularized(alpha=alpha, L1_wt=1, refit=True)
            mu = fit.predict(sm.add_constant(X.iloc[test]))
            losses.append(deviance("binomial", y[test], mu, w[test]))
        expected.append(np.mean(losses))
    selector = LassoFeatureSelection(
        family="binomial", score="deviance", n_iterations=3, n_jobs=1
    ).fit(features, y, w)
    assert selector.best_estimator_.alpha == grid[np.argmin(expected)]
    assert np.isfinite(selector.best_estimator_.coef_).all()


@pytest.mark.parametrize("as_array", [False, True])
def test_poisson_signal_grid_minimizes_actual_deviance(as_array):
    X, y, w = data("poisson")
    grid = np.logspace(-3, 3, 7)
    losses = []
    for alpha in grid:
        fold_losses = []
        for train, test in KFold(5).split(X):
            fit = sm.GLM(
                y[train],
                sm.add_constant(X.iloc[train]),
                var_weights=w[train],
                family=sm.families.Poisson(),
            ).fit_regularized(alpha=alpha, L1_wt=1, refit=True)
            mu = fit.predict(sm.add_constant(X.iloc[test]))
            fold_losses.append(deviance("poisson", y[test], mu, w[test]))
        losses.append(np.mean(fold_losses))
    model = grid_search_cv(
        X.to_numpy() if as_array else X,
        y,
        w,
        n_iterations=7,
        family="poisson",
        score="deviance",
        n_jobs=1,
    )
    assert model.alpha == grid[np.argmin(losses)]
    assert model.coef_[1] != 0
    assert min(losses) < losses[-1] / 2


@pytest.mark.parametrize("as_array", [False, True])
def test_bic_search_still_minimizes_direct_backend_criterion(as_array):
    X, y, w = data()
    grid = np.logspace(-3, 3, 7)
    criteria = [
        sm.GLM(y, sm.add_constant(X), var_weights=w)
        .fit_regularized(alpha=alpha, L1_wt=1, refit=True)
        .bic_llf
        for alpha in grid
    ]
    model = grid_search_cv(X.to_numpy() if as_array else X, y, w, n_iterations=7)
    assert model.alpha == grid[np.argmin(criteria)]


def test_real_external_sklearn_search_maximizes_negative_deviance():
    X, y, _ = data()
    search = GridSearchCV(
        EnetGLM(L1_wt=1), {"alpha": [0.001, 1000]}, cv=3, error_score="raise"
    ).fit(X, y)
    assert search.best_params_["alpha"] == 0.001
    assert np.all(search.cv_results_["mean_test_score"] < 0)
