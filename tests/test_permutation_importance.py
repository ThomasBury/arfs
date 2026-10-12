"""Weighted scoring and signed BoostAGroota permutation evidence."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.inspection import permutation_importance

import arfs.feature_selection.allrelevant as a


def data():
    rng = np.random.RandomState(31)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    return X, pd.Series(3 * X.a + rng.normal(size=len(X)))


def model(classification=False):
    cls = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
    return cls(
        n_estimators=15,
        num_leaves=7,
        min_child_samples=8,
        n_jobs=1,
        verbosity=-1,
        random_state=0,
    )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("classification", [False, True])
def test_shared_importance_matches_real_sklearn_scoring(weighted, classification):
    X, y = data()
    if classification:
        y = (y > y.median()).astype(int)
    before = X.copy(deep=True)
    weights = np.where(X.b > 0, 30.0, 1.0) if weighted else None
    fitted, X_test, y_test, w_test = a._split_fit_estimator(
        model(classification), X, y, weights
    )
    expected = permutation_importance(
        fitted,
        X_test,
        y_test,
        sample_weight=w_test,
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    ).importances_mean
    if weighted:
        unweighted = permutation_importance(
            fitted, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
        ).importances_mean
        assert not np.allclose(expected, unweighted)
    actual = a._get_perm_imp(model(classification), X, y, weights)
    assert_allclose(actual, expected)
    assert_frame_equal(X, before)


@pytest.mark.parametrize("selector_name", ["leshy", "boost"])
def test_public_callers_forward_aligned_validation_weights(monkeypatch, selector_name):
    X, y = data()
    weights = np.where(X.b > 0, 30.0, 1.0)
    observed = []

    def check_scoring(fitted, X_test, y_test, **kwargs):
        supplied = kwargs.get("sample_weight")
        assert supplied is not None
        assert_array_equal(supplied, weights[X_test.index])
        assert_array_equal(np.asarray(y_test), y.loc[X_test.index])
        observed.append(True)
        return permutation_importance(fitted, X_test, y_test, **kwargs)

    monkeypatch.setattr(a, "permutation_importance", check_scoring)
    selector = (
        a.Leshy(model(), importance="pimp", max_iter=2, n_estimators=15, random_state=0)
        if selector_name == "leshy"
        else a.BoostAGroota(model(), importance="pimp", iters=1, max_rounds=1)
    )
    before = X.copy(deep=True)
    selector.fit(X, y, weights)
    assert len(observed) == (2 if selector_name == "leshy" else 1)
    assert_frame_equal(X, before)
    assert selector.transform(X).shape[1] == selector.support_.sum()


@pytest.mark.parametrize(
    "scores, expected, selected",
    [
        ([-0.1, -0.5, -0.2, -0.2], [-0.1, -0.5, -0.2, -0.2], []),
        ([-1.0, 1.0, 0.0, 0.0], [-0.5, 0.5, 0.0, 0.0], ["b"]),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], []),
        ([2.0, -1.0, 1.0, 0.0], [0.5, -0.25, 0.25, 0.0], ["a"]),
        ([3.0, 1.0, 1.0, 1.0], [0.5, 1 / 6, 1 / 6, 1 / 6], ["a", "b"]),
        ([0.0, -1.0, -2.0, -2.0], [0.0, -0.2, -0.4, -0.4], []),
    ],
    ids=[
        "negative-total",
        "cancellation",
        "all-zero",
        "mixed",
        "positive",
        "zero-real",
    ],
)
def test_boost_signed_normalization_and_positive_evidence(
    monkeypatch, scores, expected, selected
):
    X, y = data()
    X = X[["a", "b"]]
    monkeypatch.setattr(a, "_get_perm_imp", lambda *args, **kwargs: np.array(scores))
    with np.errstate(divide="raise", invalid="raise"):
        _, kept, table, threshold = a._reduce_vars_sklearn(
            X, y, model(), 1, 1, 1, 0.1, True, None, "pimp", None
        )
    assert_allclose(table.fscore1, expected)
    assert_allclose(table.Mean, expected)
    assert np.isfinite(threshold)
    assert_array_equal(kept, selected)
    assert (table.loc[table.feature.isin(kept), "Mean"] > 0).all()


@pytest.mark.parametrize(
    "scores", [[-0.1, -0.5, -0.2, -0.2], [0.0] * 4, [-1.0, 1.0, 0.0, 0.0]]
)
def test_public_boost_empty_and_signed_selections(monkeypatch, scores):
    X, y = data()
    X = X[["a", "b"]]
    monkeypatch.setattr(a, "_get_perm_imp", lambda *args, **kwargs: np.array(scores))
    selector = a.BoostAGroota(
        model(), importance="pimp", cutoff=1, iters=1, max_rounds=1
    ).fit(X, y)
    expected = [False, True] if scores[1] > 0 else [False, False]
    assert_array_equal(selector.support_, expected)
    assert_array_equal(selector.selected_features_, X.columns[expected])
    assert_frame_equal(selector.transform(X), X.loc[:, expected])


def test_boost_each_iteration_has_its_own_positive_scale(monkeypatch):
    X, y = data()
    scores = iter([np.array([2.0, -1.0, 1.0, 0.0]), np.array([1.0, -1.0, 0.0, 0.0])])
    monkeypatch.setattr(a, "_get_perm_imp", lambda *args, **kwargs: next(scores))
    _, kept, table, _ = a._reduce_vars_sklearn(
        X[["a", "b"]], y, model(), 1, 1, 2, 0.1, True, None, "pimp", None
    )
    assert_allclose(table.fscore1, [0.5, -0.25, 0.25, 0.0])
    assert_allclose(table.fscore2, [0.5, -0.5, 0.0, 0.0])
    assert_allclose(table.Mean, [0.5, -0.375, 0.125, 0.0])
    assert_array_equal(kept, ["a"])
