"""Leshy validates fitted state and preserves its iteration/importance contracts."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.stats import binom
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.multitest import fdrcorrection

from arfs.feature_selection import allrelevant as a


def data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["x", "z", "noise"])
    X["category"] = pd.Categorical(rng.randint(0, 4, len(X)))
    return X, 3 * X.x - X.z, np.where(X.x > 0, 4.0, 1.0)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("max_iter", 0),
        ("max_iter", 1),
        ("max_iter", True),
        ("max_iter", 2.5),
        ("max_iter", np.nan),
        ("n_estimators", 0),
        ("n_estimators", -3),
        ("n_estimators", True),
        ("n_estimators", 2.5),
        ("n_estimators", "automatic"),
        ("importance", "fastshap"),
        ("importance", "typo"),
        ("perc", 0),
        ("perc", 101),
        ("perc", np.nan),
        ("perc", np.inf),
        ("perc", "90"),
        ("alpha", 0),
        ("alpha", 1.1),
        ("alpha", np.nan),
        ("alpha", np.inf),
        ("alpha", None),
    ],
)
@pytest.mark.parametrize("set_params", [False, True])
def test_current_parameters_are_validated_at_fit(parameter, value, set_params):
    X, y, _ = data()
    selector = a.Leshy(lgb.LGBMRegressor())
    if set_params:
        selector.set_params(**{parameter: value})
    else:
        selector = a.Leshy(selector.estimator, **{parameter: value})
    clone(selector)  # Construction and cloning only store parameters.
    with pytest.raises(ValueError, match=parameter):
        selector.fit(X, y)
    with pytest.raises(NotFittedError):
        selector.get_support()


@pytest.mark.parametrize(
    "operation",
    ["get_support", "transform", "plot_importance", "get_feature_names_out"],
)
def test_selection_requires_completed_fit(operation):
    selector = a.Leshy(lgb.LGBMRegressor())
    args = (pd.DataFrame({"x": [1.0]}),) if operation == "transform" else ()
    with pytest.raises(NotFittedError):
        getattr(selector, operation)(*args)


@pytest.mark.parametrize("depth", [-1, None, 3, 10000])
def test_auto_tree_count_handles_unlimited_and_large_depth(depth):
    selector = a.Leshy(lgb.LGBMRegressor(max_depth=depth), n_estimators="auto")
    selector.estimator_ = clone(selector.estimator)
    selector.is_cat = False
    effective = 10 if depth is None or depth < 0 else depth
    expected = max(1, int((8 / (np.sqrt(8) * effective)) * 100))
    assert selector._get_tree_num(4) == expected


def test_auto_tree_count_default_lightgbm_runs():
    X, y, w = data()
    selector = a.Leshy(
        lgb.LGBMRegressor(verbosity=-1, n_jobs=1, min_child_samples=5),
        n_estimators="auto",
        max_iter=2,
        random_state=2,
        importance="native",
    ).fit(X, y, sample_weight=w)
    assert selector.estimator_.n_estimators > 0
    assert selector.n_iter_ == 2
    assert selector.n_features_in_ == X.shape[1]


def test_duplicate_names_fail_before_training():
    X, y, _ = data()
    X.columns = ["same"] * X.shape[1]
    with pytest.raises(ValueError, match="unique"):
        a.Leshy(lgb.LGBMRegressor()).fit(X, y)


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 2.0])
def test_hit_counting_preserves_importances_and_historical_signed_rule(threshold):
    real = np.array([np.nan, -0.5, 0.0, 2.0, 3.0])
    before = real.copy()
    hits = np.zeros(5, dtype=int)
    result = a.Leshy._assign_hits(hits, (real, np.array([threshold])), threshold)
    assert result is hits
    assert_array_equal(real, before)
    assert_array_equal(hits, np.where(np.isnan(real), 0, real) > threshold)


@pytest.mark.parametrize("two_step", [False, True])
def test_decisions_match_independent_binomial_reference(two_step):
    selector = a.Leshy(lgb.LGBMRegressor(), two_step=two_step)
    decisions = np.array([0, 0, 0, 1, -1])
    hits = np.array([40, 0, 20, 40, 0])
    active = np.flatnonzero(decisions >= 0)
    acceptance = binom.sf(hits[active] - 1, 40, 0.5)
    rejection = binom.cdf(hits[active], 40, 0.5)
    threshold = selector.alpha / (40 if two_step else len(decisions))
    accept, reject = acceptance <= threshold, rejection <= threshold
    if two_step:
        accept &= fdrcorrection(acceptance, alpha=selector.alpha)[0]
        reject &= fdrcorrection(rejection, alpha=selector.alpha)[0]
    expected = decisions.copy()
    expected[active[(decisions[active] == 0) & accept]] = 1
    expected[active[(decisions[active] == 0) & reject]] = -1
    assert_array_equal(selector._do_tests(decisions, hits, 40), expected)
    assert_array_equal(expected, [1, -1, 0, 1, -1])


def test_two_step_fdr_keeps_a_candidate_that_passes_the_iteration_cutoff_undecided():
    selector = a.Leshy(lgb.LGBMRegressor(), two_step=True)
    hits = np.full(100, 5)
    hits[0] = 10
    decisions = np.zeros(100, dtype=int)
    assert_array_equal(selector._do_tests(decisions, hits, 10), np.zeros_like(decisions))


@pytest.mark.parametrize("early_stop", [False, True])
def test_iteration_count_reporting_and_nan_history(monkeypatch, capsys, early_stop):
    X, y, _ = data()
    real_vectors = []

    def importance(self, frame, target, weight, decisions):
        real = np.where(decisions >= 0, 3.0, np.nan)
        real_vectors.append(real)
        return real, np.array([1.0])

    def decisions(self, state, hits, iteration):
        state[0] = -1
        if early_stop:
            state[1:] = 1
        return state

    monkeypatch.setattr(a.Leshy, "_add_shadows_get_imps", importance)
    monkeypatch.setattr(a.Leshy, "_do_tests", decisions)
    selector = a.Leshy(
        lgb.LGBMRegressor(verbosity=-1),
        max_iter=4,
        n_estimators=2,
        random_state=1,
        verbose=1,
    ).fit(X, y)
    expected_count = 1 if early_stop else 4
    assert selector.n_iter_ == len(selector.imp_real_hist) == expected_count
    assert f"Iteration: {expected_count} / 4" in capsys.readouterr().out
    assert_array_equal(selector.imp_real_hist, real_vectors)
    if not early_stop:
        assert np.isnan(selector.imp_real_hist[1:, 0]).all()
        assert all(np.isnan(real[0]) for real in real_vectors[1:])
        assert not selector.get_support().any()
        check_is_fitted(selector)
    selector.set_params(max_iter=1)
    with pytest.raises(ValueError, match="max_iter"):
        selector.fit(X, y)
    with pytest.raises(NotFittedError):
        selector.transform(X)
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)


def test_summary_failure_leaves_leshy_unfitted(monkeypatch):
    X, y, _ = data()
    selector = a.Leshy(
        lgb.LGBMRegressor(verbosity=-1, n_jobs=1),
        n_estimators=4,
        max_iter=2,
        importance="native",
        random_state=1,
    ).fit(X, y)

    def closed_pipe(*args, **kwargs):
        raise BrokenPipeError("closed selector output")

    monkeypatch.setattr(a, "print", closed_pipe, raising=False)
    selector.set_params(verbose=1)
    with pytest.raises(BrokenPipeError, match="closed selector output"):
        selector.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)
    with pytest.raises(NotFittedError):
        selector.transform(X)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("rf", [False, True])
def test_shared_native_shap_matches_legacy_reduction(monkeypatch, task, rf):
    X, y, w = data()
    before = X.copy(deep=True)
    if task == "binary":
        y = (y > 0).astype(int)
    elif task == "multiclass":
        y = pd.Series(np.digitize(y, [-1, 1]))
    cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    estimator = cls(
        n_estimators=8,
        max_depth=3,
        num_leaves=7,
        min_child_samples=5,
        random_state=3,
        n_jobs=1,
        verbosity=-1,
        boosting_type="rf" if rf else "gbdt",
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
    )
    reference, valid, _, _ = a._split_fit_estimator(clone(estimator), X, y, w)
    values = shap.TreeExplainer(reference).shap_values(valid)
    if isinstance(values, list):
        if task == "binary":
            expected = np.abs(values[1]).mean(axis=0)
        elif task == "regression":
            expected = np.abs(values[0]).mean(axis=0)
        else:
            expected = np.abs(np.stack(values, axis=2)).sum(axis=2).mean(axis=0)
    else:
        expected = np.abs(values)
        if expected.ndim == 3:
            expected = expected.sum(axis=2)
        expected = expected.mean(axis=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("LightGBM must use native contributions")

    monkeypatch.setattr(a.shap, "TreeExplainer", forbidden)
    assert_allclose(
        a._get_shap_imp(estimator, X, y, w), expected, rtol=1e-12, atol=1e-14
    )
    # Both callers must reach the same native backend, without optional imports.
    for selector in (
        a.Leshy(estimator, n_estimators=8, max_iter=2, random_state=3),
        a.BoostAGroota(estimator, iters=1, max_rounds=1, random_state=3),
    ):
        selector.fit(X, y, sample_weight=w)
        assert len(selector.support_) == X.shape[1]
    assert_frame_equal(X, before)
