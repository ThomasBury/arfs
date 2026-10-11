"""Unit 1B review evidence against the frozen wheel, not a package test gate.

Strict xfails assert intended behavior. Tests using monkeypatch substitute model
outputs or record arguments to isolate selector logic; they do not repair ARFS.
"""

from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.stats import binom
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import fdrcorrection

import arfs.feature_selection.allrelevant as a


def data():
    rng = np.random.RandomState(31)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    return X, pd.Series(3 * X.a + rng.normal(size=len(X)))


def model():
    return lgb.LGBMRegressor(n_estimators=15, num_leaves=7,
                            min_child_samples=8, n_jobs=1, verbosity=-1,
                            random_state=0)


def test_fdr_matches_independent_library():
    p = np.array([0.8, 0.001, 0.03, 0.002, 0.5, 0.01])
    actual, reference = a.Leshy._fdrcorrection(p), fdrcorrection(p)
    assert_array_equal(actual[0], reference[0])
    assert_allclose(actual[1], reference[1])


@pytest.mark.parametrize("two_step", [True, False])
def test_binomial_decisions_reference(two_step):
    selector = a.Leshy(model(), two_step=two_step)
    hits = np.array([20, 0, 10, 17, 3])
    accept, reject = binom.sf(hits - 1, 20, 0.5), binom.cdf(hits, 20, 0.5)
    if two_step:
        accept = fdrcorrection(accept)[0] & (accept <= 0.05 / 20)
        reject = fdrcorrection(reject)[0] & (reject <= 0.05 / 20)
    else:
        accept, reject = accept <= 0.05 / 5, reject <= 0.05 / 5
    expected = np.zeros(5)
    expected[accept], expected[reject] = 1, -1
    assert_array_equal(selector._do_tests(np.zeros(5), hits, 20), expected)


def test_shadow_preserves_values_index_and_input():
    X = pd.DataFrame({"a": np.arange(20.), "cat": pd.Categorical(["x", "y"] * 10)},
                     index=np.arange(20) + 100)
    before = X.copy(deep=True)
    np.random.seed(0)
    augmented, names = a._create_shadow(X)
    assert_frame_equal(X, before)
    assert_array_equal(augmented.index, X.index)
    for original, shadow in zip(X, names):
        assert_array_equal(np.sort(augmented[shadow]), np.sort(X[original]))
        assert augmented[shadow].dtype == X[original].dtype


@pytest.mark.xfail(strict=True, reason="F1: ranking promotes every tentative feature despite keep_weak=False")
def test_leshy_does_not_select_only_tentative_features(monkeypatch):
    X, y = data()
    monkeypatch.setattr(a, "_get_imp", lambda est, frame, *args, **kw:
                        np.r_[np.full(3, 2.), np.ones(frame.shape[1] - 3)])
    selector = a.Leshy(model(), importance="native", max_iter=2,
                       n_estimators=2, random_state=0, keep_weak=False).fit(X, y)
    assert selector.n_features_ == 0
    assert selector.support_weak_.all()
    assert not selector.support_.any()


@pytest.mark.xfail(strict=True, reason="F2: real first iteration is discarded as an obsolete zero row")
def test_tentative_rule_uses_all_recorded_iterations():
    assert_array_equal(a._select_tentative(np.array([0]),
                       np.array([[9.], [4.]]), [5., 5.]), [0])


@pytest.mark.xfail(strict=True, reason="F3: held-out weights are omitted from permutation scoring")
def test_permutation_importance_matches_weighted_reference():
    X, y = data()
    weights = np.where(X.b > 0, 30., 1.)
    fitted, X_test, y_test, w_test = a._split_fit_estimator(model(), X.copy(), y, weights)
    expected = permutation_importance(fitted, X_test, y_test, sample_weight=w_test,
                                      n_repeats=5, random_state=42, n_jobs=1).importances_mean
    unweighted = permutation_importance(fitted, X_test, y_test,
                                       n_repeats=5, random_state=42, n_jobs=1).importances_mean
    actual = a._get_perm_imp(model(), X.copy(), y, weights)
    assert not np.allclose(expected, unweighted)
    assert_allclose(actual, unweighted)
    assert_allclose(actual, expected)


@pytest.mark.xfail(strict=True, reason="F4: dividing signed importances by a negative total reverses their order")
def test_boost_negative_importances_do_not_select_worst_feature(monkeypatch):
    X, y = data()
    X = X[["a", "b"]]
    monkeypatch.setattr(a, "_get_perm_imp", lambda *args, **kw: np.array([-.1, -.5, -.2, -.2]))
    _, selected, _, _ = a._reduce_vars_sklearn(
        X, y, model(), 1, 1, 1, .1, True, None, "pimp", None)
    assert "b" not in selected.values


@pytest.mark.parametrize("selector_name", ["boost", "groot"])
@pytest.mark.xfail(strict=True, reason="F5: fit overwrites the caller's categorical values with integer codes")
def test_fit_preserves_categorical_input(selector_name):
    X, y = data()
    X["cat"] = pd.Categorical(np.where(X.b > 0, "yes", "no"))
    original = X.copy(deep=True)
    selector = (a.BoostAGroota(model(), iters=1, max_rounds=1) if selector_name == "boost"
                else a.GrootCV(objective="l2", n_folds=2, n_iter=1, n_jobs=1))
    selector.fit(X, y)
    assert_frame_equal(X, original)


@pytest.mark.xfail(strict=True, reason="F6: Boost categorical metadata omits shadows and disappears after reduction")
def test_boost_preserves_categorical_roles_across_rounds(monkeypatch):
    X, y = data()
    X = X[["a", "b"]]
    X["a"] = pd.Categorical(np.where(X.a > 0, "yes", "no"))
    recorded = []

    def importance(estimator, frame, *args, cat_feature=None, **kwargs):
        recorded.append((list(frame.columns), cat_feature))
        return np.array([10., 0., 1., 1.]) if frame.shape[1] == 4 else np.array([10., 1.])

    monkeypatch.setattr(a, "_get_shap_imp", importance)
    a.BoostAGroota(model(), iters=1, max_rounds=2, cutoff=1).fit(X, y)
    assert len(recorded) == 2
    assert recorded == [(["a", "b", "ShadowVar1", "ShadowVar2"], [0, 2]),
                        (["a", "ShadowVar1"], [0, 1])]


@pytest.mark.xfail(strict=True, reason="F6: Groot loses categorical metadata before fitting every fold")
def test_groot_preserves_categorical_roles(monkeypatch):
    X, y = data()
    X["a"] = pd.Categorical(np.where(X.a > 0, "yes", "no"))
    recorded = []

    def train(X_train, *args, category_cols=None, **kwargs):
        recorded.append(category_cols)
        return SimpleNamespace(best_iteration=1), np.ones(X_train.shape), 1

    monkeypatch.setattr(a, "_train_lgb_model", train)
    a.GrootCV(objective="l2", n_folds=2, n_iter=1).fit(X, y)
    assert recorded == [["a", "ShadowVar1"], ["a", "ShadowVar1"]]


@pytest.mark.xfail(strict=True, reason="F6: Leshy converts numeric categorical shadows to numeric dtype")
def test_leshy_numeric_category_shadow_matches_original_role(monkeypatch):
    X = pd.DataFrame({"category": pd.Categorical([10, 20, 30] * 10)})
    recorded = []

    def importance(estimator, frame, *args, **kwargs):
        recorded.append(frame.dtypes)
        return np.ones(frame.shape[1])

    monkeypatch.setattr(a, "_get_imp", importance)
    selector = a.Leshy(model(), importance="native", random_state=np.random.RandomState(0))
    selector._add_shadows_get_imps(X, pd.Series(np.arange(30.)), None, np.zeros(1))
    assert all(dtype == X.category.dtype for dtype in recorded[0])


@pytest.mark.parametrize("objective", ["binary", "multiclass"])
@pytest.mark.xfail(strict=True, reason="F7: auto-resolved multiclass objective is not used in SHAP aggregation")
def test_groot_multiclass_objective_aliases(objective):
    X, _ = data()
    y = pd.Series(np.tile([0, 1, 2], 40))
    a.GrootCV(objective=objective, n_folds=2, n_iter=1, n_jobs=1).fit(X, y)


def test_groot_explicit_softmax_runs():
    X, _ = data()
    a.GrootCV(objective="softmax", n_folds=2, n_iter=1, n_jobs=1).fit(
        X, pd.Series(np.tile([0, 1, 2], 40)))


@pytest.mark.xfail(strict=True, reason="F8: Groot averages fold values and includes their mean in threshold")
def test_groot_documented_max_of_shadow_medians(monkeypatch):
    X, y = data()
    X = X[["a", "b"]]
    # Each column already sums to one, so normalization changes nothing.
    values = iter([np.array([.57, .03, .35, .05]),
                   np.array([.57, .03, .05, .35]),
                   np.array([.16, .09, .70, .05])])

    def train(frame, *args, **kwargs):
        return SimpleNamespace(best_iteration=1), np.tile(next(values), (len(frame), 1)), 1

    monkeypatch.setattr(a, "_train_lgb_model", train)
    selector = a.GrootCV(objective="l2", n_folds=3, n_iter=1).fit(X, y)
    assert selector.sha_cutoff == pytest.approx(.35)
    assert selector.cv_df.set_index("feature").loc["ShadowVar1", "Med"] == pytest.approx(.35)
    assert list(selector.selected_features_) == ["a"]


@pytest.mark.xfail(strict=True, reason="F9: Boost averages maxima over iterations along the wrong axis")
def test_boost_documented_max_shadow_cutoff(monkeypatch):
    X, y = data()
    X = X[["a", "b"]]
    monkeypatch.setattr(a, "_get_shap_imp", lambda *args, **kw: np.array([.3, .1, .5, .1]))
    _, selected, _, threshold = a._reduce_vars_sklearn(
        X, y, model(), 1, 1, 1, .1, True, None, "shap", None)
    assert threshold == pytest.approx(.5)
    assert "a" not in selected.values


@pytest.mark.parametrize("column", ["ShadowVar1", "ShadowVar2"])
@pytest.mark.xfail(strict=True, reason="F10: user column names collide with the generated shadow namespace")
def test_shadow_names_do_not_change_real_feature_identity(column):
    X = pd.DataFrame({column: np.arange(20.), "b": np.arange(20.) + 100})
    augmented, shadows = a._create_shadow(X)
    assert augmented.columns.is_unique
    assert set(X).isdisjoint(shadows)
    assert_frame_equal(augmented.iloc[:, :X.shape[1]], X)


def test_fold_splitting_keeps_weights_aligned():
    X, y = data()
    X.index = y.index = np.arange(len(X)) + 200
    weights = pd.Series(np.arange(len(X)) + 1., index=X.index)
    tr, val = np.array([3, 1, 8]), np.array([2, 9])
    Xtr, Xval, ytr, yval, wtr, wval = a._split_data(X, y, tr, val, weights)
    assert_array_equal(Xtr.index, ytr.index)
    assert_array_equal(Xval.index, yval.index)
    assert_array_equal(wtr, weights.iloc[tr])
    assert_array_equal(wval, weights.iloc[val])


def test_shap_multiclass_aggregation_reference():
    X = pd.DataFrame(np.zeros((2, 3)), columns=["a", "b", "c"])
    values = np.arange(18.).reshape(2, 3, 3) - 8
    actual = dict(a._compute_importance(X, values, {"num_class": 3}, "softmax", False))
    assert_allclose([actual[col] for col in X], np.abs(values).sum(axis=2).mean(axis=0))


def test_leshy_zero_information_rejects_all():
    X = pd.DataFrame(np.ones((40, 3)), columns=["a", "b", "c"])
    selector = a.Leshy(RandomForestRegressor(max_depth=2), n_estimators=5,
                       max_iter=15, importance="native", random_state=0).fit(X, np.arange(40.))
    assert not selector.support_.any()


@pytest.mark.xfail(strict=True, reason="F11: missing target values are silently changed into class or response zero")
def test_leshy_rejects_missing_response():
    X, y = data()
    y.iloc[0] = np.nan
    with pytest.raises(ValueError):
        a.Leshy(model(), n_estimators=2, max_iter=2, importance="native", random_state=0).fit(X, y)
