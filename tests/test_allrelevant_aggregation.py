"""Exact shadow aggregation and objective-independent SHAP output axes."""

from types import SimpleNamespace

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

import arfs.feature_selection.allrelevant as a


def data(n_classes=3):
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    y = pd.Series(np.tile(np.arange(n_classes), 120 // n_classes))
    return X, y


@pytest.mark.parametrize(
    "objective",
    [
        "binary",
        "multiclass",
        "softmax",
        "multiclassova",
        "multiclass_ova",
        "ova",
        "ovr",
    ],
)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_real_classification_objectives(objective, n_classes, monkeypatch):
    X, y = data(n_classes)
    before = X.copy(deep=True)
    params = {"min_data_in_leaf": 5, "seed": 42}
    original = params.copy()
    resolved = []
    train = a._train_lgb_model

    def checked_train(*args, **kwargs):
        result = train(*args, **kwargs)
        resolved.append(kwargs["objective"])
        values = result[1]
        assert values.shape[:2] == args[0].shape
        assert values.ndim == (2 if kwargs["objective"] == "binary" else 3)
        return result

    monkeypatch.setattr(a, "_train_lgb_model", checked_train)
    selector = a.GrootCV(
        objective=objective, n_folds=2, n_iter=1, n_jobs=1, lgbm_params=params
    ).fit(X, y)
    assert len(resolved) == 2
    assert params == original
    assert np.isfinite(selector.cv_df.filter(like="fscore")).all().all()
    assert np.isfinite(selector.sha_cutoff)
    assert_array_equal(selector.support_, X.columns.isin(selector.selected_features_))
    assert_frame_equal(selector.transform(X), X[selector.selected_features_])
    assert_frame_equal(X, before)


def test_objective_from_params_and_explicit_override():
    X, y = data()
    params = {"objective": "multiclass", "seed": 42}
    original = params.copy()
    from_params = a._set_lgb_parameters(X, y, None, False, True, 1, params)
    assert from_params["objective"] == "softmax"
    assert from_params["num_class"] == 3
    overridden = a._set_lgb_parameters(X, y, "l2", False, True, 1, params)
    assert overridden["objective"] == "l2"
    assert params == original
    a.GrootCV(n_folds=2, n_iter=1, n_jobs=1, lgbm_params=params).fit(X, y)
    assert params == original


@pytest.mark.parametrize("fastshap", [False, True])
@pytest.mark.parametrize("layout", ["single", "array", "list"])
def test_shap_output_axes(layout, fastshap):
    X = pd.DataFrame(np.zeros((2, 3)), columns=["a", "b", "c"])
    values = np.arange(18.0).reshape(2, 3, 3) - 8
    if layout == "single":
        values = values[:, :, 0]
        expected = np.abs(values).mean(axis=0)
    else:
        expected = np.abs(values).sum(axis=2).mean(axis=0)
        if layout == "list":
            values = [values[:, :, k] for k in range(3)]
    actual = dict(a._compute_importance(X, values, {}, "binary", fastshap))
    assert_allclose([actual[col] for col in X], expected)


@pytest.mark.parametrize("shape", [(2,), (3, 3), (2, 4), (2, 3, 2, 2)])
def test_shap_mismatched_axes_raise(shape):
    X = pd.DataFrame(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="SHAP.*shape"):
        a._compute_importance(X, np.ones(shape), {}, "l2", False)


@pytest.mark.parametrize("cutoff", [1, 2])
@pytest.mark.parametrize("repeats", [1, 2])
def test_groot_medians_use_only_fold_columns(monkeypatch, cutoff, repeats):
    X, _ = data()
    y = pd.Series(X.a)
    # Column totals are one; the middle shadow switches between folds.
    scores = np.array(
        [[0.57, 0.03, 0.35, 0.05], [0.57, 0.03, 0.05, 0.35], [0.16, 0.09, 0.70, 0.05]]
    )
    values = iter(list(scores) * repeats)

    def train(frame, *args, **kwargs):
        return (
            SimpleNamespace(best_iteration=1),
            np.tile(next(values), (len(frame), 1)),
            1,
        )

    monkeypatch.setattr(a, "_train_lgb_model", train)
    selector = a.GrootCV(objective="l2", n_folds=3, n_iter=repeats, cutoff=cutoff).fit(
        X, y
    )
    table = selector.cv_df.set_index("feature")
    assert_allclose(
        table.loc[["a", "b", "ShadowVar1", "ShadowVar2"], "Med"],
        [0.57, 0.03, 0.35, 0.05],
    )
    assert selector.sha_cutoff == pytest.approx(0.35 / cutoff)
    assert_array_equal(selector.selected_features_, ["a"])
    assert table.filter(like="fscore").shape[1] == 3 * repeats
    assert_frame_equal(selector.transform(X), X[["a"]])


def test_groot_ranking_plot_and_equality_use_medians(monkeypatch):
    X, y = data()
    # Means rank b first (.4 > .3); medians rank a first (.4 > .2).
    values = iter(
        [
            np.array([0.4, 0.2, 0.4, 0.0]),
            np.array([0.4, 0.2, 0.0, 0.4]),
            np.array([0.1, 0.8, 0.05, 0.05]),
        ]
    )

    def train(frame, *args, **kwargs):
        return (
            SimpleNamespace(best_iteration=1),
            np.tile(next(values), (len(frame), 1)),
            1,
        )

    monkeypatch.setattr(a, "_train_lgb_model", train)
    selector = a.GrootCV(objective="l2", n_folds=3, n_iter=1, cutoff=0.125).fit(X, y)
    assert selector.sha_cutoff == pytest.approx(0.4)
    assert_array_equal(selector.selected_features_, ["a"])
    assert selector.ranking_absolutes_ == ["a", "b"]
    fig = selector.plot_importance()
    try:
        assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["b", "a"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("importance", ["shap", "native", "pimp"])
@pytest.mark.parametrize("cutoff", [1, 2])
def test_boost_maximum_shadow_mean_and_equality(monkeypatch, importance, cutoff):
    X, y = data()
    scores = iter([np.array([0.3, 0.2, 0.5, 0.0]), np.array([0.3, 0.2, 0.1, 0.4])])
    helper = {"shap": "_get_shap_imp", "native": "_get_imp", "pimp": "_get_perm_imp"}[
        importance
    ]
    monkeypatch.setattr(a, helper, lambda *args, **kwargs: next(scores))
    selector = a.BoostAGroota(
        lgb.LGBMRegressor(), importance=importance, cutoff=cutoff, iters=2, max_rounds=1
    ).fit(X, y)
    assert selector.mean_shadow == pytest.approx(0.3 / cutoff)
    assert_allclose(selector.sha_cutoff_df.mean(axis=0), [0.3, 0.2, 0.3, 0.2])
    expected = ["a"] if cutoff == 1 else ["a", "b"]
    assert_array_equal(selector.selected_features_, expected)
    assert_frame_equal(selector.transform(X), X[expected])


def test_boost_single_iteration_strongest_shadow(monkeypatch):
    X, y = data()
    monkeypatch.setattr(
        a, "_get_shap_imp", lambda *args, **kwargs: np.array([0.3, 0.1, 0.5, 0.1])
    )
    _, kept, _, threshold = a._reduce_vars_sklearn(
        X, y, lgb.LGBMRegressor(), 1, 1, 1, 0.1, True, None, "shap", None
    )
    assert threshold == pytest.approx(0.5)
    assert kept.empty
