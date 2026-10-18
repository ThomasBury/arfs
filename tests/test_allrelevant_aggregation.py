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


@pytest.mark.parametrize("labels", [(0, 1), (1, 0), (3, 7)])
@pytest.mark.parametrize("count,expected", [(8, False), (9, True)])
def test_groot_imbalance_uses_largest_class_and_strict_threshold(
    labels, count, expected
):
    X = pd.DataFrame({"a": np.arange(10)})
    y = pd.Series([labels[0]] * count + [labels[1]] * (10 - count))
    params = a._set_lgb_parameters(X, y, "binary", False, True, 1)
    assert params.get("is_unbalance", False) == expected


@pytest.mark.parametrize("majority", [0, 1])
def test_groot_binary_majority_reaches_native_training(majority, monkeypatch):
    X, y = data(2)
    y[:] = majority
    y.iloc[::10] = 1 - majority
    seen = []
    train = a._train_lgb_model

    def checked_train(*args, **kwargs):
        result = train(*args, **kwargs)
        seen.append(result[0].params.get("is_unbalance", False))
        return result

    monkeypatch.setattr(a, "_train_lgb_model", checked_train)
    a.GrootCV(
        objective="binary",
        n_folds=2,
        n_iter=1,
        n_jobs=1,
        lgbm_params={"min_data_in_leaf": 5, "seed": 42},
    ).fit(X, y)
    assert seen == [True, True]


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


@pytest.mark.parametrize("n_features", [1, 4])
@pytest.mark.parametrize("layout", ["list", "array"])
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_leshy_shap_keeps_every_feature(monkeypatch, task, layout, n_features):
    X = pd.DataFrame(np.zeros((3, n_features)))
    values = np.arange(1, X.size + 1, dtype=float).reshape(X.shape)
    values[1] *= -2
    if task == "regression":
        estimator = lgb.LGBMRegressor()
        y = pd.Series([0.1, 0.2, 0.3])
        outputs = [values]
        expected = np.abs(values).mean(axis=0)
        array = values
    elif task == "binary":
        estimator = lgb.LGBMClassifier()
        y = pd.Series([0, 1, 0])
        outputs = [-2 * values, 3 * values]
        expected = 3 * np.abs(values).mean(axis=0)
        array = outputs[1]
    else:
        estimator = lgb.LGBMClassifier()
        y = pd.Series([0, 1, 2])
        outputs = [values, -3 * values, 5 * values]
        expected = 9 * np.abs(values).mean(axis=0)
        array = np.stack(outputs, axis=-1)

    monkeypatch.setattr(
        a, "_split_fit_estimator", lambda *args, **kwargs: (estimator, X, y, None)
    )
    explainer = SimpleNamespace(
        shap_values=lambda frame: outputs if layout == "list" else array
    )
    monkeypatch.setattr(a.shap, "TreeExplainer", lambda model: explainer)

    actual = a._get_shap_imp(estimator, X, y)
    assert actual.shape == (n_features,)
    assert_allclose(actual, expected)


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


@pytest.mark.parametrize(
    "scores,support,ranking",
    [
        ([0.6, 0.1, 0.2, 0.1], [True, False], [1, 2]),
        ([0.4, 0.4, 0.1, 0.1], [True, True], [1, 1]),
        ([0.1, 0.1, 0.4, 0.4], [False, False], [2, 2]),
        ([0.0, 0.0, 0.0, 0.0], [False, False], [2, 2]),
    ],
)
@pytest.mark.parametrize("selector_kind", ["boost_shap", "boost_native", "boost_pimp", "groot"])
def test_boost_and_groot_rankings_match_support_and_empty_selection(
    monkeypatch, scores, support, ranking, selector_kind
):
    X, y = data()
    if selector_kind == "groot":
        monkeypatch.setattr(
            a,
            "_train_lgb_model",
            lambda frame, *args, **kwargs: (
                SimpleNamespace(best_iteration=1),
                np.tile(scores, (len(frame), 1)),
                1,
            ),
        )
        selector = a.GrootCV(objective="l2", n_folds=2, n_iter=1, cutoff=1).fit(X, y)
    else:
        imp = selector_kind.removeprefix("boost_")
        helper = {"shap": "_get_shap_imp", "native": "_get_imp", "pimp": "_get_perm_imp"}[
            imp
        ]
        monkeypatch.setattr(a, helper, lambda *args, **kwargs: np.array(scores))
        selector = a.BoostAGroota(
            lgb.LGBMRegressor(), importance=imp, cutoff=1, iters=1, max_rounds=1
        ).fit(X, y)
    assert_array_equal(selector.support_, support)
    assert_array_equal(selector.ranking_, ranking)
    assert_array_equal(
        selector.feature_names_in_[selector.ranking_ == 1],
        selector.selected_features_,
    )
    assert_array_equal(
        selector.feature_names_in_[selector.ranking_ == 2],
        selector.feature_names_in_[~selector.support_],
    )


@pytest.mark.parametrize("importance", ["shap", "native"])
def test_boost_zero_and_mixed_zero_iterations_remain_finite(monkeypatch, importance):
    X, y = data()
    helper = "_get_shap_imp" if importance == "shap" else "_get_imp"

    # Mixed zero and nonzero iterations keep the zero iteration in the mean.
    mixed = iter([np.zeros(4), np.array([6.0, 2.0, 2.0, 0.0])])
    monkeypatch.setattr(a, helper, lambda *args, **kwargs: next(mixed))
    with np.errstate(divide="raise", invalid="raise"):
        selector = a.BoostAGroota(
            lgb.LGBMRegressor(), importance=importance, cutoff=1, iters=2, max_rounds=1
        ).fit(X, y)
    assert_allclose(selector.sha_cutoff_df.iloc[:, :4], [[0.0, 0.0, 0.0, 0.0], [0.6, 0.2, 0.2, 0.0]])
    assert_allclose(selector.sha_cutoff_df.mean(axis=0), [0.3, 0.1, 0.1, 0.0])
    assert selector.mean_shadow == pytest.approx(0.1)
    assert_array_equal(selector.selected_features_, ["a", "b"])
    assert_array_equal(selector.ranking_, [1, 1])

    # Zero shadow threshold still requires strictly positive real mean.
    monkeypatch.setattr(a, helper, lambda *args, **kwargs: np.array([1.0, 0.0, 0.0, 0.0]))
    with np.errstate(divide="raise", invalid="raise"):
        zero_shadow = a.BoostAGroota(
            lgb.LGBMRegressor(), importance=importance, cutoff=1, iters=1, max_rounds=1
        ).fit(X, y)
    assert zero_shadow.mean_shadow == pytest.approx(0.0)
    assert_array_equal(zero_shadow.selected_features_, ["a"])
    assert_array_equal(zero_shadow.ranking_, [1, 2])


@pytest.mark.parametrize("importance", ["shap", "native"])
def test_real_uninformative_boost_fit_is_finite_and_selects_nothing(importance):
    X, _ = data()
    y = pd.Series(np.zeros(len(X)))
    model = lgb.LGBMRegressor(
        n_estimators=3, num_leaves=4, min_child_samples=5, n_jobs=1, verbosity=-1
    )
    with np.errstate(divide="raise", invalid="raise"):
        selector = a.BoostAGroota(
            model, importance=importance, cutoff=1, iters=2, max_rounds=1
        ).fit(X, y)
    assert np.isfinite(selector.sha_cutoff_df.to_numpy()).all()
    assert selector.mean_shadow == pytest.approx(0.0)
    assert len(selector.selected_features_) == 0
    assert_array_equal(selector.support_, [False, False])
    assert_array_equal(selector.ranking_, [2, 2])


def test_groot_zero_and_mixed_zero_folds_remain_finite_and_retained(monkeypatch):
    X, y = data()

    # Two zero folds and one positive fold: median across 3 folds must be 0.
    three_folds = iter(
        [np.zeros(4), np.zeros(4), np.array([0.8, 0.0, 0.2, 0.0])]
    )
    monkeypatch.setattr(
        a,
        "_train_lgb_model",
        lambda frame, *args, **kwargs: (
            SimpleNamespace(best_iteration=1),
            np.tile(next(three_folds), (len(frame), 1)),
            1,
        ),
    )
    with np.errstate(divide="raise", invalid="raise"):
        selector = a.GrootCV(objective="l2", n_folds=3, n_iter=1, cutoff=1).fit(X, y)
    table = selector.cv_df.set_index("feature")
    assert np.isfinite(table.to_numpy()).all()
    assert_allclose(table["Med"], 0.0)
    assert selector.sha_cutoff == pytest.approx(0.0)
    assert len(selector.selected_features_) == 0
    assert_array_equal(selector.ranking_, [2, 2])

    # One zero fold and one positive fold: median halves the positive fold.
    two_folds = iter([np.zeros(4), np.array([6.0, 2.0, 2.0, 0.0])])
    monkeypatch.setattr(
        a,
        "_train_lgb_model",
        lambda frame, *args, **kwargs: (
            SimpleNamespace(best_iteration=1),
            np.tile(next(two_folds), (len(frame), 1)),
            1,
        ),
    )
    with np.errstate(divide="raise", invalid="raise"):
        mixed = a.GrootCV(objective="l2", n_folds=2, n_iter=1, cutoff=1).fit(X, y)
    mixed_table = mixed.cv_df.set_index("feature")
    assert_allclose(
        mixed_table.loc[["a", "b", "ShadowVar1", "ShadowVar2"], "Med"],
        [0.3, 0.1, 0.1, 0.0],
    )
    assert mixed.sha_cutoff == pytest.approx(0.1)
    assert_array_equal(mixed.selected_features_, ["a", "b"])
    assert_array_equal(mixed.ranking_, [1, 1])

    # Zero shadow median still requires strictly positive real median.
    monkeypatch.setattr(
        a,
        "_train_lgb_model",
        lambda frame, *args, **kwargs: (
            SimpleNamespace(best_iteration=1),
            np.tile([1.0, 0.0, 0.0, 0.0], (len(frame), 1)),
            1,
        ),
    )
    with np.errstate(divide="raise", invalid="raise"):
        zero_shadow = a.GrootCV(objective="l2", n_folds=2, n_iter=1, cutoff=1).fit(X, y)
    assert zero_shadow.sha_cutoff == pytest.approx(0.0)
    assert_array_equal(zero_shadow.selected_features_, ["a"])
    assert_array_equal(zero_shadow.ranking_, [1, 2])


def test_real_uninformative_groot_fit_is_finite_and_selects_nothing():
    X, _ = data()
    y = pd.Series(np.zeros(len(X)))
    with np.errstate(divide="raise", invalid="raise"):
        selector = a.GrootCV(
            objective="l2",
            n_folds=2,
            n_iter=1,
            n_jobs=1,
            lgbm_params={"min_data_in_leaf": 5, "seed": 42},
        ).fit(X, y)
    assert np.isfinite(selector.cv_df.select_dtypes("number").to_numpy()).all()
    assert selector.sha_cutoff == pytest.approx(0.0)
    assert len(selector.selected_features_) == 0
    assert_array_equal(selector.support_, [False, False])
    assert_array_equal(selector.ranking_, [2, 2])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
@pytest.mark.parametrize("importance", ["shap", "native", "pimp"])
def test_boost_rejects_nonfinite_importance_without_swallowing(
    monkeypatch, bad, importance
):
    X, y = data()
    helper = {"shap": "_get_shap_imp", "native": "_get_imp", "pimp": "_get_perm_imp"}[
        importance
    ]
    scores = np.array([bad, 1e308 if bad == 1e308 else 1.0, 0.0, 0.0])
    monkeypatch.setattr(a, helper, lambda *args, **kwargs: scores)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        a.BoostAGroota(
            lgb.LGBMRegressor(), importance=importance, cutoff=1, iters=1, max_rounds=1
        ).fit(X, y)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
def test_groot_and_shared_helpers_reject_nonfinite_importance(monkeypatch, bad):
    X, y = data()
    scores = np.array([bad, 1e308 if bad == 1e308 else 1.0, 0.0, 0.0])
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        a._compute_importance(
            pd.DataFrame(np.zeros((2, 4)), columns=["a", "b", "c", "d"]),
            np.vstack([scores, scores]),
            {},
            "l2",
            False,
        )
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        a._merge_importance_df(
            pd.DataFrame({"feature": ["a", "b", "c", "d"]}),
            list(zip(["a", "b", "c", "d"], scores)),
            0,
            2,
            ["a", "b", "c", "d"],
        )
    monkeypatch.setattr(
        a,
        "_train_lgb_model",
        lambda frame, *args, **kwargs: (
            SimpleNamespace(best_iteration=1),
            np.tile(scores, (len(frame), 1)),
            1,
        ),
    )
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        a.GrootCV(objective="l2", n_folds=2, n_iter=1, cutoff=1).fit(X, y)

    if bad != 1e308:
        monkeypatch.setattr(
            a,
            "_split_fit_estimator",
            lambda *args, **kwargs: (
                lgb.LGBMRegressor(),
                X,
                y,
                None,
            ),
        )
        monkeypatch.setattr(
            a.shap,
            "TreeExplainer",
            lambda model: SimpleNamespace(
                shap_values=lambda frame: np.full(frame.shape, bad)
            ),
        )
        with pytest.raises(ValueError, match="finite"):
            a._get_shap_imp(lgb.LGBMRegressor(), X, y)
        monkeypatch.setattr(
            a,
            "permutation_importance",
            lambda *args, **kwargs: SimpleNamespace(
                importances_mean=np.full(X.shape[1], bad)
            ),
        )
        with pytest.raises(ValueError, match="finite"):
            a._get_perm_imp(lgb.LGBMRegressor(), X, y, None)
