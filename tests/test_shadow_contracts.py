"""R07 regressions for owned encoding and faithful categorical shadows."""

import sys
from contextlib import nullcontext

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from arfs.feature_selection import allrelevant as ar


def test_fastshap_missing_backend_fails_before_cloning(monkeypatch):
    monkeypatch.setitem(sys.modules, "fasttreeshap", None)

    def forbidden(*args, **kwargs):
        pytest.fail("missing backend must fail before cloning or training")

    monkeypatch.setattr(ar, "clone", forbidden)
    X, y = data()
    with pytest.raises(ImportError, match="fasttreeshap is not installed"):
        ar._get_shap_imp_fast(model(), X, y)


@pytest.fixture(
    params=[False, True] if int(pd.__version__.split(".")[0]) < 3 else [True]
)
def copy_on_write(request):
    context = (
        pd.option_context("mode.copy_on_write", request.param)
        if int(pd.__version__.split(".")[0]) < 3
        else nullcontext()
    )
    with context:
        yield


def model():
    return lgb.LGBMRegressor(
        n_estimators=5, num_leaves=5, min_child_samples=5, n_jobs=1, verbosity=-1
    )


def data(dtype="category"):
    rng = np.random.default_rng(42)
    values = [20, 10, None, 30] * 20
    if dtype not in ["category", object]:
        values = ["b", "a", None, "c"] * 20
    X = pd.DataFrame(
        {"cat": pd.Series(values, dtype=dtype), "num": rng.normal(size=80)}
    )
    X.index = pd.Index(np.arange(80) * 3 + 20, name="row")
    y = pd.Series(3 * X.num + rng.normal(size=80), index=X.index, name="response")
    return X, y


@pytest.mark.parametrize("selector_name", ["leshy", "boost", "groot"])
@pytest.mark.parametrize("dtype", ["category", object, "string", None])
def test_real_fits_preserve_input_and_transform_values(
    selector_name, dtype, copy_on_write
):
    parent, y = data(dtype)
    X = parent[["cat", "num"]]
    before = parent.copy(deep=True)
    weights = np.linspace(1.0, 2.0, len(X))
    if selector_name == "leshy":
        selector = ar.Leshy(
            model(), importance="native", n_estimators=5, max_iter=3, random_state=0
        )
    elif selector_name == "boost":
        selector = ar.BoostAGroota(
            model(), importance="shap", iters=1, max_rounds=2, silent=True
        )
    else:
        selector = ar.GrootCV(
            objective="l2",
            n_folds=2,
            n_iter=1,
            n_jobs=1,
            lgbm_params={"num_iterations": 5, "min_data_in_leaf": 5},
        )
    assert selector.fit(X, y, sample_weight=weights) is selector
    assert_frame_equal(X, before)
    assert_frame_equal(parent, before)
    assert_frame_equal(selector.transform(X), before[selector.selected_features_])


@pytest.mark.parametrize(
    "importance", [ar._get_imp, ar._get_shap_imp, ar._get_perm_imp]
)
def test_shared_importance_encoding_owns_its_frame(importance):
    X, y = data("string")
    before = X.copy(deep=True)
    result = importance(model(), X, y, sample_weight=None)
    assert result.shape == (2,) and np.isfinite(result).all()
    assert_frame_equal(X, before)


@pytest.mark.parametrize(
    "columns",
    [
        ["ShadowVar1", "b"],
        ["ShadowVar2", "b"],
        ["ShadowVar1", "ShadowVar1_", "ShadowVar1__", "ShadowVar2"],
        [0, "ShadowVar1"],
    ],
)
def test_shadow_names_preserve_real_column_identity(columns):
    X = pd.DataFrame(np.arange(20 * len(columns)).reshape(20, -1), columns=columns)
    X.index = pd.Index([3, 1, 3, 5] * 5, name="row")
    before = X.copy(deep=True)
    augmented, shadows = ar._create_shadow(X)
    assert augmented.shape == (len(X), 2 * len(columns))
    assert augmented.columns.tolist() == columns + shadows
    assert augmented.columns.is_unique and set(columns).isdisjoint(shadows)
    assert_frame_equal(augmented.iloc[:, : len(columns)], before)
    for original, shadow in zip(columns, shadows):
        assert_series_equal(
            augmented[shadow].value_counts().sort_index(),
            X[original].value_counts().sort_index(),
            check_names=False,
        )
    assert_frame_equal(X, before)


@pytest.mark.parametrize("dtype", ["category", "string", "boolean"])
@pytest.mark.parametrize("n_columns", [1, 3])
def test_leshy_repeated_shadows_keep_dtype_values_and_unique_names(
    dtype, n_columns, monkeypatch
):
    values = [10, 20, None, 30] * 5
    if dtype == "string":
        values = ["nan", "None", None, "a"] * 5
    elif dtype == "boolean":
        values = [True, False, None, True] * 5
    X = pd.DataFrame(
        {f"Shadow_{i}": pd.Series(values, dtype=dtype) for i in range(n_columns)}
    )
    if dtype == "category":
        for col in X:
            X[col] = X[col].cat.set_categories([30, 20, 10, 99], ordered=True)
    before = X.copy(deep=True)
    recorded = []

    def importance(estimator, frame, *args, **kwargs):
        recorded.append(frame.copy(deep=True))
        return np.arange(frame.shape[1], dtype=float)

    monkeypatch.setattr(ar, "_get_imp", importance)
    selector = ar.Leshy(
        model(), importance="native", random_state=np.random.RandomState(0)
    )
    real, shadows = selector._add_shadows_get_imps(
        X, pd.Series(np.arange(len(X))), None, np.zeros(n_columns)
    )
    frame = recorded[0]
    assert frame.columns.is_unique
    assert_frame_equal(frame.iloc[:, :n_columns], before)
    assert_array_equal(real, np.arange(n_columns))
    assert_allclose(shadows, np.arange(n_columns, frame.shape[1]))
    assert len(shadows) >= 5
    for i in range(n_columns, frame.shape[1]):
        assert frame.iloc[:, i].dtype == X.iloc[:, 0].dtype
        assert_series_equal(
            frame.iloc[:, i].value_counts(dropna=False).sort_index(),
            X.iloc[:, 0].value_counts(dropna=False).sort_index(),
            check_names=False,
        )
    assert_frame_equal(X, before)


def test_boost_categorical_roles_survive_reordering_and_category_removal(monkeypatch):
    X, y = data()
    X = X[["num", "cat"]]
    X["discard"] = np.arange(len(X))
    X["ShadowVar1"] = pd.Series(["x", "y"] * 40, index=X.index, dtype="string")
    before = X.copy(deep=True)
    recorded = []

    def importance(estimator, frame, *args, cat_feature=None, **kwargs):
        recorded.append((frame.copy(deep=True), list(cat_feature)))
        scores = {8: [10.0, 20.0, 0.0, 30.0], 6: [0.0, 0.0, 30.0], 2: [30.0]}
        return np.r_[scores[frame.shape[1]], np.ones(frame.shape[1] // 2)]

    monkeypatch.setattr(ar, "_get_shap_imp", importance)
    selector = ar.BoostAGroota(
        model(), iters=1, max_rounds=3, cutoff=1, silent=True
    ).fit(X, y)
    assert [roles for _, roles in recorded] == [[1, 3, 5, 7], [1, 2, 4, 5], [0, 1]]
    assert [frame.columns[: frame.shape[1] // 2].tolist() for frame, _ in recorded] == [
        ["num", "cat", "discard", "ShadowVar1"],
        ["num", "cat", "ShadowVar1"],
        ["ShadowVar1"],
    ]
    for frame, roles in recorded:
        n_real = frame.shape[1] // 2
        assert frame.columns.is_unique
        for original in roles[: len(roles) // 2]:
            shadow = original + n_real
            assert_array_equal(
                np.sort(frame.iloc[:, original]), np.sort(frame.iloc[:, shadow])
            )
    assert_array_equal(selector.selected_features_, ["ShadowVar1"])
    assert_frame_equal(X, before)


def test_boost_drops_category_metadata_when_only_numeric_features_remain(monkeypatch):
    X, y = data()
    recorded = []

    def importance(estimator, frame, *args, cat_feature=None, **kwargs):
        recorded.append(cat_feature)
        return (
            np.array([0.0, 10.0, 1.0, 1.0])
            if frame.shape[1] == 4
            else np.array([10.0, 1.0])
        )

    monkeypatch.setattr(ar, "_get_shap_imp", importance)
    ar.BoostAGroota(model(), iters=1, max_rounds=2, cutoff=1).fit(X, y)
    assert recorded == [[0, 2], []]


def test_real_groot_folds_and_plot_use_original_feature_identity(monkeypatch):
    X, y = data("string")
    X = X.rename(columns={"cat": "ShadowVar1"})
    X["ShadowVar1_"] = pd.Categorical([10, 20, 10, 30] * 20)
    before = X.copy(deep=True)
    encoded_cat = np.tile([0, 1, -1, 2], 20)
    encoded_category = np.tile([0, 1, 0, 2], 20)
    recorded = []
    train = ar._train_lgb_model

    def checked_train(
        frame, target, weights, validation, *args, category_cols=None, **kwargs
    ):
        assert frame.columns.equals(validation.columns)
        assert frame.columns.is_unique and len(frame.columns) == 2 * len(X.columns)
        roles = [frame.columns.get_loc(col) for col in category_cols]
        assert roles == [0, 2, 3, 5]
        for batch in [frame, validation]:
            positions = (batch.index.to_numpy() - 20) // 3
            assert_array_equal(batch.iloc[:, 0], encoded_cat[positions])
            assert_array_equal(batch.iloc[:, 2], encoded_category[positions])
            for original, shadow in [(0, 3), (2, 5)]:
                assert_array_equal(
                    np.sort(batch.iloc[:, original]), np.sort(batch.iloc[:, shadow])
                )
        result = train(
            frame,
            target,
            weights,
            validation,
            *args,
            category_cols=category_cols,
            **kwargs,
        )
        infos = result[0].dump_model()["feature_infos"]
        assert all(infos[col]["values"] for col in category_cols)
        assert infos["num"]["values"] == []
        recorded.append(list(category_cols))
        return result

    monkeypatch.setattr(ar, "_train_lgb_model", checked_train)
    selector = ar.GrootCV(
        objective="l2",
        n_folds=2,
        n_iter=1,
        n_jobs=1,
        lgbm_params={"num_iterations": 5, "min_data_in_leaf": 5},
    ).fit(X, y)
    assert len(recorded) == 2 and recorded[0] == recorded[1]
    assert set(selector.ranking_absolutes_) == set(X.columns)
    assert_frame_equal(X, before)
    fig = selector.plot_importance()
    try:
        assert {tick.get_text() for tick in fig.axes[0].get_yticklabels()} == set(
            X.columns
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("selector_name", ["leshy", "boost"])
def test_real_sklearn_fits_receive_matching_original_and_shadow_roles(
    selector_name, monkeypatch
):
    X, y = data()
    X = X.rename(columns={"cat": "Shadow_0"})
    before = X.copy(deep=True)
    recorded = []
    fit = lgb.LGBMRegressor.fit

    def checked_fit(self, frame, target, *args, categorical_feature="auto", **kwargs):
        assert frame.columns.is_unique
        assert categorical_feature[0] == 0
        assert all(frame.columns[pos] != "num" for pos in categorical_feature)
        expected = [0, 2, 4, 6, 8] if selector_name == "leshy" else [0, 2]
        assert categorical_feature == expected
        recorded.append(list(categorical_feature))
        return fit(
            self,
            frame,
            target,
            *args,
            categorical_feature=categorical_feature,
            **kwargs,
        )

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", checked_fit)
    if selector_name == "leshy":
        ar.Leshy(
            model(), importance="native", max_iter=3, n_estimators=5, random_state=0
        ).fit(X, y)
    else:
        ar.BoostAGroota(model(), importance="native", iters=1, max_rounds=1).fit(X, y)
    assert recorded
    assert_frame_equal(X, before)
