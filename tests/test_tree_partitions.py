"""R19 real one-tree partitions, fit-time parameters and missing routes."""

from copy import deepcopy

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit

from arfs.preprocessing import IntervalToMidpoint, TreeDiscretizer

PARAMS = {
    "num_threads": 1,
    "min_data_in_leaf": 5,
    "learning_rate": 1,
    "verbosity": -1,
    "seed": 8,
}
ROUND_ALIASES = [
    "num_iterations",
    "num_iteration",
    "n_iter",
    "num_tree",
    "num_trees",
    "num_round",
    "num_rounds",
    "nrounds",
    "num_boost_round",
    "n_estimators",
    "max_iter",
]


def numeric_data(task="regression"):
    X = pd.DataFrame({"x": np.arange(240, dtype=float)}, index=np.arange(240) + 1000)
    y = np.repeat([0.0, 2.0, -1.0, 3.0, 0.0, 2.0], 40)
    if task == "classification":
        y = np.tile(
            np.r_[
                np.zeros(36),
                np.ones(4),
                np.zeros(28),
                np.ones(12),
                np.zeros(12),
                np.ones(28),
            ],
            2,
        )
    return X, y


def discretizer(**kwargs):
    return TreeDiscretizer(n_bins=6, boost_params=PARAMS.copy(), **kwargs)


def leaf_ids(model, frame, col="x"):
    if col in model.ordinal_encoder_dic:
        encoded = model.ordinal_encoder_dic[col].transform(frame[[col]])
        frame = pd.DataFrame(encoded, columns=[col], index=frame.index)
    return model.tree_dic[col].model.predict(frame[[col]], pred_leaf=True).ravel()


def assert_partition(actual, leaves):
    codes, _ = pd.factorize(actual)
    assert (codes >= 0).all()
    assert_array_equal(
        codes[:, None] == codes[None, :], leaves[:, None] == leaves[None, :]
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("categorical", [False, True])
def test_real_numeric_partition_and_exact_threshold_boundaries(task, categorical):
    X, y = numeric_data(task)
    before = X.copy(deep=True)
    model = discretizer(task=task, num_bins_as_category=categorical).fit(X, y)
    leaves = leaf_ids(model, X)
    assert len(np.unique(leaves)) == 6
    result = model.transform(X)
    assert_partition(result.x, leaves)
    assert isinstance(
        result.x.dtype, pd.CategoricalDtype if categorical else pd.IntervalDtype
    )
    table = model.tree_dic["x"].model.trees_to_dataframe()
    thresholds = np.sort(
        table.loc[table.split_feature.notna(), "threshold"].to_numpy(float)
    )
    assert_array_equal(
        model.bin_upper_bound_dic["x"], np.r_[-np.inf, thresholds, np.inf]
    )
    values = np.r_[
        thresholds,
        np.nextafter(thresholds, -np.inf),
        np.nextafter(thresholds, np.inf),
        120.0,
        120.25,
        -1e6,
        1e6,
        np.nan,
    ]
    query = pd.DataFrame({"x": values})
    actual = model.transform(query).x
    assert_partition(actual, leaf_ids(model, query))
    finite = np.isfinite(values)
    for value, interval in zip(values[finite], actual[finite]):
        assert value in interval
    assert_frame_equal(X, before)
    assert result.index.equals(X.index)


def test_partition_does_not_depend_on_equal_leaf_output_values():
    X, y = numeric_data()
    X["x"] = pd.Categorical(np.tile(["a", "b", "c", "d", "e", "f"], 40))
    model = discretizer().fit(X, y)
    leaves = leaf_ids(model, X)
    assert len(np.unique(leaves)) > 1
    booster = model.tree_dic["x"].model
    # A real tree can have equal-valued leaves; the partition remains its leaf IDs.
    for leaf in np.unique(leaves):
        booster.set_leaf_output(0, int(leaf), 0.0)
    assert_partition(model.transform(X).x, leaves)


def test_category_group_labels_cannot_merge_when_display_text_is_truncated():
    names = ["sameprefix" + str(i) + "samesuffix" for i in range(6)]
    X = pd.DataFrame({"cat": pd.Categorical(np.tile(names, 40))})
    y = np.tile(np.arange(6.0), 40)
    model = discretizer().fit(X, y)
    assert_partition(model.transform(X).cat, leaf_ids(model, X, "cat"))
    assert isinstance(model.transform(X).cat.dtype, pd.CategoricalDtype)


def test_generated_name_and_column_order_cannot_change_other_feature_fit():
    X, y = numeric_data()
    X["x_g"] = np.tile(np.arange(40, dtype=float), 6)
    before = X.copy(deep=True)
    model = discretizer().fit(X, y)
    reverse = discretizer().fit(X[["x_g", "x"]], y)
    for col in X:
        alone = discretizer().fit(X[[col]], y)
        assert_allclose(
            model.tree_dic[col].model.predict(X[[col]]),
            alone.tree_dic[col].model.predict(X[[col]]),
        )
        assert_allclose(
            reverse.tree_dic[col].model.predict(X[[col]]),
            alone.tree_dic[col].model.predict(X[[col]]),
        )
        assert_frame_equal(model.transform(X)[[col]], alone.transform(X[[col]]))
    assert_frame_equal(X, before)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_weighted_raw_outputs_match_independent_native_single_tree(task):
    X, y = numeric_data(task)
    weights = np.linspace(0.2, 3, len(X))
    model = discretizer(raw=True, task=task).fit(X, y, weights)
    train, _ = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X))
    objective = "rmse" if task == "regression" else "binary"
    params = dict(PARAMS, objective=objective, metric=objective, num_leaves=6)
    native = lgb.train(
        params,
        lgb.Dataset(X.iloc[train], label=y[train], weight=weights[train]),
        num_boost_round=1,
    )
    assert_allclose(model.transform(X).x, native.predict(X), atol=1e-12)
    assert model.tree_dic["x"].model.num_trees() == 1
    unweighted = discretizer(raw=True, task=task).fit(X, y)
    assert not np.allclose(model.transform(X).x, unweighted.transform(X).x)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("dtype", [object, "category", "string", None, "boolean"])
def test_categorical_raw_and_grouped_missing_unknown_use_native_routes(raw, dtype):
    values = ["a", "missing_added", "b", None] * 60
    query_values = ["a", None, "new", "missing_added"]
    if dtype == "boolean":
        values = [True, False, True, None] * 60
        query_values = [True, None, None, False]
    X = pd.DataFrame({"cat": pd.Series(values, dtype=dtype)})
    y = np.tile([0.0, 1.0, 2.0, 3.0], 60)
    before = X.copy(deep=True)
    model = discretizer(raw=raw).fit(X, y)
    query = pd.DataFrame(
        {"cat": pd.Series(query_values, dtype=dtype)}, index=[0, 1, 2, 3]
    )
    query_before = query.copy(deep=True)
    encoded = model.ordinal_encoder_dic["cat"].transform(query)
    if isinstance(encoded, np.ndarray):
        encoded = pd.DataFrame(encoded, columns=["cat"], index=query.index)
    actual = model.transform(query)
    assert actual.cat.notna().all()
    if raw:
        assert_allclose(actual.cat, model.tree_dic["cat"].model.predict(encoded))
    else:
        leaves = model.tree_dic["cat"].model.predict(encoded, pred_leaf=True).ravel()
        assert_partition(actual.cat, leaves)
    assert_frame_equal(X, before)
    assert_frame_equal(query, query_before)


@pytest.mark.parametrize("zero_as_missing", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_numeric_missing_and_zero_follow_actual_leaves(zero_as_missing, raw):
    X, y = numeric_data()
    X.loc[X.index[:40], "x"] = np.nan
    model = TreeDiscretizer(
        n_bins=6, raw=raw, boost_params=dict(PARAMS, zero_as_missing=zero_as_missing)
    )
    model.fit(X, y)
    query = pd.DataFrame({"x": [np.nan, 0.0, 1.0, -1.0, 40.0, 120.25, 240.0]})
    actual = model.transform(query).x
    if raw:
        assert_allclose(actual, model.tree_dic["x"].model.predict(query))
    else:
        assert_partition(actual, leaf_ids(model, query))


@pytest.mark.parametrize("dtype", ["float64", "Float64", "string", "category"])
@pytest.mark.parametrize("raw", [False, True])
def test_all_missing_feature_is_a_single_fitted_leaf(dtype, raw):
    X = pd.DataFrame({"x": pd.Series([None] * 80, dtype=dtype)})
    before = X.copy(deep=True)
    model = discretizer(raw=raw).fit(X, np.tile([0.0, 1.0], 40))
    actual = model.transform(X)
    assert actual.x.notna().all() and actual.x.nunique() == 1
    assert model.tree_dic["x"].model.num_trees() == 1
    if not raw and dtype in ["float64", "Float64"]:
        assert actual.x.iloc[0] == pd.Interval(-np.inf, np.inf)
        assert_array_equal(model.bin_upper_bound_dic["x"], [-np.inf, np.inf])
    assert_frame_equal(X, before)


@pytest.mark.parametrize("alias", ROUND_ALIASES)
def test_one_tree_contract_overrides_every_round_alias_without_mutation(alias):
    X, y = numeric_data()
    params = dict(PARAMS, **{alias: 3})
    before = deepcopy(params)
    model = TreeDiscretizer(n_bins=2, boost_params=params)
    assert params == before and model.get_params()["boost_params"] == before
    model.fit(X, y)
    assert params == before and model.boost_params == before
    assert model.tree_dic["x"].model.num_trees() == 1
    assert model.tree_dic["x"].model.dump_model()["tree_info"][0]["num_leaves"] <= 2


def test_current_task_bins_and_conflicting_leaf_objective_aliases():
    X, _ = numeric_data()
    y = np.tile(np.r_[np.zeros(10), np.ones(10)], 12)
    params = dict(
        PARAMS,
        num_leaves=64,
        max_leaf=48,
        max_leaves=32,
        num_leaf=24,
        max_leaf_nodes=16,
        objective="poisson",
        loss="poisson",
        app="poisson",
        application="poisson",
        objective_type="poisson",
    )
    before = deepcopy(params)
    model = TreeDiscretizer(n_bins=6, boost_params=params)
    model.set_params(n_bins=2, task="classification").fit(X, y)
    booster = model.tree_dic["x"].model
    assert booster.params["objective"] == "binary"
    assert booster.dump_model()["tree_info"][0]["num_leaves"] <= 2
    assert params == before and model.boost_params == before
    fresh = clone(model).fit(X, y)
    assert_frame_equal(fresh.transform(X), model.transform(X))


@pytest.mark.parametrize("raw", [False, True])
def test_constant_numeric_and_unselected_passthrough(raw):
    X = pd.DataFrame(
        {"x": [2.0] * 80, "pass": pd.Series(["keep"] * 80, dtype="string")}
    )
    before = X.copy(deep=True)
    model = TreeDiscretizer(
        bin_features=["x"], n_bins=3, raw=raw, boost_params=PARAMS.copy()
    )
    result = model.fit_transform(X, np.tile([0.0, 1.0], 40))
    assert_frame_equal(result[["pass"]], before[["pass"]])
    assert result.x.nunique() == 1
    assert_frame_equal(X, before)
    if not raw:
        midpoints = IntervalToMidpoint().fit_transform(result)
        assert midpoints.shape == result.shape


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_raw_ordinary_categories_match_fitted_encoded_tree(task):
    X = pd.DataFrame({"cat": np.tile(["a", "b", "c", "d"], 60)})
    y = np.tile([0.0, 1.0, 2.0, 3.0] if task == "regression" else [0, 0, 1, 1], 60)
    model = discretizer(raw=True, task=task).fit(X, y)
    query = X.iloc[:4].copy()
    encoded = model.ordinal_encoder_dic["cat"].transform(query)
    if isinstance(encoded, np.ndarray):
        encoded = pd.DataFrame(encoded, columns=["cat"], index=query.index)
    assert_allclose(
        model.transform(query).cat, model.tree_dic["cat"].model.predict(encoded)
    )


@pytest.mark.parametrize("n_bins", [0, 1, -1, 2.5, True, np.nan, np.inf])
def test_invalid_bin_limits_fail_before_training(n_bins, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid n_bins reached model training")

    monkeypatch.setattr("arfs.preprocessing.GradientBoosting", forbidden)
    X, y = numeric_data()
    model = TreeDiscretizer(boost_params=PARAMS.copy()).set_params(n_bins=n_bins)
    with pytest.raises(ValueError, match="n_bins"):
        model.fit(X, y)


@pytest.mark.parametrize(
    "params,error",
    [
        ({"task": "multiclass"}, ValueError),
        ({"task": "invalid"}, ValueError),
        ({"boost_params": []}, TypeError),
    ],
)
def test_invalid_current_parameters_rejected_at_fit(params, error):
    X, y = numeric_data()
    model = TreeDiscretizer(boost_params=PARAMS.copy()).set_params(**params)
    with pytest.raises(error):
        model.fit(X, y)


def test_categorical_selection_includes_nullable_strings_and_boolean():
    X = pd.DataFrame(
        {
            "text": pd.Series(["a", "b"] * 60, dtype="string"),
            "flag": pd.Series([True, False] * 60, dtype="boolean"),
            "numeric": np.arange(120.0),
        }
    )
    model = TreeDiscretizer(
        bin_features="categorical", n_bins=2, boost_params=PARAMS.copy()
    )
    result = model.fit_transform(X, np.tile([0.0, 1.0], 60))
    assert set(model.tree_dic) == {"text", "flag"}
    assert isinstance(result.text.dtype, pd.CategoricalDtype)
    assert isinstance(result.flag.dtype, pd.CategoricalDtype)
    assert_frame_equal(result[["numeric"]], X[["numeric"]])


@pytest.mark.parametrize("column", [None, 0])
def test_categorical_grouping_preserves_original_column_label(column):
    X = pd.DataFrame({column: pd.Categorical(np.tile(["a", "b"], 40))})
    before = X.copy(deep=True)
    model = discretizer().fit(X, np.tile([0.0, 1.0], 40))
    result = model.transform(X)
    assert_partition(result[column], leaf_ids(model, X, column))
    assert result.columns.equals(X.columns)
    assert_frame_equal(X, before)
