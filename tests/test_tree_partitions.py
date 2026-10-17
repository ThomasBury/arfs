"""R19 real one-tree partitions, fit-time parameters and missing routes."""

from copy import deepcopy

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone

from arfs.preprocessing import IntervalToMidpoint, TreeDiscretizer, TreeGrouper

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


def grouper(**kwargs):
    return TreeGrouper(n_groups=6, boost_params=PARAMS.copy(), **kwargs)


def leaf_ids(model, frame, col="x"):
    values = (
        model._feature_values(frame[col], col)
        if isinstance(model, TreeGrouper)
        else frame[[col]].to_numpy(dtype=float, na_value=np.nan)
    )
    return model.tree_dic[col].predict(values, pred_leaf=True).ravel()


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
    table = model.tree_dic["x"].trees_to_dataframe()
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
    model = grouper().fit(X, y)
    leaves = leaf_ids(model, X)
    assert len(np.unique(leaves)) > 1
    booster = model.tree_dic["x"]
    # A real tree can have equal-valued leaves; the partition remains its leaf IDs.
    for leaf in np.unique(leaves):
        booster.set_leaf_output(0, int(leaf), 0.0)
    assert_partition(model.transform(X).x, leaves)


def test_category_group_labels_cannot_merge_when_display_text_is_truncated():
    names = ["sameprefix" + str(i) + "samesuffix" for i in range(6)]
    X = pd.DataFrame({"cat": pd.Categorical(np.tile(names, 40))})
    y = np.tile(np.arange(6.0), 40)
    model = grouper().fit(X, y)
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
            model.tree_dic[col].predict(X[[col]]),
            alone.tree_dic[col].predict(X[[col]]),
        )
        assert_allclose(
            reverse.tree_dic[col].predict(X[[col]]),
            alone.tree_dic[col].predict(X[[col]]),
        )
        assert_frame_equal(model.transform(X)[[col]], alone.transform(X[[col]]))
    assert_frame_equal(X, before)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_weighted_raw_outputs_match_independent_native_single_tree(task):
    X, y = numeric_data(task)
    weights = np.linspace(0.2, 3, len(X))
    weights[::7] = 0
    keep = weights > 0
    model = discretizer(raw=True, task=task).fit(X, y, weights)
    objective = "rmse" if task == "regression" else "binary"
    params = dict(PARAMS, objective=objective, metric=objective, num_leaves=6)
    native = lgb.train(
        params,
        lgb.Dataset(X.iloc[keep], label=y[keep], weight=weights[keep]),
        num_boost_round=1,
    )
    assert_allclose(model.transform(X).x, native.predict(X), atol=1e-12)
    assert model.tree_dic["x"].num_trees() == 1
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
    model = grouper(raw=raw).fit(X, y)
    query = pd.DataFrame(
        {"cat": pd.Series(query_values, dtype=dtype)}, index=[0, 1, 2, 3]
    )
    query_before = query.copy(deep=True)
    encoded = model._feature_values(query.cat, "cat")
    actual = model.transform(query)
    assert actual.cat.notna().all()
    if raw:
        assert_allclose(actual.cat, model.tree_dic["cat"].predict(encoded))
    else:
        leaves = model.tree_dic["cat"].predict(encoded, pred_leaf=True).ravel()
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
        assert_allclose(actual, model.tree_dic["x"].predict(query))
    else:
        assert_partition(actual, leaf_ids(model, query))


@pytest.mark.parametrize("dtype", ["float64", "Float64", "string", "category"])
@pytest.mark.parametrize("raw", [False, True])
def test_all_missing_feature_is_a_single_fitted_leaf(dtype, raw):
    X = pd.DataFrame({"x": pd.Series([None] * 80, dtype=dtype)})
    before = X.copy(deep=True)
    factory = grouper if dtype in ["string", "category"] else discretizer
    model = factory(raw=raw).fit(X, np.tile([0.0, 1.0], 40))
    actual = model.transform(X)
    assert actual.x.notna().all() and actual.x.nunique() == 1
    assert model.tree_dic["x"].num_trees() == 1
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
    assert model.tree_dic["x"].num_trees() == 1
    assert model.tree_dic["x"].dump_model()["tree_info"][0]["num_leaves"] <= 2


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
    booster = model.tree_dic["x"]
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
    model = grouper(raw=True, task=task).fit(X, y)
    query = X.iloc[:4].copy()
    encoded = model._feature_values(query.cat, "cat")
    assert_allclose(model.transform(query).cat, model.tree_dic["cat"].predict(encoded))


@pytest.mark.parametrize("n_bins", [0, 1, -1, 2.5, True, np.nan, np.inf])
def test_invalid_bin_limits_fail_before_training(n_bins, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid n_bins reached model training")

    monkeypatch.setattr("arfs.preprocessing.lgb.train", forbidden)
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
    model = TreeGrouper(
        group_features="categorical", n_groups=2, boost_params=PARAMS.copy()
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
    model = grouper().fit(X, np.tile([0.0, 1.0], 40))
    result = model.transform(X)
    assert_partition(result[column], leaf_ids(model, X, column))
    assert result.columns.equals(X.columns)
    assert_frame_equal(X, before)


@pytest.mark.parametrize("selection", ["all", "numerical", None])
def test_discretizer_automatic_selection_is_numeric_only(selection):
    X, y = numeric_data()
    X["category"] = pd.Categorical(np.tile(["a", "b"], 120))
    X["boolean"] = np.tile([True, False], 120)
    model = discretizer(bin_features=selection).fit(X, y)
    assert model.bin_features_ == ["x"]
    assert_frame_equal(model.transform(X).drop(columns="x"), X.drop(columns="x"))


@pytest.mark.parametrize("selection", ["categorical", ["cat"]])
def test_discretizer_rejects_categorical_selection_with_migration(selection):
    with pytest.raises(ValueError, match="TreeGrouper"):
        TreeDiscretizer(bin_features=selection).fit(
            pd.DataFrame({"cat": ["a", "b"]}), [0, 1]
        )


def test_numeric_category_codes_require_explicit_category_dtype():
    X = pd.DataFrame({"code": np.tile([1, 2], 40)})
    with pytest.raises(ValueError, match="cast numeric codes"):
        grouper(group_features=["code"]).fit(X, np.tile([0, 1], 40))
    X.code = X.code.astype("category")
    assert grouper().fit_transform(X, np.tile([0, 1], 40)).code.notna().all()


@pytest.mark.parametrize("factory", [discretizer, grouper])
@pytest.mark.parametrize(
    "target", [[1, 2], [0, 2], [0, 1, 2], [0, np.nan], [0, np.inf], ["0", "1"]]
)
def test_classification_rejects_invalid_targets_even_on_zero_weight_rows(
    factory, target
):
    y = np.tile(target, 40)
    X = pd.DataFrame({"x": np.arange(len(y)), "cat": np.tile(["a", "b"], len(y) // 2)})
    weights = np.ones(len(y))
    weights[1::2] = 0
    with pytest.raises(ValueError, match="y"):
        factory(task="classification").fit(X, y, weights)


@pytest.mark.parametrize("factory", [discretizer, grouper])
@pytest.mark.parametrize("target", [[0, np.nan], [0, np.inf], ["0", "1"], [1j, 2j]])
def test_regression_requires_finite_numeric_targets(factory, target):
    with pytest.raises(ValueError, match="y"):
        factory().fit(pd.DataFrame({"x": [1, 2], "cat": ["a", "b"]}), target, [1, 0])


@pytest.mark.parametrize("factory", [discretizer, grouper])
@pytest.mark.parametrize("constant", [False, True, 0, 1])
def test_constant_binary_targets_are_valid(factory, constant):
    X = pd.DataFrame({"x": np.arange(80), "cat": np.tile(["a", "b"], 40)})
    model = factory(task="classification", raw=True).fit(X, np.full(len(X), constant))
    for col in model.tree_dic:
        assert model.tree_dic[col].num_trees() == 1
        assert model.transform(X)[col].between(0, 1).all()


@pytest.mark.parametrize("factory", [discretizer, grouper])
@pytest.mark.parametrize(
    "weights", [[0, 0], [-1, 1], [np.nan, 1], [np.inf, 1], [1], [[1], [1]]]
)
def test_tree_weights_validated_before_learning(factory, weights):
    with pytest.raises(ValueError, match="sample_weight"):
        factory().fit(pd.DataFrame({"x": [1, 2], "cat": ["a", "b"]}), [0, 1], weights)


@pytest.mark.parametrize("factory", [discretizer, grouper])
@pytest.mark.parametrize("raw", [False, True])
def test_zero_weight_rows_do_not_change_vocabulary_partition_or_weight_alignment(
    factory, raw
):
    X, y = numeric_data()
    X["cat"] = pd.Categorical(
        np.tile(["b", "a", "c"], 80), categories=["c", "b", "a", "unused"]
    )
    weights = np.linspace(0.2, 2, len(X))
    # Deliberately unrelated Series index: weights align by row position.
    w = pd.Series(weights, index=X.index[::-1])
    baseline = factory(raw=raw).fit(X, y, w)
    extra = pd.DataFrame({"x": [-1e9], "cat": ["zero-only"]})
    augmented = pd.concat([X.astype({"cat": object}), extra], ignore_index=True)
    augmented.cat = pd.Categorical(
        augmented.cat, categories=list(X.cat.cat.categories) + ["zero-only"]
    )
    actual = factory(raw=raw).fit(augmented, np.r_[y, 1e6], np.r_[weights, 0])
    assert_frame_equal(actual.transform(X), baseline.transform(X))
    if isinstance(actual, TreeGrouper):
        assert "zero-only" not in actual.category_dtypes_["cat"].categories
        assert "unused" not in baseline.category_dtypes_["cat"].categories
        unknown = actual.transform(
            pd.DataFrame({"x": [0, 0], "cat": ["zero-only", None]})
        )
        assert unknown.cat.iloc[0] == unknown.cat.iloc[1]


def test_nominal_relabeling_invariant_and_native_categorical_partition():
    X = pd.DataFrame({"cat": np.tile(["a", "b", "c", "d", "e", "f"], 100)})
    y = np.tile([0.0, 10.0, 0.0, 10.0, 0.0, 10.0], 100)
    params = dict(
        PARAMS, max_cat_to_onehot=1, cat_smooth=0, cat_l2=0, min_data_per_group=1
    )
    model = TreeGrouper(n_groups=2, boost_params=params).fit(X, y)
    renamed = X.replace({"a": "z", "b": "c", "c": "y", "d": "b", "e": "x", "f": "a"})
    other = clone(model).fit(renamed, y)
    assert_partition(other.transform(renamed).cat, leaf_ids(model, X, "cat"))
    assert_partition(model.transform(X).cat, y)
    node = model.tree_dic["cat"].dump_model()["tree_info"][0]["tree_structure"]
    assert node["decision_type"] == "=="
    assert not model.transform(X).cat.cat.ordered


@pytest.mark.parametrize("zero_as_missing", [False, True])
def test_ordered_groups_keep_declared_relative_order_and_are_contiguous(
    zero_as_missing,
):
    levels = ["sixth", "first", "fifth", "second", "fourth", "third"]
    X = pd.DataFrame(
        {"cat": pd.Categorical(np.tile(levels, 80), categories=levels, ordered=True)}
    )
    y = np.tile([0.0, 0.0, 10.0, 10.0, 0.0, 0.0], 80)
    model = TreeGrouper(
        n_groups=3, boost_params=dict(PARAMS, zero_as_missing=zero_as_missing)
    ).fit(X, y)
    assert list(model.category_dtypes_["cat"].categories) == levels
    result = model.transform(X.iloc[:6]).cat
    codes = result.cat.codes.to_numpy()
    assert result.cat.ordered
    assert (np.diff(codes) >= 0).all()
    for code in np.unique(codes):
        positions = np.flatnonzero(codes == code)
        assert positions[-1] - positions[0] + 1 == len(positions)
    assert (
        model.tree_dic["cat"].dump_model()["tree_info"][0]["tree_structure"][
            "decision_type"
        ]
        == "<="
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("ordered", [False, True])
def test_grouped_raw_matches_independent_native_positive_weight_tree(task, ordered):
    levels = ["d", "a", "c", "b"]
    X = pd.DataFrame(
        {
            "cat": pd.Categorical(
                np.tile(levels, 100), categories=levels, ordered=ordered
            )
        }
    )
    y = np.tile([0.0, 0.0, 1.0, 1.0], 100)
    weights = np.linspace(0.2, 2, len(X))
    weights[::5] = 0
    model = grouper(task=task, raw=True).fit(X, y, weights)
    codes = X.cat.cat.codes.to_numpy(dtype=float)
    if ordered:
        codes += 1
    params = dict(
        PARAMS, objective="rmse" if task == "regression" else "binary", num_leaves=6
    )
    keep = weights > 0
    native = lgb.train(
        params,
        lgb.Dataset(
            codes[keep, None],
            label=y[keep],
            weight=weights[keep],
            categorical_feature=[] if ordered else [0],
        ),
        num_boost_round=1,
    )
    assert_allclose(model.transform(X).cat, native.predict(codes[:, None]), atol=1e-12)


def test_group_categories_are_frozen_across_batches_missing_and_unknown_levels():
    X = pd.DataFrame({"cat": np.tile(["a", "b", "c", None], 80)})
    y = np.tile([0.0, 1.0, 2.0, 3.0], 80)
    model = grouper().fit(X, y)
    query = pd.DataFrame({"cat": ["a", "new", None, "b"]}, index=[50, 20, 70, 10])
    whole = model.transform(query)
    pieces = pd.concat([model.transform(query.iloc[[i]]) for i in range(len(query))])
    assert_frame_equal(whole, pieces)
    assert whole.cat.dtype == model.group_dtypes_["cat"]
    assert whole.cat.iloc[1] == whole.cat.iloc[2]


def test_group_refit_clone_arrays_schema_names_and_serialization(tmp_path):
    import joblib

    X = pd.DataFrame({"cat": np.tile(["a", "b"], 80), "pass": np.arange(160)})
    y = np.tile([0, 1], 80)
    model = grouper().fit(X, y)
    assert_frame_equal(
        model.transform(X.to_numpy())[["cat"]], model.transform(X)[["cat"]]
    )
    assert_array_equal(model.transform(X.to_numpy())["pass"], X["pass"])
    assert_array_equal(model.get_feature_names_out(), X.columns)
    with pytest.raises(ValueError, match="input_features"):
        model.get_feature_names_out(["wrong", "pass"])
    for query in [
        X[["pass", "cat"]],
        X.rename(columns={"cat": "new"}),
        X.assign(other=1),
        X.to_numpy()[:, :1],
    ]:
        with pytest.raises(ValueError, match="match"):
            model.transform(query)
    with pytest.raises(ValueError, match="unique"):
        model.fit(pd.concat([X, X], axis=1), y)
    new = X.rename(columns={"cat": "kind"})
    assert_frame_equal(model.fit_transform(new, y), clone(model).fit_transform(new, y))
    assert set(model.tree_dic) == set(model.category_dtypes_) == {"kind"}
    path = tmp_path / "tree.joblib"
    joblib.dump(model, path)
    assert_frame_equal(joblib.load(path).transform(new), model.transform(new))
    array_model = grouper().fit(X[["cat"]].to_numpy(), y)
    assert_array_equal(array_model.get_feature_names_out(), ["pred_0"])


def test_numeric_feature_names_and_same_environment_serialization(tmp_path):
    import joblib

    X, y = numeric_data()
    model = discretizer().fit(X, y)
    assert_array_equal(model.get_feature_names_out(["x"]), ["x"])
    path = tmp_path / "numeric.joblib"
    joblib.dump(model, path)
    assert_frame_equal(joblib.load(path).transform(X), model.transform(X))


def test_pipeline_forwards_weights_to_both_supervised_steps():
    from sklearn.pipeline import Pipeline

    X, y = numeric_data()
    X["cat"] = np.tile(["a", "b", "c"], 80)
    weights = np.linspace(0.1, 5, len(X))
    pipe = Pipeline([("group", grouper()), ("bin", discretizer())])
    result = pipe.fit_transform(
        X, y, group__sample_weight=weights, bin__sample_weight=weights
    )
    group = grouper().fit(X, y, weights)
    expected = discretizer().fit_transform(group.transform(X), y, sample_weight=weights)
    assert_frame_equal(result, expected)
    assert_frame_equal(
        X[["cat"]], pd.DataFrame({"cat": np.tile(["a", "b", "c"], 80)}, index=X.index)
    )


def test_all_row_contract_overrides_sampling_without_parameter_mutation():
    X, y = numeric_data()
    params = dict(
        PARAMS,
        boosting_type="goss",
        subsample=0.5,
        subsample_freq=1,
        pos_bagging_fraction=0.3,
        neg_bagging_fraction=0.3,
    )
    before = deepcopy(params)
    model = TreeDiscretizer(n_bins=6, boost_params=params, raw=True).fit(X, y)
    assert_allclose(model.transform(X).x, discretizer(raw=True).fit_transform(X, y).x)
    assert params == before
