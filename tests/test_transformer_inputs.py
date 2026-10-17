"""R20 reusable transformer options, input normalization and Patsy rows."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from patsy import INTERCEPT, EvalEnvironment, ModelDesc, NAAction, PatsyError, dmatrix
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from arfs.preprocessing import (
    IntervalToMidpoint,
    PatsyTransformer,
    TreeDiscretizer,
    TreeGrouper,
    transform_interval_to_midpoint,
)
from tests.test_tree_partitions import PARAMS, numeric_data


@pytest.mark.parametrize("rule", ["all", "numerical", None])
def test_tree_refit_resolves_current_schema_without_changing_options(rule):
    X, y = numeric_data()
    X["cat"] = pd.Categorical(np.where(X.x < 100, "a", "b"))
    model = TreeDiscretizer(bin_features=rule, n_bins=3, boost_params=PARAMS.copy())
    model.fit(X, y)
    assert model.bin_features == rule
    new = X.rename(columns={"x": "z", "cat": "kind"})
    actual = model.fit_transform(new, y)
    expected = clone(model).fit_transform(new, y)
    assert_frame_equal(actual, expected)
    assert model.bin_features == rule
    assert set(model.tree_dic).issubset(new.columns)
    assert set(model.leaf_interval_dic_).issubset(new.columns)
    assert_array_equal(model.feature_names_in_, new.columns)
    assert model.n_features_in_ == 2


@pytest.mark.parametrize("raw", [False, True])
def test_tree_array_fit_transform_matches_named_frame_and_preserves_input(raw):
    X, y = numeric_data()
    array = np.column_stack([X.x, X.x % 10])
    before = array.copy()
    frame = pd.DataFrame(array, columns=["pred_0", "pred_1"])
    model = TreeDiscretizer(n_bins=3, raw=raw, boost_params=PARAMS.copy())
    actual = model.fit_transform(array, y)
    assert_frame_equal(actual, clone(model).fit_transform(frame, y))
    assert_frame_equal(model.transform(array), actual)
    assert_array_equal(array, before)
    with pytest.raises(ValueError, match="match"):
        model.transform(array[:, :1])


def test_tree_explicit_columns_and_filter_remain_constructor_options():
    X, y = numeric_data()
    X["small"] = X.x % 2
    columns = ["x", "small"]
    model = TreeDiscretizer(
        bin_features=columns, n_bins_max=3, boost_params=PARAMS.copy()
    ).fit(X, y)
    assert model.bin_features is columns
    assert columns == ["x", "small"]
    assert model.bin_features_ == ["x"]
    assert_array_equal(model.transform(X).small, X.small)
    new = X.copy()
    new["small"] = new.x
    model.fit(new, y)
    assert model.bin_features_ == columns
    assert model.get_params()["bin_features"] == columns


@pytest.mark.parametrize("change", ["reorder", "replace", "duplicate"])
def test_tree_rejects_changed_full_named_schema(change):
    X, y = numeric_data()
    X["other"] = X.x % 10
    model = TreeDiscretizer(bin_features=["x"], boost_params=PARAMS.copy()).fit(X, y)
    if change == "reorder":
        query = X[["other", "x"]]
    elif change == "replace":
        query = X.rename(columns={"other": "new"})
    else:
        query = X.copy()
        query.columns = ["x", "x"]
    with pytest.raises(ValueError, match="match"):
        model.transform(query)


def interval_data():
    return pd.DataFrame(
        {
            "interval": pd.arrays.IntervalArray.from_tuples([(0, 2), (2, 4)]),
            "second": pd.arrays.IntervalArray.from_tuples([(4, 6), (6, 8)]),
            "number": [11, 12],
        },
        index=[30, 10],
    )


@pytest.mark.parametrize("api", ["estimator", "function"])
@pytest.mark.parametrize(
    "cols", ["interval", ["interval"], pd.Index(["interval"]), "all"]
)
def test_midpoint_scalar_and_sequence_selection_preserve_input(api, cols):
    X = interval_data()
    before = X.copy(deep=True)
    if api == "estimator":
        model = IntervalToMidpoint(cols=cols)
        actual = model.fit_transform(X)
        assert model.cols is cols
    else:
        actual = transform_interval_to_midpoint(X, cols=cols)
    assert_allclose(actual.interval, [1, 3])
    if isinstance(cols, str) and cols == "all":
        assert_allclose(actual.second, [5, 7])
    else:
        assert_array_equal(actual.second, X.second)
    assert_frame_equal(X, before)
    assert_array_equal(actual.index, X.index)
    assert_array_equal(actual.columns, X.columns)


def test_midpoint_refit_and_clone_rediscover_interval_columns():
    X = interval_data()
    model = IntervalToMidpoint().fit(X)
    model.fit(X)
    assert model.cols == "all"
    assert model.columns_to_transform_ == ["interval", "second"]
    new = X.rename(columns={"interval": "third", "second": "fourth"})
    assert_frame_equal(model.fit_transform(new), clone(model).fit_transform(new))
    assert model.columns_to_transform_ == ["third", "fourth"]


@pytest.mark.parametrize("return_type", ["dataframe", "ndarray"])
@pytest.mark.parametrize("input_type", ["frame", "dict", "array"])
def test_patsy_fit_and_transform_use_same_names_and_owned_inputs(
    return_type, input_type
):
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0]}, index=[10, 20, 30])
    formula = "center(x) + z"
    if input_type == "dict":
        data = X.to_dict("list")
    elif input_type == "array":
        data = X.to_numpy()
        formula = "center(pred_0) + pred_1"
    else:
        data = X
    before = X.copy(deep=True)
    model = PatsyTransformer(formula, return_type=return_type)
    actual = model.fit_transform(data)
    assert_allclose(model.transform(data), actual)
    assert model.formula == formula
    assert model.n_features_in_ == 2
    if input_type == "frame":
        assert_array_equal(
            actual.index if return_type == "dataframe" else X.index, X.index
        )
    assert_frame_equal(X, before)
    assert_allclose(pd.DataFrame(data).to_numpy(), X.to_numpy())


@pytest.mark.parametrize("return_type", ["dataframe", "ndarray"])
def test_patsy_array_transform_uses_fitted_frame_names_and_validates_width(return_type):
    X = pd.DataFrame({"x": [1.0, 2.0], "z": [3.0, 4.0]})
    model = PatsyTransformer("x + z", return_type=return_type).fit(X)
    assert_allclose(model.transform(X.to_numpy()), model.transform(X))
    for width in [1, 3]:
        with pytest.raises(ValueError, match="width"):
            model.transform(np.ones((2, width)))


def test_patsy_automatic_formula_refit_clone_target_exclusion_and_current_options():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [7.0, 8.0, 9.0]})
    model = PatsyTransformer().fit(X, y=X.target)
    assert model.formula is None
    assert "target" not in model.design_.column_names
    new = pd.DataFrame({"z": [10.0, 20.0]})
    model.set_params(add_intercept=False)
    actual = model.fit_transform(new)
    assert_frame_equal(actual, clone(model).fit_transform(new))
    assert list(actual.columns) == ["z"]
    assert model.formula is None
    model.set_params(formula="center(z)")
    model.fit(new)
    assert_allclose(model.transform({"z": [30.0]})["center(z)"], [15])
    assert model.formula == "center(z)"


@pytest.mark.parametrize("return_type", ["dataframe", "ndarray"])
@pytest.mark.parametrize("policy", ["raise", NAAction(on_NA="raise")])
def test_patsy_missing_raise_applies_during_fit_and_transform(return_type, policy):
    model = PatsyTransformer("x", NA_action=policy, return_type=return_type)
    with pytest.raises(PatsyError, match="missing"):
        model.fit(pd.DataFrame({"x": [1.0, np.nan]}))
    model.fit(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(PatsyError, match="missing"):
        model.transform(pd.DataFrame({"x": [3.0, np.nan]}, index=[10, 20]))


@pytest.mark.parametrize("return_type", ["dataframe", "ndarray"])
@pytest.mark.parametrize("policy", ["drop", NAAction(on_NA="drop")])
def test_patsy_explicit_drop_preserves_retained_dataframe_indices_only(
    return_type, policy
):
    X = pd.DataFrame({"x": [1.0, np.nan, 3.0]}, index=[30, 10, 20])
    y = pd.Series([10, 20, 30], index=X.index)
    before = X.copy(deep=True)
    model = PatsyTransformer("x", NA_action=policy, return_type=return_type)
    actual = model.fit_transform(X, y)
    reference = dmatrix("x", X, NA_action=policy, return_type="dataframe")
    assert_allclose(actual, reference)
    query = pd.DataFrame({"x": [4.0, np.nan, 5.0]}, index=[70, 50, 60])
    result = model.transform(query)
    assert_allclose(result, [[1, 4], [1, 5]])
    if return_type == "dataframe":
        assert_array_equal(actual.index, [30, 20])
        assert_array_equal(result.index, [70, 60])
    assert len(y) == 3
    assert_frame_equal(X, before)


@pytest.mark.parametrize(
    "formula", ["x", "1 + x", "0 + x", ModelDesc.from_formula("x")]
)
@pytest.mark.parametrize("add_intercept", [False, True])
def test_patsy_intercept_choice_matches_native_without_mutating_formula(
    formula, add_intercept
):
    X = pd.DataFrame({"x": [1.0, 2.0]})
    before = formula.describe() if isinstance(formula, ModelDesc) else formula
    actual = PatsyTransformer(formula, add_intercept=add_intercept).fit_transform(X)
    desc = ModelDesc.from_formula(formula) if isinstance(formula, str) else formula
    native = ModelDesc(
        [], [term for term in desc.rhs_termlist if add_intercept or term != INTERCEPT]
    )
    assert_frame_equal(actual, dmatrix(native, X, return_type="dataframe"))
    assert (formula.describe() if isinstance(formula, ModelDesc) else formula) == before


def test_patsy_pipeline_reuses_fitted_center_and_row_policy():
    pipe = Pipeline([("formula", PatsyTransformer("center(x)", NA_action="raise"))])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    pipe.fit(X)
    result = pipe.transform(pd.DataFrame({"x": [10.0]}, index=[40]))
    assert_allclose(result["center(x)"], [8])
    assert_array_equal(result.index, [40])
    with pytest.raises(PatsyError, match="missing"):
        pipe.transform(pd.DataFrame({"x": [np.nan]}))


@pytest.mark.parametrize(
    "model", [TreeDiscretizer(), TreeGrouper(), IntervalToMidpoint(), PatsyTransformer()]
)
def test_transformer_unfitted_errors(model):
    with pytest.raises(NotFittedError):
        model.transform(pd.DataFrame({"x": [1.0]}))


def test_tree_array_query_uses_named_fit_schema():
    X, y = numeric_data()
    model = TreeDiscretizer(boost_params=PARAMS.copy()).fit(X, y)
    assert_frame_equal(
        model.transform(X.to_numpy()), model.transform(X).reset_index(drop=True)
    )
    duplicate = pd.concat([X, X], axis=1)
    with pytest.raises(ValueError, match="unique"):
        model.fit(duplicate, y)


@pytest.mark.parametrize("api", ["estimator", "function"])
def test_midpoint_repeated_or_absent_names_keep_selection_unique(api):
    X = interval_data()
    columns = ["interval", "interval", "absent"]
    actual = (
        IntervalToMidpoint(cols=columns).fit_transform(X)
        if api == "estimator"
        else transform_interval_to_midpoint(X, cols=columns)
    )
    assert_allclose(actual.interval, [1, 3])
    assert_array_equal(actual.second, X.second)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_patsy_keeps_explicit_evaluation_environment(method):
    def double(value):
        return value * 2

    X = pd.DataFrame({"x": [1.0, 2.0]})
    model = PatsyTransformer(
        "double(x)", add_intercept=False, eval_env=EvalEnvironment.capture(0)
    )
    getattr(model, method)(X)
    assert_allclose(model.transform({"x": [3.0]})["double(x)"], [6])
    assert_allclose(double(X.x), [2, 4])
