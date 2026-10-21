"""Independent scalar references for mixed, weighted, incomplete distances."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.sparse import csr_matrix

from arfs import sampling


def reference(X, Y=None, categorical=(), weights=None):
    """Scalar weighted mean over pairwise-observed features; pooled ranges."""
    Y = X if Y is None else Y
    pooled = pd.concat([X, Y], ignore_index=True)
    weights = np.ones(X.shape[1]) if weights is None else weights
    ranges = {
        c: float(pooled[c].max() - pooled[c].min())
        for c in X.columns
        if c not in categorical
    }
    out = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            numerator = denominator = 0.0
            for c, weight in zip(X.columns, weights):
                left, right = X.iloc[i][c], Y.iloc[j][c]
                if pd.isna(left) or pd.isna(right) or weight == 0:
                    continue
                if c in categorical:
                    delta = float(left != right)
                else:
                    delta = (
                        abs(float(left) - float(right)) / ranges[c]
                        if ranges[c]
                        else 0.0
                    )
                numerator += weight * delta
                denominator += weight
            out[i, j] = numerator / denominator if denominator else np.nan
    return out


def mixed():
    return pd.DataFrame(
        {"x": [0.0, 5.0, 10.0], "c": ["a", "a", "b"], "constant": [7.0, 7.0, 7.0]},
        index=[8, 8, 90],
    )


@pytest.mark.parametrize(
    "mask",
    [
        "auto",
        ["c"],
        [False, True, False],
        [1],
        np.array([1]),
        np.array([False, True, False]),
        ("c",),
    ],
)
@pytest.mark.parametrize("weight_kind", ["array", "list", "series"])
def test_hand_weighted_mixed_matrix_and_input_preservation(mask, weight_kind):
    X = mixed()
    before = X.copy(deep=True)
    weights = {
        "array": np.array([2.0, 1.0, 1.0]),
        "list": [2.0, 1.0, 1.0],
        "series": pd.Series([2.0, 1.0, 1.0], index=[90, 8, 7]),
    }[weight_kind]
    expected = [[0.0, 0.25, 0.75], [0.25, 0.0, 0.5], [0.75, 0.5, 0.0]]
    assert_allclose(reference(X, categorical=["c"], weights=weights), expected)
    result = sampling.gower_matrix(X, weight=weights, cat_features=mask)
    assert result.dtype == np.float64
    assert_allclose(result, expected)
    assert_allclose(
        sampling.gower_matrix(X, weight=np.asarray(weights) * 11, cat_features=mask),
        result,
    )
    assert_frame_equal(X, before)


@pytest.mark.parametrize(
    "values",
    [[-3.0, -1.0], [-3.0, 0.0], [1e9, 1e9 + 1.0], [0.0, 1e-12], [1e-100, 2e-100]],
)
def test_numeric_range_is_independent_of_origin_and_small_units(values):
    X = pd.DataFrame({"x": values})
    assert_allclose(sampling.gower_matrix(X), [[0.0, 1.0], [1.0, 0.0]])


def test_translation_and_rescaling_preserve_mixed_distance():
    X = mixed()
    expected = sampling.gower_matrix(X)
    changed = X.copy()
    changed["x"] = changed.x * 4 - 100
    changed["constant"] = changed.constant * -3 + 200
    assert_allclose(sampling.gower_matrix(changed), expected)


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("y_values", [[1.0, 3.0], [1.0, 3.0, 20.0]])
def test_cross_matrix_positions_ranges_and_transpose(as_array, y_values):
    X = pd.DataFrame({"x": [0.0, 10.0]}, index=[50, 50])
    Y = pd.DataFrame({"x": y_values}, index=np.arange(len(y_values)) + 200)
    expected = reference(X, Y)
    before = Y.copy(deep=True)
    left, right = (X.to_numpy(), Y.to_numpy()) if as_array else (X, Y)
    assert_allclose(sampling.gower_matrix(left, right), expected)
    assert_allclose(sampling.gower_matrix(right, left), expected.T)
    assert_frame_equal(Y, before)


def test_cross_ranges_include_reference_and_series_is_one_row():
    X = mixed()
    Y = X.iloc[0].copy()
    Y["x"] = 20.0
    before = Y.copy(deep=True)
    assert_allclose(sampling.gower_matrix(X, Y), reference(X, Y.to_frame().T, ["c"]))
    assert_series_equal(Y, before)
    assert_allclose(
        sampling.gower_matrix(
            pd.DataFrame({"x": [0.0, 1.0]}), pd.DataFrame({"x": [10.0]})
        ),
        [[1.0], [0.9]],
    )


@pytest.mark.parametrize(
    "kind", ["numeric", "categorical", "all-missing-column", "all-missing-row"]
)
def test_missingness_uses_pairwise_available_mass_without_warnings(kind):
    X = mixed().reset_index(drop=True)
    if kind == "numeric":
        X.loc[1, "x"] = np.nan
    elif kind == "categorical":
        X.loc[1, "c"] = None
    elif kind == "all-missing-column":
        X["x"] = np.nan
    else:
        X.loc[1] = [np.nan, None, np.nan]
    before = X.copy(deep=True)
    expected = reference(X, categorical=["c"], weights=[2.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = sampling.gower_matrix(X, weight=[2.0, 1.0, 1.0])
    assert_allclose(result, expected, equal_nan=True)
    assert_allclose(result, result.T, equal_nan=True)
    assert_frame_equal(X, before)


def test_undefined_pairs_are_nan_and_observed_constants_are_zero():
    X = pd.DataFrame({"a": [0.0, np.nan, np.nan], "b": [np.nan, 1.0, np.nan]})
    assert_allclose(
        sampling.gower_matrix(X),
        [[0.0, np.nan, np.nan], [np.nan, 0.0, np.nan], [np.nan, np.nan, np.nan]],
        equal_nan=True,
    )
    assert np.isnan(sampling.gower_matrix(X, weight=[0.0, 1.0])[0]).all()
    assert_allclose(sampling.gower_matrix(pd.DataFrame({"a": [7.0, 7.0]})), 0.0)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "category", "string", "boolean"])
def test_extension_dtypes_and_missing_values(dtype):
    values = [False, True, None] if dtype == "boolean" else [1, 2, None]
    X = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    assert_allclose(
        sampling.gower_matrix(X),
        [[0.0, 1.0, np.nan], [1.0, 0.0, np.nan], [np.nan, np.nan, np.nan]],
        equal_nan=True,
    )


def test_category_identity_is_not_stringified():
    X = pd.DataFrame({"cat": pd.Series([1, "1", None, "nan"], dtype=object)})
    assert_allclose(
        sampling.gower_matrix(X), reference(X, categorical=["cat"]), equal_nan=True
    )
    assert sampling.gower_matrix(X)[0, 1] == 1.0
    strings = pd.DataFrame({"cat": ["a", "b"]})
    assert_allclose(sampling.gower_matrix(strings), [[0.0, 1.0], [1.0, 0.0]])


@pytest.mark.parametrize("mask", ["auto", [1], [False, True], np.array([False, True])])
def test_mixed_object_array_soft_inference_skips_leading_missing(mask):
    X = np.array([[None, "a"], [5.0, "a"], [10.0, "b"]], dtype=object)
    frame = pd.DataFrame({"x": [np.nan, 5.0, 10.0], "c": ["a", "a", "b"]})
    before = X.copy()
    assert_allclose(
        sampling.gower_matrix(X, cat_features=mask),
        reference(frame, categorical=["c"]),
        equal_nan=True,
    )
    assert X.tolist() == before.tolist()


@pytest.mark.parametrize("mask", [[], [1], np.array([], dtype=int)])
def test_array_positions_and_empty_mask(mask):
    assert_allclose(
        sampling.gower_matrix(np.array([[0.0, 0.0], [1.0, 1.0]]), cat_features=mask),
        [[0.0, 1.0], [1.0, 0.0]],
    )


@pytest.mark.parametrize("missing", [None, pd.NA, np.nan])
def test_object_array_numeric_inference_ignores_missing_sentinels(missing):
    X = np.array([[missing], [0.0], [5.0], [10.0]], dtype=object)
    expected = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]])
    result = sampling.gower_matrix(X)
    assert np.isnan(result[0]).all()
    assert np.isnan(result[:, 0]).all()
    assert_allclose(result[1:, 1:], expected)


@pytest.mark.parametrize(
    "weights",
    [
        [-1.0, 2.0],
        [0.0, 0.0],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [1.0],
        [[1.0, 1.0]],
        2.0,
        np.array([1 + 1j, 1]),
    ],
)
def test_invalid_feature_weights_fail(weights):
    with pytest.raises(ValueError):
        sampling.gower_matrix(mixed()[["x", "constant"]], weight=weights)


def test_zero_and_extreme_feature_weights_are_stable_per_pair():
    X = pd.DataFrame({"a": [0.0, 10.0, 20.0], "b": [None, None, 5.0]})
    result = sampling.gower_matrix(X, weight=[1e-308, 1e308], cat_features=[])
    assert_allclose(result, [[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]])
    complete = mixed()[["x", "constant"]]
    assert_allclose(
        sampling.gower_matrix(complete, weight=[1e308, 1e308]), reference(complete)
    )
    assert_allclose(sampling.gower_matrix(complete, weight=[0.0, 1.0]), 0.0)


@pytest.mark.parametrize(
    "mask, error",
    [
        ([True], ValueError),
        ([3], ValueError),
        ([-1], ValueError),
        (["absent"], ValueError),
        ([True, 0], TypeError),
        ([1.5], TypeError),
        ([[True, False, False]], TypeError),
        (None, TypeError),
        ("unknown", TypeError),
    ],
)
def test_invalid_category_masks_fail(mask, error):
    with pytest.raises(error):
        sampling.gower_matrix(mixed(), cat_features=mask)


def test_named_masks_need_frame_and_integer_names_do_not_mean_positions():
    with pytest.raises(TypeError, match="Named"):
        sampling.gower_matrix(np.zeros((2, 2)), cat_features=["x"])
    X = mixed()
    X.columns = [10, 50, 100]
    assert_allclose(
        sampling.gower_matrix(X, cat_features=[1]), reference(X, categorical=[50])
    )


@pytest.mark.parametrize("side", ["x", "y"])
def test_sparse_is_rejected_before_shape_or_schema_checks(side):
    X = np.zeros((2, 2))
    sparse = csr_matrix(X)
    with pytest.raises(TypeError, match="Sparse"):
        sampling.gower_matrix(
            sparse if side == "x" else X, sparse if side == "y" else X
        )


@pytest.mark.parametrize(
    "X", [np.arange(5), np.zeros((2, 2, 1)), [[1.0, 2.0]], pd.Series([1.0, 2.0])]
)
def test_non_matrix_inputs_fail(X):
    with pytest.raises(TypeError, match="two-dimensional"):
        sampling.gower_matrix(X)


def test_schema_width_duplicate_and_empty_columns_fail():
    X = mixed()
    with pytest.raises(TypeError, match="same columns"):
        sampling.gower_matrix(X, X.iloc[:, ::-1])
    with pytest.raises(TypeError, match="same columns"):
        sampling.gower_matrix(X, X.iloc[0].iloc[::-1])
    with pytest.raises(ValueError, match="number of features"):
        sampling.gower_matrix(X.to_numpy(), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="unique"):
        sampling.gower_matrix(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(ValueError, match="one feature"):
        sampling.gower_matrix(np.empty((2, 0)))


@pytest.mark.parametrize("values", [[0.0, np.inf], [-np.inf, 0.0], [-1e308, 1e308]])
def test_invalid_numeric_values_and_overflowing_range_fail(values):
    with pytest.raises(ValueError, match="finite|float64"):
        sampling.gower_matrix(pd.DataFrame({"x": values}))


def test_declared_numeric_feature_must_be_numeric():
    with pytest.raises(TypeError, match="must be numeric"):
        sampling.gower_matrix(pd.DataFrame({"x": ["a", "b"]}), cat_features=[])


@pytest.mark.parametrize("sizes", [(0, 0), (0, 2), (2, 0), (1, 1)])
def test_empty_row_shapes_and_singleton(sizes):
    n_x, n_y = sizes
    result = sampling.gower_matrix(np.zeros((n_x, 1)), np.zeros((n_y, 1)))
    assert result.shape == sizes
    assert_allclose(result, 0.0)


def test_sampling_caller_uses_nominal_columns_with_integer_names(monkeypatch):
    X = pd.DataFrame({10: [0.0, 2.0, 4.0], 50: ["a", "a", "b"]})
    original = sampling._gower_condensed
    seen = {}

    def capture(X, num_cols, non_num_cols):
        seen["nominal"] = non_num_cols
        seen["distances"] = original(X, num_cols, non_num_cols)
        return seen["distances"]

    monkeypatch.setattr(sampling, "_gower_condensed", capture)
    result = sampling.sample(X, n=1)
    assert seen["nominal"] == [50]
    assert_allclose(
        seen["distances"], reference(X, categorical=[50])[np.triu_indices(len(X), k=1)]
    )
    assert result[10].iloc[0] == 2.0
    assert result[50].iloc[0] == "a"


def test_nearest_neighbor_caller_without_key_uses_repaired_numeric_distance():
    X = pd.DataFrame({"x": [-3.0, -2.0, -1.0]})
    query = pd.DataFrame({"x": [-3.0]})
    result = sampling.gower_topn(X, query, n=2)
    assert list(result["index"]) == [0, 1]
    assert_allclose(result["values"], [0.0, 0.5])
