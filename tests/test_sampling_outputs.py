"""Sampling output regressions promoted from review 05 F6-F8/F10."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from arfs import sampling


def test_numeric_imputation_uses_each_column_mean():
    X = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [100.0, np.nan, 200.0]})
    before = X.copy(deep=True)
    result = sampling.sample(X, n=1)
    assert_allclose(result.iloc[0], [2.0, 150.0])
    assert_frame_equal(X, before)


@pytest.mark.parametrize("label", [None, "numeric", "category"])
def test_cluster_labels_do_not_replace_features(label):
    X = pd.DataFrame({"x": [0.0, 2.0, 10.0, 12.0]})
    if label:
        X["label"] = X.x if label == "numeric" else ["a", "a", "b", "b"]
    result = sampling.sample(X, n=1)
    assert result.columns.equals(X.columns)
    assert result.index.tolist() == ["clus_0"]
    if label == "numeric":
        assert_allclose(result.label, [6.0])
    elif label == "category":
        assert result.label.iloc[0] == "a"


@pytest.mark.parametrize("columns", [["c", "x"], [50, 10]])
def test_summary_preserves_column_order_and_input(columns):
    X = pd.DataFrame(
        [["a", 0.0], ["a", 2.0], ["b", 10.0], ["b", 12.0]],
        columns=columns,
        index=[8, 8, 90, 30],
    )
    before = X.copy(deep=True)
    result = sampling.sample(X, n=2)
    assert result.columns.equals(X.columns)
    assert_allclose(sorted(result[columns[1]]), [1.0, 11.0])
    assert set(result[columns[0]]) == {"a", "b"}
    assert_frame_equal(X, before)


def test_unknown_method_raises():
    with pytest.raises(NotImplementedError, match="unknown"):
        sampling.sample(pd.DataFrame({"x": [0.0, 1.0]}), method="unknown")


@pytest.mark.parametrize("dtype", [object, "category", "string", "boolean"])
def test_isolation_returns_original_values_indexes_and_dtypes(monkeypatch, dtype):
    values = [True, False, True] if dtype == "boolean" else ["b", None, "a"]
    X = pd.DataFrame({"c": pd.Series(values, dtype=dtype), "x": [0.0, 1.0, 2.0]})
    X.index = [8, 8, 90]
    before = X.copy(deep=True)
    monkeypatch.setattr(
        sampling, "isof_find_sample", lambda X, sample_weight: np.array([2, 0])
    )
    result = sampling.sample(X, method="isoforest")
    assert_frame_equal(result, X.iloc[[2, 0]])
    assert_frame_equal(X, before)


@pytest.mark.parametrize("container", [list, np.array, pd.Series])
def test_isolation_forwards_positional_row_weights(monkeypatch, container):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]}, index=[5, 5, 3])
    weights = container([1.0, 2.0, 3.0])
    seen = {}

    def select(X, sample_weight):
        seen["weights"] = sample_weight
        return np.array([2, 0])

    monkeypatch.setattr(sampling, "isof_find_sample", select)
    assert_frame_equal(
        sampling.sample(X, sample_weight=weights, method="isoforest"), X.iloc[[2, 0]]
    )
    assert_array_equal(seen["weights"], weights)


@pytest.mark.parametrize("rows", [2, 3])
def test_gower_rejects_row_weights_even_when_rows_equal_features(rows):
    X = pd.DataFrame({"a": np.arange(rows), "b": np.arange(rows)})
    with pytest.raises(ValueError, match="row weights"):
        sampling.sample(X, n=1, sample_weight=np.ones(rows))


@pytest.mark.parametrize(
    "entry", [sampling.sample, sampling.isof_find_sample, sampling.isolation_forest]
)
@pytest.mark.parametrize(
    "weights",
    [[1, 2], [[1], [2], [3]], [0, 0, 0], [1, -1, 1], [1, np.nan, 1], [1, np.inf, 1]],
)
def test_invalid_isolation_weights_rejected_before_fitting(entry, weights, monkeypatch):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    if entry is sampling.sample:
        kwargs = {"method": "isoforest"}
    else:
        kwargs = {}
    with pytest.raises(ValueError, match="sample_weight"):
        entry(X, sample_weight=weights, **kwargs)


@pytest.mark.parametrize(
    "values",
    [
        [np.nan, np.nan],
        pd.Series([None, None], dtype=object),
        pd.Series([None, None], dtype="category"),
        pd.Series([None, None], dtype="string"),
    ],
)
def test_gower_rejects_entirely_missing_columns(values):
    with pytest.raises(ValueError, match="observed value"):
        sampling.sample(pd.DataFrame({"x": [0, 1], "missing": values}), n=1)


@pytest.mark.parametrize("n", [0, -1, 4, 1.5, True])
def test_invalid_cluster_count_rejected(n):
    with pytest.raises(ValueError, match="n"):
        sampling.sample(pd.DataFrame({"x": [0, 1, 2]}), n=n)


@pytest.mark.parametrize("rows", [0, 1])
def test_gower_tiny_inputs_rejected(rows):
    with pytest.raises(ValueError, match="at least two rows"):
        sampling.sample(pd.DataFrame({"x": np.arange(rows)}), n=1)


def test_all_categorical_summary_and_mode_imputation():
    X = pd.DataFrame(
        {
            "c": pd.Series(["a", None, "b"], dtype="category"),
            "d": pd.Series([True, None, True], dtype="boolean"),
        }
    )
    result = sampling.sample(X, n=1)
    assert result.columns.equals(X.columns)
    assert result.c.iloc[0] == "a"
    assert result.d.iloc[0]


def test_real_isolation_fit_forwards_weights(monkeypatch):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    seen = {}
    original = sampling.IsolationForest.fit

    def fit(self, X, y=None, sample_weight=None):
        seen["rows"] = X.copy()
        seen["weights"] = sample_weight.copy()
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(sampling.IsolationForest, "fit", fit)
    scores = sampling.isolation_forest(X, [1, 0, 3])
    assert_array_equal(seen["weights"], [1, 3])
    assert_frame_equal(seen["rows"], X.iloc[[0, 2]])
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()


def test_nullable_integer_imputation_can_have_fractional_mean():
    X = pd.DataFrame({"x": pd.Series([0, 3, None], dtype="Int64")})
    assert_allclose(sampling.sample(X, n=1).x, [1.5])


@pytest.mark.parametrize("method", ["gower", "isoforest"])
def test_invalid_feature_schema_rejected(method):
    with pytest.raises(ValueError, match="unique columns"):
        sampling.sample(
            pd.DataFrame([[0, 1], [2, 3]], columns=["x", "x"]), n=1, method=method
        )
    with pytest.raises(ValueError, match="at least one feature"):
        sampling.sample(pd.DataFrame(index=[0, 1]), n=1, method=method)
    with pytest.raises(TypeError, match="DataFrame"):
        sampling.sample(np.ones((2, 2)), n=1, method=method)
