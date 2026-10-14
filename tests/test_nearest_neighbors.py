"""Query/count regressions promoted from review 05 F11-F12."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from arfs import sampling


@pytest.mark.parametrize("query_position, expected", [(2, [1, 0]), (0, [1, 2])])
def test_query_changes_key_neighbors_and_excludes_its_identity(
    query_position, expected
):
    X = pd.DataFrame({"id": ["a", "b", "c"], "x": [0.0, 5.0, 10.0]}, index=[20, 20, 90])
    before = X.copy(deep=True)
    query = X.iloc[[query_position]].copy()
    result = sampling.gower_topn(X, query, key="id", n=2)
    assert_array_equal(result["index"], expected)
    assert_allclose(result["values"], [0.5, 1.0])
    assert_frame_equal(X, before)
    assert_frame_equal(query, X.iloc[[query_position]])


def test_duplicate_ids_choose_the_nearest_row_without_overshooting():
    X = pd.DataFrame({"id": ["a", "a", "a", "b", "c"], "x": [0.0, 0.0, 0.0, 5.0, 10.0]})
    query = pd.DataFrame({"id": ["q"], "x": [9.0]})
    result = sampling.gower_topn(X, query, key="id", n=3)
    assert_array_equal(result["index"], [4, 3, 0])
    assert_allclose(result["values"], [0.1, 0.4, 0.9])


def test_nearest_duplicate_row_is_retained_and_all_query_id_rows_excluded():
    X = pd.DataFrame({"id": ["a", "a", "b", "q", "q"], "x": [0.0, 9.0, 5.0, 10.0, 8.0]})
    query = pd.DataFrame({"id": ["q"], "x": [10.0]})
    result = sampling.gower_topn(X, query, key="id", n=2)
    assert_array_equal(result["index"], [1, 2])
    assert_allclose(result["values"], [0.1, 0.5])
    with pytest.raises(ValueError, match="eligible"):
        sampling.gower_topn(X, query, key="id", n=3)


@pytest.mark.parametrize("kind", ["array", "frame", "series"])
def test_no_key_keeps_self_neighbors_and_supports_query_containers(kind):
    X = pd.DataFrame({"x": [0.0, 2.0, 10.0]}, index=[50, 50, 90])
    query = X.iloc[[1]]
    if kind == "array":
        X, query = X.to_numpy(), query.to_numpy()
    elif kind == "series":
        query = query.iloc[0]
    result = sampling.gower_topn(X, query, n=3)
    assert_array_equal(result["index"], [1, 0, 2])
    assert_allclose(result["values"], [0.0, 0.2, 0.8])


def test_key_series_query_and_feature_weights_keep_candidate_dtype_roles():
    X = pd.DataFrame(
        {"id": ["a", "b", "c"], "x": [0.0, 5.0, 10.0], "c": ["u", "u", "v"]}
    )
    query = X.iloc[2]
    result = sampling.gower_topn(
        X, query, key="id", n=2, weight=[3.0, 1.0], cat_features=["c"]
    )
    assert_array_equal(result["index"], [1, 0])
    assert_allclose(result["values"], [0.625, 1.0])


def test_only_one_candidate_by_query_distance_vector_is_allocated(monkeypatch):
    X = pd.DataFrame({"id": ["a", "b", "c"], "x": [0.0, 5.0, 10.0]})
    query = X.iloc[[2]].copy()
    original = sampling.gower_matrix
    shapes = []

    def distance(candidates, reference, *args, **kwargs):
        shapes.append((candidates.shape, reference.shape))
        assert candidates.columns.tolist() == reference.columns.tolist() == ["x"]
        assert_frame_equal(reference, query.drop(columns=["id"]))
        result = original(candidates, reference, *args, **kwargs)
        assert result.shape == (len(X), 1)
        return result

    monkeypatch.setattr(sampling, "gower_matrix", distance)
    sampling.gower_topn(X, query, key="id", n=2)
    assert shapes == [((3, 1), (1, 1))]


def test_mixed_and_tuple_ids_are_not_coerced_and_missing_ids_are_ineligible():
    X = pd.DataFrame({"id": [1, "1", (2, "x"), None], "x": [0.0, 8.0, 5.0, 9.0]})
    query = pd.DataFrame({"id": [(9, "q")], "x": [9.0]})
    result = sampling.gower_topn(X, query, key="id", n=3)
    assert_array_equal(result["index"], [1, 2, 0])
    assert_allclose(result["values"], [1 / 9, 4 / 9, 1.0])


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_missing_query_id_rejected(missing):
    X = pd.DataFrame({"id": ["a", "b"], "x": [0.0, 1.0]})
    query = pd.DataFrame({"id": [missing], "x": [0.0]})
    with pytest.raises(ValueError, match="nonmissing"):
        sampling.gower_topn(X, query, key="id", n=1)


def test_undefined_distances_are_excluded_without_nan_replacement():
    X = pd.DataFrame({"x": [np.nan, 0.0, 2.0]})
    query = pd.DataFrame({"x": [0.0]})
    result = sampling.gower_topn(X, query, n=2)
    assert_array_equal(result["index"], [1, 2])
    assert_allclose(result["values"], [0.0, 1.0])
    with pytest.raises(ValueError, match="eligible"):
        sampling.gower_topn(X, query, n=3)


def test_nearest_finite_duplicate_is_used_when_first_duplicate_is_undefined():
    X = pd.DataFrame({"id": ["a", "a", "b"], "x": [np.nan, 0.0, 2.0]})
    query = pd.DataFrame({"id": ["q"], "x": [0.0]})
    result = sampling.gower_topn(X, query, key="id", n=2)
    assert_array_equal(result["index"], [1, 2])
    assert_allclose(result["values"], [0.0, 1.0])


@pytest.mark.parametrize("key", [None, "id"])
def test_zero_count_returns_empty_and_no_eligible_neighbors_raise_for_positive_count(
    key,
):
    X = pd.DataFrame({"id": ["a", "a"], "x": [np.nan, np.nan]})
    query = pd.DataFrame({"id": ["a"], "x": [np.nan]})
    result = sampling.gower_topn(X, query, key=key, n=0)
    assert result["index"].dtype.kind in "iu"
    assert result["index"].shape == result["values"].shape == (0,)
    if key:
        with pytest.raises(ValueError, match="eligible"):
            sampling.gower_topn(X, query, key=key, n=1)


def test_empty_candidate_axis():
    X = pd.DataFrame({"x": pd.Series([], dtype=float)})
    query = pd.DataFrame({"x": [0.0]})
    assert sampling.gower_topn(X, query, n=0)["index"].size == 0
    with pytest.raises(ValueError, match="eligible"):
        sampling.gower_topn(X, query, n=1)


@pytest.mark.parametrize("n", [-1, 1.5, True, np.nan, "1"])
@pytest.mark.parametrize("entry", ["smallest", "topn"])
def test_invalid_neighbor_counts_rejected(n, entry):
    with pytest.raises(ValueError, match="nonnegative integer"):
        if entry == "smallest":
            sampling.smallest_indices(np.array([0.0, 1.0]), n)
        else:
            sampling.gower_topn(np.array([[0.0], [1.0]]), np.array([[0.0]]), n=n)


def test_smallest_indices_excludes_all_nonfinite_values_and_stabilizes_ties():
    values = np.array([np.nan, np.inf, -np.inf, 0.5, 0.1, 0.1])
    before = values.copy()
    result = sampling.smallest_indices(values, 3)
    assert_array_equal(result["index"], [4, 5, 3])
    assert_allclose(result["values"], [0.1, 0.1, 0.5])
    assert_array_equal(values, before)
    with pytest.raises(ValueError, match="finite"):
        sampling.smallest_indices(values, 4)
    assert sampling.smallest_indices(values, 0)["index"].size == 0


def test_smallest_indices_preserves_flattening_and_integer_values():
    result = sampling.smallest_indices(np.array([[3, 1], [1, 2]]), np.int64(3))
    assert_array_equal(result["index"], [1, 2, 3])
    assert_array_equal(result["values"], [1, 1, 2])
    assert result["values"].dtype.kind in "iu"
    result = sampling.smallest_indices(np.array([]), 0)
    assert result["index"].shape == result["values"].shape == (0,)


@pytest.mark.parametrize(
    "query", [None, np.array([0.0]), np.empty((0, 1)), np.zeros((2, 1))]
)
def test_explicit_one_row_query_required(query):
    with pytest.raises((TypeError, ValueError), match="data_y|1 row"):
        sampling.gower_topn(np.array([[0.0], [1.0]]), query, n=1)


@pytest.mark.parametrize(
    "X, query, key",
    [
        (np.array([[0.0], [1.0]]), np.array([[0.0]]), "id"),
        (pd.DataFrame({"x": [0.0, 1.0]}), pd.DataFrame({"x": [0.0]}), "id"),
        (
            pd.DataFrame([["a", "b", 0.0]], columns=["id", "id", "x"]),
            pd.DataFrame({"id": ["q"], "x": [0.0]}),
            "id",
        ),
    ],
)
def test_invalid_key_schema_rejected(X, query, key):
    with pytest.raises((TypeError, ValueError), match="key|unique"):
        sampling.gower_topn(X, query, key=key, n=0)


def test_stable_key_ties_and_integer_named_key():
    X = pd.DataFrame({10: ["a", "a", "b"], 50: [0.0, 0.0, 0.0]}, index=[8, 8, 30])
    query = pd.DataFrame({10: ["q"], 50: [0.0]})
    result = sampling.gower_topn(X, query, key=10, n=2)
    assert_array_equal(result["index"], [0, 2])
    assert_allclose(result["values"], [0.0, 0.0])


@pytest.mark.parametrize("dtype", ["string", "category"])
def test_nullable_identifier_dtypes_keep_nonmissing_unique_neighbors(dtype):
    X = pd.DataFrame(
        {"id": pd.Series(["a", "a", None, "b"], dtype=dtype), "x": [0.0, 8.0, 9.0, 5.0]}
    )
    query = pd.DataFrame({"id": ["q"], "x": [9.0]})
    result = sampling.gower_topn(X, query, key="id", n=2)
    assert_array_equal(result["index"], [1, 3])
    assert_allclose(result["values"], [1 / 9, 4 / 9])


@pytest.mark.parametrize(
    "values", [np.array([1 + 2j]), np.array(["1"]), np.array([None])]
)
def test_smallest_indices_requires_real_numeric_values(values):
    with pytest.raises(TypeError, match="real numeric"):
        sampling.smallest_indices(values, 0)


def test_zero_neighbors_still_validates_distance_schema_masks_and_weights():
    X = pd.DataFrame({"x": [0.0, 1.0]})
    query = X.iloc[[0]]
    with pytest.raises(TypeError, match="same columns"):
        sampling.gower_topn(X, query.rename(columns={"x": "y"}), n=0)
    with pytest.raises(ValueError, match="weight"):
        sampling.gower_topn(X, query, weight=[1.0, 2.0], n=0)
    with pytest.raises(ValueError, match="cat_features"):
        sampling.gower_topn(X, query, cat_features=["missing"], n=0)


def test_all_undefined_query_distances_have_no_neighbor():
    X = pd.DataFrame({"x": [0.0, 1.0]})
    query = pd.DataFrame({"x": [np.nan]})
    assert sampling.gower_topn(X, query, n=0)["index"].size == 0
    with pytest.raises(ValueError, match="eligible"):
        sampling.gower_topn(X, query, n=1)
