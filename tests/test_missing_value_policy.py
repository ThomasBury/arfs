"""C03: one explicit row policy, with nominal missingness kept distinct."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from arfs import association as a


@pytest.mark.parametrize("policy", ["typo", "", False, 0, [], pd.NA])
@pytest.mark.parametrize("caller", ["matrix", "series", "regression", "classification"])
def test_invalid_missing_policy_is_rejected(policy, caller):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "z": [3.0, 1.0, 2.0, 0.0]})
    with pytest.raises(ValueError, match="handle_na"):
        if caller == "matrix":
            a.association_matrix(X, handle_na=policy)
        elif caller == "series":
            a.association_series(X, "z", handle_na=policy)
        elif caller == "regression":
            a.f_stat_regression_parallel(X, X.z, handle_na=policy)
        else:
            a.f_stat_classification_parallel(X, [0, 1, 0, 1], handle_na=policy)


@pytest.mark.parametrize("dtype", [object, "category", "string", "boolean"])
@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_fill_nominal_missingness_is_distinct_and_preserves_inputs(dtype, missing):
    values = (
        [True, False, missing, True, missing, False]
        if dtype == "boolean"
        else ["a", "__arfs_missing__", missing, "a", missing, "__arfs_missing___"]
    )
    X = pd.DataFrame(
        {
            "cat": pd.Series(values, dtype=dtype),
            "num": pd.Series([1, None, 3, 4, 5, 6], dtype="Int64"),
        }
    )
    X.index = [8, 3, 9, 2, 7, 1]
    before = X.copy(deep=True)
    weights = np.arange(1.0, 7.0)
    checked, retained = a._check_association_input(X, weights, "fill")
    assert not checked.isna().any().any()
    assert checked.cat.iloc[2] == checked.cat.iloc[4]
    assert checked.cat.iloc[2] not in X.cat.dropna().tolist()
    assert_array_equal(checked.cat.iloc[[0, 1, 3, 5]], X.cat.iloc[[0, 1, 3, 5]])
    assert checked.num.tolist() == [1, 0, 3, 4, 5, 6]
    assert checked.num.dtype == X.num.dtype
    assert_array_equal(checked.index, X.index)
    assert_array_equal(retained, weights)
    assert_frame_equal(X, before)


def test_fill_avoids_unused_category_collisions_and_preserves_order():
    dtype = pd.CategoricalDtype(
        ["a", "__arfs_missing__", "__arfs_missing___"], ordered=True
    )
    X = pd.DataFrame({"cat": pd.Series(["a", None, "a"], dtype=dtype)})
    before = X.copy(deep=True)
    checked, _ = a._check_association_input(X, handle_na="fill")
    assert checked.cat.iloc[1] not in dtype.categories
    assert checked.cat.cat.ordered
    assert_array_equal(checked.cat.cat.categories[:3], dtype.categories)
    assert_frame_equal(X, before)


@pytest.mark.parametrize("dtype", [object, "category", "string", "boolean"])
def test_fill_handles_all_missing_nominal_columns(dtype):
    X = pd.DataFrame({"cat": pd.Series([None, None, None], dtype=dtype)})
    checked, _ = a._check_association_input(X, handle_na="fill")
    assert not checked.cat.isna().any()
    assert checked.cat.nunique() == 1
    assert X.cat.isna().all()


@pytest.mark.parametrize("dtype", [object, "category"])
def test_fill_preserves_mixed_nominal_identities_including_numeric_zero(dtype):
    X = pd.DataFrame({"cat": pd.Series([0, "0", None, 1, None, "1"], dtype=dtype)})
    before = X.copy(deep=True)
    checked, _ = a._check_association_input(X, handle_na="fill")
    codes, _ = pd.factorize(checked.cat)
    assert_array_equal(codes, [0, 1, 2, 3, 2, 4])
    assert_frame_equal(X, before)


def test_fill_leaves_complete_columns_and_numeric_zero_policy_unchanged():
    X = pd.DataFrame(
        {
            "cat": pd.Series([True, False, True], dtype="boolean"),
            "text": pd.Series(["a", "b", "c"], dtype="string"),
            "num": [0.0, np.nan, 2.0],
        }
    )
    before = X.copy(deep=True)
    checked, _ = a._check_association_input(X, handle_na="fill")
    assert_frame_equal(checked[["cat", "text"]], X[["cat", "text"]])
    assert_array_equal(checked.num, [0.0, 0.0, 2.0])
    assert_frame_equal(X, before)


@pytest.mark.parametrize("jobs", [1, 2])
def test_public_fill_matches_explicit_nominal_reference(jobs):
    X = pd.DataFrame(
        {
            "cat": pd.Series(["a", None, "a", None, "b", "b"], dtype="string"),
            "flag": pd.Series([True, None, False, None, True, False], dtype="boolean"),
            "num": [1.0, 3.0, np.nan, 4.0, 8.0, 7.0],
        }
    )
    before = X.copy(deep=True)
    reference = X.astype({"cat": object, "flag": object}).fillna(
        {"cat": "missing", "flag": "missing", "num": 0.0}
    )
    weights = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
    actual = a.association_matrix(X, weights, n_jobs=jobs, handle_na="fill")
    expected = a.association_matrix(reference, weights, n_jobs=jobs)
    assert_frame_equal(actual, expected)
    assert_allclose(
        a.association_series(
            X, "num", sample_weight=weights, n_jobs=jobs, handle_na="fill"
        ),
        a.association_series(reference, "num", sample_weight=weights, n_jobs=jobs),
    )
    for func, target in [
        (a.f_stat_regression_parallel, np.arange(6.0)),
        (
            a.f_stat_classification_parallel,
            pd.Series([0, 1, 0, 1, 0, 1], dtype="category"),
        ),
    ]:
        assert_allclose(
            func(X, target, weights, n_jobs=jobs, handle_na="fill"),
            func(reference, target, weights, n_jobs=jobs),
        )
    assert_frame_equal(X, before)


def test_drop_remains_listwise_and_aligns_response_and_frequency_weights():
    X = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0, 5.0], "b": [2.0, 4.0, np.nan, 8.0, 7.0]},
        index=[9, 2, 8, 1, 7],
    )
    y = pd.Series([5.0, 1.0, 9.0, 6.0, 2.0])
    weights = [1.0, 20.0, 30.0, 2.0, 3.0]
    checked, target, retained = a._check_relevance_input(X, y, weights, "drop")
    assert_array_equal(checked.index, [9, 1, 7])
    assert_array_equal(target, [5.0, 6.0, 2.0])
    assert_array_equal(retained, [1.0, 2.0, 3.0])
    expected = a.f_cont_regression_parallel(X.iloc[[0, 3, 4]], target, retained)
    assert_allclose(a.f_cont_regression_parallel(X, y, weights), expected)


@pytest.mark.parametrize("policy", ["drop", "fill", None])
def test_fill_does_not_hide_invalid_response_or_weights(policy):
    X = pd.DataFrame({"x": [1.0, np.nan, 2.0, 3.0]})
    with pytest.raises(ValueError, match="responses"):
        a.f_stat_regression_parallel(X, [1.0, np.nan, 2.0, 3.0], handle_na=policy)
    with pytest.raises(ValueError, match="sample_weight"):
        a.association_matrix(X, [1.0, -1.0, 1.0, 1.0], handle_na=policy)
