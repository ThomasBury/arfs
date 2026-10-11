"""Column replacement preserves caller data under pandas Copy-on-Write."""

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.pipeline import Pipeline

from arfs.feature_selection.allrelevant import _create_shadow
from arfs.preprocessing import (
    IntervalToMidpoint,
    TreeDiscretizer,
    find_interval_midpoint,
    transform_interval_to_midpoint,
)


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


def test_shadow_preserves_values_dtypes_index_and_input(copy_on_write):
    X = pd.DataFrame(
        {
            "number": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "integer": np.arange(6, dtype=np.int32),
            "category": pd.Categorical(
                [2, 1, None, 2, 3, 1], categories=[3, 2, 1, 0], ordered=True
            ),
            "text": pd.array(["a", "b", None, "c", "b", "a"], dtype="string"),
            "nullable": pd.array([1, None, 3, 4, 5, 6], dtype="Int64"),
        },
        index=pd.Index([40, 10, 40, 20, 70, 60], name="row"),
    )
    before = X.copy(deep=True)
    state = np.random.get_state()
    try:
        np.random.seed(42)
        augmented, names = _create_shadow(X)
    finally:
        np.random.set_state(state)

    assert augmented.shape == (len(X), 2 * X.shape[1])
    assert augmented.columns.tolist() == X.columns.tolist() + names
    assert_frame_equal(augmented[X.columns], before)
    assert_frame_equal(X, before)
    assert augmented.index.equals(X.index)
    assert not augmented[names[0]].equals(X.iloc[:, 0].rename(names[0]))
    for column, shadow in zip(X.columns, names):
        assert augmented[shadow].dtype == X[column].dtype
        expected = X[column].value_counts(dropna=False)
        actual = augmented[shadow].value_counts(dropna=False).reindex(expected.index)
        assert_series_equal(actual, expected, check_names=False)

    augmented.loc[10, names[0]] = -99.0
    assert_frame_equal(X, before)
    assert_frame_equal(augmented[X.columns], before)


@pytest.mark.parametrize("api", ["estimator", "function"])
def test_interval_public_midpoints(api, copy_on_write):
    X = pd.DataFrame(
        {
            "interval": pd.arrays.IntervalArray.from_tuples(
                [(-np.inf, 2.0), (2.0, 6.0), (6.0, np.inf), (np.nan, np.nan)]
            ),
            "number": [1, 2, 3, 4],
            "category": pd.Categorical(["a", "b", "a", "b"]),
        },
        index=[40, 10, 40, 20],
    )
    before = X.copy(deep=True)
    actual = (
        IntervalToMidpoint().fit_transform(X)
        if api == "estimator"
        else transform_interval_to_midpoint(X)
    )
    expected = before.copy()
    expected["interval"] = [2.0, 4.0, 6.0, np.nan]
    assert_frame_equal(actual, expected)
    assert_frame_equal(X, before)


def test_interval_midpoint_reference():
    intervals = pd.Series(
        pd.arrays.IntervalArray.from_tuples(
            [(-np.inf, 2.0), (2.0, 6.0), (6.0, np.inf), (np.nan, np.nan)]
        )
    )
    assert_allclose(find_interval_midpoint(intervals), [2.0, 4.0, 6.0, np.nan])


def test_tree_midpoint_pipeline(copy_on_write):
    X = pd.DataFrame(
        {"number": np.arange(240, dtype=float)}, index=np.arange(240) + 1000
    )
    y = np.repeat([0.0, 2.0, -1.0, 3.0, 0.0, 2.0], 40)
    before = X.copy(deep=True)
    tree = TreeDiscretizer(
        n_bins=6,
        boost_params={
            "num_threads": 1,
            "min_data_in_leaf": 5,
            "learning_rate": 1,
            "verbosity": -1,
            "seed": 8,
        },
    )
    intervals = tree.fit_transform(X, y)
    assert isinstance(intervals["number"].dtype, pd.IntervalDtype)
    actual = Pipeline(
        [("tree", tree), ("midpoint", IntervalToMidpoint())]
    ).fit_transform(X, y)
    assert actual.shape == X.shape
    assert actual.index.equals(X.index)
    assert np.isfinite(actual["number"]).all()
    assert_allclose(actual["number"], find_interval_midpoint(intervals["number"]))
    assert_frame_equal(X, before)
