"""Bounded isolation regressions promoted from review 05 F9."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from arfs import sampling


@pytest.mark.parametrize("length", [1, 2, 3, 10, 21, 40, 99, 100, 101])
def test_sizes_are_distinct_rounded_up_five_percent_proper_subsets(length):
    expected = sorted(
        {
            min(length - 1, math.ceil(length * percent / 100))
            for percent in range(5, 100, 5)
        }
        - {0}
    )
    assert_array_equal(sampling.get_5_percent_splits(length), expected)


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_invalid_split_lengths_fail_clearly(length):
    with pytest.raises(ValueError, match="positive integer"):
        sampling.get_5_percent_splits(length)


@pytest.mark.parametrize("length", [2, 10, 21, 100])
def test_rejection_retries_each_size_twenty_times_then_returns_full_data(
    monkeypatch, length
):
    X = pd.DataFrame({"x": np.arange(length)})
    sizes = []
    fits = []

    def scores(X, sample_weight):
        fits.append(sample_weight)
        return np.arange(len(X), dtype=float)

    def reject(full, part):
        sizes.append(len(part))
        assert len(sizes) <= 380, "retry bound exceeded"
        assert_array_equal(full, np.arange(length))
        assert len(set(part)) == len(part)
        return SimpleNamespace(pvalue=0.0)

    monkeypatch.setattr(sampling, "isolation_forest", scores)
    monkeypatch.setattr(sampling, "ks_2samp", reject)
    result = sampling.isof_find_sample(X, np.ones(length))
    expected = sorted(
        {
            min(length - 1, math.ceil(length * percent / 100))
            for percent in range(5, 100, 5)
        }
    )
    assert sizes == [size for size in expected for _ in range(20)]
    assert_array_equal(result, np.arange(length))
    assert len(fits) == 1
    assert_array_equal(fits[0], np.ones(length))


def test_acceptance_stops_at_first_accepted_draw_and_keeps_strict_threshold(
    monkeypatch,
):
    sizes = []
    monkeypatch.setattr(
        sampling, "isolation_forest", lambda X, sample_weight: np.arange(len(X))
    )

    def compare(full, part):
        sizes.append(len(part))
        return SimpleNamespace(pvalue=0.95 if len(sizes) < 22 else 0.96)

    monkeypatch.setattr(sampling, "ks_2samp", compare)
    result = sampling.isof_find_sample(pd.DataFrame({"x": np.arange(100)}))
    assert sizes == [5] * 20 + [10] * 2
    assert len(result) == len(set(result)) == 10
    assert ((result >= 0) & (result < 100)).all()


@pytest.mark.parametrize("rows", [0, 1])
def test_tiny_isolation_inputs_return_identity_without_fitting(monkeypatch, rows):
    X = pd.DataFrame(
        {"x": np.arange(rows), "c": pd.Series(["a"] * rows, dtype="category")}
    )
    X.index = [8] * rows

    def unexpected(*args, **kwargs):
        pytest.fail("tiny input must not fit a forest or evaluate KS")

    monkeypatch.setattr(sampling, "isolation_forest", unexpected)
    monkeypatch.setattr(sampling, "ks_2samp", unexpected)
    assert_array_equal(sampling.isof_find_sample(X), np.arange(rows))
    result = sampling.sample(X, method="isoforest")
    assert_frame_equal(result, X)
    assert result is not X


def test_public_exhaustion_returns_original_schema_and_order(monkeypatch):
    X = pd.DataFrame(
        {"x": np.arange(10), "c": pd.Series(["a", "b"] * 5, dtype="category")}
    )
    X.index = [3, 3, 8, 5, 6, 7, 8, 9, 10, 11]
    before = X.copy(deep=True)
    monkeypatch.setattr(
        sampling, "isolation_forest", lambda X, sample_weight: np.arange(len(X))
    )
    calls = []

    def reject(full, part):
        calls.append(len(part))
        assert len(calls) <= 380, "retry bound exceeded"
        return SimpleNamespace(pvalue=0.0)

    monkeypatch.setattr(sampling, "ks_2samp", reject)
    result = sampling.sample(X, method="isoforest")
    assert_frame_equal(result, before)
    assert_frame_equal(X, before)


@pytest.mark.parametrize("rows", [2, 10, 40])
def test_real_constant_forest_accepts_first_five_percent_draw(rows):
    X = pd.DataFrame({"x": np.ones(rows), "c": ["a"] * rows})
    result = sampling.sample(X, method="isoforest")
    assert len(result) == max(1, math.ceil(rows * 0.05))
    assert_frame_equal(result, X.loc[result.index])


@pytest.mark.parametrize("X", [np.zeros(1), np.zeros((1, 0)), np.zeros((0, 0)), [[1]]])
def test_invalid_direct_predictor_matrix_rejected_before_tiny_identity(X):
    with pytest.raises(ValueError, match="predictor matrix"):
        sampling.isof_find_sample(X)


def test_seed_replay_for_real_weighted_ten_row_sampler():
    X = pd.DataFrame(
        {"x": np.arange(10, dtype=float), "c": ["a", "b"] * 5}, index=np.arange(10) + 50
    )
    weights = np.arange(10, dtype=float)
    state = np.random.get_state()
    try:
        np.random.seed(61)
        first = sampling.sample(X, sample_weight=weights, method="isoforest")
        np.random.seed(61)
        second = sampling.sample(X, sample_weight=weights, method="isoforest")
    finally:
        np.random.set_state(state)
    assert 1 <= len(first) <= len(X)
    assert first.index.is_unique
    assert_frame_equal(first, second)
    assert_frame_equal(first, X.loc[first.index])
