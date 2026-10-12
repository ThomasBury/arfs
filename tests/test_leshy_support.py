"""Leshy support and ranking use the actual recorded iterations."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import arfs.feature_selection.allrelevant as a


@pytest.mark.parametrize("keep_weak", [False, True])
@pytest.mark.parametrize(
    "decisions",
    [[1, 1, 1], [0, 0, 0], [1, 0, -1], [-1, -1, -1], [1, 0, 0]],
    ids=["confirmed", "tentative", "mixed", "empty", "no-rejected"],
)
def test_fitted_mask_count_names_and_transform_agree(monkeypatch, decisions, keep_weak):
    X = pd.DataFrame(np.arange(60.0).reshape(20, 3), columns=["a", "b", "c"])
    before = X.copy(deep=True)
    decisions = np.array(decisions)
    history = np.full((2, 3), 2.0)
    monkeypatch.setattr(
        a.Leshy,
        "select_features",
        lambda self, **kwargs: (decisions, [1.0, 1.0], history, 1.0),
    )
    selector = a.Leshy(
        lgb.LGBMRegressor(verbosity=-1), keep_weak=keep_weak, random_state=0
    ).fit(X, np.arange(len(X)))
    weak = decisions == 0
    expected = (decisions == 1) | (weak & keep_weak)
    assert_array_equal(selector.support_, expected)
    assert_array_equal(selector.get_support(), expected)
    assert_array_equal(selector.support_weak_, weak)
    assert selector.n_features_ == expected.sum()
    assert_array_equal(selector.selected_features_, X.columns[expected])
    assert_array_equal(selector.not_selected_features_, X.columns[~expected])
    assert_frame_equal(selector.transform(X), X.loc[:, expected])
    assert_frame_equal(X, before)
    assert_array_equal(selector.ranking_[decisions == 1], 1)
    assert_array_equal(selector.ranking_[weak], 2)


@pytest.mark.parametrize("keep_weak", [False, True])
def test_two_real_iterations_are_recorded_and_tentative_policy_survives(
    monkeypatch, keep_weak
):
    X = pd.DataFrame(np.arange(60.0).reshape(20, 3), columns=["a", "b", "c"])
    outputs = iter([9.0, 4.0])
    monkeypatch.setattr(
        a,
        "_get_imp",
        lambda est, frame, *args, **kwargs: np.r_[
            np.full(3, next(outputs)), np.full(frame.shape[1] - 3, 5.0)
        ],
    )
    selector = a.Leshy(
        lgb.LGBMRegressor(verbosity=-1),
        importance="native",
        max_iter=2,
        n_estimators=2,
        random_state=0,
        keep_weak=keep_weak,
    ).fit(X, np.arange(len(X)))
    assert_array_equal(selector.imp_real_hist, [[9.0] * 3, [4.0] * 3])
    assert selector.support_weak_.all()
    assert_array_equal(selector.support_, np.full(3, keep_weak))
    assert selector.n_features_ == (3 if keep_weak else 0)
    assert selector.transform(X).shape[1] == selector.n_features_


@pytest.mark.parametrize(
    "history, shadows, expected",
    [
        ([[9.0], [4.0]], [5.0, 5.0], [0]),
        ([[6.0]], [5.0], [0]),
        ([[5.0], [5.0]], [5.0, 5.0], []),
    ],
)
def test_tentative_median_uses_every_actual_row(history, shadows, expected):
    assert_array_equal(
        a._select_tentative(np.array([0]), np.array(history), shadows), expected
    )


@pytest.mark.parametrize("tentative", [np.array([], dtype=int), np.array([2])])
def test_rejected_ranking_uses_both_iterations_and_preserves_support(tentative):
    selector = a.Leshy(lgb.LGBMRegressor(verbosity=-1), keep_weak=False)
    selector._calculate_support(np.array([], dtype=int), tentative, 3)
    before = selector.support_.copy()
    selector._calculate_relative_ranking(
        3,
        tentative,
        np.array([], dtype=int),
        np.array([[9.0, 1.0, 0.0], [1.0, 9.0, 0.0]]),
    )
    base = 3 if len(tentative) else 2
    assert_array_equal(selector.ranking_[:2], [base, base])
    assert_array_equal(selector.support_, before)
