"""C05: selected-column storage agrees with the former dense MRMR calculation."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from arfs.feature_selection import MinRedundancyMaxRelevance as MRMR
from arfs.feature_selection.mrmr import FLOOR


def spread(values):
    """An order-sensitive callable that must keep receiving a labeled Series."""
    assert isinstance(values, pd.Series)
    return values.iloc[0] + values.iloc[-1] / 2


def dense_reference(model):
    """Small square reference using the pre-C05 row-wise reduction."""
    features = model.relevance_.index.tolist()
    table = pd.DataFrame(FLOOR, index=features, columns=features)
    selected, remaining, records = [], features.copy(), []
    for _ in range(model.n_features_to_select_):
        denominator = pd.Series(1.0, index=features)
        if selected:
            last = selected[-1]
            peers = [
                c
                for c in remaining
                if not model.only_same_domain or c.split("_")[0] == last.split("_")[0]
            ]
            if peers:
                table.loc[peers, last] = (
                    model.redundancy_func_(
                        target=last, features=peers, **model.redundancy_args
                    )
                    .fillna(FLOOR)
                    .abs()
                    .clip(FLOOR)
                )
                denominator = (
                    table.loc[remaining, selected]
                    .apply(model.denominator_func, axis=1)
                    .replace(1.0, np.inf)
                )
        score = model.relevance_.loc[remaining] / denominator
        best = score.sort_values(ascending=False, kind="stable").idxmax()
        records.append([score[best], model.relevance_[best], denominator[best]])
        selected.append(best)
        remaining.remove(best)
    ranking = pd.DataFrame(
        records, index=selected, columns=model.ranking_.columns, dtype=float
    )
    ranking["redundancy"] = ranking["redundancy"].replace(1.0, 0.0)
    if selected:
        ranking.iloc[0, 0] = np.inf
    return selected, ranking, table


@pytest.mark.parametrize("reducer", ["mean", "max", np.mean, np.max, spread])
@pytest.mark.parametrize("same_domain", [False, True])
@pytest.mark.parametrize("count", [0, 1, 4, 10])
def test_storage_and_ranking_match_dense_reference(reducer, same_domain, count):
    names = ["a_first", "b_tie", "a_duplicate", "b_zero", "c_alone", "unavailable"]
    X = pd.DataFrame(np.arange(36).reshape(6, 6), columns=names)
    values = np.array(
        [
            [1, -0.5, -1, 0, np.nan, 0.2],
            [0.4, 1, 0.2, 0.5, 1, 0.2],
            [0.3, 0.6, 1, 0, 0.7, 0.2],
            [0.2, 0.1, 0.8, 1, 0.1, 0.2],
            [0.5, 0.2, 0.6, 0.3, 1, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2, 1],
        ]
    )
    pairs = pd.DataFrame(values, index=names, columns=names)
    calls = []

    def redundancy(target, features, **kwargs):
        calls.append((target, features.copy()))
        # Reversed labels verify alignment rather than positional assignment.
        return pairs.loc[features, target].iloc[::-1]

    model = MRMR(
        count,
        relevance_func=lambda **kw: pd.Series([3, 2, 2, 0, 1, np.nan], index=names),
        redundancy_func=redundancy,
        denominator_func=reducer,
        only_same_domain=same_domain,
        show_progress=False,
    ).fit(X, pd.Series(np.arange(6)))
    actual_calls = calls.copy()
    calls.clear()
    selected, ranking, table = dense_reference(model)
    assert model.selected_features_ == selected
    assert_frame_equal(model.ranking_, ranking, check_index_type=False)
    assert calls == actual_calls
    assert model.redundancy.columns.tolist() == selected[:-1]
    assert_frame_equal(
        model.redundancy, table.loc[:, selected[:-1]], check_column_type=False
    )
    assert model.redundancy.shape == (len(model.features), max(len(selected) - 1, 0))
    # Refitting must discard the old columns, including when the new fit is empty.
    model.set_params(n_features_to_select=0).fit(X, pd.Series(np.arange(6)))
    assert model.redundancy.shape == (0, 0)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("reducer", [np.mean, np.max, spread])
@pytest.mark.parametrize("sample_size", [None, 24])
def test_weighted_sampled_default_callbacks_match_dense_reference(
    task, reducer, sample_size
):
    rng = np.random.default_rng(91)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "d"])
    X["category"] = pd.Categorical(np.tile(["x", "y", "z"], 20))
    y = X.a + rng.normal(size=60)
    if task == "classification":
        y = (y > 0).astype("category")
    weights = np.tile([0, 1, 2, 3, 4], 12)
    before = X.copy(deep=True)
    model = MRMR(
        4,
        task=task,
        denominator_func=reducer,
        sample_size=sample_size,
        random_state=4,
        show_progress=False,
    ).fit(X, y, sample_weight=weights)
    selected, ranking, table = dense_reference(model)
    assert model.selected_features_ == selected
    assert_allclose(model.ranking_, ranking, rtol=1e-14)
    assert_frame_equal(
        model.redundancy, table.loc[:, selected[:-1]], check_column_type=False
    )
    assert_frame_equal(X, before)
    rows = np.arange(len(X)) if sample_size is None else model.sample_indices_
    repeated = np.repeat(np.arange(len(rows)), weights[rows])
    replicated = MRMR(4, task=task, denominator_func=reducer, show_progress=False).fit(
        X.iloc[rows].iloc[repeated], y.iloc[rows].iloc[repeated]
    )
    # Frequency-weight replication applies to relevance, not weighted Spearman ranks.
    assert_allclose(replicated.relevance_, model.relevance_, rtol=1e-12, atol=1e-14)
    assert_array_equal(model.get_support(), X.columns.isin(selected))


@pytest.mark.parametrize("reducer", ["mean", "max", np.mean, np.max])
def test_builtin_reducers_do_not_use_row_apply(monkeypatch, reducer):
    def reject_apply(*args, **kwargs):
        raise AssertionError("Built-in reducers must reduce all rows together")

    monkeypatch.setattr(pd.DataFrame, "apply", reject_apply)
    X = pd.DataFrame(np.zeros((3, 8)), columns=[f"x_{i}" for i in range(8)])
    model = MRMR(
        3,
        relevance_func=lambda **kw: pd.Series(1.0, index=X.columns),
        redundancy_func=lambda features, **kw: pd.Series(0.0, index=features),
        denominator_func=reducer,
        show_progress=False,
    ).fit(X, pd.Series([0, 1, 2]))
    assert model.selected_features_ == X.columns[:3].tolist()
    assert model.redundancy.shape == (8, 2)


@pytest.mark.parametrize("reducer", [np.mean, np.max, spread])
def test_large_selection_avoids_fragmentation_warnings(reducer):
    X = pd.DataFrame(np.zeros((3, 140)), columns=[f"x_{i}" for i in range(140)])
    model = MRMR(
        130,
        relevance_func=lambda **kw: pd.Series(1.0, index=X.columns),
        redundancy_func=lambda features, **kw: pd.Series(0.0, index=features),
        denominator_func=reducer,
        show_progress=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        model.fit(X, pd.Series([0, 1, 2]))
    selected, ranking, table = dense_reference(model)
    assert selected == model.selected_features_ == X.columns[:130].tolist()
    assert_frame_equal(model.ranking_, ranking)
    assert_frame_equal(model.redundancy, table.loc[:, selected[:-1]])
    assert model.redundancy.shape == (140, 129)
