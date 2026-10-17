"""Opt-in MRMR sampling preserves alignment and exposes its statistical limits."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import clone

from arfs.feature_selection import MinRedundancyMaxRelevance as MRMR


def data(task):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=60),
            "noise": rng.normal(size=60),
            "group": pd.Categorical(np.tile(["a", "b", "c"], 20)),
        },
        index=np.arange(60) // 2,
    )
    y = pd.Series(X.signal.to_numpy() + rng.normal(size=60), index=np.arange(60) + 100)
    if task == "classification":
        y = (y > 0).astype("category")
    return X, y


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("weighted", [False, True])
def test_sample_matches_manual_subset_and_transform_keeps_all_rows(task, weighted):
    X, y = data(task)
    weights = (
        pd.Series(np.arange(1, 61), index=np.arange(60)[::-1]) if weighted else None
    )
    before, before_y = X.copy(deep=True), y.copy(deep=True)
    before_w = None if weights is None else weights.copy(deep=True)
    model = MRMR(2, task=task, sample_size=24, random_state=9, show_progress=False)
    transformed = model.fit_transform(X, y, sample_weight=weights)
    rows = model.sample_indices_
    assert model.n_samples_fit_ == 24
    assert len(rows) == len(np.unique(rows)) == 24
    assert np.all(np.diff(rows) > 0)
    expected_w = None if weights is None else weights.to_numpy()[rows]
    reference = MRMR(2, task=task, show_progress=False).fit(
        X.iloc[rows], y.iloc[rows], sample_weight=expected_w
    )
    assert model.selected_features_ == reference.selected_features_
    assert_frame_equal(model.ranking_, reference.ranking_)
    assert_frame_equal(model.relevance_args["X"], X.iloc[rows])
    assert_series_equal(model.relevance_args["y"], y.iloc[rows].rename("target"))
    assert model.relevance_args["X"] is model.redundancy_args["X"]
    if weighted:
        assert_array_equal(model.relevance_args["sample_weight"], expected_w)
        assert_array_equal(model.redundancy_args["sample_weight"], expected_w)
        assert_series_equal(weights, before_w)
    assert_frame_equal(transformed, X[model.selected_features_])
    assert_frame_equal(X, before)
    assert_series_equal(y, before_y)
    if task == "classification":
        assert_allclose(
            y.iloc[rows].value_counts(normalize=True).sort_index(),
            y.value_counts(normalize=True).sort_index(),
            atol=1 / len(rows),
        )


@pytest.mark.parametrize("cap", [None, 60, 100])
def test_disabled_or_large_cap_is_identical_to_full_data(cap):
    X, y = data("regression")
    baseline = MRMR(2, show_progress=False).fit(X, y)
    sampled = MRMR(2, sample_size=cap, random_state=7, show_progress=False).fit(X, y)
    assert sampled.sample_indices_ is None
    assert sampled.n_samples_fit_ == len(X)
    assert_frame_equal(sampled.ranking_, baseline.ranking_)


def test_integer_seed_is_repeatable_cloneable_and_refit_resets_sample():
    X, y = data("regression")
    model = MRMR(2, sample_size=20, random_state=8, show_progress=False).fit(X, y)
    rows, ranking = model.sample_indices_.copy(), model.ranking_.copy(deep=True)
    repeated = clone(model).fit(X, y)
    assert_array_equal(repeated.sample_indices_, rows)
    assert_frame_equal(repeated.ranking_, ranking)
    model.fit(X, y)
    assert_array_equal(model.sample_indices_, rows)
    assert model.random_state == 8
    different = clone(model).set_params(random_state=9).fit(X, y)
    assert not np.array_equal(different.sample_indices_, rows)
    model.set_params(sample_size=None).fit(X, y)
    assert model.sample_indices_ is None
    assert model.n_samples_fit_ == len(X)


@pytest.mark.parametrize("cap", [0, -1, True, np.bool_(False), 3.5, "10"])
def test_invalid_cap_is_rejected(cap):
    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        MRMR(1, sample_size=cap, show_progress=False).fit(*data("regression"))


def test_stratification_does_not_silently_drop_a_rare_target_class():
    X = pd.DataFrame({"x": np.arange(100)})
    y = pd.Series(["common"] * 99 + ["rare"])
    with pytest.raises(ValueError, match="lost a positive-weight target class"):
        MRMR(
            1,
            task="classification",
            sample_size=10,
            random_state=0,
            show_progress=False,
        ).fit(X, y)


def test_weight_validation_happens_before_sampling_and_after_subsetting():
    X, y = data("regression")
    model = MRMR(1, sample_size=20, random_state=8, show_progress=False).fit(X, y)
    omitted = np.setdiff1d(np.arange(len(X)), model.sample_indices_)[0]
    weights = np.ones(len(X))
    weights[omitted] = -1
    with pytest.raises(ValueError, match="finite, nonnegative"):
        model.fit(X, y, sample_weight=weights)
    weights = np.zeros(len(X))
    weights[omitted] = 1
    with pytest.raises(ValueError, match="positive mass"):
        model.fit(X, y, sample_weight=weights)
    bad_target = y.copy()
    bad_target.iloc[omitted] = np.nan
    with pytest.raises(ValueError, match="finite, nonmissing"):
        model.fit(X, bad_target)


def test_sampling_preserves_mixed_target_label_identity():
    X, _ = data("classification")
    y = pd.Series([0, "0"] * 30)
    model = MRMR(
        2, task="classification", sample_size=20, random_state=3, show_progress=False
    ).fit(X, y)
    assert model.relevance_args["y"].nunique() == 2
    assert model.n_samples_fit_ == 20


def test_weighted_class_guard_checks_positive_mass_not_just_labels():
    X, y = data("classification")
    model = MRMR(
        1, task="classification", sample_size=24, random_state=9, show_progress=False
    ).fit(X, y)
    rows = model.sample_indices_
    omitted_positive = next(i for i in range(len(X)) if i not in rows and y.iloc[i])
    weights = (~y.astype(bool)).astype(float).to_numpy(copy=True)
    weights[omitted_positive] = 1
    with pytest.raises(ValueError, match="lost a positive-weight target class"):
        model.fit(X, y, sample_weight=weights)


def test_custom_callbacks_share_the_sample_with_return_scores():
    from arfs import association as a

    X, y = data("regression")
    seen = []

    def relevance(X, y, sample_weight):
        seen.append(("relevance", X))
        assert len(X) == len(y) == 20
        return a.f_stat_regression_parallel(X, y, sample_weight, n_jobs=1)

    def redundancy(X, **kwargs):
        seen.append(("redundancy", X))
        return a.association_series(X, n_jobs=1, normalize=False, **kwargs)

    model = MRMR(
        2,
        relevance_func=relevance,
        redundancy_func=redundancy,
        return_scores=True,
        sample_size=20,
        random_state=8,
        show_progress=False,
    )
    selected, _, _ = model.fit(X, y)
    assert selected == model.selected_features_
    assert [name for name, _ in seen] == ["relevance", "redundancy"]
    assert seen[0][1] is seen[1][1]
    assert_frame_equal(seen[0][1], X.iloc[model.sample_indices_])


@pytest.mark.parametrize("empty", ["zero_request", "no_columns"])
def test_empty_selection_does_not_sample(empty):
    X = pd.DataFrame({"x": [1, 2, 3]})
    count = 0 if empty == "zero_request" else 2
    if empty == "no_columns":
        X = X.iloc[:, :0]
    model = MRMR(
        count, task="classification", sample_size=1, random_state=0, show_progress=False
    ).fit(X, pd.Series(["a", "a", "b"]))
    assert model.selected_features_ == []
    assert model.sample_indices_ is None
    assert model.n_samples_fit_ == 3
