"""R09: bounded mixed relevance and hand-checked MRMR ranking."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from arfs import association as a
from arfs.feature_selection import MinRedundancyMaxRelevance as MRMR


def mixed_data():
    X = pd.DataFrame(
        {
            "num": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": np.array(["a", "a", "b", "b", "b", "a"], dtype=object),
        }
    )
    return X, pd.Series([0.0, 1.0, 1.5, 3.0, 4.0, 4.5], name="label")


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_mixed_singletons_remain_eligible(task, jobs):
    X, y = mixed_data()
    if task == "classification":
        y = pd.Series(["a", "b", "b", "a", "a", "b"])
    model = MRMR(2, task=task, n_jobs=jobs, show_progress=False).fit(X, y)
    assert set(model.selected_features_) == set(X.columns)
    assert model.relevance_.notna().all()
    assert model.relevance_.between(0, 1).all()
    assert_array_equal(model.get_support(), [True, True])


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_mixed_relevance_uses_same_bounded_f_map(task, jobs):
    X, y = mixed_data()
    X["num2"] = [1.0, 2.0, 5.0, 3.0, 4.0, 0.0]
    weights = np.array([1, 2, 3, 1, 2, 0])
    if task == "classification":
        y = pd.Series([0, 0, 1, 1, 0, 1], dtype="category")
        cont, cat = a.f_cont_classification_parallel, a.f_cat_classification_parallel
        combined = a.f_stat_classification_parallel
    else:
        cont, cat = a.f_cont_regression_parallel, a.f_cat_regression_parallel
        combined = a.f_stat_regression_parallel
    before, before_y = X.copy(deep=True), y.copy(deep=True)
    raw = pd.concat([cont(X, y, weights), cat(X, y, weights)]).reindex(X.columns)
    expected = raw / (1 + raw)
    actual = combined(X, y, weights, n_jobs=jobs).reindex(X.columns)
    assert_allclose(actual, expected)
    assert actual.between(0, 1).all()
    assert_series_equal(actual, combined(X, y, weights, n_jobs=1).reindex(X.columns))
    # Adding unrelated predictor types must not rescale an existing feature.
    singleton = combined(X[["cat"]], y, weights, n_jobs=jobs)
    assert_allclose(actual["cat"], singleton["cat"])
    repeated = np.repeat(np.arange(len(X)), weights)
    assert_allclose(
        actual,
        combined(
            X.iloc[repeated].reset_index(drop=True),
            y.iloc[repeated].reset_index(drop=True),
            n_jobs=jobs,
        ).reindex(X.columns),
    )
    assert_frame_equal(X, before)
    assert_series_equal(y, before_y)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("force_finite", [True, False])
def test_perfect_equal_scores_are_one_and_ties_follow_input_order(task, force_finite):
    y = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    X = pd.DataFrame({"z": y, "a": -y, "cat": y.astype("category")})
    fn = a.f_stat_regression_parallel
    if task == "classification":
        y = y.astype("category")
        fn = a.f_stat_classification_parallel
    assert_allclose(fn(X, y, force_finite=force_finite, n_jobs=1), 1.0)
    model = MRMR(3, task=task, show_progress=False).fit(X, y)
    assert model.selected_features_ == list(X.columns)
    assert_allclose(model.ranking_.mrmr.iloc[1:], 0.0)
    assert np.isinf(model.ranking_.mrmr.iloc[0])
    assert np.isinf(model.ranking_.redundancy.iloc[1:]).all()


def test_default_relevance_is_nonnegative():
    X, y = mixed_data()
    X["num2"] = [1.0, 2.0, 5.0, 3.0, 4.0, 0.0]
    result = a.f_stat_regression_parallel(X, y, n_jobs=1)
    assert result.notna().all() and result.between(0, 1).all()


def test_negative_duplicate_gets_full_penalty():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "neg": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "b": [1.0, 3.0, 2.0, 5.0, 4.0],
        }
    )
    model = MRMR(
        3,
        relevance_func=lambda **kw: pd.Series({"a": 3.0, "neg": 2.0, "b": 1.0}),
        show_progress=False,
    ).fit(X, pd.Series([1.0, 3.0, 2.0, 4.0, 5.0]))
    assert model.selected_features_ == ["a", "b", "neg"]
    assert_allclose(model.redundancy.loc[["b", "neg"], "a"], [0.8, 1.0])
    assert_allclose(
        model.ranking_.loc["b", ["mrmr", "relevance", "redundancy"]], [1.25, 1, 0.8]
    )
    # After b is selected the mean penalty for neg is (1 + .8) / 2.
    assert_allclose(model.ranking_.loc["neg", "mrmr"], 2 / 0.9)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_constant_predictors_and_response_have_zero_default_relevance(task):
    X = pd.DataFrame({"z": [1.0] * 6, "a": np.array(["x"] * 6, dtype=object)})
    y = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    if task == "classification":
        y = pd.Series(["a", "a", "a", "b", "b", "b"])
    model = MRMR(2, task=task, show_progress=False).fit(X, y)
    assert model.selected_features_ == ["z", "a"]
    assert_allclose(model.relevance_, 0.0)
    varying, _ = mixed_data()
    constant_y = pd.Series([0.0] * 6)
    if task == "classification":
        # Keep R08's one-class ANOVA rejection rather than inventing a null test.
        with pytest.raises(ValueError, match="two positive-mass groups"):
            model.fit(varying, constant_y)
    else:
        model.fit(varying, constant_y)
        assert_allclose(model.relevance_, 0.0)
        assert model.selected_features_ == list(varying.columns)


@pytest.mark.parametrize(
    "kind", ["zero_request", "no_columns", "unavailable", "empty_callback"]
)
@pytest.mark.parametrize("return_scores", [False, True])
def test_empty_selection_has_consistent_outputs_and_can_refit(kind, return_scores):
    X, y = mixed_data()
    count, relevance = 2, None
    if kind == "zero_request":
        count = 0
    elif kind == "no_columns":
        X = X.iloc[:, :0]
    elif kind == "unavailable":
        relevance = lambda **kw: pd.Series(np.nan, index=kw["X"].columns)
    else:
        relevance = lambda **kw: pd.Series(dtype=float)
    model = MRMR(
        count,
        relevance_func=relevance,
        return_scores=return_scores,
        show_progress=False,
    )
    output = model.fit(X, y)
    assert (output[0] if return_scores else output.selected_features_) == []
    assert model.ranking_.empty
    assert list(model.ranking_.columns) == ["mrmr", "relevance", "redundancy"]
    assert_array_equal(model.get_support(), np.zeros(X.shape[1], dtype=bool))
    assert model.n_features_in_ == X.shape[1]
    assert_frame_equal(model.transform(X), X.iloc[:, :0])
    assert model.n_features_to_select == count
    if relevance is None and count > 0:
        model.fit(*mixed_data())
        assert len(model.selected_features_) == 2


@pytest.mark.parametrize("scores", [[-1.0, 1.0], [np.inf, 1.0], [-np.inf, 1.0]])
def test_invalid_custom_relevance_rejected(scores):
    X, y = mixed_data()
    with pytest.raises(ValueError, match="finite.*nonnegative"):
        MRMR(
            1,
            relevance_func=lambda **kw: pd.Series(scores, index=X.columns),
            show_progress=False,
        ).fit(X, y)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_selection_count_rejected(count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        MRMR(count, show_progress=False).fit(*mixed_data())


def test_custom_relevance_order_does_not_change_tie_order():
    X, y = mixed_data()
    model = MRMR(
        1,
        relevance_func=lambda **kw: pd.Series({"cat": 0.0, "num": 0.0}),
        show_progress=False,
    ).fit(X, y)
    assert model.selected_features_ == ["num"]


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("seed", range(5))
def test_strong_mixed_signal_and_null_screen(task, seed):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.normal(size=160))
    X = pd.DataFrame({"signal": y + 0.1 * rng.normal(size=160)})
    if task == "classification":
        y = (y > 0).astype("category")
        X["signal"] = y.astype(float) + 0.1 * rng.normal(size=160)
    X["noise"] = rng.normal(size=160)
    X["group_noise"] = pd.Series(rng.integers(0, 3, size=160), dtype="category")
    model = MRMR(1, task=task, show_progress=False).fit(X, y)
    assert model.selected_features_ == ["signal"]
    # MRMR still fills the requested slots on null data; this is not error control.
    null = X.drop(columns="signal")
    model = MRMR(2, task=task, show_progress=False).fit(null, y)
    assert set(model.selected_features_) == set(null.columns)


@pytest.mark.parametrize(
    "result",
    [
        [1.0, 2.0],
        pd.Series([1.0, 2.0], index=["num", "num"]),
        pd.Series({"unknown": 1.0}),
    ],
)
def test_custom_relevance_requires_known_unique_labels(result):
    with pytest.raises(ValueError, match="predictor labels"):
        MRMR(1, relevance_func=lambda **kw: result, show_progress=False).fit(
            *mixed_data()
        )


@pytest.mark.parametrize("force_finite", [True, False])
def test_undefined_numeric_score_policy_is_explicit(force_finite):
    X, y = mixed_data()
    X["constant"] = 1.0
    scores = a.f_stat_regression_parallel(X, y, n_jobs=1, force_finite=force_finite)
    if force_finite:
        assert scores["constant"] == 0.0
    else:
        assert np.isnan(scores["constant"])


@pytest.mark.parametrize("kind", ["zero_request", "no_columns"])
@pytest.mark.parametrize("custom", [False, True])
def test_empty_request_needs_no_statistic_even_with_one_observation(kind, custom):
    X, y = mixed_data()
    X, y = X.iloc[:1], y.iloc[:1]
    count = 0 if kind == "zero_request" else 2
    if kind == "no_columns":
        X = X.iloc[:, :0]

    def should_not_run(**kwargs):
        raise AssertionError("An empty selection needs no relevance computation.")

    model = MRMR(
        count, relevance_func=should_not_run if custom else None, show_progress=False
    ).fit(X, y)
    assert model.selected_features_ == []
    assert model.ranking_.empty
    assert model.n_features_in_ == X.shape[1]
