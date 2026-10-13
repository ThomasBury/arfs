"""Observation validation regressions promoted from reviews 1A, 1B and 04."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from arfs import association as a
from arfs import benchmark, utils
from arfs import feature_selection as fs
from arfs.feature_selection import allrelevant as ar
from arfs.feature_selection.lasso import EnetGLM, weighted_cross_val_score
from arfs.gbm import GradientBoosting, _make_split
from arfs.utils import validate_sample_weight

INVALID_WEIGHTS = [
    [0.0] * 4,
    [1.0, 1.0, 1.0, -0.9],
    [1.0, np.nan, 1.0, 1.0],
    [1.0, np.inf, 1.0, 1.0],
    [1.0] * 3,
    [[1.0]] * 4,
]


def frame():
    return pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 2.0, 1.0, 3.0]})


@pytest.mark.parametrize("weights", INVALID_WEIGHTS)
@pytest.mark.parametrize(
    "statistic", [a.association_series, a.association_matrix, a.weighted_corr]
)
def test_invalid_weights_rejected(weights, statistic):
    X = frame()
    with pytest.raises(ValueError):
        if statistic is a.association_series:
            statistic(X, "y", sample_weight=weights)
        elif statistic is a.association_matrix:
            statistic(X, sample_weight=weights)
        else:
            statistic(X.x, X.y, weights, method="pearson")


@pytest.mark.parametrize("container", [list, np.array, pd.Series])
def test_list_array_series_weights_and_complete_case_alignment(container):
    X = frame()
    X.index = [40, 10, 90, 20]
    X.loc[10, "x"] = np.nan
    before = X.copy(deep=True)
    weights = container([1.0, 8.0, 2.0, 3.0])
    result = a.wcorr_series(X, "y", weights, method="pearson")
    valid = X.x.notna()
    expected = a.weighted_corr(
        X.loc[valid, "x"], X.loc[valid, "y"], [1.0, 2.0, 3.0], method="pearson"
    )
    assert_allclose(result["x"], expected)
    checked, retained = a._check_association_input(X, weights)
    assert_frame_equal(checked, X.loc[valid])
    assert_array_equal(retained, [1.0, 2.0, 3.0])
    assert_frame_equal(X, before)
    assert_array_equal(weights, [1.0, 8.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "statistic",
    [
        a.correlation_ratio,
        a.f_cat_regression,
        a.cramer_v,
        a.weighted_theils_u,
        a.weighted_conditional_entropy,
        a.f_cont_classification,
    ],
)
def test_zero_weight_category_equivalent_to_removal(statistic):
    x = pd.Series(["a", "a", "b", "b", "z"], name="x", dtype=object)
    y = pd.Series([1.0, 2.0, 4.0, 6.0, 100.0], name="y")
    if statistic in [a.cramer_v, a.weighted_theils_u, a.weighted_conditional_entropy]:
        y = pd.Series(["x", "y", "x", "y", "q"], name="y")
    if statistic is a.f_cont_classification:
        x, y = y, x
    assert_allclose(
        statistic(x, y, [1, 1, 1, 1, 0]),
        statistic(x.iloc[:4], y.iloc[:4], [1, 1, 1, 1]),
    )


@pytest.mark.parametrize(
    "statistic",
    [
        a.correlation_ratio,
        a.f_cat_regression,
        a.cramer_v,
        a.weighted_theils_u,
        a.weighted_conditional_entropy,
        a.f_cont_classification,
    ],
)
@pytest.mark.parametrize("weights", INVALID_WEIGHTS)
def test_grouped_statistics_validate_weights(statistic, weights):
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series([1.0, 2.0, 4.0, 6.0])
    if statistic in [a.cramer_v, a.weighted_theils_u, a.weighted_conditional_entropy]:
        y = x.copy()
    if statistic is a.f_cont_classification:
        x, y = y, x
    with pytest.raises(ValueError):
        statistic(x, y, weights)


@pytest.mark.parametrize(
    "statistic", [a.f_oneway_weighted, a._weighted_correlation_ratio]
)
def test_direct_grouped_input_zero_mass_and_invalid_groups(statistic):
    groups = [([1, 2], [1, 1]), ([4, 6], [1, 1])]
    assert_allclose(statistic(*groups, ([100], [0])), statistic(*groups))
    for extra in [([100], [-1]), ([100], [np.nan]), ([100], [1, 2]), ([[100]], [1])]:
        with pytest.raises(ValueError):
            statistic(*groups, extra)
    with pytest.raises(ValueError):
        statistic(([100], [0]))
    with pytest.raises(ValueError):
        statistic()


@pytest.mark.parametrize(
    "X, weights",
    [
        (pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, 2.0, np.nan]}), None),
        (pd.DataFrame({"x": [np.nan, 1.0], "y": [2.0, 3.0]}), [1, 0]),
        (pd.DataFrame(columns=["x", "y"], dtype=float), None),
    ],
)
def test_no_usable_observations_rejected(X, weights):
    with pytest.raises(ValueError):
        a.association_series(X, "y", sample_weight=weights)


@pytest.mark.parametrize("jobs", [1, 2])
def test_handle_na_none_preserves_rows_and_weights_for_custom_callable(jobs):
    X = frame()
    X.loc[1, "y"] = np.nan
    before = X.copy(deep=True)

    def capture(x, y, sample_weight, as_frame):
        assert len(y) == 4 and y.isna().any()
        assert_array_equal(sample_weight, [1, 2, 3, 4])
        return 0.5

    result = a.association_series(
        X,
        "y",
        sample_weight=[1, 2, 3, 4],
        handle_na=None,
        num_num_assoc=capture,
        n_jobs=jobs,
    )
    assert_allclose(result, 0.5)
    assert_frame_equal(X, before)


def fit(kind, X, y, weights=None):
    if kind == "gbm":
        return GradientBoosting(
            params={"objective": "regression", "n_estimators": 2, "num_threads": 1},
            verbose_eval=False,
        ).fit(X, y, sample_weight=weights)
    if kind == "leshy":
        return fs.Leshy(
            DecisionTreeRegressor(),
            n_estimators=5,
            max_iter=2,
            importance="native",
            verbose=0,
        ).fit(X, y, sample_weight=weights)
    if kind == "boost":
        return fs.BoostAGroota(
            DecisionTreeRegressor(),
            iters=1,
            max_rounds=1,
            importance="native",
            silent=True,
        ).fit(X, y, sample_weight=weights)
    if kind == "groot":
        return fs.GrootCV(
            objective="regression", n_folds=2, n_iter=1, silent=True, n_jobs=1
        ).fit(X, y, sample_weight=weights)
    return EnetGLM().fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("kind", ["gbm", "leshy", "boost", "groot", "glm"])
@pytest.mark.parametrize(
    "y",
    [
        [0, np.nan, 1, 2],
        [0, np.inf, 1, 2],
        [0, -np.inf, 1, 2],
        [[0], [1], [2], [3]],
        [0, 1, 2],
        [],
    ],
)
def test_training_rejects_invalid_targets(kind, y):
    with pytest.raises(ValueError):
        fit(kind, frame(), y)


@pytest.mark.parametrize("kind", ["gbm", "leshy", "boost", "groot", "glm"])
@pytest.mark.parametrize("weights", INVALID_WEIGHTS)
def test_training_rejects_invalid_weights(kind, weights):
    with pytest.raises(ValueError):
        fit(kind, frame(), [0, 1, 2, 3], weights)


@pytest.mark.parametrize(
    "statistic",
    [
        a.f_cat_regression_parallel,
        a.f_cont_regression_parallel,
        a.f_cont_classification_parallel,
        a.f_cat_classification_parallel,
    ],
)
@pytest.mark.parametrize("handle_na", ["drop", "fill", None])
def test_relevance_rejects_missing_response_before_row_policy(statistic, handle_na):
    with pytest.raises(ValueError):
        statistic(frame(), [0, np.nan, 1, 2], handle_na=handle_na)


@pytest.mark.parametrize("zero_split", ["train", "valid"])
def test_split_weight_mass_checked_before_training(zero_split):
    X = pd.DataFrame({"x": np.arange(30.0)})
    y = pd.Series(np.arange(30.0), index=np.arange(30) + 50)
    weights = pd.Series(np.ones(30), index=np.arange(30) + 100)
    tr, va = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X))
    weights.iloc[tr if zero_split == "train" else va] = 0
    with pytest.raises(ValueError, match="positive"):
        _make_split(X, y, weights, None, None, False)
    with pytest.raises(ValueError, match="positive"):
        fit("gbm", X, y, weights)
    with pytest.raises(ValueError, match="positive"):
        ar._split_data(X, y, tr, va, weights)
    tr, va = train_test_split(np.arange(30), random_state=42)
    weights[:] = 1
    weights.iloc[tr if zero_split == "train" else va] = 0
    with pytest.raises(ValueError, match="positive"):
        ar._split_fit_estimator(DecisionTreeRegressor(), X.copy(), y, weights)


def test_glm_cross_validation_omits_zero_rows_before_forming_folds():
    X = pd.DataFrame({"x": np.arange(30.0)})
    weights = np.ones(30)
    weights[:6] = 0
    y = X.x.to_numpy() + np.sin(X.x.to_numpy())
    actual = weighted_cross_val_score(EnetGLM(), X, y, weights, n_jobs=1)
    expected = weighted_cross_val_score(
        EnetGLM(), X.iloc[6:], y[6:], weights[6:], n_jobs=1
    )
    assert_allclose(actual, expected)


def test_split_preserves_positional_alignment_and_inputs():
    X = pd.DataFrame({"x": np.arange(30.0)}, index=np.arange(30) + 400)
    y = pd.Series(np.arange(30.0), index=np.arange(30) + 50, name="response")
    weights = pd.Series(np.arange(30.0) + 1, index=np.arange(30) + 100, name="weights")
    before = X.copy(), y.copy(), weights.copy()
    tr, va = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X))
    Xtr, ytr, Xva, yva, wva, wtr, _, _ = _make_split(X, y, weights, None, None, False)
    assert_frame_equal(Xtr, X.iloc[tr])
    assert_frame_equal(Xva, X.iloc[va])
    assert_series_equal(ytr, y.iloc[tr])
    assert_series_equal(yva, y.iloc[va])
    assert_series_equal(wtr, weights.iloc[tr])
    assert_series_equal(wva, weights.iloc[va])
    assert_frame_equal(X, before[0])
    assert_series_equal(y, before[1])
    assert_series_equal(weights, before[2])


def test_relative_weights_are_not_normalized():
    weights = [0.1, 0.2, 0, 0.3]
    assert_array_equal(validate_sample_weight(weights), weights)


@pytest.mark.parametrize("statistic", [a.wm, a.wrank])
@pytest.mark.parametrize("weights", INVALID_WEIGHTS)
def test_low_level_statistics_validate_weights(statistic, weights):
    with pytest.raises(ValueError):
        statistic([0, 1, 2, 3], weights)


@pytest.mark.parametrize("container", [list, np.array, pd.Series])
def test_targets_accept_array_likes_and_preserve_response_values(container):
    y = container([0, 1, 2, 3])
    assert_array_equal(utils.validate_target(y, 4), y)
    labels = pd.Series(["a", "b", "a", "b"], index=[9, 4, 5, 8], name="class")
    assert_series_equal(utils.validate_target(labels, 4), labels)


@pytest.mark.parametrize(
    "x, y",
    [
        (1, [1]),
        ([1], 1),
        ([], []),
        ([1, np.nan], [1, 2]),
        ([1, 2], [1, np.inf]),
        ([[1], [2]], [1, 2]),
    ],
)
def test_pairwise_statistics_reject_invalid_observations(x, y):
    with pytest.raises(ValueError):
        a.weighted_corr(x, y)


def test_undefined_grouped_statistics_rejected():
    with pytest.raises(ValueError, match="two positive-mass"):
        a.f_cat_regression(pd.Series(["a", "a", "b"]), [1, 2, 3], [1, 1, 0])
    with pytest.raises(ValueError, match="two positive-mass"):
        a.cramer_v(["a", "a", "b"], ["x", "y", "z"], [1, 1, 0])


def test_pairwise_series_are_aligned_by_position():
    x = pd.Series([0.0, 1.0, 2.0, 3.0], index=[50, 40, 30, 20])
    y = pd.Series([0.0, 2.0, 1.0, 3.0], index=[3, 6, 8, 9])
    assert_allclose(
        a.weighted_corr(x, y, method="pearson"),
        a.weighted_corr(x.to_numpy(), y.to_numpy(), method="pearson"),
    )
    categories = pd.Series(["a", "a", "b", "b"], index=x.index)
    labels = pd.Series(["x", "y", "x", "y"], index=y.index)
    assert_allclose(
        a.cramer_v(categories, labels),
        a.cramer_v(categories.to_numpy(), labels.to_numpy()),
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_benchmark_weights_match_independent_reference(task, monkeypatch):
    rng = np.random.default_rng(742)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["signal", "noise", "other"])
    y = 2 * X.signal + rng.normal(size=120)
    if task == "classification":
        y = (y > 0).astype(int)
    weights = pd.Series(np.where(X.signal > 0, 20.0, 1.0), index=X.index + 300)
    estimator = (
        DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    )
    actual = {}

    def capture(model, features, target, **kwargs):
        actual["weights"] = kwargs["sample_weight"].copy()
        actual["result"] = permutation_importance(model, features, target, **kwargs)
        return actual["result"]

    monkeypatch.setattr(benchmark, "permutation_importance", capture)
    model = estimator(max_depth=3, random_state=8)
    fig = benchmark.sklearn_pimp_bench(model, X, y, task=task, sample_weight=weights)
    try:
        tr, va = train_test_split(
            np.arange(len(X)),
            stratify=y if task == "classification" else None,
            random_state=42,
        )
        reference = estimator(max_depth=3, random_state=8).fit(
            X.iloc[tr], y.iloc[tr], sample_weight=weights.iloc[tr]
        )
        expected = permutation_importance(
            reference,
            X.iloc[va],
            y.iloc[va],
            sample_weight=weights.iloc[va],
            n_repeats=10,
            random_state=42,
            n_jobs=1,
        )
        unweighted = permutation_importance(
            reference, X.iloc[va], y.iloc[va], n_repeats=10, random_state=42, n_jobs=1
        )
        assert_array_equal(actual["weights"], weights.iloc[va])
        assert_allclose(actual["result"].importances, expected.importances)
        assert not np.allclose(expected.importances, unweighted.importances)
        fig.canvas.draw()
    finally:
        plt.close(fig)
