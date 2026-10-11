"""Real caller regressions for the supported dependency APIs."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.stats import f_oneway
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from arfs import association
from arfs import feature_selection as fs
from arfs.benchmark import sklearn_pimp_bench
from arfs.preprocessing import OrdinalEncoderPandas


def test_public_lasso_selector_fits():
    x = np.arange(30.0)
    X = pd.DataFrame({"x": x})
    model = fs.LassoFeatureSelection(n_iterations=3).fit(X, x + np.sin(x))
    assert "x" in model.selected_features_
    assert np.isfinite(model.best_estimator_.coef_).all()
    assert np.isfinite(model.best_estimator_.predict(X)).all()


def test_summary_real_pipeline_renders_html():
    X = pd.DataFrame(
        {
            "missing": [np.nan] * 4,
            "constant": [1] * 4,
            "varying": [1, 2, 3, 4],
            "cat": pd.Categorical(["a", "b", "a", "b"]),
        }
    )
    pipe = Pipeline(
        [
            ("missing", fs.MissingValueThreshold(0.5)),
            ("unique", fs.UniqueValuesThreshold()),
            ("encoder", OrdinalEncoderPandas()),
            ("last", fs.UniqueValuesThreshold(10)),
        ]
    )
    assert pipe.fit_transform(X).shape == (4, 0)
    style = fs.make_fs_summary(pipe)
    expected = pd.DataFrame(
        {
            "predictor": X.columns,
            "missing": [0.0, 1, 1, 1],
            "unique": [np.nan, 0, 1, 1],
            "encoder": [np.nan] * 4,
            "last": [np.nan, np.nan, 0, 0],
        }
    )
    assert_frame_equal(style.data, expected, check_dtype=False)
    html = style.to_html()
    assert all(color in html for color in ["#ba0202", "#0c8a30", "#f57505"])


def test_benchmark_real_permutation_plot():
    rng = np.random.default_rng(614)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["genuine", "other", "random"])
    y = 3 * X.genuine - X.other + rng.normal(size=200)
    fig = sklearn_pimp_bench(DecisionTreeRegressor(max_depth=2, random_state=8), X, y)
    try:
        fig.canvas.draw()
        assert sorted(t.get_text() for t in fig.axes[0].get_yticklabels()) == sorted(
            X.columns
        )
    finally:
        plt.close(fig)


def test_association_clustering_and_plot_preserve_data_and_owner():
    frame = pd.DataFrame(
        [[1.0, -0.8, 0.2], [-0.8, 1.0, 0.4], [0.2, 0.4, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    before = frame.copy()
    clustered = association.cluster_sq_matrix(frame)
    assert_frame_equal(clustered, before.loc[clustered.index, clustered.columns])
    assert sorted(clustered.columns) == sorted(frame.columns)
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        result, target_ax = association.plot_association_matrix(
            frame, ax=ax, show=False, annotate=True
        )
        assert result is fig and target_ax is ax
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert_allclose(ax.images[0].get_array(), before.loc[labels, labels])
        assert ax.images[0].get_clim() == (-1, 1)
        assert "-0.8" in [t.get_text().replace("−", "-") for t in ax.texts]
        fig.canvas.draw()
        assert_frame_equal(frame, before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("weighted", [False, True])
def test_scalar_anova_preserves_reference_statistic_and_shape(weighted):
    groups = [np.array([1.0, 2.0, 4.0]), np.array([4.0, 5.0, 7.0])]
    weights = (
        [np.array([1, 2, 1]), np.array([2, 1, 3])]
        if weighted
        else [np.ones(3, dtype=int)] * 2
    )
    expected = f_oneway(*(np.repeat(g, w) for g, w in zip(groups, weights))).statistic
    actual = association.f_oneway_weighted(*zip(groups, weights))
    assert actual.shape == (1,)
    assert_allclose(actual[0], expected)


@pytest.mark.parametrize("different_levels", [False, True])
def test_scalar_anova_constant_handling(different_levels):
    groups = [np.full(3, 2.0), np.full(3, 4.0 if different_levels else 2.0)]
    with (
        warnings.catch_warnings(record=True) as caught,
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("always", UserWarning)
        actual = association.f_oneway_weighted(*((g, np.ones(3)) for g in groups))
    assert actual.shape == (1,)
    assert np.isinf(actual[0]) if different_levels else np.isnan(actual[0])
    constant_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(constant_warnings) == (0 if different_levels else 1)
    if constant_warnings:
        assert str(constant_warnings[0].message) == "Features [0] are constant."


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_parallel_relevance_agrees(weighted, task):
    X = pd.DataFrame(
        {
            "number": [0.0, 1.0, 2.0, 3.0],
            "group": pd.Series(["a", "a", "b", "b"], dtype=object),
        }
    )
    w = np.array([1.0, 2.0, 1.0, 3.0]) if weighted else None
    y = pd.Series([1.0, 3.0, 2.0, 7.0], name="response")
    fn = association.f_cat_regression_parallel
    if task == "classification":
        y = pd.Series([0, 0, 1, 1], name="response")
        fn = association.f_cont_classification_parallel
    assert_series_equal(
        fn(X, y, sample_weight=w, n_jobs=1), fn(X, y, sample_weight=w, n_jobs=2)
    )
