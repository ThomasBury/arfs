"""R29 plots follow feature coordinates, fitted membership and full-data shares."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from matplotlib.colors import to_rgba
from pandas.testing import assert_frame_equal

from arfs.association import cluster_sq_matrix, plot_association_matrix
from arfs.feature_selection import CollinearityThreshold, Leshy, VariableImportance
from arfs.feature_selection import variable_importance as vi
from arfs.feature_selection.allrelevant import BLUE, YELLOW
from arfs.utils import plot_y_vs_X


@pytest.fixture(autouse=True)
def restore_plots():
    with mpl.rc_context():
        yield
    plt.close("all")


@pytest.mark.parametrize(
    "columns,ncols",
    [
        (["a", "b", "c", "d"], 2),
        ([0, 1, 2, 3], 2),
        (["a"], 1),
        (["a", "b"], 2),
        (["a", "b", "c"], 2),
        ([10, 20], 1),
    ],
)
def test_target_plot_uses_every_feature_and_hides_unused_axes(columns, ncols):
    frame = pd.DataFrame(np.arange(12 * len(columns)).reshape(12, -1), columns=columns)
    before = frame.copy(deep=True)
    y = np.arange(12)
    other = plt.figure()
    fig = plot_y_vs_X(frame, y, ncols=ncols)
    fig.canvas.draw()
    for ax, column in zip(fig.axes, columns):
        np.testing.assert_array_equal(
            ax.collections[0].get_offsets()[:, 0], frame[column]
        )
        np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 1], y)
        assert ax.get_title() == str(column)
    assert all(not ax.axison for ax in fig.axes[len(columns) :])
    assert_frame_equal(frame, before)
    plt.close(fig)
    assert plt.fignum_exists(other.number)


@pytest.mark.parametrize("ncols", [0, -1, True, 1.5])
def test_target_plot_rejects_invalid_columns_before_creating_figure(ncols):
    with pytest.raises(ValueError, match="ncols"):
        plot_y_vs_X(pd.DataFrame({"a": [1]}), [1], ncols=ncols)
    assert not plt.get_fignums()


def test_target_plot_rejects_no_features_before_creating_figure():
    with pytest.raises(ValueError, match="feature"):
        plot_y_vs_X(pd.DataFrame(index=[0]), [1])
    assert not plt.get_fignums()


@pytest.mark.parametrize("keep_weak", [False, True])
@pytest.mark.parametrize("names", [["selected", "rejected", "tentative"], [10, 20, 30]])
def test_leshy_colors_follow_masks_after_sorting(keep_weak, names):
    selector = Leshy(LGBMRegressor(), keep_weak=keep_weak)
    selector.feature_names_in_ = np.array(names)
    selector.imp_real_hist = np.array(
        [[1.0, 0.0, 2.0], [1.0, 10.0, 2.0], [1.0, 0.0, 2.0]]
    )
    selector.support_ = np.array([True, False, keep_weak])
    selector.support_weak_ = np.array([False, False, True])
    selector.sha_max = 0.5
    before = selector.imp_real_hist.copy()
    fig = selector.plot_importance(n_feat_per_inch=0.5)
    fig.canvas.draw()
    colors = dict(
        zip(
            [t.get_text() for t in fig.axes[0].get_yticklabels()],
            [p.get_facecolor() for p in fig.axes[0].patches],
        )
    )
    assert colors[str(names[0])] == to_rgba(BLUE)
    assert colors[str(names[1])] == to_rgba("gray")
    assert colors[str(names[2])] == to_rgba(YELLOW)
    np.testing.assert_array_equal(selector.imp_real_hist, before)


def fitted_importance(monkeypatch, threshold=0.9, scores=(6.0, 3.0, 1.0), names=None):
    names = list(range(len(scores))) if names is None else names
    scores = np.array(scores)
    shares = scores / scores.sum() if scores.sum() else scores
    summary = pd.DataFrame(
        {
            "feature": names,
            "importance": scores,
            "normalized_importance": shares,
            "cumulative_importance": scores.cumsum() / scores.sum()
            if scores.sum()
            else scores,
        }
    )
    monkeypatch.setattr(vi, "_compute_varimp_lgb", lambda **kwargs: summary)
    return VariableImportance(threshold=threshold).fit(
        pd.DataFrame(np.ones((2, len(scores))), columns=names), [0.0, 1.0]
    )


@pytest.mark.parametrize(
    "threshold,count", [(0, 0), (0.6, 1), (0.7, 2), (0.9, 2), (1, 3)]
)
@pytest.mark.parametrize("log", [False, True])
def test_importance_cutoff_matches_real_selection(monkeypatch, threshold, count, log):
    selector = fitted_importance(monkeypatch, threshold)
    before = selector.feature_importances_summary_.copy(deep=True)
    fig = selector.plot_importance(log=log, figsize=(8, 4))
    fig.canvas.draw()
    ax1, ax2 = fig.axes
    assert selector.support_.sum() == count
    assert ax2.get_title() == f"{count} features selected"
    assert ax2.get_xscale() == "linear"
    assert ax2.get_ylim() == (0, 1.05)
    assert ax2.get_xlim()[0] <= 0 <= ax2.get_xlim()[1]
    assert [t.get_text() for t in ax2.get_xticklabels()] == ["0", "1", "2"]
    assert len(ax1.lines) == (1 if count else 0)
    assert len(ax2.lines) == (2 if count else 1)
    if count:
        np.testing.assert_allclose(
            ax1.lines[0].get_xdata(), [before.normalized_importance.iloc[count - 1]] * 2
        )
        np.testing.assert_allclose(ax2.lines[1].get_xdata(), [count - 1] * 2)
    assert_frame_equal(selector.feature_importances_summary_, before)


@pytest.mark.parametrize("plot_n,expected", [(1, 1), (2, 2), (3, 3), (50, 3)])
@pytest.mark.parametrize("names", [["signal", "second", "noise"], [10, 20, 30]])
def test_importance_limit_keeps_full_data_cumulative_shares(
    monkeypatch, plot_n, expected, names
):
    selector = fitted_importance(monkeypatch, threshold=1, names=names)
    fig = selector.plot_importance(plot_n=plot_n, log=False)
    fig.canvas.draw()
    ax1, ax2 = fig.axes
    assert len(ax1.collections[0].get_offsets()) == expected
    assert len(ax2.collections[0].get_offsets()) == expected
    np.testing.assert_allclose(
        ax2.collections[0].get_offsets()[:, 0], np.arange(expected)
    )
    np.testing.assert_allclose(
        ax2.collections[0].get_offsets()[:, 1], [0.6, 0.9, 1.0][:expected]
    )
    assert [t.get_text() for t in ax1.get_yticklabels()] == [
        str(x) for x in names[:expected]
    ]
    assert fig.get_size_inches()[1] == max(2, expected / 3)
    assert len(ax1.lines) == (1 if expected == 3 else 0)
    assert len(ax2.lines) == (2 if expected == 3 else 1)


@pytest.mark.parametrize(
    "scores,threshold,count",
    [
        ((1.0,), 1, 1),
        ((0.0,), 1, 0),
        ((0.0, 0.0, 0.0), 0.9, 0),
        ((1.0, 1e-20, 0.0), 1, 2),
    ],
)
def test_importance_singleton_zero_total_and_rounded_terminal_share(
    monkeypatch, scores, threshold, count
):
    selector = fitted_importance(monkeypatch, threshold=threshold, scores=scores)
    fig = selector.plot_importance(figsize=(8, 4))
    fig.canvas.draw()
    assert len(fig.axes[0].collections[0].get_offsets()) == len(scores)
    assert fig.axes[1].get_title() == f"{count} features selected"
    assert len(fig.axes[0].lines) == (1 if count else 0)


@pytest.mark.parametrize("plot_n", [0, -1, True, 1.5, "2"])
def test_importance_invalid_limit_rejected_before_figure(monkeypatch, plot_n):
    selector = fitted_importance(monkeypatch)
    with pytest.raises(ValueError, match="plot_n"):
        selector.plot_importance(plot_n=plot_n)
    assert not plt.get_fignums()


@pytest.mark.parametrize("density", [0, -1, True, np.nan, np.inf])
def test_importance_invalid_density_rejected_before_figure(monkeypatch, density):
    selector = fitted_importance(monkeypatch)
    with pytest.raises(ValueError, match="n_feat_per_inch"):
        selector.plot_importance(n_feat_per_inch=density)
    assert not plt.get_fignums()


def test_importance_requires_fit():
    with pytest.raises(ValueError, match="fitted"):
        VariableImportance().plot_importance()
    assert not plt.get_fignums()


@pytest.mark.parametrize("value", [1.0, np.nan])
def test_singleton_association_preserves_value_and_owner(value):
    frame = pd.DataFrame([[value]], index=["a"], columns=["a"])
    before = frame.copy(deep=True)
    assert_frame_equal(cluster_sq_matrix(frame), before)
    fig, ax = plt.subplots(figsize=(4, 3))
    result, target_ax = plot_association_matrix(frame, ax=ax, show=False, annotate=True)
    fig.canvas.draw()
    assert result is fig and target_ax is ax
    np.testing.assert_allclose(ax.images[0].get_array().filled(np.nan), [[value]])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a"]
    assert_frame_equal(frame, before)
    plt.close(fig)
    assert not plt.get_fignums()


def test_singleton_collinearity_plot_caller():
    selector = CollinearityThreshold().fit(pd.DataFrame({"a": [1.0, 2.0, 4.0]}))
    fig = selector.plot_association(figsize=(4, 3))
    fig.canvas.draw()
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["a_num"]


@pytest.mark.parametrize("shape", [(0, 0), (2, 1)])
def test_invalid_association_rejected_before_figure(shape):
    frame = pd.DataFrame(np.ones(shape))
    with pytest.raises(ValueError, match="nonempty square"):
        cluster_sq_matrix(frame)
    with pytest.raises(ValueError, match="nonempty square"):
        plot_association_matrix(frame, show=False)
    assert not plt.get_fignums()
