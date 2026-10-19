"""C04: round-local diagnostics, quiet controls and runnable API examples."""

import doctest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from matplotlib.colors import to_rgba
from sklearn.ensemble import RandomForestClassifier

from arfs import feature_selection as fs
from arfs.feature_selection import allrelevant as a
from arfs.gbm import GradientBoosting


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


def data():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    return X, pd.Series(2 * X.a + rng.normal(size=len(X)))


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("cutoff", [1, 2])
def test_boost_round_diagnostics_and_complete_first_ranking(monkeypatch, rounds, cutoff):
    X, y = data()
    scores = iter(
        [
            [3, 6, 0, 2, 0, 1],
            [5, 4, 0, 2, 0, 1],
            [8, 0, 1, 1],
            [6, 0, 2, 2],
            [3, 1],
            [4, 1],
        ][: 2 * rounds]
    )
    seen = []

    def importance(estimator, frame, target, **kwargs):
        seen.append(list(frame.columns))
        return np.array(next(scores))

    monkeypatch.setattr(a, "_get_shap_imp", importance)
    selector = fs.BoostAGroota(
        LGBMRegressor(), cutoff=cutoff, iters=2, max_rounds=rounds, random_state=5
    ).fit(X, y)
    assert len(seen) == 2 * rounds
    assert selector.ranking_absolutes_ == ["b", "a", "c"]
    candidates = {1: ["a", "b", "c"], 2: ["a", "b"], 3: ["a"]}[rounds]
    history = selector.sha_cutoff_df
    assert list(history.columns[: len(candidates)]) == candidates
    assert list(history.columns) == seen[-1]
    assert history.shape == (2, 2 * len(candidates))
    expected = np.array(
        {
            1: [[3, 6, 0, 2, 0, 1], [5, 4, 0, 2, 0, 1]],
            2: [[8, 0, 1, 1], [6, 0, 2, 2]],
            3: [[3, 1], [4, 1]],
        }[rounds]
    )
    expected = expected / expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(history, expected)
    assert selector.mean_shadow == pytest.approx(
        history.iloc[:, len(candidates) :].mean().max() / cutoff
    )
    assert set(selector.selected_features_) == ({"a", "b"} if rounds == 1 else {"a"})
    fig = selector.plot_importance()
    fig.canvas.draw()
    ax = fig.axes[0]
    assert {t.get_text() for t in ax.get_yticklabels()} == set(candidates)
    np.testing.assert_allclose(ax.lines[-1].get_xdata(), selector.mean_shadow)


def test_boost_first_round_ties_and_refit_reset_diagnostics(monkeypatch):
    X, y = data()
    monkeypatch.setattr(a, "_get_shap_imp", lambda *args, **kwargs: np.ones(6))
    selector = fs.BoostAGroota(iters=1, max_rounds=1)
    for names in [["a", "b", "c"], [30, 10, 20]]:
        selector.fit(X.set_axis(names, axis=1), y)
        assert selector.ranking_absolutes_ == names
        assert list(selector.sha_cutoff_df.columns[:3]) == names


@pytest.mark.parametrize("names", [["selected", "rejected"], [10, 20]])
def test_boost_plot_colors_follow_names_instead_of_mean_order(names):
    selector = fs.BoostAGroota()
    selector.sha_cutoff_df = pd.DataFrame(
        [[0.2, 0.0, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1]],
        columns=names + ["shadow_1", "shadow_2"],
    )
    selector.mean_shadow = 0.1
    selector.selected_features_ = np.array(names[:1])
    fig = selector.plot_importance()
    fig.canvas.draw()
    colors = dict(
        zip(
            [t.get_text() for t in fig.axes[0].get_yticklabels()],
            [p.get_facecolor() for p in fig.axes[0].patches],
        )
    )
    assert colors[str(names[0])] == to_rgba(a.BLUE)
    assert colors[str(names[1])] == to_rgba("gray")


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("kind", ["leshy", "boost", "groot", "importance", "gbm"])
def test_existing_output_controls(kind, noisy, capfd, tmp_path):
    X, y = data()
    params = {"objective": "regression", "n_estimators": 5, "num_threads": 1}
    if kind == "leshy":
        selector = fs.Leshy(
            RandomForestClassifier(max_depth=3, random_state=5),
            n_estimators=5,
            max_iter=2,
            importance="shap",
            random_state=5,
            verbose=int(noisy),
        )
        selector.fit(X, (y > y.median()).astype(int))
    elif kind == "boost":
        selector = fs.BoostAGroota(
            LGBMRegressor(**params, verbosity=-1),
            iters=1,
            max_rounds=1,
            random_state=5,
            silent=not noisy,
        ).fit(X, y)
    elif kind == "groot":
        selector = fs.GrootCV(
            objective="regression", n_folds=2, n_iter=1, n_jobs=1, silent=not noisy
        ).fit(X, y)
    elif kind == "importance":
        selector = fs.VariableImportance(
            lgb_kwargs=params, fastshap=False, verbose=1 if noisy else -1
        ).fit(X, y)
    else:
        model = GradientBoosting(
            params=params, show_learning_curve=False, verbose_eval=int(noisy)
        )
        model.fit(X, y)
        path = model.save(str(tmp_path), "quiet")
        restored = GradientBoosting(verbose_eval=int(noisy))
        restored.load(path)
        np.testing.assert_allclose(restored.predict(X), model.predict(X))
    captured = capfd.readouterr()
    if noisy:
        assert captured.out or captured.err
    else:
        assert captured.out == captured.err == ""


def test_quiet_mode_keeps_warnings_visible(monkeypatch, capfd):
    X, y = data()
    monkeypatch.setattr(a, "_get_imp", lambda *args, **kwargs: np.ones(6))
    with pytest.warns(UserWarning, match="native variable importance"):
        fs.BoostAGroota(importance="native", iters=1, max_rounds=1).fit(X, y)
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("example", [fs.Leshy, fs.make_fs_summary])
def test_bounded_docstring_examples_execute(example):
    namespace = {}
    for entry in doctest.DocTestParser().get_examples(example.__doc__):
        exec(compile(entry.source, example.__name__, "exec"), namespace)  # noqa: S102
    if example is fs.Leshy:
        assert isinstance(namespace["X"], pd.DataFrame)
        selected = namespace["feat_selector"].selected_features_
        assert list(namespace["X_filtered"].columns) == list(selected)
    else:
        table = namespace["fs_summary_df"].data
        assert table.predictor.tolist() == list(namespace["X"].columns)
        assert "lowimp" in table.columns


@pytest.mark.parametrize("method", ["association", "correlation"])
def test_collinearity_retains_unrelated_features_and_zero_diagonal(method):
    X = pd.DataFrame({"a": [-1, -1, 1, 1], "b": [-1, 1, -1, 1]})
    selector = fs.CollinearityThreshold(threshold=0, method=method).fit(X)
    np.testing.assert_allclose(selector.assoc_matrix_, 0)
    assert list(selector.selected_features_) == list(X.columns)
