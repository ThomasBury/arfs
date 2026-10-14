"""Benchmark preprocessing, input labels and actual comparison dispatch."""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from arfs import benchmark
from arfs.feature_selection import allrelevant as a
from arfs.preprocessing import OrdinalEncoderPandas


def data(classification=False):
    rng = np.random.RandomState(47)
    X = pd.DataFrame(
        rng.normal(size=(120, 3)),
        columns=["genuine_signal", "other", "random_noise"],
        index=np.arange(120) * 7 + 100,
    )
    y = 3 * X.genuine_signal + rng.normal(size=len(X))
    return X, (y > 0).astype(int) if classification else y


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_encoder_fits_training_only_and_unknowns_stay_unknown(
    monkeypatch, classification, weighted
):
    X, y = data(classification)
    tr, va = train_test_split(
        np.arange(len(X)), stratify=y if classification else None, random_state=42
    )
    X["category"] = "training"
    X.iloc[va, -1] = "heldout_only"
    before = X.copy(deep=True)
    records = {}
    fit = OrdinalEncoderPandas.fit
    importance = benchmark.permutation_importance

    def observe_fit(self, frame, *args, **kwargs):
        result = fit(self, frame, *args, **kwargs)
        records["encoder"] = self
        records["training"] = frame.copy()
        return result

    def observe_importance(model, frame, target, **kwargs):
        records["validation"] = frame.copy()
        records["weights"] = kwargs["sample_weight"]
        result = importance(model, frame, target, **kwargs)
        assert np.isfinite(result.importances).all()
        return result

    monkeypatch.setattr(OrdinalEncoderPandas, "fit", observe_fit)
    monkeypatch.setattr(benchmark, "permutation_importance", observe_importance)
    cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
    weights = np.linspace(1, 4, len(X)) if weighted else None
    fig = benchmark.sklearn_pimp_bench(
        cls(max_depth=3, random_state=8),
        X,
        y,
        task="classification" if classification else "regression",
        sample_weight=weights,
    )
    try:
        assert_frame_equal(records["training"], X.iloc[tr])
        assert_array_equal(records["encoder"].categories_[0], ["training"])
        assert_array_equal(records["validation"].index, X.iloc[va].index)
        assert records["validation"].category.isna().all()
        if weighted:
            assert_array_equal(records["weights"], weights[va])
        else:
            assert records["weights"] is None
        assert_frame_equal(X, before)
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["array", "integer", "named"])
def test_real_plot_labels_and_model_input_order(kind):
    X, y = data()
    if kind == "array":
        X = X.to_numpy()
        labels = ["x0", "x1", "x2"]
    elif kind == "integer":
        X.columns = [10, 5, 20]
        labels = ["10", "5", "20"]
    else:
        labels = list(X.columns)
    before = X.copy()
    model = DecisionTreeRegressor(max_depth=3, random_state=8)
    fig = benchmark.sklearn_pimp_bench(model, X, y)
    try:
        fig.canvas.draw()
        ticks = fig.axes[0].get_yticklabels()
        assert sorted(t.get_text() for t in ticks) == sorted(labels)
        if kind == "named":
            colors = {t.get_text(): t.get_color() for t in ticks}
            assert colors["genuine_signal"] == "green"
            assert colors["random_noise"] == "red"
        tr, va = train_test_split(np.arange(len(X)), random_state=42)
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=labels)
        ref = DecisionTreeRegressor(max_depth=3, random_state=8).fit(
            frame.iloc[tr], y.iloc[tr]
        )
        assert_allclose(model.predict(frame.iloc[va]), ref.predict(frame.iloc[va]))
        if kind != "integer":
            assert_array_equal(model.feature_names_in_, frame.columns)
        if isinstance(X, pd.DataFrame):
            assert_frame_equal(X, before)
        else:
            assert_array_equal(X, before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("X", [np.arange(10), np.zeros((10, 2, 1)), [[1, 2]]])
def test_bad_benchmark_input_fails_before_training(X):
    model = DecisionTreeRegressor()
    with pytest.raises(TypeError, match="two-dimensional ndarray"):
        benchmark.sklearn_pimp_bench(model, X, np.arange(len(X)))
    assert not hasattr(model, "tree_")


@pytest.mark.parametrize("selector", [a.GrootCV(objective="regression"), object()])
def test_unsupported_comparison_fails_before_training(monkeypatch, selector):
    X, y = data()

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported comparison started native training")

    monkeypatch.setattr(lgb, "train", unexpected)
    with pytest.raises(TypeError, match="Leshy or BoostAGroota.*GrootCV"):
        benchmark.compare_varimp(selector, [DecisionTreeRegressor()], X, y)
    assert not hasattr(selector, "estimator")


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("selector_cls", [a.Leshy, a.BoostAGroota])
def test_comparisons_run_each_supplied_model_and_method(
    monkeypatch, capsys, classification, selector_cls
):
    X, y = data(classification)
    before = X.copy(deep=True)
    weights = np.linspace(1, 4, len(X))
    forest = RandomForestClassifier if classification else RandomForestRegressor
    light = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
    models = [
        forest(n_estimators=3, max_depth=3, n_jobs=1, random_state=8),
        light(n_estimators=3, num_leaves=4, n_jobs=1, verbosity=-1, random_state=8),
    ]
    if selector_cls is a.Leshy:
        selector = selector_cls(models[0], n_estimators=3, max_iter=2, random_state=11)
    else:
        selector = selector_cls(models[0], iters=1, max_rounds=1)
    runs, calls, fitted = [], [], []
    fit = selector_cls.fit

    def observe_fit(self, X, y, sample_weight=None):
        runs.append(self)
        assert_array_equal(sample_weight, weights)
        if selector_cls is a.Leshy:
            assert self.random_state == 11
            assert self.is_lgb == isinstance(self.estimator, light)
        return fit(self, X, y, sample_weight=sample_weight)

    def observer(method, original):
        def observe(estimator, X, y, sample_weight=None, **kwargs):
            calls.append((type(estimator).__name__, method))
            assert_array_equal(sample_weight, weights)
            return original(estimator, X, y, sample_weight=sample_weight, **kwargs)

        return observe

    def fit_observer(original):
        def observe(self, *args, **kwargs):
            fitted.append(type(self).__name__)
            return original(self, *args, **kwargs)

        return observe

    for method, name in [
        ("shap", "_get_shap_imp"),
        ("pimp", "_get_perm_imp"),
        ("native", "_get_imp"),
    ]:
        monkeypatch.setattr(a, name, observer(method, getattr(a, name)))
    for cls in [forest, light]:
        monkeypatch.setattr(cls, "fit", fit_observer(cls.fit))
    monkeypatch.setattr(selector_cls, "fit", observe_fit)
    monkeypatch.setattr(selector_cls, "plot_importance", lambda *args, **kwargs: None)
    benchmark.compare_varimp(selector, models, X, y, sample_weight=weights)
    expected = {
        (type(m).__name__, mode) for m in models for mode in ["shap", "pimp", "native"]
    }
    assert set(calls) == expected
    assert set(fitted) == {type(m).__name__ for m in models}
    assert len(runs) == len({id(run) for run in runs}) == 6
    assert len({id(run.estimator) for run in runs}) == 6
    assert all(run.estimator not in models for run in runs)
    assert not hasattr(selector, "selected_features_")
    assert selector.importance == "shap"
    assert selector.estimator is models[0]
    assert all(not hasattr(m, "n_features_in_") for m in models)
    assert_frame_equal(X, before)
    output = capsys.readouterr().out
    assert all(name in output and method in output for name, method in expected)


@pytest.mark.parametrize("seed_kind", ["integer", "instance"])
def test_comparison_seed_state_is_copied_even_from_fitted_selector(
    monkeypatch, seed_kind
):
    X, y = data()
    seed = 11 if seed_kind == "integer" else np.random.RandomState(11)
    model = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=8)
    selector = a.Leshy(
        model, n_estimators=3, max_iter=2, random_state=seed, importance="native"
    ).fit(X, y)
    history = selector.imp_real_hist.copy()
    state = selector.random_state.get_state()
    runs = []
    fit = a.Leshy.fit

    def observe(self, *args, **kwargs):
        runs.append(self)
        assert self.random_state is not selector.random_state
        current = self.random_state.get_state()
        assert_array_equal(current[1], state[1])
        assert current[2:] == state[2:]
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(a.Leshy, "fit", observe)
    monkeypatch.setattr(a.Leshy, "plot_importance", lambda *args, **kwargs: None)
    benchmark.compare_varimp(selector, [model], X, y)
    assert len(runs) == 3
    assert_array_equal(selector.random_state.get_state()[1], state[1])
    assert selector.random_state.get_state()[2:] == state[2:]
    assert_array_equal(selector.imp_real_hist, history)
    assert selector.importance == "native"


@pytest.mark.parametrize("to_lightgbm", [False, True])
def test_leshy_set_params_refreshes_backend_on_real_fit(to_lightgbm):
    X, y = data()
    forest = RandomForestRegressor(n_estimators=3, max_depth=3, n_jobs=1)
    light = lgb.LGBMRegressor(n_estimators=3, num_leaves=4, n_jobs=1, verbosity=-1)
    old, new = (forest, light) if to_lightgbm else (light, forest)
    selector = a.Leshy(
        old, n_estimators=3, max_iter=2, random_state=11, importance="native"
    )
    selector.set_params(estimator=new).fit(X, y)
    assert selector.is_lgb == to_lightgbm
    assert selector.is_cat is False
    assert (
        isinstance(new.random_state, (int, np.integer))
        if to_lightgbm
        else isinstance(new.random_state, np.random.RandomState)
    )
