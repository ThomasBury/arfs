"""C02: fitted state owns learners, callbacks and reproducible randomness."""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline

from arfs import sampling
from arfs.feature_selection import allrelevant as a
from arfs.feature_selection import mrmr as m


def data():
    rng = np.random.RandomState(52)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    return X, 4 * X.a - X.b + rng.normal(size=len(X)) * 0.1


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize("n_estimators", [4, "auto"])
def test_leshy_owns_learner_and_replays_integer_seed_on_refit_and_clone(
    monkeypatch, n_estimators
):
    X, y = data()
    learner = RandomForestRegressor(n_estimators=7, max_depth=3, random_state=91)
    learner.fit(X, y)
    params, predictions = learner.get_params(), learner.predict(X)
    seeds = []
    original = a._get_imp

    def importance(estimator, *args, **kwargs):
        assert estimator is not learner
        assert isinstance(estimator.random_state, int)
        seeds.append(estimator.random_state)
        return original(estimator, *args, **kwargs)

    monkeypatch.setattr(a, "_get_imp", importance)
    selector = a.Leshy(
        learner,
        n_estimators=n_estimators,
        max_iter=3,
        importance="native",
        random_state=12,
    )
    state = np.random.get_state()
    assert selector.fit(X, y) is selector
    history, support = selector.imp_real_hist.copy(), selector.support_.copy()
    owned = selector.estimator_
    for candidate in [selector, clone(selector)]:
        candidate.fit(X, y)
        assert candidate.random_state == 12
        assert candidate.importance == candidate.importance_ == "native"
        assert candidate.n_estimators == n_estimators
        assert candidate.estimator_.n_estimators > 0
        assert candidate.estimator_ is not owned
        assert_array_equal(candidate.imp_real_hist, history)
        assert_array_equal(candidate.support_, support)
    assert seeds[:3] == seeds[3:6] == seeds[6:]
    assert len(set(seeds[:3])) > 1
    assert learner.get_params() == params
    assert_array_equal(learner.predict(X), predictions)
    assert selector.estimator is learner
    assert_rng_equal(state, np.random.get_state())


@pytest.mark.parametrize("kind", ["leshy", "boost"])
def test_fastshap_fallback_is_runtime_state_and_is_resolved_on_refit(monkeypatch, kind):
    X, y = data()
    learner = RandomForestRegressor(n_estimators=5, random_state=91)
    selector = (
        a.Leshy(
            learner, n_estimators=4, max_iter=2, importance="fastshap", random_state=1
        )
        if kind == "leshy"
        else a.BoostAGroota(
            learner, iters=1, max_rounds=1, importance="fastshap", random_state=1
        )
    )
    calls = []

    def importance(backend):
        def compute(estimator, frame, *args, **kwargs):
            calls.append(backend)
            return np.ones(frame.shape[1])

        return compute

    monkeypatch.setattr(a, "_get_shap_imp", importance("shap"))
    monkeypatch.setattr(a, "_get_shap_imp_fast", importance("fastshap"))
    monkeypatch.setitem(sys.modules, "fasttreeshap", None)
    with pytest.warns(UserWarning, match="Fallback to shap"):
        selector.fit(X, y)
    assert selector.importance == "fastshap"
    assert selector.importance_ == "shap"
    assert set(calls) == {"shap"}
    assert clone(selector).importance == "fastshap"
    calls.clear()
    monkeypatch.setitem(
        sys.modules, "fasttreeshap", SimpleNamespace(TreeExplainer=object)
    )
    selector.fit(X, y)
    assert selector.importance_ == "fastshap"
    assert set(calls) == {"fastshap"}
    assert learner.n_estimators == 5 and learner.random_state == 91


def test_mrmr_resolves_current_default_callbacks_at_each_fit(monkeypatch):
    X, y = data()
    calls = []

    def relevance(kind):
        def compute(X, y, sample_weight, n_jobs):
            calls.append((kind, n_jobs))
            return pd.Series([0.9, 0.5, 0.1], index=X.columns)

        return compute

    def redundancy(X, features, n_jobs, normalize, **kwargs):
        calls.append(("redundancy", n_jobs))
        assert normalize is False
        return pd.Series(0.2, index=features)

    monkeypatch.setattr(m, "f_stat_regression_parallel", relevance("regression"))
    monkeypatch.setattr(
        m, "f_stat_classification_parallel", relevance("classification")
    )
    monkeypatch.setattr(m, "association_series", redundancy)
    selector = m.MinRedundancyMaxRelevance(2, show_progress=False)
    assert selector.relevance_func is selector.redundancy_func is None
    assert selector.fit(X, y) is selector
    selector.set_params(task="classification", n_jobs=2).fit(X, y > y.median())
    clone(selector).fit(X, y > y.median())
    assert (
        calls
        == [("regression", 1), ("redundancy", 1)]
        + [("classification", 2), ("redundancy", 2)] * 2
    )
    assert selector.relevance_func is selector.redundancy_func is None


def test_mrmr_custom_callbacks_and_pipeline_fit_contract():
    X, y = data()

    def relevance(X, **kwargs):
        return pd.Series([0.9, 0.5, 0.1], index=X.columns)

    def redundancy(features, **kwargs):
        return pd.Series(0.2, index=features)

    selector = m.MinRedundancyMaxRelevance(
        2, relevance_func=relevance, redundancy_func=redundancy, show_progress=False
    )
    selector.set_params(n_jobs=2, task="classification")
    assert selector.fit(X, y > y.median()) is selector
    assert selector.relevance_func_ is selector.relevance_func is relevance
    assert selector.redundancy_func_ is selector.redundancy_func is redundancy
    assert_frame_equal(selector.fit_transform(X, y > y.median()), X[["a", "b"]])
    pipeline = Pipeline([("select", clone(selector)), ("predict", Ridge())])
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()
    assert "return_scores" not in selector.get_params()
    assert selector.ranking_.index.tolist() == selector.selected_features_
    assert selector.relevance_.index.tolist() == X.columns.tolist()
    assert selector.redundancy_.index.tolist() == selector.selected_features_


@pytest.mark.parametrize("kind", ["boost", "groot"])
def test_shadow_selectors_replay_integer_seeds_without_global_draws(monkeypatch, kind):
    X, y = data()
    frames = []
    create_shadow = a._create_shadow

    def shadow(frame, *args, **kwargs):
        result = create_shadow(frame, *args, **kwargs)
        frames.append(result[0].copy())
        return result

    monkeypatch.setattr(a, "_create_shadow", shadow)
    if kind == "boost":
        learner = RandomForestRegressor(n_estimators=4, max_depth=3, random_state=91)
        params = learner.get_params()
        seen_seeds = []
        original = a._get_imp

        def importance(estimator, *args, **kwargs):
            seen_seeds.append(estimator.random_state)
            return original(estimator, *args, **kwargs)

        monkeypatch.setattr(a, "_get_imp", importance)
        selector = a.BoostAGroota(
            learner, iters=2, max_rounds=2, importance="native", random_state=12
        )
    else:
        selector = a.GrootCV(
            objective="l2",
            n_folds=2,
            n_iter=2,
            n_jobs=1,
            random_state=12,
            lgbm_params={"num_iterations": 4, "min_data_in_leaf": 5},
        )
    state = np.random.get_state()
    selector.fit(X, y)
    initial_frames, frames = frames, []
    table = (selector.sha_cutoff_df if kind == "boost" else selector.cv_df).copy()
    support = selector.support_.copy()
    for candidate in [selector, clone(selector)]:
        candidate.fit(X, y)
        assert len(frames) == len(initial_frames)
        for first, second in zip(initial_frames, frames):
            assert_frame_equal(first, second)
        frames.clear()
        assert_array_equal(candidate.support_, support)
        assert_frame_equal(
            candidate.sha_cutoff_df if kind == "boost" else candidate.cv_df, table
        )
        assert candidate.random_state == 12
    selector.set_params(random_state=13).fit(X, y)
    assert not frames[0].equals(initial_frames[0])
    if kind == "boost":
        assert learner.get_params() == params
        assert set(seen_seeds) == {91}
        assert not hasattr(learner, "estimators_")
    assert_rng_equal(state, np.random.get_state())


@pytest.mark.parametrize("custom", [False, True])
def test_groot_default_and_custom_folds_keep_seed_contract(monkeypatch, custom):
    assert a.GrootCV().random_state == 2652124
    X, y = data()
    folds = KFold(n_splits=3, shuffle=True, random_state=81) if custom else None
    reference = (
        folds
        if custom
        else RepeatedKFold(n_splits=2, n_repeats=1, random_state=2652124)
    )
    expected = [(X.index[tr], X.index[va]) for tr, va in reference.split(X, y)]
    seen = []

    def train(frame, target, weight, validation, *args, **kwargs):
        seen.append((frame.index, validation.index))
        return SimpleNamespace(best_iteration=1), None, 1

    monkeypatch.setattr(a, "_train_lgb_model", train)
    monkeypatch.setattr(
        a,
        "_compute_importance",
        lambda frame, *args: list(zip(frame.columns, np.ones(frame.shape[1]))),
    )
    selector = a.GrootCV(objective="l2", n_folds=2, n_iter=1, folds=folds)
    selector.fit(X, y)
    assert len(seen) == len(expected)
    for actual, wanted in zip(seen, expected):
        for left, right in zip(actual, wanted):
            assert_array_equal(left, right)
    if custom:
        seen.clear()
        selector.set_params(random_state=42, n_folds=5, n_iter=4).fit(X, y)
        assert selector.folds is folds
        assert folds.random_state == 81
        assert len(seen) == 3
        for actual, wanted in zip(seen, expected):
            for left, right in zip(actual, wanted):
                assert_array_equal(left, right)


def test_leshy_pipeline_refreshes_owned_estimator_after_set_params(monkeypatch):
    import lightgbm as lgb

    X, y = data()
    selector = a.Leshy(lgb.LGBMRegressor(), max_iter=2, random_state=1)
    learner = RandomForestRegressor(n_estimators=7, max_depth=3, random_state=91)
    selector.set_params(estimator=learner, n_estimators="auto", importance="native")

    def importance(estimator, frame, *args, **kwargs):
        assert isinstance(estimator, RandomForestRegressor)
        assert estimator is not learner
        assert estimator.n_estimators != 7
        return np.r_[np.ones(X.shape[1]) * 4, np.ones(frame.shape[1] - X.shape[1])]

    monkeypatch.setattr(a, "_get_imp", importance)
    output = Pipeline([("select", selector)]).fit_transform(X, y)
    assert_frame_equal(output, selector.transform(X))
    assert not selector.is_lgb
    assert learner.n_estimators == 7 and learner.random_state == 91


@pytest.mark.parametrize("kind", ["leshy", "boost", "groot", "mrmr", "isolation"])
@pytest.mark.parametrize("seed", [None, "instance"])
def test_none_and_randomstate_instances_keep_sklearn_rng_semantics(
    kind, seed, monkeypatch
):
    X, y = data()
    rng = np.random.mtrand._rand if seed is None else np.random.RandomState(41)
    random_state = None if seed is None else rng
    state = rng.get_state()
    learner = RandomForestRegressor(n_estimators=2, random_state=9)
    monkeypatch.setattr(
        a, "_get_imp", lambda est, frame, *args, **kwargs: np.ones(frame.shape[1])
    )
    monkeypatch.setattr(
        a,
        "_train_lgb_model",
        lambda *args, **kwargs: (SimpleNamespace(best_iteration=1), None, 1),
    )
    monkeypatch.setattr(
        a,
        "_compute_importance",
        lambda frame, *args: list(zip(frame.columns, np.ones(frame.shape[1]))),
    )
    try:
        if kind == "isolation":
            monkeypatch.setattr(
                sampling, "ks_2samp", lambda *args: SimpleNamespace(pvalue=1.0)
            )
            sampling.isof_find_sample(X, random_state=random_state)
        else:
            if kind == "leshy":
                selector = a.Leshy(
                    learner,
                    n_estimators=2,
                    max_iter=2,
                    importance="native",
                    random_state=random_state,
                )
            elif kind == "boost":
                selector = a.BoostAGroota(
                    learner,
                    iters=1,
                    max_rounds=1,
                    importance="native",
                    random_state=random_state,
                )
            elif kind == "groot":
                selector = a.GrootCV(
                    objective="l2", n_folds=2, n_iter=1, random_state=random_state
                )
            else:
                selector = m.MinRedundancyMaxRelevance(
                    1, sample_size=20, show_progress=False, random_state=random_state
                )
            selector.fit(X, y)
            assert selector.random_state is random_state
            if kind != "mrmr":
                assert selector.random_state_ is rng
        assert (
            not np.array_equal(state[1], rng.get_state()[1])
            or state[2] != rng.get_state()[2]
        )
    finally:
        rng.set_state(state)


@pytest.mark.parametrize(
    "entry", [sampling.sample, sampling.isof_find_sample, sampling.isolation_forest]
)
def test_isolation_integer_seed_replays_forest_and_draws(entry, monkeypatch):
    X, _ = data()
    weights = np.arange(len(X)) + 1
    # Accept a nontrivial first subset so a full-data fallback cannot hide random draws.
    monkeypatch.setattr(sampling, "ks_2samp", lambda *args: SimpleNamespace(pvalue=1.0))
    kwargs = {"method": "isoforest"} if entry is sampling.sample else {}
    state = np.random.get_state()
    first = entry(X, sample_weight=weights, random_state=12, **kwargs)
    second = entry(X, sample_weight=weights, random_state=12, **kwargs)
    other = entry(X, sample_weight=weights, random_state=13, **kwargs)
    if entry is sampling.sample:
        assert_frame_equal(first, second)
        assert_frame_equal(first, X.loc[first.index])
        assert not first.equals(other)
    else:
        assert_array_equal(first, second)
        assert not np.array_equal(first, other)
    assert_rng_equal(state, np.random.get_state())


@pytest.mark.parametrize(
    "module",
    [
        "arfs.preprocessing",
        "arfs.feature_selection.base",
        "arfs.feature_selection.unsupervised",
        "arfs.feature_selection.allrelevant",
        "arfs.feature_selection.mrmr",
        "arfs.sampling",
    ],
)
def test_import_preserves_global_numpy_rng_in_fresh_process(module):
    script = f"""
import importlib
import numpy as np
np.random.seed(812)
expected = np.random.RandomState(812).random_sample(10)
importlib.import_module({module!r})
np.testing.assert_array_equal(np.random.random_sample(10), expected)
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())
