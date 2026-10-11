"""Review 04: native references and strict counterexamples, no library repairs.

Run from /tmp with the installed-wheel interpreter; see the report for commands.
Monkeypatches only observe boundaries or isolate plotting from computation.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import arfs.benchmark as benchmark
from arfs.gbm import GradientBoosting, _make_split, gbm_flavour
from arfs.feature_selection.allrelevant import GrootCV, Leshy
from arfs.preprocessing import OrdinalEncoderPandas
from arfs.utils import LightForestClassifier, LightForestRegressor, is_lightgbm


def defect(number, reason, **kwargs):
    return pytest.mark.xfail(strict=True, reason=f"F{number}: {reason}", **kwargs)


def frame():
    rng = np.random.default_rng(614)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["genuine", "other", "random"])
    X.index = np.arange(200) * 7 + 1000
    y = pd.Series(3 * X.genuine - X.other + rng.normal(size=200), index=X.index)
    w = pd.Series(np.where(X.genuine > 0, 12., .2), index=X.index)
    return X, y, w


def params(objective="regression", **updates):
    return dict(objective=objective, metric="l2", n_estimators=25, num_threads=1,
                num_leaves=7, min_data_in_leaf=5, force_col_wise=True,
                seed=91, **updates)


def wrapper(p=None, **kwargs):
    return GradientBoosting(params=params() if p is None else p,
                            show_learning_curve=False, verbose_eval=0, **kwargs)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def native_fit(X, y, w, p, stratified=False, initial=None, categorical="auto", feval=None):
    splitter = StratifiedShuffleSplit if stratified else ShuffleSplit
    tr, va = next(splitter(n_splits=1, test_size=.2, random_state=42).split(X, y))
    train = lgb.Dataset(X.iloc[tr], label=y.iloc[tr],
                        weight=None if w is None else w.iloc[tr],
                        init_score=None if initial is None else np.asarray(initial)[tr],
                        categorical_feature=categorical)
    valid = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=train,
                        weight=None if w is None else w.iloc[va],
                        init_score=None if initial is None else np.asarray(initial)[va],
                        categorical_feature=categorical)
    p = p.copy()
    rounds = p.pop("n_estimators")
    p["verbosity"] = -1
    return lgb.train(p, train, num_boost_round=rounds, valid_sets=[train, valid],
                     valid_names=["train", "valid"], feval=feval,
                     callbacks=[lgb.early_stopping(10, verbose=False)]), tr, va


@pytest.mark.parametrize("mode", ["plain", "stratified", "groups"])
def test_split_positions_and_isolation(mode):
    X, y, w = frame()
    y = pd.Series(np.arange(len(X)) % 2, index=X.index + 500)
    initial = pd.Series(np.arange(len(X)) / 10, index=X.index[::-1])
    groups = pd.Series(np.arange(len(X)) // 5, index=X.index)
    splitter = {"plain": ShuffleSplit, "stratified": StratifiedShuffleSplit,
                "groups": GroupShuffleSplit}[mode](n_splits=1, test_size=.2, random_state=42)
    tr, va = next(splitter.split(X, y, groups=groups if mode == "groups" else None))
    result = _make_split(X, y, w, initial, groups if mode == "groups" else None, mode == "stratified")
    for actual, expected in zip(result, [X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va],
                                        w.iloc[va], w.iloc[tr], initial.iloc[va], initial.iloc[tr]]):
        (assert_frame_equal if isinstance(expected, pd.DataFrame) else assert_series_equal)(actual, expected)
    assert set(tr).isdisjoint(va)
    if mode == "groups":
        assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
    if mode == "stratified":
        assert y.iloc[va].mean() == y.mean()


@defect(1, "stratification silently defeats requested group isolation")
def test_stratification_and_groups_do_not_leak():
    X, _, _ = frame()
    y = pd.Series(np.arange(len(X)) % 2, index=X.index)
    groups = pd.Series(np.arange(len(X)) // 5, index=X.index)
    m = wrapper(params("binary"), stratified=True, return_valid_features=True)
    m.fit(X, y, groups=groups)
    valid_groups = set(groups.loc[m.valid_features.index])
    train_groups = set(groups.loc[X.index.difference(m.valid_features.index)])
    assert train_groups.isdisjoint(valid_groups), sorted(train_groups & valid_groups)


@pytest.mark.parametrize("field", ["y", "sample_weight", "init_score", "groups"])
def test_invalid_lengths_rejected(field):
    X, y, w = frame()
    kw = {}
    if field == "y":
        y = y.iloc[:-1]
    else:
        kw[field] = w.iloc[:-1]
    with pytest.raises(ValueError):
        wrapper().fit(X, y, **kw)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass", "poisson"])
def test_weighted_native_reference(task):
    X, y, w = frame()
    p = params(task)
    stratified = task in ("binary", "multiclass")
    if task == "binary":
        y = (y > 0).astype(int)
        p["metric"] = "binary_logloss"
    elif task == "multiclass":
        y = pd.qcut(y, 3, labels=False)
        p.update(num_class=3, metric="multi_logloss")
    elif task == "poisson":
        y = np.exp(y / 4)
        p["metric"] = "poisson"
    original_X, original_y, original_w, original_p = X.copy(), y.copy(), w.copy(), copy.deepcopy(p)
    m = wrapper(p, stratified=stratified, return_valid_features=True)
    assert m.fit(X, y, sample_weight=w) is None  # This wrapper is not a sklearn estimator.
    reference, tr, va = native_fit(X, y, w, p, stratified)
    assert_frame_equal(m.valid_features, X.iloc[va])
    assert_allclose(m.predict_raw(X), reference.predict(X), rtol=1e-12, atol=1e-12)
    assert m.model.best_iteration == reference.best_iteration
    assert_allclose(m.model.best_score["valid"][p["metric"]], reference.best_score["valid"][p["metric"]])
    unweighted, _, _ = native_fit(X, y, None, p, stratified)
    assert np.max(np.abs(reference.predict(X) - unweighted.predict(X))) > 1e-4
    assert_frame_equal(X, original_X)
    assert_series_equal(y, original_y)
    assert_series_equal(w, original_w)
    assert p == original_p


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_prediction_shapes_labels_and_raw_scores(task):
    X, y, _ = frame()
    p = params(task)
    if task == "binary":
        y = (y > 0).astype(int)
        p["metric"] = "binary_logloss"
    if task == "multiclass":
        y = pd.qcut(y, 3, labels=False)
        p.update(num_class=3, metric="multi_logloss")
    m = wrapper(p)
    m.fit(X, y)
    native = m.model.predict(X)
    expected = (native > .5).astype(int) if task == "binary" else (native.argmax(1) if task == "multiclass" else native)
    assert_allclose(m.predict(X), expected)
    assert_allclose(m.predict(X, predict_proba=True), native)
    assert_allclose(m.predict_raw(X, raw_score=True), m.model.predict(X, raw_score=True))
    assert_allclose(m.predict(X.iloc[:1]), expected[:1])
    assert_allclose(m.predict(X.to_numpy()), expected)
    assert m.predict_raw(X.iloc[:1], pred_leaf=True).shape[0] == 1


@pytest.mark.parametrize("method", ["predict", "predict_raw", "save"])
def test_unfitted_rejected(method, tmp_path):
    with pytest.raises(ValueError, match="not been trained"):
        getattr(wrapper(), method)(str(tmp_path) if method == "save" else frame()[0])


@pytest.mark.parametrize("method", ["predict", "predict_raw"])
@defect(2, "named prediction inputs are used positionally without schema validation")
def test_reordered_features_preserve_prediction_or_raise(method):
    X, y, _ = frame()
    m = wrapper()
    m.fit(X, y)
    expected = getattr(m, method)(X)
    try:
        actual = getattr(m, method)(X[X.columns[::-1]])
    except (ValueError, lgb.basic.LightGBMError):
        return
    assert_allclose(actual, expected)


@defect(2, "same-width replacement column is accepted")
def test_replacement_feature_rejected():
    X, y, _ = frame()
    m = wrapper()
    m.fit(X, y)
    with pytest.raises((ValueError, lgb.basic.LightGBMError)):
        m.predict(X.rename(columns={"genuine": "unrelated"}))


def test_missing_feature_rejected_and_native_schema_guard_available():
    X, y, _ = frame()
    m = wrapper()
    m.fit(X, y)
    with pytest.raises(lgb.basic.LightGBMError):
        m.predict(X.drop(columns="genuine"))
    with pytest.raises(lgb.basic.LightGBMError):
        m.predict_raw(X[X.columns[::-1]], validate_features=True)


@pytest.mark.parametrize("objective", ["softmax", "ova", "ovr"])
@defect(3, "valid multiclass aliases bypass class-label postprocessing")
def test_multiclass_aliases_return_labels(objective):
    X, y, _ = frame()
    y = pd.qcut(y, 3, labels=False)
    p = params(objective, num_class=3)
    p["metric"] = "multi_logloss"
    m = wrapper(p)
    m.fit(X, y)
    assert_array_equal(m.predict(X), m.model.predict(X).argmax(1))


@pytest.mark.parametrize("dtype", ["object", "string", "datetime64[ns]", "timedelta64[ns]"])
@defect(4, "auto-detection does not convert advertised unsupported categorical dtypes")
def test_advertised_auto_categorical_input(dtype):
    X, y, _ = frame()
    if dtype in ("object", "string"):
        X["category"] = pd.Series(["a", "b"] * 100, index=X.index, dtype=dtype)
    else:
        X["category"] = pd.Series(np.arange(200), index=X.index).astype(dtype)
    wrapper().fit(X, y)


def test_native_categories_and_boolean_work():
    X, y, w = frame()
    X["category"] = pd.Categorical(["red", "green"] * 100)
    X["flag"] = np.arange(200) % 2 == 0
    p = params()
    m = wrapper(p)
    m.fit(X, y, sample_weight=w)
    ref, _, _ = native_fit(X, y, w, p, categorical=["category", "flag"])
    assert_allclose(m.predict(X), ref.predict(X))
    assert m.cat_feat == ["category", "flag"]


@defect(5, "cat_feat=None still enables native categorical inference")
def test_none_disables_categorical_features():
    X, y, _ = frame()
    X["category"] = pd.Categorical(["red", "green"] * 100)
    m = wrapper(cat_feat=None)
    m.fit(X, y)
    assert m.model.dump_model()["feature_infos"]["category"]["values"] == []


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@defect(6, "nonfinite regression targets are accepted")
def test_invalid_targets_rejected(invalid):
    X, y, _ = frame()
    y.iloc[1] = invalid
    with pytest.raises((ValueError, lgb.basic.LightGBMError)):
        wrapper().fit(X, y)


def test_missing_target_matches_fabricated_zero():
    X, y, _ = frame()
    y.iloc[1] = np.nan
    m, zero = wrapper(), wrapper()
    m.fit(X, y)
    zero.fit(X, y.fillna(0))
    assert_allclose(m.predict(X), zero.predict(X), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["negative", "nan", "infinite", "all_zero"])
@defect(7, "invalid weights reach LightGBM without rejection")
def test_invalid_weights_rejected(invalid):
    X, y, w = frame()
    if invalid == "all_zero":
        w[:] = 0
    else:
        w.iloc[1] = {"negative": -1., "nan": np.nan, "infinite": np.inf}[invalid]
    with pytest.raises((ValueError, lgb.basic.LightGBMError)):
        wrapper().fit(X, y, sample_weight=w)


@pytest.mark.parametrize("task", ["regression", "binary", "poisson"])
def test_initial_scores_match_native_and_require_external_baseline(task):
    X, y, w = frame()
    p = params(task)
    if task == "binary":
        y = (y > 0).astype(int)
        p["metric"] = "binary_logloss"
    elif task == "poisson":
        y = np.exp(y / 4)
        p["metric"] = "poisson"
    initial = pd.Series(np.full(len(X), .7), index=X.index)
    m = wrapper(p)
    m.fit(X, y, sample_weight=w, init_score=initial)
    reference, _, va = native_fit(X, y, w, p, initial=initial)
    assert_allclose(m.predict_raw(X, raw_score=True), reference.predict(X, raw_score=True))
    with pytest.raises(AttributeError, match="init_score"):
        m.predict(X)
    raw = m.predict_raw(X, raw_score=True)
    full = raw + initial.to_numpy()
    if task == "regression":
        loss = (full[va] - y.iloc[va]) ** 2
    elif task == "binary":
        prob = 1 / (1 + np.exp(-full[va]))
        loss = -y.iloc[va] * np.log(prob) - (1 - y.iloc[va]) * np.log(1 - prob)
        assert_allclose(m.predict_raw(X), 1 / (1 + np.exp(-raw)))
    else:
        loss = np.exp(full[va]) - y.iloc[va] * full[va]
        assert_allclose(m.predict_raw(X), np.exp(raw))
    assert_allclose(np.average(loss, weights=w.iloc[va]), m.model.best_score["valid"][p["metric"]], rtol=1e-6)


@defect(8, "multiclass initial-score matrices are coerced to a 1D Series")
def test_multiclass_initial_score_matrix():
    X, y, w = frame()
    y = pd.qcut(y, 3, labels=False)
    p = params("multiclass", num_class=3)
    p["metric"] = "multi_logloss"
    initial = np.tile([.2, .3, .4], (len(X), 1))
    ref, _, _ = native_fit(X, y, w, p, initial=initial)
    m = wrapper(p)
    m.fit(X, y, sample_weight=w, init_score=initial)
    assert_allclose(m.predict_raw(X, raw_score=True), ref.predict(X, raw_score=True))


@defect(9, "a normal refit retains the previous initial-score guard")
def test_refit_resets_initial_score_flag():
    X, y, _ = frame()
    m = wrapper()
    m.fit(X, y, init_score=y * 0)
    m.fit(X, y)
    assert_allclose(m.predict(X), m.model.predict(X))


@defect(9, "disabled per-fit outputs retain old validation data and curve")
def test_refit_resets_optional_outputs():
    X, y, _ = frame()
    m = GradientBoosting(params=params(), return_valid_features=True, verbose_eval=0)
    m.fit(X, y)
    assert m.valid_features is not None and m.learning_curve is not None
    m.return_valid_features = False
    m.show_learning_curve = False
    m.fit(X.rename(columns={"genuine": "new"}), y)
    assert m.valid_features is None and m.learning_curve is None


def test_custom_metric_early_stopping_and_weights():
    X, y, w = frame()
    observed = []
    def metric(predictions, data):
        observed.append((len(predictions), data.get_weight().copy()))
        return "weighted_mse", float(np.average((predictions - data.get_label()) ** 2,
                                                weights=data.get_weight())), False
    p = params()
    p["metric"] = metric
    m = wrapper(p)
    m.fit(X, y, sample_weight=w)
    direct = p.copy()
    direct["metric"] = "None"
    ref, tr, va = native_fit(X, y, w, direct, feval=metric)
    assert_allclose(m.predict(X), ref.predict(X))
    assert_allclose(observed[0][1], w.iloc[tr])
    assert_allclose(observed[1][1], w.iloc[va])
    assert set(m.model.best_score["valid"]) == {"weighted_mse"}
    assert p["metric"] is metric


def test_early_stopping_uses_validation_metric():
    X, y, _ = frame()
    seen = {160: 0, 40: 0}
    def metric(predictions, data):
        n = len(predictions)
        seen[n] += 1
        return "contrary", -seen[n] if n == 160 else seen[n], False
    p = params()
    p["metric"] = metric
    m = wrapper(p)
    m.fit(X, y)
    assert m.model.best_iteration == 1
    assert seen == {160: 11, 40: 11}


def test_custom_objective_native_boundary():
    X, y, _ = frame()
    def objective(predictions, data):
        return predictions - data.get_label(), np.ones_like(predictions)
    p = params(objective)
    m = wrapper(p)
    m.fit(X, y)
    ref, _, _ = native_fit(X, y, None, p)
    assert_allclose(m.predict(X), ref.predict(X))
    assert m.model_params["objective"] == "none"


@pytest.mark.parametrize("objective", ["regression", "binary", "multiclassova", "tweedie"])
def test_default_metric_matches_native(objective):
    X, y, w = frame()
    p = params(objective)
    del p["metric"]
    if objective == "binary":
        y = (y > 0).astype(int)
    elif objective == "multiclassova":
        y = pd.qcut(y, 3, labels=False)
        p["num_class"] = 3
    elif objective == "tweedie":
        y = np.exp(y / 4)
    m = wrapper(p)
    m.fit(X, y, sample_weight=w)
    reference, _, _ = native_fit(X, y, w, p)
    assert_allclose(m.predict_raw(X), reference.predict(X))
    assert m.model.best_score["valid"] == reference.best_score["valid"]


def test_parameter_alias_precedence_and_array_fit():
    X, y, _ = frame()
    p = params(num_boost_round=1, num_iterations=2)
    p["n_estimators"] = 3
    m = wrapper(p)
    m.fit(X.to_numpy(), y.to_numpy())
    assert m.model.current_iteration() <= 2
    assert m.model.params["num_iterations"] == 2
    assert p["n_estimators"] == 3 and p["num_boost_round"] == 1
    assert m.predict(X.to_numpy()).shape == (len(X),)


def reload_fresh(m, X, tmp_path):
    model_path = m.save(str(tmp_path / "nested"), "trusted_model")
    data_path, output_path = tmp_path / "inputs.joblib", tmp_path / "output.json"
    joblib.dump(X, data_path)
    subprocess.run([sys.executable, str(Path(__file__).with_name("reload.py")),
                    model_path, str(data_path), str(output_path)], check=True,
                   cwd=tmp_path, capture_output=True, text=True, timeout=60)
    return json.loads(output_path.read_text())


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass", "category"])
def test_fresh_process_reload_predictions(task, tmp_path):
    X, y, _ = frame()
    p = params("regression" if task == "category" else task)
    if task == "category":
        X["category"] = pd.Categorical(["a", "b"] * 100)
    if task == "binary":
        y = (y > 0).astype(int)
        p["metric"] = "binary_logloss"
    if task == "multiclass":
        y = pd.qcut(y, 3, labels=False)
        p.update(num_class=3, metric="multi_logloss")
    m = wrapper(p)
    m.fit(X, y)
    result = reload_fresh(m, X, tmp_path)
    assert_allclose(result["prediction"], m.predict(X), rtol=0, atol=0)
    assert_allclose(result["raw"], m.predict_raw(X, raw_score=True), rtol=0, atol=0)
    assert result["feature_names"] == m.model.feature_name()
    assert result["best_iteration"] == m.model.best_iteration


@defect(10, "saved Booster loses the initial-score prediction guard")
def test_reload_preserves_initial_score_contract(tmp_path):
    X, y, _ = frame()
    m = wrapper()
    m.fit(X, y, init_score=pd.Series(np.full(len(y), 10.)))
    result = reload_fresh(m, X, tmp_path)
    assert_allclose(result["raw"], m.predict_raw(X, raw_score=True))
    assert result.get("prediction_error") == "AttributeError", result


@defect(10, "load treats category levels as categorical feature names")
def test_reload_preserves_category_names(tmp_path):
    X, y, _ = frame()
    X["category"] = pd.Categorical(["a", "b"] * 100)
    m = wrapper()
    m.fit(X, y)
    assert reload_fresh(m, X, tmp_path)["cat_feat"] == ["category"]


@defect(10, "loading a normal Booster retains a previous initial-score flag")
def test_load_into_previously_fitted_wrapper(tmp_path):
    X, y, _ = frame()
    old, new = wrapper(), wrapper()
    old.fit(X, y, init_score=y * 0)
    new.fit(X, y)
    old.load(new.save(str(tmp_path), "trusted_normal"))
    assert_allclose(old.predict(X), new.predict(X))


def test_save_paths_overwrite_curve_and_missing_load(tmp_path, monkeypatch):
    X, y, _ = frame()
    m = GradientBoosting(params=params(), verbose_eval=0)
    m.fit(X, y)
    path = Path(m.save(str(tmp_path / "new"), "trusted"))
    assert path.is_file() and path.with_name("trusted_learning_curve.png").is_file()
    before = path.read_bytes()
    m.fit(X, y + 100)
    assert m.save(str(tmp_path / "new"), "trusted") == str(path)
    assert path.read_bytes() != before  # Existing names overwrite silently.
    monkeypatch.chdir(tmp_path)
    relative = m.save(name="relative")
    assert not Path(relative).is_absolute() and Path(relative).is_file()
    with pytest.raises(FileNotFoundError):
        m.load(str(tmp_path / "absent.joblib"))
    occupied = tmp_path / "not_directory"
    occupied.write_text("sentinel")
    with pytest.raises(FileExistsError):
        m.save(str(occupied), "model")
    assert occupied.read_text() == "sentinel"


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_benchmark_weighted_reference_and_feature_labels(task, monkeypatch):
    X, y, w = frame()
    if task == "classification":
        y = (y > 0).astype(int)
    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    model = cls(max_depth=3, random_state=8)
    records = {}
    real = benchmark.permutation_importance
    def observe(model, features, target, **kwargs):
        records.update(X=features.copy(), y=target.copy(), weights=kwargs["sample_weight"].copy())
        records["result"] = real(model, features, target, **kwargs)
        return records["result"]
    def plot(self, values, **kwargs):
        records.update(plotted=values.copy(), labels=list(kwargs["labels"]))
        self.set_yticks(np.arange(1, len(kwargs["labels"]) + 1), kwargs["labels"])
        return {}
    monkeypatch.setattr(benchmark, "permutation_importance", observe)
    monkeypatch.setattr(Axes, "boxplot", plot)  # Isolate computation from F13.
    benchmark.sklearn_pimp_bench(model, X, y, task=task, sample_weight=w)
    tr, va = train_test_split(np.arange(len(X)), stratify=y if task == "classification" else None, random_state=42)
    ref = cls(max_depth=3, random_state=8).fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
    expected = permutation_importance(ref, X.iloc[va], y.iloc[va], sample_weight=w.iloc[va],
                                      n_repeats=10, random_state=42, n_jobs=1)
    unweighted = permutation_importance(ref, X.iloc[va], y.iloc[va], n_repeats=10, random_state=42, n_jobs=1)
    assert_frame_equal(records["X"], X.iloc[va])
    assert_series_equal(records["weights"], w.iloc[va])
    assert_allclose(records["result"].importances, expected.importances)
    assert not np.allclose(expected.importances, unweighted.importances)
    order = expected.importances_mean.argsort()
    assert records["labels"] == list(X.columns[order])
    assert_allclose(records["plotted"], expected.importances[order].T)
    metric = accuracy_score if task == "classification" else r2_score
    assert_allclose(model.score(X.iloc[va], y.iloc[va], sample_weight=w.iloc[va]),
                    metric(y.iloc[va], ref.predict(X.iloc[va]), sample_weight=w.iloc[va]))


@defect(11, "benchmark fits its category encoder before the holdout split", raises=AssertionError)
def test_benchmark_preprocessing_uses_training_rows_only(monkeypatch):
    X, y, _ = frame()
    X.columns = ["signal", "other", "noise"]
    tr, va = train_test_split(np.arange(len(X)), random_state=42)
    X["category"] = pd.Series("train", index=X.index, dtype=object)
    X.iloc[va, -1] = "heldout_only"
    reference = OrdinalEncoderPandas().fit(X.iloc[tr])
    assert reference.categories_[0].tolist() == ["train"]
    assert reference.transform(X.iloc[va])["category"].isna().all()
    records = []
    original = OrdinalEncoderPandas.fit
    def observe(self, data, *args, **kwargs):
        result = original(self, data, *args, **kwargs)
        records.append((data.index.copy(), [a.tolist() for a in self.categories_]))
        return result
    monkeypatch.setattr(OrdinalEncoderPandas, "fit", observe)
    monkeypatch.setattr(Axes, "boxplot", lambda *args, **kwargs: {})
    benchmark.sklearn_pimp_bench(DecisionTreeRegressor(max_depth=2, random_state=8), X, y)
    assert set(records[0][0]).isdisjoint(X.index[va]), records


@defect(12, "documented ndarray input requires DataFrame.columns")
def test_benchmark_accepts_documented_array(monkeypatch):
    X, y, _ = frame()
    monkeypatch.setattr(Axes, "boxplot", lambda *args, **kwargs: {})
    benchmark.sklearn_pimp_bench(DecisionTreeRegressor(max_depth=2), X.to_numpy(), y)


@defect(12, "integer feature names fail substring highlighting")
def test_benchmark_integer_feature_names(monkeypatch):
    X, y, _ = frame()
    X.columns = [0, 1, 2]
    monkeypatch.setattr(Axes, "boxplot", lambda *args, **kwargs: {})
    benchmark.sklearn_pimp_bench(DecisionTreeRegressor(max_depth=2), X, y)


@pytest.mark.xfail(tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 11),
                   strict=True, reason="F13: removed matplotlib boxplot labels argument", raises=TypeError)
def test_benchmark_real_plot():
    X, y, _ = frame()
    fig = benchmark.sklearn_pimp_bench(DecisionTreeRegressor(max_depth=2, random_state=8), X, y)
    assert sorted(t.get_text() for t in fig.axes[0].get_yticklabels()) == sorted(X.columns)


@pytest.mark.xfail(
    strict=True,
    reason="Inherited unit 0: pandas 3 shadow arrays are read-only" if int(pd.__version__.split(".")[0]) >= 3
    else "F14: GrootCV ignores injected estimator and importance mode",
    raises=ValueError if int(pd.__version__.split(".")[0]) >= 3 else AssertionError,
)
def test_compare_grootcv_configuration_is_real(monkeypatch):
    X, y, w = frame()
    records = []
    supplied_fits = []
    original_train, original_fit = lgb.train, DecisionTreeRegressor.fit
    def observe_train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        records.append(type(result).__name__)
        return result
    def observe_fit(self, *args, **kwargs):
        supplied_fits.append(type(self).__name__)
        return original_fit(self, *args, **kwargs)
    monkeypatch.setattr(lgb, "train", observe_train)
    monkeypatch.setattr(DecisionTreeRegressor, "fit", observe_fit)
    monkeypatch.setattr(GrootCV, "plot_importance", lambda *args, **kwargs: None)
    selector = GrootCV(objective="regression", n_folds=2, n_iter=1, n_jobs=1,
                       lgbm_params={"num_leaves": 3, "learning_rate": .3})
    np.random.seed(15)
    benchmark.compare_varimp(selector, [DecisionTreeRegressor()], X, y, sample_weight=w)
    assert supplied_fits, {"native_models_fitted": records, "supplied_model_fits": supplied_fits}


@defect(15, "comparison reuses selector with stale estimator flags and advanced RNG", raises=AssertionError)
def test_compare_reinitializes_selector_for_each_model(monkeypatch):
    X, y, w = frame()
    original = RandomForestRegressor(n_estimators=2, random_state=4)
    selector = Leshy(original, n_estimators=2, max_iter=2, random_state=11)
    records = []
    original_fit = Leshy.fit
    def observe(self, X, y, sample_weight=None):
        fresh = Leshy(clone(self.estimator), n_estimators=2, max_iter=2, random_state=11)
        records.append((id(self), self.is_lgb, fresh.is_lgb, self.random_state))
        assert_series_equal(sample_weight, w)
        return original_fit(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(Leshy, "fit", observe)
    monkeypatch.setattr(Leshy, "plot_importance", lambda *args, **kwargs: None)
    models = [lgb.LGBMRegressor(n_estimators=2, n_jobs=1, verbosity=-1)]
    benchmark.compare_varimp(selector, models, X, y, sample_weight=w)
    assert all(cached == actual and isinstance(seed, int) for _, cached, actual, seed in records), records


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_lightforest_and_model_detection(task):
    X, y, w = frame()
    factory = LightForestClassifier if task == "classification" else LightForestRegressor
    if task == "classification":
        y = (y > 0).astype(int)
    m = factory(X.shape[1], n_estimators=3).set_params(n_jobs=1, random_state=8)
    assert m.get_params()["boosting_type"] == "rf"
    assert_allclose(m.get_params()["colsample_bytree"], 1 / np.sqrt(3) if task == "classification" else 1 / 3)
    m.fit(X, y, sample_weight=w)
    assert np.isfinite(m.predict(X)).all()
    assert is_lightgbm(m) and gbm_flavour(m) == "lgb"
    assert gbm_flavour(m.booster_) == "lgb"
    assert not is_lightgbm(m.booster_)  # This helper recognizes sklearn LGBM classes only.
    assert gbm_flavour(DecisionTreeRegressor()) == "unknown"
