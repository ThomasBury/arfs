"""R21: named predictions, supported categories, positional offsets and refits."""

import copy

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.model_selection import (
    GroupShuffleSplit,
    ShuffleSplit,
    StratifiedShuffleSplit,
)

from arfs.gbm import GradientBoosting, _make_split


@pytest.fixture
def data():
    rng = np.random.default_rng(614)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["signal", "other", "noise"])
    X.index = np.arange(200) * 7 + 1000
    y = pd.Series(
        (3 * X.signal - X.other).to_numpy() + rng.normal(size=200), index=X.index + 500
    )
    weight = pd.Series(np.where(X.signal > 0, 12.0, 0.2), index=X.index[::-1])
    return X, y, weight


def params(objective="regression", **updates):
    return dict(
        objective=objective,
        metric="l2",
        n_estimators=25,
        num_threads=1,
        num_leaves=7,
        min_data_in_leaf=5,
        force_col_wise=True,
        seed=91,
        **updates,
    )


def wrapper(p=None, **kwargs):
    return GradientBoosting(
        params=params() if p is None else p,
        show_learning_curve=False,
        verbose_eval=0,
        **kwargs,
    )


def native_fit(X, y, weight, p, *, initial=None, categorical=None, stratified=False):
    split = StratifiedShuffleSplit if stratified else ShuffleSplit
    tr, va = next(split(n_splits=1, test_size=0.2, random_state=42).split(X, y))
    train = lgb.Dataset(
        X.iloc[tr],
        label=y.iloc[tr],
        weight=weight.iloc[tr],
        init_score=None if initial is None else np.asarray(initial)[tr],
        categorical_feature=[] if categorical is None else categorical,
    )
    valid = lgb.Dataset(
        X.iloc[va],
        label=y.iloc[va],
        weight=weight.iloc[va],
        reference=train,
        init_score=None if initial is None else np.asarray(initial)[va],
        categorical_feature=[] if categorical is None else categorical,
    )
    p = p.copy()
    rounds = p.pop("n_estimators")
    p["verbosity"] = -1
    model = lgb.train(
        p,
        train,
        num_boost_round=rounds,
        valid_sets=[train, valid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(10, verbose=False)],
    )
    return model, tr, va


@pytest.mark.parametrize("entry", ["wrapper", "split"])
def test_conflicting_splits_rejected(data, entry):
    X, y, _ = data
    y = (y > 0).astype(int)
    groups = np.arange(len(X)) // 5
    with pytest.raises(ValueError, match="groups.*stratified"):
        if entry == "wrapper":
            wrapper(params("binary"), stratified=True).fit(X, y, groups=groups)
        else:
            _make_split(X, y, None, None, groups, True)


def test_grouped_fit_uses_disjoint_groups(data):
    X, y, weight = data
    groups = pd.Series(np.arange(len(X)) // 5, index=X.index[::-1])
    tr, va = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            X, y, groups
        )
    )
    model = wrapper(return_valid_features=True)
    model.fit(X, y, sample_weight=weight, groups=groups)
    assert_frame_equal(model.valid_features, X.iloc[va])
    assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])


@pytest.mark.parametrize("method", ["predict", "predict_raw"])
@pytest.mark.parametrize(
    "change", ["reorder", "replace", "missing", "extra", "duplicate"]
)
def test_prediction_requires_full_ordered_schema(data, method, change):
    X, y, _ = data
    model = wrapper()
    model.fit(X, y)
    bad = {
        "reorder": X[X.columns[::-1]],
        "replace": X.rename(columns={"signal": "unrelated"}),
        "missing": X.drop(columns="signal"),
        "extra": X.assign(extra=1),
        "duplicate": X.set_axis(["signal", "signal", "noise"], axis=1),
    }[change]
    with pytest.raises(ValueError, match="feature names and order"):
        getattr(model, method)(bad)
    if method == "predict_raw":
        with pytest.raises(ValueError, match="feature names and order"):
            model.predict_raw(bad, validate_features=False)


@pytest.mark.parametrize("names", [["signal", "other", "noise"], [2, 7, 9]])
def test_array_prediction_is_positional_and_keeps_native_options(data, names):
    X, y, _ = data
    X.columns = names
    model = wrapper()
    model.fit(X, y)
    assert_allclose(model.predict(X.to_numpy()), model.predict(X))
    assert_allclose(model.predict(X.iloc[:1]), model.model.predict(X.iloc[:1]))
    for options in [{"raw_score": True}, {"pred_leaf": True}, {"pred_contrib": True}]:
        assert_allclose(
            model.predict_raw(X.to_numpy(), **options),
            model.model.predict(X, **options),
        )
    for invalid in [X.to_numpy()[:, :2], X.to_numpy()[0], np.zeros((1, 3, 1))]:
        with pytest.raises(ValueError, match="two-dimensional"):
            model.predict_raw(invalid)


def test_array_fit_prediction(data):
    X, y, _ = data
    model = wrapper()
    with pytest.warns(UserWarning, match="not a pandas DataFrame"):
        model.fit(X.to_numpy(), y.to_numpy())
    assert_allclose(model.predict(X.to_numpy()), model.model.predict(X.to_numpy()))


@pytest.mark.parametrize("names", [["x", "x", "z"], [1, "1", "z"]])
def test_fit_rejects_ambiguous_names(data, names):
    X, y, _ = data
    X.columns = names
    with pytest.raises(ValueError, match="unique"):
        wrapper().fit(X, y)


@pytest.mark.parametrize(
    "objective",
    ["multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr"],
)
def test_classification_aliases_match_native(data, objective):
    X, y, weight = data
    y = pd.qcut(y, 3, labels=False)
    p = params(objective, num_class=3)
    p["metric"] = "multi_logloss"
    model = wrapper(p, stratified=True)
    model.fit(X, y, sample_weight=weight)
    reference, _, _ = native_fit(X, y, weight, p, stratified=True)
    probability = reference.predict(X)
    assert_allclose(model.predict(X, predict_proba=True), probability)
    assert_array_equal(model.predict(X), probability.argmax(axis=1))
    assert model.predict(X.iloc[:1]).shape == (1,)


@pytest.mark.parametrize("categorical", ["auto", None, [], ["category", "flag"]])
def test_category_selection_matches_native_and_owns_inputs(data, categorical):
    X, y, weight = data
    X["category"] = pd.Categorical(["red", "green"] * 100)
    X["flag"] = np.arange(len(X)) % 2 == 0
    original = X.copy(deep=True)
    original_selection = copy.deepcopy(categorical)
    model = wrapper(cat_feat=categorical)
    model.fit(X, y, sample_weight=weight)
    names = ["category", "flag"] if categorical == "auto" else categorical
    reference, _, _ = native_fit(X, y, weight, params(), categorical=names)
    assert_allclose(model.predict(X), reference.predict(X))
    assert model.cat_feat == (names or None)
    if not names:
        assert model.model.dump_model()["feature_infos"]["category"]["values"] == []
    assert_frame_equal(X, original)
    assert categorical == original_selection
    query = X.iloc[:2].copy()
    query["category"] = pd.Categorical(["new", None])
    assert_allclose(model.predict(query), reference.predict(query))


def test_explicit_integer_codes_and_integer_feature_names(data):
    X, y, weight = data
    X[7] = np.arange(len(X)) % 3
    model = wrapper(cat_feat=[7])
    model.fit(X, y, sample_weight=weight)
    reference, _, _ = native_fit(X, y, weight, params(), categorical=["7"])
    assert_allclose(model.predict(X), reference.predict(X))


@pytest.mark.parametrize(
    "dtype", ["object", "string", "datetime64[ns]", "timedelta64[ns]", "complex128"]
)
def test_unsupported_feature_dtype_has_conversion_error(data, dtype):
    X, y, _ = data
    values = ["a", "b"] * 100 if dtype in ("object", "string") else np.arange(len(X))
    X["unsupported"] = pd.Series(values, index=X.index).astype(dtype)
    with pytest.raises(ValueError, match="Unsupported feature dtypes.*Convert"):
        wrapper().fit(X, y)


@pytest.mark.parametrize("container", ["array", "frame"])
@pytest.mark.parametrize("objective", ["multiclass", "ovr"])
def test_multiclass_offsets_match_native_and_validation_loss(
    data, container, objective
):
    X, y, weight = data
    y = pd.qcut(y, 3, labels=False)
    p = params(objective, num_class=3)
    p["metric"] = "multi_logloss"
    scores = np.column_stack([X.signal / 8, X.other / 5, X.noise / 4])
    initial = (
        pd.DataFrame(scores, index=X.index[::-1]) if container == "frame" else scores
    )
    original = initial.copy()
    model = wrapper(p, stratified=True)
    model.fit(X, y, sample_weight=weight, init_score=initial)
    reference, _, va = native_fit(X, y, weight, p, initial=initial, stratified=True)
    raw = model.predict_raw(X, raw_score=True)
    assert_allclose(raw, reference.predict(X, raw_score=True))
    full = raw + scores
    if objective == "multiclass":
        probability = np.exp(full - full.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
    else:
        probability = 1 / (1 + np.exp(-full))
    loss = -np.log(probability[va, y.iloc[va].to_numpy()])
    assert_allclose(
        np.average(loss, weights=weight.iloc[va]),
        model.model.best_score["valid"]["multi_logloss"],
        rtol=1e-6,
    )
    with pytest.raises(AttributeError, match="init_score"):
        model.predict(X)
    assert_array_equal(initial, original)


@pytest.mark.parametrize("task", ["regression", "binary", "poisson"])
def test_one_dimensional_offsets_match_native(data, task):
    X, y, weight = data
    p = params(task)
    if task == "binary":
        y = (y > 0).astype(int)
        p["metric"] = "binary_logloss"
    elif task == "poisson":
        y = np.exp(y / 4)
        p["metric"] = "poisson"
    initial = pd.Series(np.linspace(-0.3, 0.7, len(X)), index=X.index[::-1])
    model = wrapper(p)
    model.fit(X, y, sample_weight=weight, init_score=initial)
    reference, _, _ = native_fit(X, y, weight, p, initial=initial)
    assert_allclose(
        model.predict_raw(X, raw_score=True), reference.predict(X, raw_score=True)
    )
    with pytest.raises(AttributeError, match="init_score"):
        model.predict(X)


@pytest.mark.parametrize("multiclass", [False, True])
@pytest.mark.parametrize("bad", ["short", "wide", "flat", "nan", "inf"])
def test_invalid_initial_score_shape_or_values_rejected(data, multiclass, bad):
    X, y, _ = data
    p = params("multiclass", num_class=3) if multiclass else params()
    y = pd.qcut(y, 3, labels=False) if multiclass else y
    initial = np.zeros((len(X), 3)) if multiclass else np.zeros(len(X))
    if bad == "short":
        initial = initial[:-1]
    elif bad == "wide":
        initial = np.zeros((len(X), 2))
    elif bad == "flat":
        initial = initial.ravel() if multiclass else initial[:, None]
    else:
        initial[0] = np.nan if bad == "nan" else np.inf
    with pytest.raises(ValueError, match="init_score.*finite.*shape"):
        wrapper(p).fit(X, y, init_score=initial)


def test_successful_refits_reset_contract_and_outputs(data):
    X, y, _ = data
    model = wrapper(return_valid_features=True)
    model.show_learning_curve = True
    model.fit(X, y, init_score=np.zeros(len(X)))
    old_curve = model.learning_curve
    assert old_curve is not None and model.valid_features is not None
    model.show_learning_curve = False
    model.return_valid_features = False
    new = X.rename(columns={"signal": "new"})
    model.fit(new, y)
    assert model.is_init_score is False
    assert model.valid_features is None and model.learning_curve is None
    assert_allclose(model.predict(new), model.model.predict(new))
    with pytest.raises(ValueError, match="feature names and order"):
        model.predict(X)
    # Clearing the wrapper reference does not close a figure owned by its caller.
    assert plt.fignum_exists(old_curve.number)
    plt.close(old_curve)
    model.fit(new, y, init_score=np.zeros(len(X)))
    assert model.is_init_score is True
    with pytest.raises(AttributeError, match="init_score"):
        model.predict(new)


def test_failed_refit_preserves_previous_fitted_contract(data):
    X, y, _ = data
    X["category"] = pd.Categorical(["red", "green"] * 100)
    model = wrapper()
    model.fit(X, y)
    expected = model.predict(X)
    old_model = model.model
    model.cat_feat_input = None
    with pytest.raises(ValueError, match="init_score"):
        model.fit(X, y, init_score=np.zeros(len(X) - 1))
    assert model.model is old_model and model.cat_feat == ["category"]
    assert model.is_init_score is False
    assert_allclose(model.predict(X), expected)


def test_native_feature_name_normalization_keeps_fitted_schema(data):
    X, y, _ = data
    X.columns = ["a b", 7, "noise"]
    model = wrapper()
    model.fit(X, y)
    assert model.model.feature_name() == ["a_b", "7", "noise"]
    assert_allclose(model.predict(X), model.model.predict(X))
    assert_allclose(
        model.predict_raw(X, validate_features=True), model.model.predict(X)
    )
    assert_allclose(model.predict(X.to_numpy()), model.predict(X))
    with pytest.raises(ValueError, match="feature names and order"):
        model.predict(X.rename(columns={"a b": "a_b"}))


def test_refit_refreshes_category_names_and_native_failure_keeps_state(data):
    X, y, _ = data
    X["category"] = pd.Categorical(["red", "green"] * 100)
    model = wrapper()
    model.fit(X, y)
    new = X.drop(columns="category")
    model.fit(new, y)
    assert model.cat_feat is None
    expected = model.predict(new)
    old_model = model.model
    model.params = params("not_an_objective")
    with pytest.raises(lgb.basic.LightGBMError):
        model.fit(X, y, init_score=np.zeros(len(X)))
    assert model.model is old_model and model.cat_feat is None
    assert model.is_init_score is False
    assert_allclose(model.predict(new), expected)


def test_load_replaces_new_schema_state(data, tmp_path):
    X, y, _ = data
    saved = wrapper()
    saved.fit(X, y)
    path = saved.save(str(tmp_path), "ordinary")
    loaded = wrapper()
    loaded.fit(X.rename(columns={"signal": "old"}), y)
    loaded.load(path)
    assert_allclose(loaded.predict(X), saved.predict(X))
