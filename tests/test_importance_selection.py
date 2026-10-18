"""R17 cumulative selection and explanation population references."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from arfs import feature_selection as fs
from arfs.feature_selection import variable_importance as vi
from arfs.preprocessing import OrdinalEncoderPandas

PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 6,
    "num_leaves": 4,
    "min_data_in_leaf": 5,
    "num_threads": 1,
    "verbosity": -1,
}


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        rng.normal(size=(120, 3)), columns=["a", "b", "c"], index=np.arange(120) + 100
    )
    return X, pd.Series(4 * X.a + 0.3 * X.b, index=X.index)


def controlled_backend(monkeypatch, values, fastshap=False):
    """Replace training/output only, leaving aggregation and selection real."""
    calls = []
    n_rows = len(values[0]) if isinstance(values, list) else len(values)

    class TrainingDouble:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, sample_weight=None):
            calls.append((X.copy(), y, sample_weight, self.kwargs))
            self.valid_features = X.iloc[:n_rows]
            self.model = SimpleNamespace(predict=lambda data, pred_contrib: values)

    monkeypatch.setattr(vi, "GradientBoosting", TrainingDouble)
    if fastshap:

        class Explainer:
            def __init__(self, *args, **kwargs):
                pass

            def shap_values(self, X):
                return values

        monkeypatch.setitem(
            sys.modules, "fasttreeshap", SimpleNamespace(TreeExplainer=Explainer)
        )
    return calls


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (0, []),
        (0.5, ["a"]),
        (0.6, ["a"]),
        (0.7, ["a", "b"]),
        (0.9, ["a", "b"]),
        (1, ["a", "b", "c"]),
    ],
)
@pytest.mark.parametrize("scores", [[0.6, 0.3, 0.1], [6, 3, 1]])
def test_smallest_positive_prefix_includes_crossing_feature(
    monkeypatch, data, threshold, expected, scores
):
    X, y = data
    controlled_backend(monkeypatch, np.array([[*scores, 1000.0]]))
    selector = fs.VariableImportance(encode=False, threshold=threshold).fit(X, y)
    assert_array_equal(selector.selected_features_, expected)
    assert_frame_equal(selector.transform(X), X.loc[:, expected])
    assert_array_equal(selector.get_feature_names_out(), expected)


@pytest.mark.parametrize("threshold", [0.5, 0.99, 1])
def test_zero_importance_excluded_at_every_positive_cutoff(
    monkeypatch, data, threshold
):
    X, y = data
    controlled_backend(monkeypatch, np.array([[0, -1, 0, 9000.0]]))
    selector = fs.VariableImportance(encode=False, threshold=threshold).fit(X, y)
    assert_array_equal(selector.selected_features_, ["b"])
    assert_allclose(
        selector.feature_importances_summary_.normalized_importance, [1, 0, 0]
    )


def test_zero_total_summary_is_finite_and_selects_nothing(monkeypatch, data):
    X, y = data
    controlled_backend(monkeypatch, np.array([[0, 0, 0, 99.0]]))
    selector = fs.VariableImportance(encode=False).fit(X, y)
    assert not selector.get_support().any()
    summary = selector.feature_importances_summary_
    assert_allclose(
        summary[["importance", "normalized_importance", "cumulative_importance"]], 0
    )
    assert_frame_equal(selector.transform(X), X.iloc[:, :0])


def test_cutoff_one_keeps_tiny_positive_importance(monkeypatch, data):
    X, y = data
    controlled_backend(monkeypatch, np.array([[1, 1e-20, 0, 0]]))
    selector = fs.VariableImportance(encode=False, threshold=1).fit(X, y)
    assert_array_equal(selector.selected_features_, ["a", "b"])


def test_tied_importance_uses_input_order_and_valid_refit(monkeypatch, data):
    X, y = data
    controlled_backend(monkeypatch, np.array([[1, 1, 1, 0]]))
    selector = fs.VariableImportance(encode=False, threshold=0.3).fit(X, y)
    assert_array_equal(selector.selected_features_, ["a"])
    selector.fit(X.loc[:, ["c", "b", "a"]], y)
    assert_array_equal(selector.selected_features_, ["c"])
    assert_array_equal(selector.feature_importances_summary_.feature, ["c", "b", "a"])
    assert_array_equal(
        selector.set_params(threshold=1).fit(X, y).selected_features_, X.columns
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("fastshap", [False, True])
def test_nonfinite_contributions_fail_clearly(monkeypatch, data, value, fastshap):
    X, y = data
    values = np.array([[value, 0, 0] + ([] if fastshap else [0])])
    controlled_backend(monkeypatch, values, fastshap)
    with pytest.raises(ValueError, match="finite"):
        fs.VariableImportance(encode=False, fastshap=fastshap).fit(X, y)


def test_overflowed_importance_total_is_rejected(monkeypatch, data):
    X, y = data
    controlled_backend(monkeypatch, np.array([[1e308, 1e308, 0, 0]]))
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        fs.VariableImportance(encode=False).fit(X, y)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, -np.inf, "bad", True])
def test_current_threshold_validated_before_training(monkeypatch, data, threshold):
    X, y = data
    calls = controlled_backend(monkeypatch, np.ones((2, 4)))
    with pytest.raises(ValueError, match="threshold"):
        fs.VariableImportance().set_params(threshold=threshold).fit(X, y)
    assert calls == []


@pytest.mark.parametrize("layout", ["native", "array", "list"])
def test_multiclass_layouts_sum_output_importances(monkeypatch, data, layout):
    X, _ = data
    blocks = np.array(
        [
            [[1, -2, 3, 100], [2, 3, -4, 200], [-3, 4, 5, 300]],
            [[-1, 2, -3, 400], [-2, -3, 4, 500], [3, -4, -5, 600]],
        ],
        dtype=float,
    )
    fastshap = layout != "native"
    values = (
        blocks.reshape(2, -1) if not fastshap else blocks[:, :, :-1].transpose(0, 2, 1)
    )
    if layout == "list":
        values = [values[:, :, k] for k in range(3)]
    controlled_backend(monkeypatch, values, fastshap)
    selector = fs.VariableImportance(
        task="multiclass", encode=False, fastshap=fastshap
    ).fit(X, np.tile([0, 1, 2], 40))
    summary = selector.feature_importances_summary_.set_index("feature").loc[X.columns]
    assert_allclose(summary.importance, [6, 9, 12])
    assert_allclose(summary.normalized_importance, [2 / 9, 3 / 9, 4 / 9])


@pytest.mark.parametrize("shape", [(2,), (3, 3), (2, 5), (2, 0), (2, 4, 3)])
def test_native_contribution_shape_validated(monkeypatch, data, shape):
    X, y = data
    controlled_backend(monkeypatch, np.ones(shape))
    with pytest.raises(ValueError, match="shape"):
        fs.VariableImportance(encode=False).fit(X, y)


@pytest.mark.parametrize("fastshap", [False, True])
def test_single_output_uses_absolute_mean_not_signed_mean(monkeypatch, data, fastshap):
    X, y = data
    values = np.array([[1, -2, 0], [-3, 4, 0]], dtype=float)
    if not fastshap:
        values = np.column_stack([values, [900, -900]])
    controlled_backend(monkeypatch, values, fastshap)
    summary = (
        vi._compute_varimp_lgb(X, y, fastshap=fastshap)
        .set_index("feature")
        .loc[X.columns]
    )
    assert_allclose(summary.importance, [2, 3, 0])


def test_exactly_one_fit_and_removed_iteration_parameter(monkeypatch, data):
    X, y = data
    calls = controlled_backend(monkeypatch, np.ones((2, 4)))
    selector = clone(fs.VariableImportance(encode=False)).fit(X, y)
    assert len(calls) == 1
    assert "n_iterations" not in selector.get_params()
    with pytest.raises(TypeError):
        fs.VariableImportance(n_iterations=1)
    with pytest.raises(ValueError):
        selector.set_params(n_iterations=1)
    with pytest.raises(TypeError):
        vi._compute_varimp_lgb(X, y, n_iterations=1)


def test_missing_optional_backend_fails_before_encoding_or_training(monkeypatch, data):
    X, y = data
    monkeypatch.setitem(sys.modules, "fasttreeshap", None)

    def forbidden(*args, **kwargs):
        pytest.fail("missing backend must fail before encoding/training")

    monkeypatch.setattr(vi, "GradientBoosting", forbidden)
    monkeypatch.setattr(vi, "OrdinalEncoderPandas", forbidden)
    with pytest.raises(ImportError, match="fasttreeshap"):
        fs.VariableImportance(fastshap=True).fit(X, y)


@pytest.mark.parametrize("constant", [False, True])
def test_real_single_signal_and_constant_response(data, constant):
    X, y = data
    X = X.assign(b=0.0, c=0.0)
    selector = fs.VariableImportance(encode=False, lgb_kwargs=PARAMS.copy()).fit(
        X, y * 0 if constant else 4 * X.a
    )
    assert_array_equal(selector.selected_features_, [] if constant else ["a"])
    assert (
        np.isfinite(selector.feature_importances_summary_.select_dtypes("number"))
        .all()
        .all()
    )


@pytest.mark.parametrize(
    "task,objective", [("classification", "binary"), ("multiclass", "multiclass")]
)
def test_real_classification_array_series_parity(data, task, objective):
    X, _ = data
    y = (
        (X.a > 0).astype(int).to_numpy()
        if task == "classification"
        else np.tile([0, 1, 2], 40)
    )
    params = {
        **PARAMS,
        "objective": objective,
        "metric": "binary_logloss" if task == "classification" else "multi_logloss",
    }
    if task == "multiclass":
        params["num_class"] = 3
    options = {"task": task, "encode": False, "lgb_kwargs": params}
    array_fit = fs.VariableImportance(**options).fit(X, y)
    series_fit = fs.VariableImportance(**options).fit(X, pd.Series(y, index=X.index))
    assert_frame_equal(
        array_fit.feature_importances_summary_, series_fit.feature_importances_summary_
    )
    assert_array_equal(array_fit.support_, series_fit.support_)


def test_weighted_training_equal_validation_row_population(monkeypatch, data):
    X, y = data
    X["cat"] = pd.Categorical(np.tile(["a", "b", "c"], 40))
    weights = pd.Series(np.where(X.a > 0, 20.0, 1.0), index=X.index)
    before = X.copy(deep=True)
    observed = []
    original = vi.GradientBoosting

    class ObservedTraining(original):
        def fit(self, X, y, sample_weight=None):
            super().fit(X, y, sample_weight=sample_weight)
            observed.append((self, X.copy(), sample_weight.copy()))

    monkeypatch.setattr(vi, "GradientBoosting", ObservedTraining)
    pipe = Pipeline(
        [
            ("missing", fs.MissingValueThreshold()),
            ("encoder", OrdinalEncoderPandas()),
            (
                "importance",
                fs.VariableImportance(encode=False, lgb_kwargs=PARAMS.copy()),
            ),
        ]
    )
    out = pipe.fit_transform(X, y, importance__sample_weight=weights)
    assert len(observed) == 1
    model, encoded, forwarded = observed[0]
    assert_array_equal(forwarded, weights)
    train, valid = next(
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(encoded, y)
    )
    reference = lgb.train(
        {k: v for k, v in PARAMS.items() if k != "n_estimators"},
        lgb.Dataset(
            encoded.iloc[train], label=y.iloc[train], weight=weights.iloc[train]
        ),
        num_boost_round=PARAMS["n_estimators"],
        valid_sets=[
            lgb.Dataset(
                encoded.iloc[valid], label=y.iloc[valid], weight=weights.iloc[valid]
            )
        ],
        callbacks=[lgb.early_stopping(10, verbose=False)],
    )
    contributions = np.abs(
        reference.predict(encoded.iloc[valid], pred_contrib=True)[:, :-1]
    )
    expected = contributions.mean(axis=0)
    summary = (
        pipe[-1].feature_importances_summary_.set_index("feature").loc[encoded.columns]
    )
    assert_allclose(summary.importance, expected, rtol=1e-6, atol=1e-8)
    weighted_mean = np.average(contributions, axis=0, weights=weights.iloc[valid])
    assert not np.allclose(expected, weighted_mean), (
        "fixture must distinguish explanation populations"
    )
    assert_array_equal(model.valid_features.index, encoded.iloc[valid].index)
    assert_frame_equal(X, before)
    assert_frame_equal(out, encoded[out.columns])


@pytest.mark.parametrize(
    "task,expected_objective,expected_stratified,expected_num_class",
    [
        ("regression", "rmse", False, None),
        ("classification", "binary", True, None),
        ("multiclass", "multiclass", True, 3),
    ],
)
@pytest.mark.parametrize(
    "supplied",
    [None, {"min_data_in_leaf": 5, "zero_as_missing": True}],
)
def test_task_defaults_and_omitted_objective_preserve_caller_dict(
    monkeypatch,
    data,
    task,
    expected_objective,
    expected_stratified,
    expected_num_class,
    supplied,
):
    X, y = data
    if task == "classification":
        y = (X.a > 0).astype(int)
    elif task == "multiclass":
        y = np.tile([0, 1, 2], 40)
    values = (
        np.ones((2, 12))
        if expected_num_class is not None
        else np.array([[3.0, 1.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0]])
    )
    calls = controlled_backend(monkeypatch, values)
    original = None if supplied is None else supplied.copy()
    fs.VariableImportance(task=task, encode=False, lgb_kwargs=supplied).fit(X, y)
    assert supplied == original
    assert len(calls) == 1
    kwargs = calls[0][3]
    assert kwargs["stratified"] is expected_stratified
    assert kwargs["params"]["objective"] == expected_objective
    assert kwargs["params"]["zero_as_missing"] is (
        False if supplied is None else supplied["zero_as_missing"]
    )
    if supplied is not None:
        assert kwargs["params"]["min_data_in_leaf"] == 5
    if expected_num_class is None:
        assert "num_class" not in kwargs["params"]
    else:
        assert kwargs["params"]["num_class"] == expected_num_class


@pytest.mark.parametrize(
    "task,override,expected_stratified,expected_num_class",
    [
        ("classification", "rmse", False, None),
        ("classification", "poisson", False, None),
        ("multiclass", "rmse", False, None),
        ("multiclass", "binary", True, None),
        ("regression", "binary", True, None),
        ("regression", "softmax", True, 3),
        ("classification", "multiclassova", True, 3),
        ("regression", "ovr", True, 3),
    ],
)
def test_explicit_objective_overrides_task_for_training_and_stratification(
    monkeypatch, data, task, override, expected_stratified, expected_num_class
):
    X, y = data
    if override == "binary":
        y = (X.a > 0).astype(int)
    elif expected_num_class is not None or task == "multiclass":
        y = np.tile([0, 1, 2], 40)
    values = (
        np.ones((2, 12))
        if expected_num_class is not None
        else np.array([[2.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]])
    )
    calls = controlled_backend(monkeypatch, values)
    lgb_kwargs = {"objective": override, "min_data_in_leaf": 7}
    original = lgb_kwargs.copy()
    fs.VariableImportance(task=task, encode=False, lgb_kwargs=lgb_kwargs).fit(X, y)
    assert lgb_kwargs == original
    kwargs = calls[0][3]
    assert kwargs["stratified"] is expected_stratified
    assert kwargs["params"]["objective"] == override
    assert kwargs["params"]["min_data_in_leaf"] == 7
    if expected_num_class is None:
        assert "num_class" not in kwargs["params"]
    else:
        assert kwargs["params"]["num_class"] == expected_num_class


@pytest.mark.parametrize("task", ["regression", "classification", "multiclass"])
@pytest.mark.parametrize("explicit_binary", [False, True])
@pytest.mark.parametrize("matching_key", [None, "num_class", "num_classes"])
def test_binary_promoted_to_multiclass_when_more_than_two_classes(
    monkeypatch, data, task, explicit_binary, matching_key
):
    if not explicit_binary and task != "classification":
        return
    X, _ = data
    y = np.tile([0, 1, 2, 3], 30)
    calls = controlled_backend(monkeypatch, np.ones((2, 16)))
    supplied = {}
    if explicit_binary:
        supplied["objective"] = "binary"
    if matching_key is not None:
        supplied[matching_key] = 4
    lgb_kwargs = supplied if supplied else None
    original = None if lgb_kwargs is None else lgb_kwargs.copy()
    with pytest.warns(UserWarning, match="multiclass"):
        fs.VariableImportance(task=task, encode=False, lgb_kwargs=lgb_kwargs).fit(X, y)
    assert lgb_kwargs == original
    params = calls[0][3]["params"]
    assert calls[0][3]["stratified"] is True
    assert params["objective"] == "multiclass"
    if matching_key == "num_classes":
        assert params["num_classes"] == 4
        assert "num_class" not in params
    else:
        assert params["num_class"] == 4


def test_real_binary_promotion_trains_multiclass_model(data):
    X, _ = data
    y = pd.Series(np.tile([0, 1, 2], 40), index=X.index)
    X = X.assign(a=y.to_numpy(dtype=float) * 3 + X.a * 0.1)
    lgb_kwargs = {
        "objective": "binary",
        "n_estimators": 6,
        "num_leaves": 4,
        "min_data_in_leaf": 5,
        "num_threads": 1,
    }
    original = lgb_kwargs.copy()
    with pytest.warns(UserWarning, match="multiclass"):
        selector = fs.VariableImportance(
            task="regression", encode=False, lgb_kwargs=lgb_kwargs
        ).fit(X, y)
    assert lgb_kwargs == original
    assert "a" in selector.selected_features_


@pytest.mark.parametrize(
    "key,metric,expected",
    [
        ("metric", "binary_logloss", {"multi_logloss"}),
        ("metric", "binary", {"multi_logloss"}),
        ("metric", "binary_error", {"multi_error"}),
        ("metric", "binary_logloss,binary_error", {"multi_logloss", "multi_error"}),
        ("metric", ["binary_logloss", "multi_error"], {"multi_logloss", "multi_error"}),
        ("metric", ["binary_logloss,binary_error"], {"multi_logloss", "multi_error"}),
        (
            "metric",
            ("multi_error", "binary_logloss,softmax"),
            {"multi_error", "multi_logloss"},
        ),
        ("metric", ("binary_error", "multi_logloss"), {"multi_error", "multi_logloss"}),
        ("metric", {"binary_error"}, {"multi_error"}),
        ("metrics", "binary_error", {"multi_error"}),
        (
            "metric_types",
            ["binary_logloss", "binary_error"],
            {"multi_logloss", "multi_error"},
        ),
        ("metric", "multi_logloss", {"multi_logloss"}),
        ("metric", "auc_mu", {"auc_mu"}),
        (
            "metric",
            "multiclass,softmax,multiclassova,multiclass_ova,ova,ovr",
            {"multi_logloss"},
        ),
        ("metric", "multi_logloss,None,na,null,custom", {"multi_logloss"}),
        ("metric", "", {"multi_logloss"}),
        ("metric", None, {"multi_logloss"}),
        ("metric", lambda preds, dataset: ("custom_error", 0.0, False), {"custom_error"}),
    ],
)
def test_real_binary_promotion_translates_metrics_without_mutating_params(
    monkeypatch, data, key, metric, expected
):
    X, _ = data
    y = pd.Series(np.tile([0, 1, 2], 40), index=X.index)
    X = X.assign(a=y.to_numpy(dtype=float) * 3 + X.a * 0.1)
    observed = []
    original_gbm = vi.GradientBoosting

    class ObservedGBM(original_gbm):
        def fit(self, X, y, sample_weight=None):
            super().fit(X, y, sample_weight=sample_weight)
            observed.append(self)

    monkeypatch.setattr(vi, "GradientBoosting", ObservedGBM)
    params = {**PARAMS, "objective": "binary"}
    params.pop("metric")
    if key == "metric":
        # LightGBM's primary name takes precedence over conflicting aliases.
        params.update(metrics="auc", metric_types="average_precision")
    params[key] = metric
    original = deepcopy(params)
    with pytest.warns(UserWarning, match="multiclass"):
        selector = fs.VariableImportance(
            task="classification", encode=False, lgb_kwargs=params
        ).fit(X, y)
    assert params == original
    assert set(observed[0].model.best_score["valid"]) == expected
    assert "a" in selector.selected_features_


@pytest.mark.parametrize(
    "key,metric",
    [
        ("metric", "auc"),
        ("metrics", "average_precision"),
        ("metric_types", ["binary_logloss", "auc"]),
        ("metric", "cross_entropy"),
        ("metrics", "xentropy"),
        ("metric_types", "cross_entropy_lambda"),
        ("metric", "xentlambda"),
        ("metrics", "kullback_leibler"),
        ("metric_types", "kldiv"),
        ("metric", "multi_logloss,rmse"),
        ("metrics", ["multi_error", "ndcg"]),
        ("metric", ["multi_logloss,cross_entropy"]),
        ("metric_types", ["multi_error,auc"]),
    ],
)
def test_promotion_rejects_metrics_without_equivalent_before_encoding(
    monkeypatch, data, key, metric
):
    X, _ = data
    y = np.tile([0, 1, 2], 40)

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible metrics must fail before encoding or training")

    monkeypatch.setattr(vi, "OrdinalEncoderPandas", forbidden)
    monkeypatch.setattr(vi, "GradientBoosting", forbidden)
    params = {"objective": "binary", key: metric}
    original = deepcopy(params)
    with pytest.raises(ValueError, match="multiclass.*metric"):
        fs.VariableImportance(lgb_kwargs=params).fit(X, y)
    assert params == original


@pytest.mark.parametrize("task", ["regression", "classification", "multiclass"])
@pytest.mark.parametrize("key", ["num_class", "num_classes"])
@pytest.mark.parametrize("bad_count", [2, 3, 5])
def test_promotion_rejects_mismatched_num_class_without_overwriting(
    monkeypatch, data, task, key, bad_count
):
    X, _ = data
    y = np.tile([0, 1, 2, 3], 30)
    calls = controlled_backend(monkeypatch, np.ones((2, 16)))
    lgb_kwargs = {"objective": "binary", key: bad_count}
    original = lgb_kwargs.copy()
    with pytest.raises(ValueError, match="num_class"):
        fs.VariableImportance(task=task, encode=False, lgb_kwargs=lgb_kwargs).fit(X, y)
    assert lgb_kwargs == original
    assert calls == []


@pytest.mark.parametrize("key", ["num_class", "num_classes"])
def test_multiclass_num_class_preserved_and_missing_classes_accepted(
    monkeypatch, data, key
):
    X, _ = data
    # Labels {0, 2} are a subset of {0, 1, 2, 3} with missing classes 1 and 3.
    y = np.tile([0, 2], 60)
    calls = controlled_backend(monkeypatch, np.ones((2, 16)))
    explicit = {key: 4}
    original = explicit.copy()
    fs.VariableImportance(task="multiclass", encode=False, lgb_kwargs=explicit).fit(
        X, y
    )
    assert explicit == original
    assert calls[0][3]["params"][key] == 4
    if key == "num_classes":
        assert "num_class" not in calls[0][3]["params"]

    with pytest.raises(ValueError, match="num_class"):
        fs.VariableImportance(
            task="multiclass",
            encode=False,
            lgb_kwargs={"num_class": 3, "num_classes": 2},
        ).fit(X, np.tile([0, 1, 2], 40))


@pytest.mark.parametrize("key", ["num_class", "num_classes"])
def test_real_multiclass_explicit_num_class_accepts_missing_classes(data, key):
    X, _ = data
    y = pd.Series(np.tile([0, 2], 60), index=X.index)
    X = X.assign(a=y.to_numpy(dtype=float) * 3 + X.a * 0.1)
    lgb_kwargs = {
        "n_estimators": 6,
        "num_leaves": 4,
        "min_data_in_leaf": 5,
        "num_threads": 1,
        key: 4,
    }
    original = lgb_kwargs.copy()
    selector = fs.VariableImportance(
        task="multiclass", encode=False, lgb_kwargs=lgb_kwargs
    ).fit(X, y)
    assert lgb_kwargs == original
    assert "a" in selector.selected_features_


@pytest.mark.parametrize("key", ["num_class", "num_classes"])
def test_real_multiclass_conflicting_class_count_is_not_silently_overwritten(
    monkeypatch, data, key
):
    X, _ = data
    y = np.tile([0, 1, 2], 40)
    bad = {
        "n_estimators": 4,
        "num_leaves": 4,
        "min_data_in_leaf": 5,
        "num_threads": 1,
        key: 2,
    }
    original = bad.copy()
    calls = controlled_backend(monkeypatch, np.ones((2, 8)))
    with pytest.raises(ValueError, match="subset"):
        fs.VariableImportance(task="multiclass", encode=False, lgb_kwargs=bad).fit(
            X, y
        )
    assert bad == original
    assert calls == []


@pytest.mark.parametrize(
    "task,raw_labels,expected_unique",
    [
        ("classification", np.tile([False, True], 60), [0, 1]),
        ("classification", pd.Series(np.tile([False, True], 60), dtype="boolean"), [0, 1]),
        ("classification", np.tile([0.0, 1.0], 60), [0, 1]),
        ("classification", pd.Series(np.tile([0.0, 1.0], 60), dtype="Float64"), [0, 1]),
        ("multiclass", np.tile([0.0, 1.0, 2.0], 40), [0, 1, 2]),
        ("multiclass", pd.Series(np.tile([0, 1, 2], 40), dtype="Int64"), [0, 1, 2]),
    ],
)
def test_classification_label_conversion_to_int(
    monkeypatch, data, task, raw_labels, expected_unique
):
    X, _ = data
    n_outputs = 1 if task == "classification" else 3
    calls = controlled_backend(monkeypatch, np.ones((2, 4 * n_outputs)))
    fs.VariableImportance(task=task, encode=False).fit(X, raw_labels)
    passed_y = calls[0][1]
    assert pd.api.types.is_integer_dtype(np.asarray(passed_y).dtype)
    assert_array_equal(np.unique(passed_y), expected_unique)


@pytest.mark.parametrize("task", ["classification", "multiclass"])
@pytest.mark.parametrize(
    "bad_labels",
    [
        np.tile([0.0, 0.5], 60),
        np.tile([0.0, np.nan], 60),
        np.tile([0.0, np.inf], 60),
        np.tile([0.0, -np.inf], 60),
        np.tile(["0", "1"], 60),
        pd.Series(np.tile(["0", "1"], 60), dtype="string"),
        np.tile(["neg", "pos"], 60),
        pd.Categorical(np.tile([0, 1], 60)),
        pd.Categorical(np.tile(["a", "b"], 60)),
        np.tile([0j, 1j], 60),
    ],
)
def test_classification_rejects_non_integral_nonfinite_and_non_numeric_labels(
    monkeypatch, data, task, bad_labels
):
    X, _ = data
    calls = controlled_backend(monkeypatch, np.ones((2, 4)))
    with pytest.raises(ValueError):
        fs.VariableImportance(task=task, encode=False).fit(X, bad_labels)
    assert calls == []


@pytest.mark.parametrize(
    "task,lgb_kwargs,bad_labels",
    [
        # Binary labels not a subset of {0, 1}
        ("classification", None, np.tile([-1, 1], 60)),
        ("classification", None, np.tile([1, 2], 60)),
        ("classification", None, np.tile([0, 2], 60)),
        ("regression", {"objective": "binary"}, np.tile([1, 2], 60)),
        # Multiclass inferred num_class: labels must equal {0..K-1}
        ("multiclass", None, np.tile([1, 2, 3], 40)),
        ("multiclass", None, np.tile([0, 1, 3], 40)),
        ("multiclass", None, np.tile([-1, 0, 1], 40)),
        ("classification", None, np.tile([0, 1, 3], 40)),  # promoted binary
        ("regression", {"objective": "softmax"}, np.tile([0, 2], 60)),
        # Multiclass explicit num_class=n: labels must be a subset of {0..n-1}
        ("multiclass", {"num_class": 3}, np.tile([0, 3], 60)),
        ("multiclass", {"num_classes": 3}, np.tile([-1, 1], 60)),
        ("multiclass", {"num_class": 1}, np.tile([0, 1], 60)),
        ("multiclass", {"num_class": True}, np.tile([0, 1], 60)),
        # Classification families need at least 2 distinct classes
        ("classification", None, np.zeros(120, dtype=int)),
        ("classification", None, np.ones(120, dtype=int)),
        ("classification", None, np.ones(120, dtype=bool)),
        ("multiclass", None, np.zeros(120, dtype=int)),
        ("multiclass", {"num_class": 3}, np.zeros(120, dtype=int)),
        ("regression", {"objective": "binary"}, np.zeros(120, dtype=int)),
        ("regression", {"objective": "ovr"}, np.zeros(120, dtype=int)),
    ],
)
def test_classification_label_domain_and_class_count_rules(
    monkeypatch, data, task, lgb_kwargs, bad_labels
):
    X, _ = data
    calls = controlled_backend(monkeypatch, np.ones((2, 12)))
    with pytest.raises(ValueError):
        fs.VariableImportance(task=task, encode=False, lgb_kwargs=lgb_kwargs).fit(
            X, bad_labels
        )
    assert calls == []


@pytest.mark.parametrize(
    "task,lgb_kwargs,bad_labels",
    [
        ("classification", None, np.r_[0, np.ones(119, dtype=int)]),
        ("multiclass", None, np.r_[0, np.tile([1, 2], 59), 1]),
        ("classification", None, np.tile(["0", "1"], 60)),
        ("multiclass", {"num_class": 2}, np.tile([0, 1, 2], 40)),
    ],
)
def test_precheck_fails_before_encoding_or_training(
    monkeypatch, data, task, lgb_kwargs, bad_labels
):
    X, _ = data
    X = X.assign(cat=np.tile(["x", "y"], 60))

    def forbidden(*args, **kwargs):
        pytest.fail("pre-check must fail before encoding or training")

    monkeypatch.setattr(vi, "OrdinalEncoderPandas", forbidden)
    monkeypatch.setattr(vi, "GradientBoosting", forbidden)
    with pytest.raises(ValueError):
        fs.VariableImportance(task=task, encode=True, lgb_kwargs=lgb_kwargs).fit(
            X, bad_labels
        )


@pytest.mark.parametrize(
    "task,override,expected_objective,expected_stratified",
    [
        ("regression", None, "rmse", False),
        ("classification", None, "binary", True),
        ("multiclass", None, "multiclass", True),
        ("classification", "rmse", "rmse", False),
        ("regression", "binary", "binary", True),
        ("regression", "softmax", "softmax", True),
    ],
)
def test_real_lightgbm_task_defaults_and_stratification(
    monkeypatch, data, task, override, expected_objective, expected_stratified
):
    X, y_reg = data
    if expected_objective == "rmse":
        y = y_reg
    elif expected_objective == "binary":
        # Choose a binary target where ShuffleSplit and StratifiedShuffleSplit differ.
        y = pd.Series(np.r_[np.ones(15, dtype=int), np.zeros(105, dtype=int)], index=X.index)
        X = X.assign(a=y.to_numpy(dtype=float) * 4 + X.a * 0.1)
    else:
        y = pd.Series(
            np.r_[np.zeros(10, dtype=int), np.ones(20, dtype=int), np.full(90, 2, dtype=int)],
            index=X.index,
        )
        X = X.assign(a=y.to_numpy(dtype=float) * 3 + X.a * 0.1)

    unstrat_val = X.iloc[
        next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y))[1]
    ].index
    if expected_stratified:
        strat_val = X.iloc[
            next(
                StratifiedShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=42
                ).split(X, y)
            )[1]
        ].index
        assert not strat_val.equals(unstrat_val)

    observed = []
    original_gbm = vi.GradientBoosting

    class ObservedGBM(original_gbm):
        def fit(self, X, y, sample_weight=None):
            super().fit(X, y, sample_weight=sample_weight)
            observed.append(self)

    monkeypatch.setattr(vi, "GradientBoosting", ObservedGBM)
    lgb_kwargs = {
        "n_estimators": 6,
        "num_leaves": 4,
        "min_data_in_leaf": 5,
        "num_threads": 1,
    }
    if override is not None:
        lgb_kwargs["objective"] = override
    original_kwargs = lgb_kwargs.copy()

    selector = fs.VariableImportance(
        task=task, encode=False, lgb_kwargs=lgb_kwargs
    ).fit(X, y)
    assert lgb_kwargs == original_kwargs
    assert len(observed) == 1
    gbm = observed[0]
    assert gbm.stratified is expected_stratified
    assert gbm.model_params["objective"] == expected_objective
    if expected_objective in {"multiclass", "softmax"}:
        assert gbm.model_params["num_class"] == 3
    expected_val_index = strat_val if expected_stratified else unstrat_val
    assert_array_equal(gbm.valid_features.index, expected_val_index)
    assert np.isfinite(
        selector.feature_importances_summary_.select_dtypes("number").to_numpy()
    ).all()
    assert "a" in selector.selected_features_
