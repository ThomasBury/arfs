"""R17 cumulative selection and explanation population references."""

import sys
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
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
