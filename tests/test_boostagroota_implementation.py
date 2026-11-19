"""BoostAGroota preserves reduction decisions and validates completed fits."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from arfs.feature_selection import allrelevant as a


def data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    return X, 3 * X.a - X.b


@pytest.mark.parametrize(
    "parameter,value",
    [
        (name, value)
        for name in ("iters", "max_rounds")
        for value in (0, -1, True, 1.5, np.nan, None)
    ]
    + [
        (name, value)
        for name in ("cutoff", "delta")
        for value in (0, -1, True, np.nan, np.inf, "1", None)
    ]
    + [
        ("delta", 1.1),
        ("importance", "fastshap"),
        ("importance", "typo"),
        ("estimator", None),
        ("estimator", Ridge()),
    ],
)
@pytest.mark.parametrize("set_params", [False, True])
def test_parameters_are_checked_at_fit(parameter, value, set_params):
    X, y = data()
    selector = a.BoostAGroota(lgb.LGBMRegressor(), iters=1, max_rounds=1)
    if set_params:
        selector.set_params(**{parameter: value})
    else:
        selector = a.BoostAGroota(
            **(selector.get_params(deep=False) | {parameter: value})
        )
    clone(selector)
    with pytest.raises(ValueError, match=parameter):
        selector.fit(X, y)
    with pytest.raises(NotFittedError):
        selector.get_support()


@pytest.mark.parametrize(
    "operation",
    ["get_support", "transform", "plot_importance", "get_feature_names_out"],
)
@pytest.mark.parametrize("failed_refit", [False, True])
def test_selection_requires_successful_fit(monkeypatch, operation, failed_refit):
    X, y = data()
    selector = a.BoostAGroota(lgb.LGBMRegressor(), iters=1, max_rounds=1)
    if failed_refit:
        monkeypatch.setattr(
            a,
            "_get_shap_imp",
            lambda model, frame, *args, **kwargs: np.ones(frame.shape[1]),
        )
        selector.fit(X, y)
        selector.set_params(iters=0)
        with pytest.raises(ValueError, match="iters"):
            selector.fit(X, y)
    args = (X,) if operation == "transform" else ()
    with pytest.raises(NotFittedError):
        getattr(selector, operation)(*args)


@pytest.mark.parametrize(
    "bad", [np.ones((8, 1)), np.ones((1, 8)), np.ones(7), np.array(1.0)]
)
def test_importance_shape_failure_invalidates_previous_fit(monkeypatch, bad):
    X, y = data()
    monkeypatch.setattr(a, "_get_shap_imp", lambda *args, **kwargs: np.ones(8))
    selector = a.BoostAGroota(lgb.LGBMRegressor(), iters=1, max_rounds=1).fit(X, y)
    monkeypatch.setattr(a, "_get_shap_imp", lambda *args, **kwargs: bad)
    with pytest.raises(ValueError, match="one value"):
        selector.fit(X, y)
    with pytest.raises(NotFittedError):
        selector.transform(X)


@pytest.mark.parametrize(
    "operation",
    ["get_support", "transform", "plot_importance", "get_feature_names_out"],
)
def test_first_training_failure_leaves_selector_unfitted(monkeypatch, operation):
    X, y = data()
    monkeypatch.setattr(a, "_get_shap_imp", lambda *args, **kwargs: np.ones(3))
    selector = a.BoostAGroota(lgb.LGBMRegressor(), iters=1, max_rounds=1)
    with pytest.raises(ValueError, match="one value"):
        selector.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)
    args = (X,) if operation == "transform" else ()
    with pytest.raises(NotFittedError):
        getattr(selector, operation)(*args)


@pytest.mark.parametrize(
    "summary", ["BoostAGroota ran successfully", "The feature selection BoostAGroota running time"]
)
def test_summary_failure_leaves_boostagroota_unfitted(monkeypatch, summary):
    X, y = data()
    monkeypatch.setattr(a, "_get_shap_imp", lambda *args, **kwargs: np.ones(8))
    selector = a.BoostAGroota(lgb.LGBMRegressor(), iters=1, max_rounds=1).fit(X, y)

    def closed_pipe(message, *args, **kwargs):
        if message.startswith(summary):
            raise BrokenPipeError("closed selector output")

    monkeypatch.setattr(a, "print", closed_pipe, raising=False)
    selector.set_params(silent=False)
    with pytest.raises(BrokenPipeError, match="closed selector output"):
        selector.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)
    with pytest.raises(NotFittedError):
        selector.get_support()
    with pytest.raises(NotFittedError):
        selector.transform(X)


@pytest.mark.parametrize("kind", ["duplicate", "empty_width", "empty_rows", "array"])
def test_invalid_frames_fail_before_training(monkeypatch, kind):
    X, y = data()
    if kind == "duplicate":
        X.columns = ["same"] * X.shape[1]
    elif kind == "empty_width":
        X = X.iloc[:, :0]
    elif kind == "empty_rows":
        X, y = X.iloc[:0], y.iloc[:0]
    else:
        X = X.to_numpy()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid inputs must fail before training")

    monkeypatch.setattr(a, "_get_shap_imp", forbidden)
    with pytest.raises(TypeError if kind == "array" else ValueError):
        a.BoostAGroota(lgb.LGBMRegressor()).fit(X, y)


@pytest.mark.parametrize(
    "kept,delta,rounds",
    [(0, 0.5, 1), (4, 0.5, 1), (3, 0.5, 1), (2, 0.5, 2), (1, 1.0, 1)],
)
def test_stopping_boundaries_and_round_count(monkeypatch, kept, delta, rounds):
    X, y = data()
    widths = []

    def importance(model, frame, *args, **kwargs):
        n = frame.shape[1] // 2
        widths.append(n)
        return np.r_[np.r_[np.full(kept, 3.0), np.zeros(n - kept)], np.ones(n)]

    monkeypatch.setattr(a, "_get_shap_imp", importance)
    selector = a.BoostAGroota(
        lgb.LGBMRegressor(), iters=2, max_rounds=2, cutoff=1, delta=delta
    ).fit(X, y)
    assert selector.n_rounds_ == rounds
    assert selector.n_features_in_ == 4
    assert widths == [4] * 2 + ([kept] * 2 if rounds == 2 else [])
    assert_array_equal(selector.selected_features_, X.columns[:kept])
    assert selector.sha_cutoff_df.index.tolist() == ["fscore1", "fscore2"]
    assert_array_equal(selector.get_feature_names_out(), selector.selected_features_)
    assert_frame_equal(selector.transform(X), X.iloc[:, :kept])


@pytest.mark.parametrize("empty", [False, True])
def test_shadow_construction_matches_legacy_frames_and_rng(empty):
    X = pd.DataFrame(
        {
            7: pd.array([1, None, 3, 4], dtype="Int64"),
            "float": [1.0, np.nan, 2.0, 3.0],
            "category": pd.Categorical(
                [20, 10, None, 20], categories=[10, 20, 30], ordered=True
            ),
            "text": pd.array(["a", "b", None, "b"], dtype="string"),
            "boolean": pd.array([True, None, False, True], dtype="boolean"),
            "ShadowVar1": pd.date_range("2020-01-01", periods=4, tz="UTC"),
            "ShadowVar1_": [0, 1, 2, 3],
        },
        index=[9, 2, 2, 5],
    )
    if empty:
        X = X.iloc[:, :0]
    before = X.copy(deep=True)
    old_rng, new_rng = np.random.RandomState(3), np.random.RandomState(3)
    shadow = X.copy()
    for col in shadow:
        shadow[col] = shadow[col].array.take(old_rng.permutation(len(shadow)))
    names = a._make_shadow_names(
        X.columns, [f"ShadowVar{i + 1}" for i in range(X.shape[1])]
    )
    shadow.columns = names
    actual, actual_names = a._create_shadow(X, new_rng)
    assert_frame_equal(actual, pd.concat([X, shadow], axis=1))
    assert actual_names == names
    left, right = old_rng.get_state(), new_rng.get_state()
    assert left[0] == right[0] and left[2:] == right[2:]
    assert_array_equal(left[1], right[1])
    assert_frame_equal(X, before)
