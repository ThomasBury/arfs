"""Native contributions preserve the historical GrootCV selection arithmetic."""

import numpy as np
import pandas as pd
import pytest
import shap
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from arfs.feature_selection import allrelevant as a


@pytest.mark.parametrize("objective", ["l2", "binary", "multiclass"])
@pytest.mark.parametrize("rf", [False, True])
def test_native_fold_scores_match_tree_explainer_and_legacy_table(
    monkeypatch, objective, rf
):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(180, 3)), columns=["z", "a", "ShadowVar1"])
    X["category"] = pd.Categorical(rng.integers(0, 4, len(X)))
    signal = 3 * X.z + 2 * X.a
    y = pd.Series(signal + rng.normal(size=len(X)))
    if objective == "binary":
        y = (signal > 0).astype(int)
    elif objective == "multiclass":
        y = pd.Series(np.digitize(signal, [-1, 1]))
    weights = np.where(X.z > 0, 4.0, 1.0)
    before = X.copy(deep=True)
    expected = pd.DataFrame({"feature": X.columns})
    train = a._train_lgb_model
    folds = []

    def checked_train(frame, *args, **kwargs):
        nonlocal expected
        bst = train(frame, *args, **kwargs)
        values = shap.TreeExplainer(bst).shap_values(frame)
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        values = np.abs(values)
        if values.ndim == 3:
            values = values.sum(axis=2)
        scores = values.mean(axis=0)
        # Independent reference: the old path sorted scores before normalizing.
        order = np.argsort(scores, kind="stable")
        scores, names = scores[order], frame.columns[order]
        normalized = scores / scores.sum() if scores.sum() > 0 else scores
        column = f"fscore{len(folds)}"
        expected = expected.merge(
            pd.DataFrame({"feature": names, column: normalized}),
            on="feature",
            how="outer",
        )
        folds.append(bst.best_iteration)
        return bst

    monkeypatch.setattr(a, "_train_lgb_model", checked_train)
    selector = a.GrootCV(
        objective=objective,
        n_folds=3,
        n_iter=2,
        n_jobs=1,
        rf=rf,
        random_state=42,
        lgbm_params={"num_iterations": 8, "min_data_in_leaf": 5, "seed": 42},
    ).fit(X, y, sample_weight=weights)
    expected["Med"] = expected.filter(like="fscore").median(axis=1)
    assert len(folds) == 6
    assert_frame_equal(selector.cv_df, expected, rtol=1e-12, atol=1e-14)
    threshold = expected.loc[~expected.feature.isin(X.columns), "Med"].max()
    assert selector.sha_cutoff == pytest.approx(threshold)
    selected = expected.loc[
        expected.feature.isin(X.columns)
        & (expected.Med >= threshold)
        & (expected.Med > 0),
        "feature",
    ].to_numpy()
    assert_array_equal(selector.selected_features_, selected)
    assert_frame_equal(X, before)


@pytest.mark.parametrize(
    "limits,maximum",
    [
        ({"num_iterations": 3}, 3),
        ({"n_estimators": 4}, 4),
        ({"num_boost_round": 5}, 5),
        ({"num_iterations": 3, "n_estimators": 4}, 3),
    ],
)
def test_iteration_limits_reach_training_and_leave_parameters_owned(
    monkeypatch, limits, maximum
):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(240, 2)), columns=["a", "b"])
    y = 5 * X.a + rng.normal(scale=0.1, size=len(X))
    params = {**limits, "min_data_in_leaf": 5}
    before = params.copy()
    train = a._train_lgb_model
    iterations = []

    def checked_train(*args, **kwargs):
        bst = train(*args, **kwargs)
        iterations.append(bst.current_iteration())
        return bst

    monkeypatch.setattr(a, "_train_lgb_model", checked_train)
    a.GrootCV(objective="l2", n_folds=2, n_iter=1, n_jobs=1, lgbm_params=params).fit(
        X, y
    )
    assert iterations == [maximum, maximum]
    assert params == before


def test_failed_refit_preserves_previous_selection_and_feature_order():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(240, 2)), columns=["a", "b"])
    y = 5 * X.a + rng.normal(scale=0.1, size=len(X))
    selector = a.GrootCV(
        objective="l2",
        n_folds=2,
        n_iter=1,
        n_jobs=1,
        lgbm_params={"num_iterations": 5, "min_data_in_leaf": 5},
    )
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)

    selector.fit(X, y)
    selected = selector.transform(X)
    selected_names = selector.get_feature_names_out()
    selector.set_params(lgbm_params={"num_iterations": 0})
    with pytest.raises(ValueError, match="boosting rounds"):
        selector.fit(X[["b", "a"]], y)

    assert_array_equal(selector.feature_names_in_, X.columns)
    assert_array_equal(selector.get_feature_names_out(), selected_names)
    assert_frame_equal(selector.transform(X), selected)


def test_native_explanations_use_early_stopping_best_iteration():
    rng = np.random.default_rng(19)
    X = pd.DataFrame(rng.normal(size=(160, 2)), columns=["a", "b"])
    train, valid = X.iloc[:100], X.iloc[100:]
    bst = a._train_lgb_model(
        train,
        5 * train.a,
        None,
        valid,
        -5 * valid.a,
        None,
        objective="l2",
        num_threads=1,
        num_iterations=100,
        min_data_in_leaf=5,
        verbosity=-1,
    )
    assert 0 < bst.best_iteration < 100
    native = a._compute_importance(
        train, bst.predict(train, pred_contrib=True), native=True
    )
    reference = np.abs(shap.TreeExplainer(bst).shap_values(train)).mean(axis=0)
    assert_allclose(native, reference, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("shape", [(2,), (3, 4), (2, 0), (2, 3), (2, 4, 1)])
def test_malformed_native_layout_is_rejected(shape):
    X = pd.DataFrame(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Native SHAP.*shape"):
        a._compute_importance(X, np.ones(shape), native=True)


def test_native_bias_is_excluded_for_every_output():
    X = pd.DataFrame(np.zeros((2, 2)))
    values = np.array([[1, -2, 100, -3, 4, -200], [-5, 6, 300, 7, -8, -400]])
    assert_allclose(a._compute_importance(X, values, native=True), [8, 10])
