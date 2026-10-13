"""R14: one fitted GLM design across prediction, selection and refits."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arfs.feature_selection import make_fs_summary
from arfs.feature_selection.lasso import EnetGLM, LassoFeatureSelection


def data():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(90, 3)), columns=["a", "b", "c"])
    w = rng.uniform(0.2, 2, len(X))
    y = 2 + 3 * X.a.to_numpy() + rng.normal(size=len(X))
    return X, y, w


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_fitted_design_matches_weighted_least_squares_and_single_row(
    as_array, fit_intercept
):
    X, y, w = data()
    features = X.to_numpy() if as_array else X
    before = X.copy(deep=True)
    model = EnetGLM(alpha=0.001, L1_wt=1, fit_intercept=fit_intercept)
    assert model.fit(features, y, w) is model
    design = sm.add_constant(X).to_numpy() if fit_intercept else X.to_numpy()
    beta = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    assert_allclose(model.coef_, beta, atol=1e-8)
    assert_allclose(model.predict(features), design @ beta, atol=1e-8)
    single = features[[0]] if as_array else features.iloc[[0]]
    assert_allclose(model.predict(single), np.asarray(model.predict(features))[[0]])
    assert model.n_features_in_ == X.shape[1]
    expected_names = ["pred_0", "pred_1", "pred_2"] if as_array else X.columns
    assert_array_equal(model.feature_names_in_, expected_names)
    assert len(model.design_feature_names_) == X.shape[1] + fit_intercept
    assert_frame_equal(X, before)


@pytest.mark.parametrize("name", ["const", "Intercept", "intercept", "constant"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_supplied_constant_is_preserved_without_duplicate_intercept(
    name, fit_intercept
):
    X, y, w = data()
    X[name] = 2.0
    before = X.copy(deep=True)
    model = EnetGLM(alpha=0.001, L1_wt=1, fit_intercept=fit_intercept).fit(X, y, w)
    assert_array_equal(model.design_feature_names_, X.columns)
    assert model.added_intercept_ is None
    reference = sm.WLS(y, X, weights=w).fit()
    assert_allclose(model.predict(X), reference.predict(X), atol=1e-8)
    assert_allclose(model.predict(X.iloc[[0]]), reference.predict(X.iloc[[0]]))
    assert_frame_equal(X, before)


@pytest.mark.parametrize("name", ["const", "Intercept", "Intercept_1"])
def test_varying_intercept_names_are_predictors_and_generated_name_cannot_collide(name):
    X, y, w = data()
    X = X.rename(columns={"b": name})
    if name == "Intercept_1":
        X = X.rename(columns={"c": "Intercept"})
    model = EnetGLM(alpha=0.001, L1_wt=1).fit(X, y, w)
    assert model.added_intercept_ not in X.columns
    assert len(set(model.design_feature_names_)) == X.shape[1] + 1
    design = np.column_stack([np.ones(len(X)), X.to_numpy()])
    reference = sm.WLS(y, design, weights=w).fit()
    assert_allclose(model.predict(X), reference.predict(design), atol=1e-8)


@pytest.mark.parametrize("selector", [False, True])
@pytest.mark.parametrize("constant", [None, "const", "constant"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_named_schema_alignment_and_positional_array_inference(
    selector, constant, fit_intercept
):
    X, y, w = data()
    if constant:
        X[constant] = 1.0
    if selector:
        model = LassoFeatureSelection(n_iterations=3, fit_intercept=fit_intercept).fit(
            X, y, w
        )
        method = model.transform
    else:
        model = EnetGLM(alpha=0.001, L1_wt=1, fit_intercept=fit_intercept).fit(X, y, w)
        method = model.predict
    expected = method(X)
    actual = method(X[X.columns[::-1]])
    array = method(X.to_numpy())
    if selector:
        assert_frame_equal(actual, expected)
        assert_frame_equal(array, expected)
    else:
        assert_allclose(actual, expected)
        assert_allclose(array, expected)


@pytest.mark.parametrize("selector", [False, True])
@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "replacement", "duplicate", "narrow", "wide", "onedim"],
)
def test_inference_rejects_changed_or_ambiguous_schema(selector, mutation):
    X, y, w = data()
    if selector:
        model = LassoFeatureSelection(n_iterations=3).fit(X, y, w)
        method = model.transform
    else:
        model = EnetGLM(alpha=0.001, L1_wt=1).fit(X, y, w)
        method = model.predict
    bad = {
        "missing": X.drop(columns="b"),
        "extra": X.assign(extra=1.0),
        "replacement": X.rename(columns={"b": "other"}),
        "duplicate": X.set_axis(["a", "a", "c"], axis=1),
        "narrow": X.to_numpy()[:, :2],
        "wide": np.column_stack([X, np.ones(len(X))]),
        "onedim": X.a.to_numpy(),
    }[mutation]
    with pytest.raises(ValueError, match="columns|features|two-dimensional"):
        method(bad)


@pytest.mark.parametrize("selector", [False, True])
def test_duplicate_training_names_are_rejected(selector):
    X, y, w = data()
    X.columns = ["a", "a", "c"]
    model = LassoFeatureSelection(n_iterations=3) if selector else EnetGLM()
    with pytest.raises(ValueError, match="duplicate"):
        model.fit(X, y, w)


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("constant", [None, "const", "constant"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_selector_metadata_describes_inputs_and_actual_selected_design(
    as_array, constant, fit_intercept
):
    X, y, w = data()
    if constant:
        X[constant] = 1.0
    features = X.to_numpy() if as_array else X
    selector = LassoFeatureSelection(n_iterations=3, fit_intercept=fit_intercept).fit(
        features, y, w
    )
    output = selector.transform(features)
    expected_inputs = (
        [f"pred_{i}" for i in range(X.shape[1])] if as_array else X.columns
    )
    assert selector.n_features_in_ == X.shape[1]
    assert_array_equal(selector.feature_names_in_, expected_inputs)
    assert len(selector.support_) == X.shape[1]
    coefficients = pd.Series(
        np.asarray(selector.best_estimator_.coef_),
        index=selector.best_estimator_.model.exog_names,
    )
    assert_array_equal(selector.design_feature_names_, coefficients.index)
    assert_array_equal(selector.design_support_, coefficients.to_numpy() != 0)
    assert_array_equal(
        selector.support_,
        coefficients.reindex(expected_inputs).to_numpy() != 0,
    )
    assert_array_equal(output.columns, selector.get_feature_names_out())
    assert_array_equal(output.columns, selector.selected_features_)
    single = features[[0]] if as_array else features.iloc[[0]]
    assert_frame_equal(selector.transform(single), output.iloc[[0]])


@pytest.mark.parametrize("params", [{"family": "poisson"}, {"link": "log"}])
def test_set_params_objective_matches_fresh_construction(params):
    X, y, w = data()
    y = np.exp(y / 6)
    model = EnetGLM(alpha=0.001, L1_wt=1).set_params(**params).fit(X, y, w)
    fresh = EnetGLM(alpha=0.001, L1_wt=1, **params).fit(X, y, w)
    assert type(model.objective) is type(fresh.objective)
    assert type(model.objective.link) is type(fresh.objective.link)
    assert_allclose(model.predict(X), fresh.predict(X))
    assert_allclose(model.score(X, y, w), fresh.score(X, y, w))


@pytest.mark.parametrize("selector", [False, True])
def test_successful_refit_replaces_design_and_uses_current_parameters(selector):
    X, y, w = data()
    model = (
        LassoFeatureSelection(n_iterations=3)
        if selector
        else EnetGLM(alpha=0.001, L1_wt=1)
    )
    model.fit(X, y, w)
    old_prediction = model.best_estimator_.predict(X) if selector else model.predict(X)
    model.set_params(fit_intercept=False, family="poisson")
    if not selector:
        assert_allclose(model.predict(X), old_prediction)
    new_X = X[["c", "a"]].rename(columns={"c": "new"})
    new_y = np.exp(y / 6)
    assert model.fit(new_X, new_y, w) is model
    fresh = clone(model).fit(new_X, new_y, w)
    if selector:
        assert_frame_equal(model.transform(new_X), fresh.transform(new_X))
        estimator, reference = model.best_estimator_, fresh.best_estimator_
    else:
        estimator, reference = model, fresh
    assert_array_equal(model.feature_names_in_, new_X.columns)
    assert model.n_features_in_ == 2
    assert_array_equal(estimator.design_feature_names_, new_X.columns)
    assert estimator.added_intercept_ is None
    assert_allclose(estimator.predict(new_X), reference.predict(new_X))


def test_zero_alpha_fit_does_not_modify_constructor_parameters():
    X, y, w = data()
    model = EnetGLM(alpha=0.0)
    before = model.get_params()
    model.fit(X, y, w)
    assert model.get_params() == before


@pytest.mark.parametrize("selector", [False, True])
def test_inference_before_fit_is_actionable(selector):
    X, _, _ = data()
    model = LassoFeatureSelection() if selector else EnetGLM()
    with pytest.raises(NotFittedError):
        (model.transform if selector else model.predict)(X)
    if selector:
        with pytest.raises(NotFittedError):
            model.get_feature_names_out()


def test_real_pipeline_feature_names_summary_and_pandas_output():
    X, y, w = data()
    pipe = Pipeline(
        [
            ("scale", StandardScaler().set_output(transform="pandas")),
            (
                "select",
                LassoFeatureSelection(n_iterations=3).set_output(transform="pandas"),
            ),
        ]
    ).fit(X, y, select__sample_weight=w)
    output = pipe.transform(X)
    assert_array_equal(output.columns, pipe.get_feature_names_out())
    summary = make_fs_summary(pipe)
    assert_array_equal(summary.data.predictor, X.columns)
    selector = pipe.named_steps["select"]
    assert_array_equal(summary.data["select"], selector.support_.astype(int))
    assert "background-color" in summary.to_html()
    with pytest.raises(ValueError, match="input_features"):
        selector.get_feature_names_out(["wrong", "b", "c"])


@pytest.mark.parametrize(
    "labels", [[10, 20, 30], [0, "b", "c"], [("a", 1), ("b", 2), ("c", 3)]]
)
@pytest.mark.parametrize("selector", [False, True])
def test_original_label_objects_survive_fitted_design_and_reordering(labels, selector):
    X, y, w = data()
    X.columns = pd.Index(labels, tupleize_cols=False)
    model = (
        LassoFeatureSelection(n_iterations=3)
        if selector
        else EnetGLM(alpha=0.001, L1_wt=1)
    )
    model.fit(X, y, w)
    assert_array_equal(model.feature_names_in_, X.columns.to_numpy())
    method = model.transform if selector else model.predict
    expected = method(X)
    actual = method(X.iloc[:, ::-1])
    if selector:
        assert_frame_equal(actual, expected)
    else:
        assert_allclose(actual, expected)


def test_zero_constant_stays_an_input_and_does_not_replace_a_nonzero_intercept():
    X, y, w = data()
    X["const"] = 0.0
    model = EnetGLM(alpha=0.001, L1_wt=1).fit(X, y, w)
    assert model.added_intercept_ == "Intercept"
    assert_array_equal(model.design_feature_names_, ["Intercept", *X.columns])
    design = np.column_stack([np.ones(len(X)), X.to_numpy()])
    assert_allclose(
        model.predict(X), sm.WLS(y, design, weights=w).fit().predict(design)
    )
