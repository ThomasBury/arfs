"""R16 selector identity, parameter and fitted-state regressions."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from arfs import feature_selection as fs
from arfs.feature_selection.unsupervised import _recursive_collinear_elimination

CLASSES = [
    fs.MissingValueThreshold,
    fs.UniqueValuesThreshold,
    fs.CardinalityThreshold,
    fs.CollinearityThreshold,
    fs.VariableImportance,
]


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"z": [1, 2, 3, 4], "constant": [1] * 4, "a": [4, 2, 3, 1]},
        index=[3, 6, 8, 12],
    )


@pytest.fixture
def importance_backend(monkeypatch):
    def compute(X, **kwargs):
        return pd.DataFrame(
            {
                "feature": X.columns,
                "importance": [0.6, 0.3, 0.1][: X.shape[1]],
                "cumulative_importance": [0.6, 0.9, 1.0][: X.shape[1]],
            }
        )

    monkeypatch.setattr(
        "arfs.feature_selection.variable_importance._compute_varimp_lgb", compute
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_duplicate_fit_and_transform_names_rejected(cls, frame, importance_backend):
    duplicate = frame.copy()
    duplicate.columns = ["a", "a", "b"]
    with pytest.raises(ValueError, match="unique"):
        cls().fit(duplicate, np.arange(len(frame)))
    selector = cls().fit(frame, np.arange(len(frame)))
    with pytest.raises(ValueError, match="unique"):
        selector.transform(pd.concat([frame, frame.iloc[:, [0]]], axis=1))


@pytest.mark.parametrize("cls", CLASSES)
def test_unfitted_and_partial_state_raise_not_fitted(cls, frame):
    selector = cls()
    for operation in [
        selector.transform,
        selector.get_support,
        selector.get_feature_names_out,
    ]:
        with pytest.raises(NotFittedError):
            operation(frame) if operation == selector.transform else operation()
    selector.feature_names_in_ = frame.columns.to_numpy()
    with pytest.raises(NotFittedError):
        selector.transform(frame)
    with pytest.raises(NotFittedError):
        selector.get_support()


@pytest.mark.parametrize("cls", CLASSES)
def test_metadata_named_projection_clone_and_refit(cls, frame, importance_backend):
    before = frame.copy(deep=True)
    selector = clone(cls()).fit(frame, np.arange(len(frame)))
    assert selector.n_features_in_ == 3
    out = selector.transform(frame)
    assert_array_equal(out.columns, selector.get_feature_names_out())
    assert_array_equal(
        selector.get_support(indices=True), np.flatnonzero(selector.support_)
    )
    assert_frame_equal(out, frame.iloc[:, selector.get_support()])
    projection = frame.loc[:, selector.selected_features_].copy()
    projection["extra"] = 8
    assert_frame_equal(selector.transform(projection[projection.columns[::-1]]), out)
    assert_frame_equal(selector.transform(frame.iloc[:0]), out.iloc[:0])
    if len(out.columns):
        with pytest.raises(KeyError):
            selector.transform(frame.drop(columns=out.columns[0]))
    assert_frame_equal(frame, before)
    renamed = frame.rename(columns=lambda name: "new_" + name)
    assert selector.fit(renamed, np.arange(len(frame))) is selector
    assert_array_equal(selector.feature_names_in_, renamed.columns)
    assert_array_equal(
        selector.get_feature_names_out(), selector.transform(renamed).columns
    )
    selector.fit(renamed.iloc[:, :2], np.arange(len(frame)))
    assert selector.n_features_in_ == 2
    assert len(selector.get_support()) == 2


def test_labelled_statistic_aligned_to_input_order(frame):
    selector = fs.BaseThresholdSelector(
        1, lambda X: X.nunique().sort_values(ascending=False).rename("count"), True
    ).fit(frame)
    assert_array_equal(selector.selected_features_, ["z", "a"])
    assert_series_equal(selector.statistic_series_, frame.nunique().rename("count"))
    assert selector.statistic_df_.columns.tolist() == ["statistic"]


@pytest.mark.parametrize("names", [[2, 1, 3], [("z", 0), ("c", 0), ("a", 0)]])
def test_nonstring_callback_names_preserve_identity(frame, names):
    frame.columns = pd.Index(names, tupleize_cols=False)
    selector = fs.BaseThresholdSelector(1, lambda X: X.nunique().iloc[::-1], True).fit(
        frame
    )
    assert_frame_equal(selector.transform(frame), frame.iloc[:, [0, 2]])
    assert_array_equal(selector.get_feature_names_out(), frame.columns[[0, 2]])


def test_collinearity_uses_fitted_order_for_real_ties(frame):
    X = frame.loc[:, ["z"]].assign(a=frame.z)
    before = X.copy(deep=True)
    selector = fs.CollinearityThreshold().fit(X)
    assert_array_equal(selector.selected_features_, ["a"])
    selector.fit(X.iloc[:, ::-1])
    assert_array_equal(selector.selected_features_, ["z"])
    selector.set_params(threshold=1).fit(X)
    assert_array_equal(selector.selected_features_, X.columns)
    assert_frame_equal(X, before)


@pytest.mark.parametrize(
    "statistic",
    [
        pd.Series([1, 2], index=["z", "a"]),
        pd.Series([1, 2, 3], index=["z", "a", "extra"]),
        pd.Series([1, 2, 3], index=["z", "z", "a"]),
        pd.Series([1, np.nan, 3], index=["z", "constant", "a"]),
        pd.Series([1, np.inf, 3], index=["z", "constant", "a"]),
        pd.Series([1, "bad", 3], index=["z", "constant", "a"]),
        pd.Series([1, pd.NA, 3], index=["z", "constant", "a"], dtype="Int64"),
        pd.Series([1, 2j, 3], index=["z", "constant", "a"]),
        np.ones(3),
    ],
)
def test_invalid_callback_result_rejected(frame, statistic):
    selector = fs.BaseThresholdSelector(1, lambda X: statistic, True)
    with pytest.raises(ValueError, match="statistic"):
        selector.fit(frame)
    with pytest.raises(NotFittedError):
        selector.get_support()


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf, "bad", True])
def test_base_rejects_invalid_threshold_at_fit(frame, threshold):
    selector = fs.BaseThresholdSelector(statistic_fn=lambda X: X.nunique())
    with pytest.raises(ValueError, match="threshold"):
        selector.set_params(threshold=threshold).fit(frame)


@pytest.mark.parametrize(
    "params", [{"statistic_fn": None}, {"greater_than_threshold": "yes"}]
)
def test_base_rejects_invalid_callback_parameters(frame, params):
    selector = fs.BaseThresholdSelector(statistic_fn=lambda X: X.nunique())
    with pytest.raises(TypeError):
        selector.set_params(**params).fit(frame)


@pytest.mark.parametrize("cls", CLASSES[:-1])
@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, -np.inf, "bad", True])
def test_invalid_thresholds_after_set_params(cls, threshold, frame):
    selector = cls().set_params(threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        selector.fit(frame)


@pytest.mark.parametrize("cls", [fs.MissingValueThreshold, fs.CollinearityThreshold])
def test_fraction_threshold_upper_bound(cls, frame):
    with pytest.raises(ValueError, match="threshold"):
        cls().set_params(threshold=1.1).fit(frame)


@pytest.mark.parametrize("cls", [fs.UniqueValuesThreshold, fs.CardinalityThreshold])
@pytest.mark.parametrize("threshold", [0, 1.5])
def test_count_threshold_is_positive_integer(cls, threshold, frame):
    with pytest.raises(ValueError, match="threshold"):
        cls().set_params(threshold=threshold).fit(frame)


def test_invalid_collinearity_method_after_set_params(frame):
    with pytest.raises(ValueError, match="method"):
        fs.CollinearityThreshold().set_params(method="misspelled").fit(frame)


def test_strict_threshold_equality_and_valid_updates():
    X = pd.DataFrame(
        {"complete": [1.0] * 4, "half": [1, 2, np.nan, np.inf], "missing": [np.nan] * 4}
    )
    selector = fs.MissingValueThreshold(0.5).fit(X)
    assert_array_equal(selector.selected_features_, ["complete"])
    assert selector.set_params(threshold=0).fit(X).selected_features_.size == 0
    assert_array_equal(
        selector.set_params(threshold=1).fit(X).selected_features_, ["complete", "half"]
    )
    counts = pd.DataFrame({"two": [1, 2, 1], "three": [1, 2, 3]})
    assert_array_equal(
        fs.UniqueValuesThreshold(np.int64(2)).fit(counts).selected_features_, ["three"]
    )
    cats = counts.astype("category")
    assert_array_equal(fs.CardinalityThreshold(2).fit(cats).selected_features_, [])
    assert_array_equal(fs.CardinalityThreshold(3).fit(cats).selected_features_, ["two"])
    matrix = pd.DataFrame([[0, -0.8], [-0.8, 0]], index=["z", "a"], columns=["z", "a"])
    assert _recursive_collinear_elimination(matrix, 0.8) == []


@pytest.mark.parametrize("remaining", [0, 1])
@pytest.mark.parametrize("method", ["association", "correlation"])
def test_empty_pipeline_stages_have_consistent_metadata(remaining, method):
    X = pd.DataFrame({"constant": [1] * 4, "varying": [1, 2, 3, 4]})
    pipe = Pipeline(
        [
            ("unique", fs.UniqueValuesThreshold(1 if remaining else 4)),
            ("missing", fs.MissingValueThreshold()),
            ("cardinality", fs.CardinalityThreshold()),
            ("collinear", fs.CollinearityThreshold(method=method)),
        ]
    )
    out = pipe.fit_transform(X)
    assert out.shape == (4, remaining)
    selector = pipe[-1]
    assert selector.n_features_in_ == remaining
    assert_array_equal(selector.get_support(), np.ones(remaining, dtype=bool))
    assert_array_equal(selector.get_feature_names_out(), out.columns)
    assert_frame_equal(
        selector.assoc_matrix_,
        pd.DataFrame(0.0, index=out.columns, columns=out.columns),
    )


@pytest.mark.parametrize("cls", CLASSES[:-1])
def test_empty_observations_rejected(cls, frame):
    with pytest.raises(ValueError, match="observations"):
        cls().fit(frame.iloc[:0])


@pytest.mark.parametrize("remaining", [0, 1])
@pytest.mark.parametrize("weights", [[0, 0, 0], [1, -1, 1], [1, 1]])
def test_no_pairs_still_validate_weights(remaining, weights):
    X = pd.DataFrame(np.ones((3, remaining)))
    with pytest.raises(ValueError, match="sample_weight"):
        fs.CollinearityThreshold().fit(X, sample_weight=weights)


def test_directional_association_and_input_order_ties():
    matrix = pd.DataFrame(
        [[0, -0.9, 0.1, 0], [-0.9, 0, 0.8, 0], [0.1, 0.2, 0, 0], [0, 0, 0, 0]],
        index=list("zabc"),
        columns=list("zabc"),
    )
    before = matrix.copy(deep=True)
    assert _recursive_collinear_elimination(matrix, 0.7) == ["a"]
    assert_frame_equal(matrix, before)
    tied = pd.DataFrame([[0, 1], [1, 0]], index=["z", "a"], columns=["z", "a"])
    assert _recursive_collinear_elimination(tied, 0.8) == ["z"]


def test_ties_stable_across_hash_seeds():
    code = """import json, pandas as pd
from arfs.feature_selection import CollinearityThreshold
X = pd.DataFrame({'z': [1,2,3,4], 'a': [1,2,3,4]})
print(json.dumps(CollinearityThreshold().fit(X).selected_features_.tolist()))
"""
    outputs = []
    for seed in ["0", "1", "2", "3"]:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONHASHSEED=seed),
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs == [["a"]] * 4


def test_real_importance_fit_metadata(frame):
    rng = np.random.default_rng(16)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=frame.columns)
    selector = fs.VariableImportance(
        lgb_kwargs={
            "objective": "rmse",
            "num_threads": 1,
            "verbosity": -1,
            "num_boost_round": 4,
        },
    ).fit(X, 3 * X.z - X.a)
    assert selector.n_features_in_ == 3
    assert_array_equal(selector.get_feature_names_out(), selector.transform(X).columns)
