"""R18 full ordinal schema, input ownership and current-parameter regressions."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from arfs.preprocessing import OrdinalEncoderPandas


def mixed_frame(dtype=object):
    X = pd.DataFrame({"cat": pd.Series(["b", "a", np.nan], dtype=dtype)})
    X["num"] = pd.Series([1, 2, 3], dtype="Int64")
    X["excluded"] = pd.Series(["x", "y", "x"], dtype="string")
    X.index = [10, 30, 20]
    return X


@pytest.mark.parametrize("output", ["default", "pandas"])
@pytest.mark.parametrize("dtype", [object, "category", "string", None])
def test_full_schema_and_native_unknown_missing_reference(output, dtype):
    X = mixed_frame(dtype)
    before = X.copy(deep=True)
    encoder = OrdinalEncoderPandas(exclude_cols=["excluded"]).set_output(
        transform=output
    )
    actual = encoder.fit_transform(X)
    native = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    categorical = X[["cat"]].astype(object).where(X[["cat"]].notna(), np.nan)
    expected = X.copy()
    expected["cat"] = native.fit_transform(categorical).ravel()
    assert_frame_equal(actual, expected)
    assert_frame_equal(X, before)
    assert encoder.n_features_in_ == 3
    assert_array_equal(encoder.feature_names_in_, X.columns)
    assert_array_equal(encoder.get_feature_names_out(), X.columns)
    assert_array_equal(encoder.get_feature_names_out(X.columns), X.columns)
    assert encoder.categorical_features_ == ["cat"]
    assert_series_equal(
        pd.Series(encoder.categories_[0]), pd.Series(native.categories_[0])
    )
    new = X.iloc[[0, 2]].copy()
    new["cat"] = pd.Series(["new", np.nan], index=new.index, dtype=dtype)
    expected = new.copy()
    categorical = new[["cat"]].astype(object).where(new[["cat"]].notna(), np.nan)
    expected["cat"] = native.transform(categorical).ravel()
    assert_frame_equal(encoder.transform(new), expected)


@pytest.mark.parametrize("categorical_output", [False, True])
def test_inverse_preserves_encoded_input_and_passthrough(categorical_output):
    X = mixed_frame()
    encoder = OrdinalEncoderPandas(
        exclude_cols=["excluded"],
        return_pandas_categorical=categorical_output,
        unknown_value=-1,
    ).fit(X)
    encoded = encoder.transform(X)
    before = encoded.copy(deep=True)
    decoded = encoder.inverse_transform(encoded)
    assert_series_equal(decoded.cat, X.cat, check_dtype=False)
    assert_frame_equal(decoded.drop(columns="cat"), X.drop(columns="cat"))
    assert_frame_equal(encoded, before)
    unknown = X.iloc[[0]].copy()
    unknown["cat"] = "unknown"
    decoded_unknown = encoder.inverse_transform(encoder.transform(unknown))
    assert decoded_unknown.cat.iloc[0] is None
    assert_series_equal(decoded_unknown.num, unknown.num)


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_refit_uses_changed_dtype_and_clears_previous_schema(dtype):
    encoder = OrdinalEncoderPandas().fit(mixed_frame())
    X = pd.DataFrame({"different": ["b", "a"], "numeric": [4.0, 5.0]})
    encoder.set_params(output_dtype=dtype, unknown_value=-1, encoded_missing_value=-2)
    encoded = encoder.fit_transform(X)
    assert encoded.different.dtype == np.dtype(dtype)
    assert_array_equal(encoder.get_feature_names_out(), X.columns)
    assert encoder.n_features_in_ == 2
    assert encoder.categorical_features_ == ["different"]
    assert clone(encoder).get_params() == encoder.get_params()
    assert_frame_equal(clone(encoder).fit_transform(X), encoded)


@pytest.mark.parametrize("width", [0, 1, 2])
@pytest.mark.parametrize("output", ["default", "pandas"])
def test_no_categorical_features_are_fitted_passthrough(width, output):
    X = pd.DataFrame(np.arange(3 * width).reshape(3, width))
    encoder = OrdinalEncoderPandas().set_output(transform=output)
    actual = encoder.fit_transform(X)
    assert_frame_equal(actual, X)
    assert actual is not X
    assert encoder.n_features_in_ == width
    assert encoder.categories_ == []
    assert_array_equal(encoder.get_feature_names_out(), X.columns)
    assert_frame_equal(encoder.inverse_transform(actual), X)
    # A refit to no categories must discard the earlier category vocabulary.
    encoder.fit(mixed_frame()).fit(X)
    assert_frame_equal(encoder.transform(X), X)
    assert encoder.categories_ == []


@pytest.mark.parametrize(
    "method", ["transform", "inverse_transform", "get_feature_names_out"]
)
def test_unfitted_methods_raise(method):
    encoder = OrdinalEncoderPandas()
    with pytest.raises(NotFittedError):
        getattr(encoder, method)(
            mixed_frame()
        ) if method != "get_feature_names_out" else encoder.get_feature_names_out()


@pytest.mark.parametrize(
    "change", ["reorder", "missing", "extra", "replace", "duplicate"]
)
@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_full_schema_rejects_changed_columns(change, method):
    X = mixed_frame()
    encoder = OrdinalEncoderPandas().fit(X)
    actual = encoder.transform(X) if method == "inverse_transform" else X.copy()
    if change == "reorder":
        actual = actual[actual.columns[::-1]]
    elif change == "missing":
        actual = actual.drop(columns="num")
    elif change == "extra":
        actual["extra"] = 1
    elif change == "replace":
        actual = actual.rename(columns={"num": "replacement"})
    else:
        actual.columns = ["cat", "num", "num"]
    with pytest.raises(ValueError, match="columns"):
        getattr(encoder, method)(actual)


def test_duplicate_fit_columns_and_non_dataframe_inputs_rejected():
    X = mixed_frame()
    X.columns = ["cat", "num", "num"]
    with pytest.raises(ValueError, match="unique"):
        OrdinalEncoderPandas().fit(X)
    with pytest.raises(TypeError, match="DataFrame"):
        OrdinalEncoderPandas().fit(np.ones((3, 2)))


def test_pipeline_names_and_configured_output_match_full_frame():
    X = mixed_frame()
    pipe = Pipeline([("encode", OrdinalEncoderPandas())]).set_output(transform="pandas")
    actual = pipe.fit_transform(X)
    assert_array_equal(pipe.get_feature_names_out(), actual.columns)
    assert actual.shape == X.shape
    with pytest.raises(ValueError, match="input_features"):
        pipe.get_feature_names_out(["replacement", "num", "excluded"])


def test_boolean_missing_and_updated_unknown_policy():
    X = pd.DataFrame({"flag": pd.Series([True, False, None], dtype="boolean")})
    encoder = OrdinalEncoderPandas(encoded_missing_value=-2, unknown_value=-1)
    actual = encoder.fit_transform(X)
    assert_array_equal(actual.flag, [1, 0, -2])
    encoder.set_params(handle_unknown="error", unknown_value=None).fit(X)
    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(pd.DataFrame({"flag": ["new"]}))


def test_default_shared_nan_code_decodes_as_missing():
    X = pd.DataFrame({"cat": ["a", np.nan]})
    encoder = OrdinalEncoderPandas().fit(X)
    unknown = pd.DataFrame({"cat": ["new"]})
    encoded = encoder.transform(unknown)
    assert pd.isna(encoder.inverse_transform(encoded).cat.iloc[0])
    assert pd.isna(encoded.cat.iloc[0])
