"""R31 intentional API removals and supported categorical migration."""

import importlib

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from arfs.preprocessing import OrdinalEncoderPandas


@pytest.mark.parametrize(
    "module,name",
    [("arfs.preprocessing", "cat_var"), ("arfs.utils", "validate_pandas_input")],
)
def test_public_legacy_helpers_are_removed(module, name):
    public = importlib.import_module(module)
    assert not hasattr(public, name)
    assert name not in getattr(public, "__all__", [])


@pytest.mark.parametrize("dtype", [object, "category", "string", None])
def test_supported_encoder_keeps_missing_distinct_and_owns_input(dtype):
    X = pd.DataFrame({"cat": pd.Series(["a", None, "b"], dtype=dtype)})
    X["num"] = pd.Series([2, 3, 1], dtype="Int64")
    X.index = [10, 20, 30]
    before = X.copy(deep=True)
    encoder = OrdinalEncoderPandas(
        output_dtype=np.int64, unknown_value=-1, encoded_missing_value=-2
    )
    encoded = encoder.fit_transform(X)
    encoded_before = encoded.copy(deep=True)
    assert_array_equal(encoded.cat, [0, -2, 1])
    assert encoded.cat.dtype == np.dtype("int64")
    assert_array_equal(encoder.get_feature_names_out(), X.columns)
    decoded = encoder.inverse_transform(encoded)
    assert decoded.cat.iloc[0] == "a" and decoded.cat.iloc[2] == "b"
    assert pd.isna(decoded.cat.iloc[1])
    assert_frame_equal(decoded[["num"]], X[["num"]])
    assert_frame_equal(X, before)
    assert_frame_equal(encoded, encoded_before)
