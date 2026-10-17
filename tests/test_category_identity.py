"""R06 category identity and dtype regressions from reviews 1A, 03, 05 and 06."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from arfs import association as a
from arfs import utils
from arfs.feature_selection import CardinalityThreshold
from arfs.feature_selection import allrelevant as ar

CATEGORICAL_DTYPES = [object, "category", "string", None, bool, "boolean"]


def categorical_series(dtype, name="cat"):
    values = [True, True, False, False, True, False]
    if dtype not in [bool, "boolean"]:
        values = ["b", "b", "a", "a", "b", "a"]
    return pd.Series(values, dtype=dtype, name=name)


def test_shared_dtype_groups_preserve_numeric_time_and_interval_roles():
    X = pd.DataFrame(
        {str(dtype): categorical_series(dtype) for dtype in CATEGORICAL_DTYPES}
    )
    X["numeric_category"] = pd.Categorical([1, 2, 1, 2, 1, 2])
    X["integer"] = pd.Series(range(6), dtype="Int64")
    X["float"] = pd.Series(range(6), dtype="Float64")
    X["time"] = pd.date_range("2020", periods=6, tz="UTC")
    X["delta"] = pd.to_timedelta(range(6), unit="D")
    X["interval"] = pd.arrays.IntervalArray.from_breaks(np.arange(7.0))
    X["unknown"] = pd.period_range("2020", periods=6, freq="M")
    before = X.copy(deep=True)
    groups = utils.create_dtype_dict(X, "dtypes")
    assert groups == {
        "cat": [str(dtype) for dtype in CATEGORICAL_DTYPES] + ["numeric_category"],
        "num": ["integer", "float", "delta"],
        "time": ["time", "delta"],
        "num_interval": ["interval"],
        "unk": ["unknown"],
    }
    assert utils.create_dtype_dict(X) == {
        col: kind for kind, cols in groups.items() for col in cols
    }
    assert_frame_equal(X, before)


@pytest.mark.parametrize("dtype", CATEGORICAL_DTYPES)
def test_cardinality_counts_all_categorical_dtypes(dtype):
    X = categorical_series(dtype).to_frame()
    X["num"] = np.arange(len(X))
    before = X.copy(deep=True)
    selector = CardinalityThreshold(2).fit(X)
    assert selector.statistic_series_.to_dict() == {"cat": 2, "num": 0}
    assert_array_equal(selector.selected_features_, ["num"])
    assert_frame_equal(X, before)


@pytest.mark.parametrize("dtype", [object, "category"])
@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, "1", 1, "1"], [0, 1, 0, 1]),
        ([np.nan, "nan", None, "None", pd.NA], [-1, 0, -1, 1, -1]),
        (["z", "a", "z", "b"], [0, 1, 0, 2]),
    ],
)
def test_encoding_preserves_original_identity_and_missingness(dtype, values, expected):
    X = pd.DataFrame({"cat": pd.Series(values, dtype=dtype)})
    X.index = np.arange(len(X)) * 3 + 10
    X["num"] = np.arange(len(X), dtype=float)
    numeric = X.num.copy()
    encoded, names, positions = utils.get_pandas_cat_codes(X)
    assert encoded is X
    assert names == ["cat"] and positions == [0]
    assert_array_equal(X.cat, expected)
    assert_series_equal(X.num, numeric)
    assert_array_equal(X.index, numeric.index)


@pytest.mark.parametrize("dtype", [object, "category", "string", None, "boolean"])
def test_encoding_nullable_dtypes_and_unused_category_levels(dtype):
    values = ["nan", None, "None", "nan", pd.NA]
    if dtype == "boolean":
        values = [True, None, False, True, pd.NA]
    X = pd.Series(values, dtype=dtype, name="cat").to_frame()
    if dtype == "category":
        X["cat"] = X.cat.cat.add_categories(["unused"])
    encoded, names, positions = utils.get_pandas_cat_codes(X)
    assert names == ["cat"] and positions == [0]
    assert_array_equal(encoded.cat, [0, -1, 1, 0, -1])


def test_encoder_time_unknown_passthrough_and_no_category_metadata():
    X = pd.DataFrame(
        {
            "num": [3.0, 2.0, 1.0],
            "time": pd.to_datetime(["2020-02-01", None, "2020-02-01"]),
            "unknown": pd.PeriodIndex(["2020-02", None, "2020-01"], freq="M"),
            "interval": pd.arrays.IntervalArray.from_breaks([0.0, 1.0, 2.0, 3.0]),
        }
    )
    passthrough = X[["num", "interval"]].copy()
    encoded, names, positions = utils.get_pandas_cat_codes(X)
    assert encoded is X
    assert names == ["time", "unknown"] and positions == [1, 2]
    assert_array_equal(X.time, [0, -1, 0])
    assert_array_equal(X.unknown, [0, -1, 1])
    assert_frame_equal(X[["num", "interval"]], passthrough)
    encoded, names, positions = utils.get_pandas_cat_codes(passthrough)
    assert encoded is passthrough and names is None and positions is None


@pytest.mark.parametrize("dtype", CATEGORICAL_DTYPES)
@pytest.mark.parametrize("jobs", [1, 2])
def test_series_matrix_and_workers_agree_on_category_roles(dtype, jobs):
    X = categorical_series(dtype).to_frame()
    X["cat2"] = categorical_series(dtype).iloc[::-1].to_numpy()
    X["cat2"] = X.cat2.astype(X.cat.dtype)
    X["num"] = [0.0, 2.0, 1.0, 4.0, 3.0, 5.0]
    X["num2"] = [2.0, 0.0, 3.0, 1.0, 4.0, 5.0]
    before = X.copy(deep=True)
    weights = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0]
    matrix = a.association_matrix(X, sample_weight=weights, n_jobs=jobs)
    assert_frame_equal(
        matrix.sort_values(["row", "col"]).reset_index(drop=True),
        a.association_matrix(X, weights, n_jobs=1)
        .sort_values(["row", "col"])
        .reset_index(drop=True),
    )
    for target in X:
        result = a.association_series(X, target, sample_weight=weights, n_jobs=jobs)
        result = result.drop(target)
        expected = matrix.loc[matrix.col == target].set_index("row").val
        assert set(result.index) == set(X.columns) - {target}
        assert_allclose(result, expected.reindex(result.index))
    assert_frame_equal(X, before)


@pytest.mark.parametrize("dtype", [bool, "boolean"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_boolean_series_uses_nominal_statistic(dtype, jobs):
    X = pd.DataFrame(
        {"b1": [True, True, False, False], "b2": [True, False, False, True]},
        dtype=dtype,
    )
    assert_allclose(a.association_series(X, "b1", n_jobs=jobs)["b2"], 0.0)


@pytest.mark.parametrize("dtype", CATEGORICAL_DTYPES)
def test_correlation_ratio_accepts_shared_dtypes_in_either_orientation(dtype):
    x = categorical_series(dtype)
    y = pd.Series([0.0, 2.0, 1.0, 4.0, 3.0, 5.0], name="num")
    expected = a.correlation_ratio(x.astype(object), y)
    assert_allclose(a.correlation_ratio(x, y), expected)
    assert_allclose(a.correlation_ratio(y, x), expected)


@pytest.mark.parametrize("dtype", [object, "category"])
@pytest.mark.parametrize("statistic", [a.correlation_ratio, a.f_cat_regression])
def test_mixed_category_values_group_without_sorting(dtype, statistic):
    x = pd.Series([1, 1, "1", "1"], dtype=dtype, name="cat")
    y = pd.Series([0.0, 2.0, 4.0, 5.0], name="num")
    reference = pd.Series(["a", "a", "b", "b"], name="cat", dtype=object)
    assert_allclose(statistic(x, y), statistic(reference, y))
    assert_allclose(
        a.f_cont_classification(y, x), a.f_cont_classification(y, reference)
    )


@pytest.mark.parametrize("caller", ["native", "shap", "pimp", "boost", "groot"])
def test_shared_encoder_callers_receive_distinct_codes(caller, monkeypatch):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "mixed": pd.Series([1, "1"] * 40, dtype=object),
            "text": pd.Series(["nan", "None"] * 40, dtype="string"),
            "noise": rng.normal(size=80),
        }
    )
    y = pd.Series(np.tile([0.0, 5.0], 40) + rng.normal(scale=0.2, size=80))
    calls = []
    encode = utils.get_pandas_cat_codes

    def checked_encode(frame):
        out, names, positions = encode(frame)
        if names:
            assert names == ["mixed", "text"] and positions == [0, 1]
            assert_array_equal(out.mixed, np.tile([0, 1], 40))
            assert_array_equal(out.text, np.tile([0, 1], 40))
        calls.append(names)
        return out, names, positions

    monkeypatch.setattr(ar, "get_pandas_cat_codes", checked_encode)
    model = lgb.LGBMRegressor(
        n_estimators=5, num_leaves=3, min_child_samples=5, verbosity=-1, n_jobs=1
    )
    if caller == "boost":
        ar.BoostAGroota(
            model, iters=1, max_rounds=1, importance="native", silent=True
        ).fit(X, y)
    elif caller == "groot":
        ar.GrootCV(
            objective="l2",
            n_folds=2,
            n_iter=1,
            n_jobs=1,
            lgbm_params={"num_iterations": 5, "num_leaves": 3},
        ).fit(X, y)
    else:
        fn = {"native": ar._get_imp, "shap": ar._get_shap_imp, "pimp": ar._get_perm_imp}
        result = fn[caller](model, X, y, sample_weight=None)
        assert result.shape == (3,) and np.isfinite(result).all()
    assert calls == [["mixed", "text"]]


@pytest.mark.parametrize("dtype", ["category", object])
def test_interval_categories_with_infinite_endpoints_are_values_not_infinities(dtype):
    intervals = pd.Series([pd.Interval(-np.inf, 0), pd.Interval(0, np.inf)] * 3, dtype=dtype)
    assert_series_equal(utils.validate_target(intervals, len(intervals)), intervals)
    frame = pd.DataFrame({"bins": intervals, "target": pd.Categorical([0, 1] * 3)})
    assert_allclose(a.theils_u_series(frame, "target", n_jobs=1)["bins"], 1.0)
    for invalid in [[0, np.inf], [0, -np.inf], [0, np.nan]]:
        with pytest.raises(ValueError, match="finite"):
            utils.validate_target(pd.Series(invalid, dtype=dtype), 2)
