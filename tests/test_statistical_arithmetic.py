"""R08: independent arithmetic references and relevance boundary contracts."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy import stats
from sklearn.feature_selection import f_regression

from arfs import association as a
from arfs.feature_selection import CollinearityThreshold, MinRedundancyMaxRelevance


def mixed_data():
    return pd.DataFrame(
        {
            "target": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "target_": [0.0, 2.0, 1.0, 5.0, 3.0, 4.0],
            "group": np.array(["a", "a", "b", "b", "b", "a"], dtype=object),
            "constant": np.array(["c"] * 6, dtype=object),
        },
        index=[10, 11, 12, 13, 14, 15],
    )


@pytest.mark.parametrize("jobs", [1, 2])
def test_linear_f_uses_pearson_and_retained_rows(jobs):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.0, 1.0, 4.0, 9.0, 16.0])
    expected = f_regression(X, y)[0][0]
    assert_allclose(a.f_cont_regression_parallel(X, y, n_jobs=jobs)["x"], expected)
    extended = pd.concat([X, pd.DataFrame({"x": [np.nan]})], ignore_index=True)
    extended_y = pd.concat([y, pd.Series([99.0])], ignore_index=True)
    assert_allclose(
        a.f_cont_regression_parallel(extended, extended_y, n_jobs=jobs)["x"],
        expected,
    )


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize(
    "fn,classification",
    [
        (a.f_cont_regression_parallel, False),
        (a.f_cat_regression_parallel, False),
        (a.f_cont_classification_parallel, True),
        (a.f_cat_classification_parallel, True),
    ],
)
def test_relevance_frequency_replication_and_worker_agreement(fn, classification, jobs):
    X = mixed_data()
    y = pd.Series([0, 0, 1, 1, 0, 1], dtype="category", name="target")
    if not classification:
        y = pd.Series([1.0, 3.0, 2.0, 7.0, 4.0, 8.0], name="target")
    weights = np.array([1, 2, 3, 1, 2, 0])
    positions = np.repeat(np.arange(len(X)), weights)
    repeated_X = X.iloc[positions].reset_index(drop=True)
    repeated_y = y.iloc[positions].reset_index(drop=True)
    actual = fn(X, y, weights, n_jobs=jobs).sort_index()
    expected = fn(repeated_X, repeated_y, n_jobs=1).sort_index()
    assert_allclose(actual, expected)
    assert_series_equal(actual, fn(X, y, weights, n_jobs=1).sort_index())


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("name", [None, "target", "group", 0])
@pytest.mark.parametrize(
    "fn,classification",
    [
        (a.f_cont_regression_parallel, False),
        (a.f_cat_regression_parallel, False),
        (a.f_cont_classification_parallel, True),
        (a.f_cat_classification_parallel, True),
    ],
)
def test_response_name_cannot_overwrite_features(fn, classification, name, jobs):
    X = mixed_data()
    X[0] = [5.0, 3.0, 1.0, 2.0, 4.0, 0.0]
    y = pd.Series([0, 0, 1, 1, 0, 1], dtype="category", name=name)
    if not classification:
        y = pd.Series([1.0, 3.0, 2.0, 7.0, 4.0, 8.0], name=name)
    before, before_y = X.copy(deep=True), y.copy(deep=True)
    actual = fn(X, y, n_jobs=jobs)
    expected = fn(X, y.rename("unused_response"), n_jobs=1)
    assert_series_equal(
        actual.sort_index(key=lambda i: i.map(str)),
        expected.sort_index(key=lambda i: i.map(str)),
    )
    assert_frame_equal(X, before)
    assert_series_equal(y, before_y)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("scale", [1.0, 0.25, 1e-6, 1e6])
def test_pearson_matrix_relative_weights_match_pair(jobs, scale):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 2.0, 1.0, 3.0]})
    weights = np.array([1.0, 2.0, 1.0, 2.0]) * scale
    expected = a.weighted_corr(X.x, X.y, weights, method="pearson")
    result = a.wcorr_matrix(X, weights, n_jobs=jobs, method="pearson")
    off_diagonal = result.loc[result.row != result.col, "val"]
    assert len(off_diagonal) > 0
    assert_allclose(off_diagonal, expected)
    assert_allclose(
        expected,
        a.weighted_corr(X.x, X.y, method="pearson", sample_weight=weights / scale),
    )


@pytest.mark.parametrize("offset", [0.0, 1e10, -1e10])
def test_grouped_statistics_centered_independent_reference(offset):
    x = pd.Series(["a", "a", "b", "b", "ignored"])
    y = pd.Series([1.0, 2.0, 4.0, 6.0, 999.0]) + offset
    weights = np.array([1, 2, 1, 2, 0])
    # Hand sums: between=121/6, within=20/6, total=141/6.
    assert_allclose(a.correlation_ratio(x, y, weights), np.sqrt(121 / 141))
    expected = stats.f_oneway([1, 2, 2], [4, 6, 6]).statistic
    assert_allclose(a.f_cat_regression(x, y, weights), expected)
    assert_allclose(a.f_cont_classification(y, x, weights), expected)


@pytest.mark.parametrize("fn", [a.f_cat_regression, a.f_cont_classification])
@pytest.mark.parametrize("total", [1.0, 2.0])
def test_grouped_relevance_rejects_insufficient_frequency_mass(fn, total):
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series([1.0, 2.0, 4.0, 6.0])
    if fn is a.f_cont_classification:
        x, y = y, x
    with pytest.raises(ValueError, match="Frequency weight total"):
        fn(x, y, np.full(4, total / 4))


@pytest.mark.parametrize(
    "fn", [a.f_cont_regression_parallel, a.f_cat_classification_parallel]
)
@pytest.mark.parametrize("total", [1.0, 2.0])
def test_correlation_relevance_rejects_insufficient_frequency_mass(fn, total):
    X = mixed_data()
    y = pd.Series([0, 0, 1, 1, 0, 1], dtype="category")
    if fn is a.f_cont_regression_parallel:
        y = y.astype(float)
    with pytest.raises(ValueError, match="Frequency weight total"):
        fn(X, y, np.full(len(X), total / len(X)), n_jobs=1)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("force_finite", [False, True])
def test_numeric_constants_and_perfect_dependence(jobs, force_finite):
    X = pd.DataFrame({"perfect": [-1.0, 0.0, 1.0], "constant": [5.0] * 3})
    result = a.f_cont_regression_parallel(
        X, X.perfect, n_jobs=jobs, force_finite=force_finite
    )
    assert result["perfect"] == (np.finfo(float).max if force_finite else np.inf)
    assert result["constant"] == 0 if force_finite else np.isnan(result["constant"])
    result = a.f_cont_regression_parallel(
        X, np.ones(3), n_jobs=jobs, force_finite=force_finite
    )
    assert (result == 0).all() if force_finite else result.isna().all()


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("force_finite", [False, True])
def test_categorical_constants_have_zero_relevance(jobs, force_finite):
    X = pd.DataFrame(
        {
            "constant": ["a"] * 5,
            "weighted_constant": ["a"] * 4 + ["b"],
            "signal": ["a", "a", "b", "b", "c"],
        },
        dtype=object,
    )
    y = pd.Series([0, 1, 1, 1, 0], dtype="category")
    result = a.f_cat_classification_parallel(
        X, y, [1, 1, 1, 1, 0], n_jobs=jobs, force_finite=force_finite
    )
    assert result["constant"] == result["weighted_constant"] == 0
    assert result["signal"] > 0
    result = a.f_cat_classification_parallel(
        X, pd.Series([0] * 5, dtype="category"), n_jobs=jobs, force_finite=force_finite
    )
    assert (result == 0).all()


@pytest.mark.parametrize("jobs", [1, 2])
def test_constant_theils_u_frame_and_collinearity(jobs):
    X = pd.DataFrame(
        {"constant": ["a"] * 4, "varying": ["a", "b", "a", "b"]}, dtype=object
    )
    assert a.weighted_theils_u(X.constant, X.varying) == 1
    result = a.weighted_theils_u(X.constant, X.varying, as_frame=True)
    assert_frame_equal(
        result, pd.DataFrame({"row": ["constant"], "col": ["varying"], "val": [1.0]})
    )
    matrix = a.association_matrix(X, n_jobs=jobs)
    assert set(matrix.columns) == {"row", "col", "val"}
    CollinearityThreshold(n_jobs=jobs).fit(X)


def test_mrmr_feature_named_target_remains_eligible():
    X = mixed_data().drop(columns=["group", "constant"])
    X["third"] = [0.0, 4.0, 2.0, 1.0, 3.0, 5.0]
    y = pd.Series([0.0, 2.0, 1.0, 3.0, 6.0, 4.0])
    result = MinRedundancyMaxRelevance(3, show_progress=False).fit(X, y)
    assert set(result.selected_features_) == set(X.columns)


def test_constant_column_warning_does_not_claim_removal():
    X = pd.DataFrame({"constant": [1.0] * 4, "varying": [0.0, 1.0, 2.0, 3.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, _ = a._check_association_input(X)
    assert_frame_equal(result, X)
    assert not any("removed" in str(w.message) for w in caught)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("classification", [False, True])
def test_weighted_relevance_dropped_rows_equal_manual_filter(jobs, classification):
    X = mixed_data()
    X.loc[15, "target"] = np.nan
    y = pd.Series([0, 0, 1, 1, 0, 1], dtype="category")
    functions = [a.f_cont_classification_parallel, a.f_cat_classification_parallel]
    if not classification:
        y = pd.Series([1.0, 3.0, 2.0, 7.0, 4.0, 8.0])
        functions = [a.f_cont_regression_parallel, a.f_cat_regression_parallel]
    weights = np.array([1, 2, 3, 1, 2, 100])
    for fn in functions:
        assert_series_equal(
            fn(X, y, weights, n_jobs=jobs),
            fn(X.iloc[:5], y.iloc[:5], weights[:5], n_jobs=1),
        )


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_fractional_frequency_mass_is_not_automatically_normalized(scale):
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series([1.0, 2.0, 4.0, 6.0])
    weights = np.array([1, 2, 1, 2]) * scale
    # Rescaling leaves the ratio of sums of squares alone, but changes N-K.
    assert_allclose(a.f_cat_regression(x, y, weights), 6.05 * (6 * scale - 2))


@pytest.mark.parametrize("jobs", [1, 2])
def test_regression_constant_categories_on_retained_positive_mass_score_zero(jobs):
    X = pd.DataFrame(
        {"constant": ["a"] * 4 + ["b"], "varying": ["a", "a", "b", "b", "c"]},
        dtype=object,
    )
    result = a.f_cat_regression_parallel(
        X, [1, 2, 4, 6, 99], [1, 1, 1, 1, 0], n_jobs=jobs
    )
    assert result["constant"] == 0
    assert_allclose(result["varying"], 9.8)


def test_grouped_near_perfect_dependence_does_not_cancel_within_variation():
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series([-1e10, -1e10 + 1, 1e10, 1e10 + 1])
    # Hand sums: between=4e20, within=1; degrees of freedom are 1 and 2.
    assert_allclose(a.f_cat_regression(x, y), 8e20)


@pytest.mark.parametrize("reset_index", [False, True])
def test_matrix_to_xy_emits_only_observed_lower_triangle(reset_index):
    frame = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=["x", "y"], columns=["x", "y"])
    result = a.matrix_to_xy(frame, reset_index=reset_index)
    expected = pd.DataFrame(
        {"row": ["x", "y", "y"], "col": ["x", "x", "y"], "val": [1.0, 0.8, 1.0]}
    )
    if reset_index:
        assert_frame_equal(result, expected)
    else:
        assert_series_equal(
            result,
            expected.set_index(["row", "col"]).val.rename(None),
            check_names=False,
        )
