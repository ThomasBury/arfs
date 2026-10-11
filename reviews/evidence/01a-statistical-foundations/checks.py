"""Unit 1A reference checks; run with pytest against the isolated frozen wheel.

Strict xfails document defects, not accepted behavior. An XPASS means re-review
the finding. No installed ARFS code is patched by this file.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from sklearn.feature_selection import f_regression

import arfs.association as a
from arfs.feature_selection import CollinearityThreshold
from arfs.feature_selection import MinRedundancyMaxRelevance as MRMR


def numeric_pair():
    return pd.DataFrame({"x": [0., 1., 3., 4.], "y": [1., 3., 2., 5.]})


def mixed_data():
    X = pd.DataFrame({"num": [0., 2., 1., 4., 3., 5.],
                      "cat": ["a", "a", "b", "b", "c", "c"]})
    return X, pd.Series([0., 1., 2., 3., 4., 6.], name="label")


def grouped_data():
    return (pd.Series(["a", "a", "b", "b"], name="group"),
            pd.Series([1., 2., 4., 6.], name="value"),
            np.array([1., 2., 1., 2.]))


def constant_categories():
    return (pd.DataFrame({"constant": ["a"] * 8,
                          "signal": ["a"] * 4 + ["b"] * 4,
                          "noise": ["a", "b"] * 4}),
            pd.Series(["a", "a", "a", "a", "b", "b", "b", "a"], name="label"))


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_unweighted_correlation_reference(method):
    X = numeric_pair()
    reference = getattr(stats, "pearsonr" if method == "pearson" else "spearmanr")
    assert_allclose(a.weighted_corr(X.x, X.y, method=method), reference(X.x, X.y).statistic)


def test_weighted_pearson_frequency_reference():
    X = numeric_pair()
    w = np.array([1, 2, 3, 4])
    reference = stats.pearsonr(np.repeat(X.x, w), np.repeat(X.y, w)).statistic
    assert_allclose(a.weighted_corr(X.x, X.y, w, method="pearson"), reference)
    assert_allclose(a.weighted_corr(X.x, X.y, w / w.sum(), method="pearson"), reference)


def test_weighted_ranks_hand_reference():
    # wCorr convention: weight below + (tie count + 1) * mean tie weight / 2.
    assert_allclose(a.wrank(np.array([1, 1, 2, 2]), np.array([1., 1., 2., 4.])),
                    [1.5, 1.5, 6.5, 6.5])


def test_theils_direction_hand_reference():
    coarse = pd.Series(["a", "a", "b", "b"], name="coarse")
    fine = pd.Series(["c", "d", "e", "f"], name="fine")
    assert_allclose(a.weighted_conditional_entropy(fine, coarse), np.log(2))
    assert_allclose(a.weighted_theils_u(coarse, fine), 1.)
    assert_allclose(a.weighted_theils_u(fine, coarse), .5)


def test_theils_weighted_frequency_reference():
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series(["c", "d", "c", "d"])
    w = np.array([4., 1., 1., 4.])
    expected = 1 + (.8 * np.log(.8) + .2 * np.log(.2)) / np.log(2)
    assert_allclose(a.weighted_theils_u(x, y, w), expected)
    assert_allclose(a.weighted_theils_u(x, y, w / 10), expected)


def test_cramer_hand_reference():
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series(["c", "d", "c", "d"])
    assert_allclose(a.cramer_v(x, y, np.array([4., 1., 1., 4.])), np.sqrt(.28))


def test_eta_and_anova_frequency_reference():
    x, y, w = grouped_data()
    # Expanded group means: 5/3 and 16/3; between SS=121/6, within SS=10/3.
    assert_allclose(a.correlation_ratio(x, y, w), np.sqrt(121 / 141))
    assert_allclose(a.f_cat_regression(x, y, w), 24.2)
    expanded = [np.repeat(y[x == g], w[x == g].astype(int)) for g in ["a", "b"]]
    assert_allclose(a.f_cat_regression(x, y, w), stats.f_oneway(*expanded).statistic)
    assert_allclose(a.f_cont_classification(y, x, w), 24.2)


def test_parallel_mixed_matrix_and_series_agree():
    X, _ = mixed_data()
    X["other"] = [0., 1., 2., 3., 5., 4.]
    X["cat2"] = ["a", "b", "a", "b", "a", "b"]
    w = np.arange(1., 7.)
    one = a.association_matrix(X, w, n_jobs=1).set_index(["row", "col"]).val.sort_index()
    two = a.association_matrix(X, w, n_jobs=2).set_index(["row", "col"]).val.sort_index()
    pd.testing.assert_series_equal(one, two)
    for target in X:
        series = a.association_series(X, target, sample_weight=w)
        for other in X.columns.drop(target):
            assert_allclose(series[other], one.loc[(other, target)])


def test_complete_case_mask_preserves_weights_and_input():
    X = pd.DataFrame({"x": [0., 1., np.nan, 4.], "y": [1., 3., 2., 5.]}, index=[4, 7, 8, 9])
    original = X.copy(deep=True)
    w = pd.Series([1., 2., 3., 4.], index=X.index)
    actual = a.wcorr_series(X, "y", w, method="pearson")["x"]
    valid = X.notna().all(axis=1)
    reference = a.weighted_corr(X.loc[valid, "x"], X.loc[valid, "y"], w[valid], method="pearson")
    assert_allclose(actual, reference)
    pd.testing.assert_frame_equal(X, original)


def test_normalized_pearson_two_workers():
    X = numeric_pair()
    result = a.wcorr_matrix(X, np.ones(4) / 4, n_jobs=2, method="pearson")
    assert_allclose(result.val, stats.pearsonr(X.x, X.y).statistic)


@pytest.mark.xfail(strict=True, reason="F1: separate z-scores erase singleton predictor types")
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_mixed_singletons_can_be_selected(task):
    X, y = mixed_data()
    if task == "classification":
        y = pd.Series(["a", "b", "b", "a", "a", "b"], name="label")
    model = MRMR(1, task=task, show_progress=False).fit(X, y)
    assert len(model.selected_features_) == 1


@pytest.mark.xfail(strict=True, reason="F1: perfect equal relevances become NaN and disappear")
def test_equal_perfect_predictors_do_not_empty_selection():
    X = pd.DataFrame({"a": np.arange(6.), "b": np.arange(6.)})
    assert len(MRMR(1, show_progress=False).fit(X, pd.Series(np.arange(6.))).selected_features_) == 1


@pytest.mark.xfail(strict=True, reason="F1: negative relevance reverses the redundancy penalty")
def test_default_relevance_is_nonnegative():
    X, y = mixed_data()
    X["num2"] = [1., 2., 5., 3., 4., 0.]
    result = a.f_stat_regression_parallel(X, y, n_jobs=1)
    assert result.notna().all() and (result >= 0).all(), result


@pytest.mark.xfail(strict=True, reason="F2: min-max normalization erases perfect negative redundancy")
def test_mrmr_does_not_prefer_exact_negative_duplicate():
    X = pd.DataFrame({"a": [1., 2., 3., 4., 5.], "neg": [-1., -2., -3., -4., -5.],
                      "b": [1., 3., 2., 5., 4.]})
    model = MRMR(2, relevance_func=lambda **kw: pd.Series({"a": 3., "neg": 2., "b": 1.}),
                 show_progress=False).fit(X, pd.Series([1., 3., 2., 4., 5.]))
    assert model.selected_features_ == ["a", "b"], model.selected_features_


@pytest.mark.xfail(strict=True, reason="F3: linear regression F uses Spearman instead of Pearson")
def test_numeric_f_linear_reference():
    X = pd.DataFrame({"x": [0., 1., 2., 3., 4.]})
    y = pd.Series([0., 1., 4., 9., 16.], name="label")
    assert_allclose(a.f_cont_regression_parallel(X, y, n_jobs=1)["x"], f_regression(X, y)[0][0])


@pytest.mark.xfail(strict=True, reason="F3: regression degrees of freedom count dropped rows")
def test_numeric_f_drop_equals_manual_filter():
    X = pd.DataFrame({"x": [0., 1., 2., 3., 4., np.nan]})
    y = pd.Series([0., 1., 3., 2., 4., 5.], name="label")
    actual = a.f_cont_regression_parallel(X, y, n_jobs=1)
    reference = a.f_cont_regression_parallel(X.iloc[:5], y.iloc[:5], n_jobs=1)
    assert_allclose(actual, reference)


@pytest.mark.xfail(strict=True, reason="F4: a constant categorical predictor gets maximum relevance")
def test_constant_categorical_relevance_zero():
    X, y = constant_categories()
    assert a.f_cat_classification_parallel(X, y, n_jobs=1)["constant"] == 0


@pytest.mark.xfail(strict=True, reason="F4: default MRMR can select the constant first")
def test_mrmr_rejects_constant_before_signal():
    X, y = constant_categories()
    assert MRMR(1, task="classification", show_progress=False).fit(X, y).selected_features_ == ["signal"]


@pytest.mark.xfail(strict=True, reason="F5: constant Theil U returns scalar despite as_frame=True")
def test_constant_categorical_matrix_and_collinearity():
    X, _ = constant_categories()
    matrix = a.association_matrix(X)
    assert set(matrix.columns) == {"row", "col", "val"}
    CollinearityThreshold().fit(X)


@pytest.mark.xfail(strict=True, reason="F6: sum(weights)-1 breaks Pearson with normalized weights")
def test_normalized_pearson_matrix_matches_pair():
    X = numeric_pair()
    w = np.ones(4) / 4
    expected = a.weighted_corr(X.x, X.y, w, method="pearson")
    result = a.wcorr_matrix(X, w, n_jobs=1, method="pearson")
    assert_allclose(result.loc[(result.row == "y") & (result.col == "x"), "val"], [expected])


@pytest.mark.xfail(strict=True, reason="F7: grouped F can be negative for positive fractional weights")
def test_fractional_weights_do_not_return_negative_f():
    x, y, w = grouped_data()
    # Either a supported nonnegative statistic or a clear rejected weight contract.
    try:
        result = a.f_cat_regression(x, y, w / w.sum())
    except ValueError:
        return
    assert result >= 0, result


@pytest.mark.xfail(strict=True, reason="F7: numeric relevance violates frequency replication unlike ANOVA")
def test_numeric_f_frequency_replication():
    X = numeric_pair()[["x"]]
    y = numeric_pair().y.rename("label")
    actual = a.f_cont_regression_parallel(X, y, np.full(4, 2.), n_jobs=1)["x"]
    expanded_X = X.loc[X.index.repeat(2)].reset_index(drop=True)
    expanded_y = y.loc[y.index.repeat(2)].reset_index(drop=True)
    expected = a.f_cont_regression_parallel(expanded_X, expanded_y, n_jobs=1)["x"]
    assert_allclose(actual, expected)


@pytest.mark.xfail(strict=True, reason="F8: input boundary accepts invalid weights")
@pytest.mark.parametrize("weights", [np.zeros(4), np.array([1., 1., 1., -.9])], ids=["zero", "negative"])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        a.association_series(numeric_pair(), "y", sample_weight=weights)


@pytest.mark.xfail(strict=True, reason="F8: documented array-like list weights cannot be masked")
def test_list_weights_supported():
    result = a.association_series(numeric_pair(), "y", sample_weight=[1., 1., 1., 1.])
    assert result.notna().all()


@pytest.mark.xfail(strict=True, reason="F9: a zero-weight category changes grouped statistics")
@pytest.mark.parametrize("statistic", [a.correlation_ratio, a.f_cat_regression, a.cramer_v])
def test_zero_weight_category_equivalent_to_removal(statistic):
    x = pd.Series(["a", "a", "b", "b", "z"], name="x")
    y = pd.Series([1., 2., 4., 6., 100.], name="y")
    if statistic is a.cramer_v:
        y = pd.Series(["x", "y", "x", "y", "q"], name="y")
    weights = np.array([1., 1., 1., 1., 0.])
    assert_allclose(statistic(x, y, weights), statistic(x.iloc[:4], y.iloc[:4], weights[:4]))


@pytest.mark.xfail(strict=True, reason="F10: no complete rows are retained and reported as zero association")
def test_no_complete_rows_rejected():
    X = pd.DataFrame({"x": [1., np.nan, 3.], "y": [np.nan, 2., np.nan]})
    with pytest.raises(ValueError):
        a.association_series(X, "y")


@pytest.mark.xfail(strict=True, reason="F10: inner series helper ignores handle_na=None")
def test_handle_na_none_passes_target_missingness_to_callable():
    X = pd.DataFrame({"x": [0., 1., 2., 3.], "y": [0., 1., np.nan, 2.]})
    seen = []
    def capture(x, y, sample_weight, as_frame):
        seen.append(y.isna().any())
        return .5
    a.association_series(X, "y", handle_na=None, num_num_assoc=capture)
    assert any(seen), seen


@pytest.mark.xfail(strict=True, reason="F11: subtracting large uncentered squares destroys variation")
@pytest.mark.parametrize("statistic", [a.correlation_ratio, a.f_cat_regression])
def test_grouped_statistics_translation_invariant(statistic):
    x, y, w = grouped_data()
    assert_allclose(statistic(x, y + 1e10, w), statistic(x, y, w), rtol=1e-5)


@pytest.mark.xfail(strict=True, reason="F12: temporary target column overwrites a real predictor")
def test_feature_named_target_retained():
    X = pd.DataFrame({"target": [0., 1., 2., 3., 4., 5.],
                      "other": [0., 2., 1., 5., 3., 4.], "third": [0., 4., 2., 1., 3., 5.]})
    y = pd.Series([0., 2., 1., 3., 6., 4.], name="label")
    model = MRMR(3, show_progress=False).fit(X, y)
    assert set(model.selected_features_) == set(X.columns), model.selected_features_


@pytest.mark.xfail(strict=True, reason="F13: boolean routing disagrees with shared dtype classifier")
def test_boolean_series_matches_categorical_matrix():
    X = pd.DataFrame({"b1": [True, True, False, False], "b2": [True, False, False, True]})
    assert_allclose(a.association_series(X, "b1")["b2"], 0.)


@pytest.mark.parametrize("seed", range(5))
def test_seeded_numeric_signal_screen(seed):
    # A screen, not evidence of false discovery control or held-out usefulness.
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=["signal", "n1", "n2", "n3", "n4", "n5"])
    y = pd.Series(3 * X.signal + rng.normal(size=len(X)), name="label")
    assert MRMR(1, show_progress=False).fit(X, y).selected_features_ == ["signal"]


def test_noise_only_mrmr_still_selects_requested_count():
    rng = np.random.default_rng(20261011)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=list("abcdef"))
    y = pd.Series(rng.normal(size=len(X)), name="label")
    assert len(MRMR(2, show_progress=False).fit(X, y).selected_features_) == 2
