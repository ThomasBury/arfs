"""R28: original labels, deterministic results and bounded useful parallel work."""

from functools import partial
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from arfs import association as a
from arfs import parallel as p


def frame(columns=("a", "b", "c")):
    return pd.DataFrame(
        [[0.0, 3.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 1.0], [4.0, 0.0, 3.0]],
        columns=columns,
        index=[8, 8, 20, 50],
    )


def sum_columns(X, y, sample_weight):
    return X.mul(sample_weight, axis=0).sum()


def broken_series(X, y, sample_weight):
    raise RuntimeError("intentional worker failure")


def broken_matrix(X, sample_weight, comb_list):
    raise RuntimeError("intentional worker failure")


LABELS = [
    [0, "b", "c"],
    [0, 1, 2],
    pd.Index([("a", 1), ("b", 2), ("c", 3)], tupleize_cols=False),
    pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)], names=["name", "id"]),
]


@pytest.mark.parametrize("columns", LABELS)
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_matrix_labels_order_weights_and_input_survive_workers(columns, method):
    X = frame(columns)
    before = X.copy(deep=True)
    weights = np.array([1.0, 2.0, 1.0, 3.0])
    one = a.wcorr_matrix(X, weights, n_jobs=1, method=method)
    two = a.wcorr_matrix(X, weights, n_jobs=2, method=method)
    assert_frame_equal(one, two)
    assert one.val.is_monotonic_decreasing
    assert list(one.index) == list(range(len(one)))
    assert set(zip(one.row, one.col)) == {
        (left, right) for left in X.columns for right in X.columns if left != right
    }
    for left, right, value in one.itertuples(index=False, name=None):
        assert value == pytest.approx(
            a.weighted_corr(X[left], X[right], weights, method=method)
        )
    assert_frame_equal(X, before)


@pytest.mark.parametrize("columns", LABELS)
def test_series_preserves_label_index_and_stable_ties(columns):
    X = frame(columns)
    X.iloc[:, :] = np.arange(len(X))[:, None]
    weights = np.ones(len(X))
    expected = X.sum()
    for jobs in [1, 2, -1, -2, -100, np.int64(2)]:
        result = p.parallel_df(sum_columns, X, X.iloc[:, 0], weights, n_jobs=jobs)
        assert_series_equal(result, expected)


@pytest.mark.parametrize("jobs", [1, 2, -1, -2, -100])
def test_matrix_ties_keep_pair_and_callback_order(jobs):
    X = frame()
    X.iloc[:, :] = np.arange(len(X))[:, None]
    pairs = list(combinations(X.columns, 2))
    compute = partial(p._compute_matrix_entries, func_xyw=a.weighted_corr)
    result = p.parallel_matrix_entries(compute, X, pairs, n_jobs=jobs)
    assert list(zip(result.row, result.col)) == [
        pair for left, right in pairs for pair in [(left, right), (right, left)]
    ]
    assert list(result.index) == list(range(6))


@pytest.mark.parametrize("constant", [False, True])
def test_native_pearson_keeps_pair_ties_and_constant_scores(constant):
    X = frame()
    X.iloc[:, :] = np.arange(len(X))[:, None]
    if constant:
        X["c"] = 1.0
    one = a.wcorr_matrix(X, n_jobs=1, method="pearson")
    two = a.wcorr_matrix(X, n_jobs=2, method="pearson")
    assert_frame_equal(one, two)
    if constant:
        assert one.loc[(one.row == "c") | (one.col == "c"), "val"].isna().all()


@pytest.mark.parametrize("jobs", [1, 2, -1, -2])
def test_empty_work_does_not_call_callback_or_dispatch(monkeypatch, jobs):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty work must not dispatch")

    monkeypatch.setattr(p, "Parallel", unexpected)
    X = frame()
    result = p.parallel_df(unexpected, X.iloc[:, :0], X.iloc[:, 0], n_jobs=jobs)
    assert_series_equal(result, pd.Series(dtype=float))
    result = p.parallel_matrix_entries(unexpected, X, [], n_jobs=jobs)
    assert list(result.columns) == ["row", "col", "val"]
    assert result.empty and result.val.dtype == np.dtype(float)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("jobs", [1, 2, -1, -2])
def test_one_numeric_column_has_no_pairs(method, jobs):
    result = a.wcorr_matrix(frame().iloc[:, :1], n_jobs=jobs, method=method)
    assert result.empty
    assert list(result.columns) == ["row", "col", "val"]
    assert result.val.dtype == np.dtype(float)


def test_association_matrix_has_the_same_empty_schema():
    result = a.association_matrix(frame().iloc[:, :1], n_jobs=-2)
    assert result.empty
    assert list(result.columns) == ["row", "col", "val"]
    assert result.val.dtype == np.dtype(float)


@pytest.mark.parametrize("jobs", [1, 2])
def test_empty_callback_results_have_matrix_schema(jobs):
    def no_results(X, sample_weight, comb_list):
        return []

    result = p.parallel_matrix_entries(
        no_results, frame(), [("a", "b"), ("a", "c")], n_jobs=jobs
    )
    assert result.empty and list(result.columns) == ["row", "col", "val"]
    assert result.val.dtype == np.dtype(float)


@pytest.mark.parametrize("jobs,expected", [(-1, 8), (-2, 7), (-100, 1), (2, 2)])
def test_native_negative_count_semantics(monkeypatch, jobs, expected):
    monkeypatch.setattr(joblib._parallel_backends, "cpu_count", lambda: 8)
    assert p._effective_n_jobs(jobs, 20) == expected


def test_native_cpu_limit_is_respected(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    assert p._effective_n_jobs(-1, 20) == 1


def test_single_pair_never_starts_parallel_workers(monkeypatch):
    monkeypatch.setattr(p, "Parallel", broken_matrix)
    result = a.wcorr_matrix(frame().iloc[:, :2], n_jobs=100)
    assert len(result) == 2


@pytest.mark.parametrize("jobs", [0, True, False, np.bool_(True), 1.5, "2", None])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_worker_counts_fail_before_callbacks(jobs, empty):
    X = frame()
    with pytest.raises(ValueError, match="n_jobs.*nonzero integer"):
        p.parallel_df(broken_series, X.iloc[:, :0] if empty else X, X.a, n_jobs=jobs)
    with pytest.raises(ValueError, match="n_jobs.*nonzero integer"):
        p.parallel_matrix_entries(
            broken_matrix, X, [] if empty else [("a", "b")], n_jobs=jobs
        )


@pytest.mark.parametrize("jobs", [1, 2])
def test_callback_errors_propagate(jobs):
    X = frame()
    with pytest.raises(RuntimeError, match="intentional worker failure"):
        p.parallel_df(broken_series, X, X.a, n_jobs=jobs)
    with pytest.raises(RuntimeError, match="intentional worker failure"):
        p.parallel_matrix_entries(
            broken_matrix, X, [("a", "b"), ("a", "c")], n_jobs=jobs
        )


@pytest.mark.parametrize("jobs", [-1, -2, 100])
@pytest.mark.parametrize("matrix", [False, True])
def test_worker_count_is_capped_to_nonempty_work(monkeypatch, jobs, matrix):
    seen = {}
    native_effective = p.effective_n_jobs

    def effective(count):
        seen["requested"] = count
        return native_effective(count)

    class InlineProbe:
        def __init__(self, n_jobs):
            seen["jobs"] = n_jobs

        def __call__(self, tasks):
            tasks = list(tasks)
            seen["sizes"] = [
                len(kwargs["comb_list"]) if matrix else args[0].shape[1]
                for _, args, kwargs in tasks
            ]
            return [func(*args, **kwargs) for func, args, kwargs in tasks]

    monkeypatch.setattr(joblib._parallel_backends, "cpu_count", lambda: 8)
    monkeypatch.setattr(p, "effective_n_jobs", effective)
    monkeypatch.setattr(p, "Parallel", InlineProbe)
    X = frame()
    if matrix:
        compute = partial(p._compute_matrix_entries, func_xyw=a.weighted_corr)
        p.parallel_matrix_entries(
            compute, X, list(combinations(X.columns, 2)), n_jobs=jobs
        )
    else:
        p.parallel_df(sum_columns, X, X.a, np.ones(len(X)), n_jobs=jobs)
    assert seen == {"requested": jobs, "jobs": 3, "sizes": [1, 1, 1]}


@pytest.mark.parametrize("backend", ["threading", "sequential"])
def test_active_joblib_backend_is_respected(backend):
    X = frame([0, "b", "c"])
    with joblib.parallel_config(backend=backend):
        result = a.wcorr_matrix(X, n_jobs=-2)
    assert_frame_equal(result, a.wcorr_matrix(X, n_jobs=1))


@pytest.mark.parametrize("jobs", [-1, -2, 2])
def test_mixed_association_callers_match_serial(jobs):
    X = frame()
    X["group"] = pd.Series(["a", "a", "b", "b"], index=X.index, dtype=object)
    X["other"] = pd.Series(["a", "b", "a", "b"], index=X.index, dtype=object)
    weights = np.array([1.0, 2.0, 1.0, 3.0])
    for fn in [
        a.association_matrix,
        a.theils_u_matrix,
        a.cramer_v_matrix,
        a.correlation_ratio_matrix,
    ]:
        assert_frame_equal(fn(X, weights, n_jobs=jobs), fn(X, weights, n_jobs=1))
    for fn, target in [
        (a.association_series, "a"),
        (a.wcorr_series, "a"),
        (a.theils_u_series, "group"),
        (a.cramer_v_series, "group"),
        (a.correlation_ratio_series, "a"),
    ]:
        assert_series_equal(
            fn(X, target, sample_weight=weights, n_jobs=jobs),
            fn(X, target, sample_weight=weights, n_jobs=1),
        )
    y = pd.Series([0, 0, 1, 1], index=X.index)
    for fn in [a.f_cat_regression_parallel, a.f_cont_classification_parallel]:
        assert_series_equal(fn(X, y, weights, n_jobs=jobs), fn(X, y, weights, n_jobs=1))
