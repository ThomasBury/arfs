"""Review 05: independent small references and bounded defect reproductions.

Strict xfails identify open findings; --runxfail exposes the assertions/errors.
Control-flow probes replace expensive/random boundaries, never library source.
"""

from functools import partial
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.sparse import csr_matrix

import arfs.association as association
import arfs.parallel as parallel
import arfs.sampling as sampling


def defect(number, reason, **kwargs):
    return pytest.mark.xfail(strict=True, reason=f"F{number}: {reason}", **kwargs)


def reference_gower(X, Y=None, categorical=(), weights=None):
    """Scalar reference: pooled ranges, pairwise available features, constant=0.

    Pooled ranges match the implementation's actual scaling contract, allowing
    cross-matrix indexing to be checked independently of its misleading Notes.
    """
    Y = X if Y is None else Y
    values = pd.concat([X, Y], ignore_index=True)
    weights = np.ones(X.shape[1]) if weights is None else weights
    ranges = {c: float(values[c].max() - values[c].min())
              for c in X.columns if c not in categorical}
    result = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            numerator = denominator = 0.0
            for c, w in zip(X.columns, weights):
                a, b = X.iloc[i][c], Y.iloc[j][c]
                if pd.isna(a) or pd.isna(b) or w == 0:
                    continue
                distance = float(a != b) if c in categorical else (
                    abs(float(a) - float(b)) / ranges[c] if ranges[c] else 0.0)
                numerator += w * distance
                denominator += w
            result[i, j] = numerator / denominator if denominator else np.nan
    return result


def mixed():
    return pd.DataFrame({"x": [0., 5., 10.], "c": pd.Series(["a", "a", "b"], index=[8, 8, 90], dtype=object),
                         "constant": [7., 7., 7.]}, index=[8, 8, 90])


@pytest.mark.parametrize("mask", ["auto", ["c"], [False, True, False], [1]])
def test_mixed_gower_feature_weights_constants_and_indexes(mask):
    X = mixed()
    before = X.copy(deep=True)
    w = np.array([2., 1., 1.])
    result = sampling.gower_matrix(X, weight=w, cat_features=mask)
    expected = np.array([[0., .25, .75], [.25, 0., .5], [.75, .5, 0.]])
    assert_allclose(reference_gower(X, categorical=["c"], weights=w), expected)
    assert_allclose(result, expected)
    assert_allclose(result, result.T)
    assert_array_equal(np.diag(result), np.zeros(len(X)))
    assert_frame_equal(X, before)
    assert_allclose(sampling.gower_matrix(X, weight=11 * w, cat_features=mask), result)


@pytest.mark.parametrize("as_array", [False, True])
def test_rectangular_gower_positions(as_array):
    X = pd.DataFrame({"x": [0., 10.]}, index=[50, 50])
    Y = pd.DataFrame({"x": [1., 3., 8.]}, index=[8, 20, 90])
    expected = reference_gower(X, Y)
    args = (X.to_numpy(), Y.to_numpy()) if as_array else (X, Y)
    assert_allclose(sampling.gower_matrix(*args), expected)
    assert_allclose(sampling.gower_matrix(*args[::-1]), expected.T)


@defect(1, "maximum-based scaling gives negative/zero distances or loses distinct floats")
@pytest.mark.parametrize("values", [[-3., -1.], [-3., 0.], [1e9, 1e9 + 1.]])
def test_numeric_translation_invariance(values):
    X = pd.DataFrame({"x": values})
    assert_allclose(sampling.gower_matrix(X), [[0., 1.], [1., 0.]])


@defect(2, "equal row counts incorrectly trigger self-matrix symmetry")
def test_equal_size_cross_matrix():
    X = pd.DataFrame({"x": [0., 10.]})
    Y = pd.DataFrame({"x": [1., 3.]})
    assert_allclose(sampling.gower_matrix(X, Y), reference_gower(X, Y))


def test_actual_ranges_include_reference_frame():
    X = pd.DataFrame({"x": [0., 1.]})
    assert_allclose(sampling.gower_matrix(X, pd.DataFrame({"x": [10.]})), [[1.], [.9]])


@defect(3, "missing features are not excluded from each pair's numerator and denominator")
@pytest.mark.parametrize("kind", ["numeric", "categorical", "all-missing"])
def test_pairwise_missing_gower(kind):
    X = pd.DataFrame({"x": [0., 1., 2.], "c": pd.Series(["a", "a", "b"], dtype=object)})
    if kind == "numeric":
        X.loc[1, "x"] = np.nan
    elif kind == "categorical":
        X.loc[1, "c"] = np.nan
    else:
        X["x"] = np.nan
    assert_allclose(sampling.gower_matrix(X), reference_gower(X, categorical=["c"]), equal_nan=True)


@defect(4, "documented array integer/empty category masks access .columns")
@pytest.mark.parametrize("mask", [[1], []])
def test_array_category_index_masks(mask):
    X = np.array([[0., 0.], [1., 1.]])
    assert_allclose(sampling.gower_matrix(X, cat_features=mask), [[0., 1.], [1., 0.]])


def test_array_boolean_mask():
    assert_allclose(sampling.gower_matrix(np.array([[0., 0.], [1., 1.]]),
                                         cat_features=[False, True]), [[0., 1.], [1., 0.]])


@defect(4, "NumPy dtype inference cannot interpret pandas extension dtypes")
@pytest.mark.parametrize("dtype", ["category", "Int64", "string"])
def test_pandas_extension_dtype_auto(dtype):
    X = pd.DataFrame({"x": pd.Series([1, 2], dtype=dtype)})
    assert_allclose(sampling.gower_matrix(X), [[0., 1.], [1., 0.]])


@pytest.mark.xfail(int(pd.__version__.split('.')[0]) >= 3, strict=True,
                   reason="F4: pandas 3 infers a StringDtype that Gower auto detection rejects")
def test_inferred_string_gower_auto():
    X = pd.DataFrame({"x": ["a", "b"]})
    assert_allclose(sampling.gower_matrix(X), [[0., 1.], [1., 0.]])


@defect(4, "documented reference Series is unpacked as a two-dimensional frame")
def test_gower_reference_series():
    X = mixed()
    assert_allclose(sampling.gower_matrix(X, X.iloc[0]),
                    reference_gower(X, X.iloc[[0]], categorical=["c"]))


@defect(4, "sparse input is inspected as a frame before the deliberate rejection")
def test_sparse_has_documented_error():
    with pytest.raises(TypeError, match="Sparse"):
        sampling.gower_matrix(csr_matrix([[0., 1.], [1., 0.]]))


def test_reordered_frame_columns_are_rejected():
    X = mixed()
    with pytest.raises(TypeError, match="same columns"):
        sampling.gower_matrix(X, X.iloc[:, ::-1])


@defect(5, "feature weights have no nonnegative/finite/positive-mass validation")
@pytest.mark.parametrize("weights", [[-1., 2.], [0., 0.], [np.nan, 1.]])
def test_invalid_feature_weights_rejected(weights):
    with pytest.raises(ValueError):
        sampling.gower_matrix(pd.DataFrame({"x": [0., 1.], "y": [1., 1.]}), weight=np.array(weights))


def test_zero_feature_weight_omits_complete_feature():
    X = pd.DataFrame({"x": [0., 1.], "y": [1., 1.]})
    assert_allclose(sampling.gower_matrix(X, weight=np.array([0., 1.])), 0.)


@defect(6, "sample_weight is passed as column weights, rejecting ordinary row weights")
def test_gower_sample_accepts_row_weights():
    X = pd.DataFrame({"x": [1., 3., 9.], "y": [2., 6., 18.]})
    result = sampling.sample(X, n=1, sample_weight=np.array([1., 2., 3.]))
    assert len(result) == 1


def test_gower_square_frame_treats_row_weights_as_feature_weights(monkeypatch):
    seen = {}
    original = sampling.gower_matrix
    def capture(*args, **kwargs):
        seen["weights"] = kwargs["weight"]
        return original(*args, **kwargs)
    monkeypatch.setattr(sampling, "gower_matrix", capture)
    X = pd.DataFrame({"x": [0., 10.], "y": [0., 20.]})
    w = np.array([1., 9.])
    result = sampling.sample(X, n=1, sample_weight=w)
    assert_array_equal(seen["weights"], w)
    assert_allclose(result[["x", "y"]].iloc[0], [5., 10.])  # unweighted representatives


def test_gower_summary_contract_and_no_mutation():
    X = pd.DataFrame({"x": [0., 2., 10., 12.], "c": ["a", "a", "b", "b"]}, index=[8, 8, 90, 30])
    before = X.copy(deep=True)
    result = sampling.sample(X, n=2)
    assert_allclose(sorted(result.x), [1., 11.])
    assert set(result.c) == {"a", "b"}
    assert result.label.isna().all()  # recorded output defect, not an endorsed schema
    assert_frame_equal(X, before)
    assert not result.index.isin(X.index).any()


@defect(7, "numeric missing values are filled using only the first numeric column's mean")
def test_gower_sample_imputes_per_column():
    X = pd.DataFrame({"a": [0., 2., 4.], "b": [100., np.nan, 200.]})
    result = sampling.sample(X, n=1)
    assert_allclose(result.b, [150.])


@defect(8, "temporary label leaks into results or overwrites an existing feature")
@pytest.mark.parametrize("label", [None, "numeric", "category"])
def test_gower_sample_preserves_feature_schema(label):
    X = pd.DataFrame({"x": [0., 2., 10., 12.]})
    if label is not None:
        X["label"] = [0., 2., 10., 12.] if label == "numeric" else ["a", "a", "b", "b"]
    result = sampling.sample(X, n=1)
    assert list(result.columns) == list(X.columns)
    if label == "category":
        assert result.label.iloc[0] in {"a", "b"}


def test_gower_size_equal_and_above_available():
    X = pd.DataFrame({"x": [0., 1., 2.]})
    assert len(sampling.sample(X, n=len(X))) == len(X)
    with pytest.raises(ValueError):
        sampling.sample(X, n=len(X) + 1)


@pytest.mark.parametrize("rows", [0, 1])
def test_gower_tiny_data_rejected(rows):
    with pytest.raises(ValueError):
        sampling.sample(pd.DataFrame({"x": np.arange(rows, dtype=float)}), n=1)


@defect(8, "unknown sampling method constructs but never raises NotImplementedError")
def test_unknown_sampling_method_rejected():
    with pytest.raises((ValueError, NotImplementedError)):
        sampling.sample(mixed(), method="unknown")


@defect(9, "retry counter never advances; size never increases")
def test_isolation_retry_is_bounded_and_advances(monkeypatch):
    sizes = []
    class ReviewGuard(Exception):
        pass
    def reject(full, part):
        sizes.append(len(part))
        if len(sizes) == 45:
            raise ReviewGuard("bounded review stopped a nonadvancing sampler")
        return SimpleNamespace(pvalue=0.0)
    monkeypatch.setattr(sampling, "isolation_forest", lambda X, sample_weight: np.arange(len(X)))
    monkeypatch.setattr(sampling, "ks_2samp", reject)
    try:
        sampling.isof_find_sample(pd.DataFrame({"x": np.arange(100)}))
    except ReviewGuard:
        pass
    assert len(set(sizes)) > 1, f"45 rejected draws all used sizes {sizes}"


@defect(9, "rounding the five-percent step to zero breaks small datasets")
@pytest.mark.parametrize("length", [0, 1, 10])
def test_tiny_split_has_no_zero_division(length):
    if length == 0:
        with pytest.raises(ValueError):
            sampling.get_5_percent_splits(length)
    else:
        sizes = sampling.get_5_percent_splits(length)
        assert ((sizes > 0) & (sizes <= length)).all()


@pytest.mark.parametrize("length", [0, 1, 10])
def test_public_isoforest_tiny_data_actual_failure(length):
    with pytest.raises(ZeroDivisionError):
        sampling.sample(pd.DataFrame({"x": np.arange(length, dtype=float)}), method="isoforest")


def test_normal_split_and_first_acceptance(monkeypatch):
    assert_array_equal(sampling.get_5_percent_splits(100), np.arange(5, 100, 5))
    weights = np.arange(1., 101.)
    seen = {}
    def scores(X, sample_weight):
        seen["weights"] = sample_weight
        return np.ones(len(X))
    monkeypatch.setattr(sampling, "isolation_forest", scores)
    result = sampling.isof_find_sample(pd.DataFrame({"x": np.arange(100)}), weights)
    assert len(result) == len(set(result)) == 10  # actual initial size is 10%, not 5%
    assert ((result >= 0) & (result < 100)).all()
    assert_array_equal(seen["weights"], weights)


@defect(6, "public isolation sampler discards supplied sample weights")
def test_isoforest_sample_forwards_weights(monkeypatch):
    seen = {}
    def select(X, sample_weight):
        seen["weights"] = sample_weight
        return np.array([2, 0])
    monkeypatch.setattr(sampling, "isof_find_sample", select)
    X = mixed()
    w = np.array([1., 2., 3.])
    sampling.sample(X, method="isoforest", sample_weight=w)
    assert_array_equal(seen["weights"], w)


@defect(10, "isolation sample returns encoded categories instead of original observations")
def test_isoforest_sample_returns_original_rows(monkeypatch):
    monkeypatch.setattr(sampling, "isof_find_sample", lambda X, sample_weight: np.array([2, 0]))
    X = mixed()
    assert_frame_equal(sampling.sample(X, method="isoforest"), X.iloc[[2, 0]])


def test_isoforest_ignores_n_but_keeps_positions_and_input(monkeypatch):
    monkeypatch.setattr(sampling, "isof_find_sample", lambda X, sample_weight: np.array([2, 0]))
    X = mixed()
    before = X.copy(deep=True)
    a = sampling.sample(X, n=1, method="isoforest")
    b = sampling.sample(X, n=1000, method="isoforest")
    assert_frame_equal(a, b)
    assert_array_equal(a.index, X.index[[2, 0]])
    assert_frame_equal(X, before)


def test_real_isoforest_seed_replay_and_nonempty_sample():
    X = pd.DataFrame({"x": np.ones(40), "c": ["a"] * 40}, index=np.arange(40) % 3)
    before = X.copy(deep=True)
    state = np.random.get_state()
    try:
        np.random.seed(61)
        a = sampling.sample(X, method="isoforest")
        np.random.seed(61)
        b = sampling.sample(X, method="isoforest")
    finally:
        np.random.set_state(state)
    assert len(a) == 4
    assert_frame_equal(a, b)
    assert_frame_equal(X, before)


def test_smallest_indices_ties_and_self_neighbors():
    values = np.array([.2, .1, .1, .9])
    result = sampling.smallest_indices(values, 2)
    assert set(result["index"]) == {1, 2}
    assert_allclose(result["values"], [.1, .1])
    X = pd.DataFrame({"x": [0., 2., 10.]}, index=[50, 50, 90])
    result = sampling.gower_topn(X, X.iloc[[1]], n=3)
    assert_array_equal(result["index"], [1, 0, 2])
    assert_allclose(result["values"], [0., .2, .8])


@defect(11, "key path ignores data_y and measures from the first data_x row")
def test_topn_key_uses_query_and_excludes_query_identity():
    X = pd.DataFrame({"id": ["a", "b", "c"], "x": [0., 5., 10.]}, index=[20, 20, 90])
    query = X.iloc[[2]]
    result = sampling.gower_topn(X, query, key="id", n=2)
    assert_array_equal(result["index"], [1, 0])
    assert_allclose(result["values"], [.5, 1.])


@defect(11, "key path can overshoot available rows despite sufficient distinct IDs")
def test_topn_duplicate_ids_with_enough_candidates():
    X = pd.DataFrame({"id": ["a", "a", "a", "b", "c"], "x": [0., 0., 0., 5., 10.]})
    query = pd.DataFrame({"id": ["q"], "x": [9.]})
    result = sampling.gower_topn(X, query, key="id", n=3)
    assert_array_equal(result["index"][:2], [4, 3])
    assert len(result["index"]) == 3


@defect(12, "n=0 returns every entry; negative n silently returns a subset")
@pytest.mark.parametrize("n", [0, -1])
def test_smallest_indices_nonpositive_n(n):
    values = np.array([.3, .1, .2])
    if n < 0:
        with pytest.raises(ValueError):
            sampling.smallest_indices(values, n)
    else:
        assert sampling.smallest_indices(values, n)["index"].size == 0


def test_smallest_indices_count_limits_and_actual_nonfinite_policy():
    values = np.array([.3, .1, .2])
    assert_array_equal(sampling.smallest_indices(values, 3)["index"], [1, 2, 0])
    with pytest.raises(ValueError):
        sampling.smallest_indices(values, 4)
    result = sampling.smallest_indices(np.array([np.nan, np.inf, -np.inf, .5]), 2)
    assert_array_equal(result["index"], [2, 3])
    assert result["values"][0] == -np.finfo(float).max
    assert sampling.smallest_indices(np.array([]), 0)["index"].size == 0


@defect(12, "optional data_y is unconditionally dereferenced")
def test_topn_optional_reference():
    result = sampling.gower_topn(pd.DataFrame({"x": [0., 1.]}), n=1)
    assert len(result["index"]) == 1


def test_topn_rejects_multiple_queries_and_too_many_neighbors():
    X = pd.DataFrame({"x": [0., 1.]})
    with pytest.raises(TypeError, match="1 row"):
        sampling.gower_topn(X, X, n=1)
    with pytest.raises(ValueError):
        sampling.gower_topn(X, X.iloc[[0]], n=3)


def numeric_frame(labels=("a", "b", "c")):
    return pd.DataFrame([[0., 4., 0.], [1., 2., 4.], [2., 3., 1.], [3., 0., 5.]], columns=labels, index=[8, 8, 30, 90])


def tidy(result):
    return result.sort_values(["row", "col"]).reset_index(drop=True)


@pytest.mark.parametrize("weighted", [False, True])
def test_parallel_real_matrix_and_series_agree(weighted):
    X = numeric_frame()
    X["group"] = pd.Series(["a", "a", "b", "b"], index=X.index, dtype=object)
    before = X.copy(deep=True)
    w = np.array([1., 2., 1., 3.]) if weighted else None
    one = association.association_matrix(X, sample_weight=w, n_jobs=1)
    two = association.association_matrix(X, sample_weight=w, n_jobs=2)
    assert_frame_equal(tidy(one), tidy(two))
    for fn in [association.wcorr_series, association.association_series]:
        assert_series_equal(fn(X, "a", sample_weight=w, n_jobs=1),
                            fn(X, "a", sample_weight=w, n_jobs=2))
    assert_frame_equal(X, before)


@pytest.mark.xfail(np.__version__ == "2.5.4", strict=True,
                   reason="C2: saved newer NumPy rejects scalar nonzero in shared ANOVA")
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_parallel_relevance_agrees(weighted, task):
    X = numeric_frame()
    X["group"] = pd.Series(["a", "a", "b", "b"], index=X.index, dtype=object)
    w = np.array([1., 2., 1., 3.]) if weighted else None
    y = pd.Series([1., 3., 2., 7.], index=X.index, name="response")
    fn = association.f_cat_regression_parallel
    if task == "classification":
        y = pd.Series([0, 0, 1, 1], index=X.index, name="response")
        fn = association.f_cont_classification_parallel
    assert_series_equal(fn(X, y, sample_weight=w, n_jobs=1), fn(X, y, sample_weight=w, n_jobs=2))


@pytest.mark.xfail(int(pd.__version__.split('.')[0]) >= 3, strict=True,
                   reason="C1: inferred strings pass dtype dispatch but fail correlation_ratio")
@pytest.mark.parametrize("jobs", [1, 2])
def test_inferred_strings_association_matrix(jobs):
    X = numeric_frame()
    X["group"] = ["a", "a", "b", "b"]
    result = association.association_matrix(X, n_jobs=jobs)
    assert set(result.row) == set(X.columns)


@defect(13, "NumPy splitting coerces mixed column labels to strings")
def test_parallel_mixed_column_labels():
    X = numeric_frame((0, "b", "c"))
    one = association.wcorr_matrix(X, n_jobs=1)
    two = association.wcorr_matrix(X, n_jobs=2)
    assert set(zip(one.row, one.col)) == set(zip(two.row, two.col))


@defect(13, "NumPy splitting expands tuple labels into arrays, breaking column lookup")
def test_parallel_tuple_column_labels():
    X = numeric_frame((("a", 1), ("b", 2), ("c", 3)))
    one = association.wcorr_matrix(X, n_jobs=1)
    two = association.wcorr_matrix(X, n_jobs=2)
    assert_frame_equal(tidy(one), tidy(two))


def test_parallel_matrix_order_is_worker_dependent():
    X = numeric_frame()
    one = association.wcorr_matrix(X, n_jobs=1)
    two = association.wcorr_matrix(X, n_jobs=2)
    assert_frame_equal(tidy(one), tidy(two))
    assert one.val.is_monotonic_decreasing
    assert not two.val.is_monotonic_decreasing


def sum_columns(X, y, sample_weight):
    w = np.ones(len(X)) if sample_weight is None else sample_weight
    return X.mul(w, axis=0).sum()


def broken_series(X, y, sample_weight):
    raise RuntimeError("intentional review worker failure")


def broken_matrix(X, sample_weight, comb_list):
    raise RuntimeError("intentional review worker failure")


@pytest.mark.parametrize("jobs", [1, 2])
def test_parallel_exceptions_propagate(jobs):
    X = numeric_frame()
    with pytest.raises(RuntimeError, match="intentional review worker failure"):
        parallel.parallel_df(broken_series, X, X.a, n_jobs=jobs)
    with pytest.raises(RuntimeError, match="intentional review worker failure"):
        parallel.parallel_matrix_entries(broken_matrix, X, [("a", "b")], n_jobs=jobs)


@pytest.mark.parametrize("jobs", [1, 2])
def test_parallel_empty_work_contract(jobs):
    X = numeric_frame()
    result = parallel.parallel_df(sum_columns, X.iloc[:, :0], X.a, n_jobs=jobs)
    assert result.empty
    compute = partial(parallel._compute_matrix_entries, func_xyw=association.weighted_corr)
    with pytest.raises(ValueError, match="No objects to concatenate"):
        parallel.parallel_matrix_entries(compute, X, [], n_jobs=jobs)
    with pytest.raises(ValueError, match="No objects to concatenate"):
        association.wcorr_matrix(X[["a"]], n_jobs=jobs, method="spearman")


@pytest.mark.parametrize("jobs", [0, -2])
def test_parallel_unsupported_worker_counts(jobs):
    X = numeric_frame()
    with pytest.raises(ValueError, match="number sections"):
        parallel.parallel_df(sum_columns, X, X.a, n_jobs=jobs)


def test_all_cores_schedules_empty_chunks(monkeypatch):
    seen = {}
    class InlineProbe:
        def __init__(self, n_jobs):
            seen["jobs"] = n_jobs
        def __call__(self, tasks):
            tasks = list(tasks)
            seen["columns"] = [len(args[0].columns) for _, args, _ in tasks]
            return [fn(*args, **kwargs) for fn, args, kwargs in tasks]
    monkeypatch.setattr(parallel, "cpu_count", lambda: 8)
    monkeypatch.setattr(parallel, "Parallel", InlineProbe)
    X = numeric_frame()
    result = parallel.parallel_df(sum_columns, X, X.a, n_jobs=-1)
    assert seen == {"jobs": 8, "columns": [1, 1, 1, 0, 0, 0, 0, 0]}
    assert_series_equal(result, X.sum().sort_values(ascending=False))


def test_more_workers_than_pairs_retains_values():
    X = numeric_frame().iloc[:, :2]
    assert_frame_equal(tidy(association.wcorr_matrix(X, n_jobs=1)),
                       tidy(association.wcorr_matrix(X, n_jobs=2)))


def test_explicit_threading_backend_matches_serial():
    X = numeric_frame()
    with joblib.parallel_config(backend="threading"):
        result = association.wcorr_matrix(X, n_jobs=2)
    assert_frame_equal(tidy(result), tidy(association.wcorr_matrix(X, n_jobs=1)))
