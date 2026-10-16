"""Independent references and structural guards for the prepared kernels."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from arfs import association as a


def test_zero_weight_ties_do_not_change_positive_ranks():
    x = [1, 1, 2, 3, 4]
    w = np.array([2.0, 0, 1, 0, 3])
    ranks = a.wrank(x, w)
    assert len(ranks) == len(x)
    assert_allclose(ranks[w > 0], a.wrank(np.array(x)[w > 0], w[w > 0]))
    assert ranks[3] == 3  # zero-only group gets preceding cumulative mass
    y = [4, 2, 1, 0, 3]
    assert_allclose(
        a.wspearman(x, y, w),
        a.wspearman(np.array(x)[w > 0], np.array(y)[w > 0], w[w > 0]),
    )


def test_nullable_pearson_is_float_in_scalar_series_and_matrix():
    X = pd.DataFrame(
        {
            "x": pd.Series([1, 2, 3, 4], dtype="Int64"),
            "y": pd.Series([2, 1, 4, 3], dtype="Float64"),
        }
    )
    expected = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)[0, 1]
    assert_allclose(a.weighted_corr(X.x, X.y, method="pearson"), expected)
    assert_allclose(a.wcorr_matrix(X, method="pearson").val, expected)
    assert_allclose(a.wcorr_series(X, "x", method="pearson")["y"], expected)
    square = a.weighted_correlation_1cpu(X)
    assert_allclose(square, [[1, expected], [expected, 1]])
    assert all(dtype == np.dtype(float) for dtype in square.dtypes)


def pearson_reference(x, y, w):
    keep = w > 0
    x, y, w = (
        np.asarray(x, dtype=float)[keep],
        np.asarray(y, dtype=float)[keep],
        w[keep],
    )
    x, y = x - x[0], y - y[0]
    x, y = x - np.average(x, weights=w), y - np.average(y, weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum(w * x * y) / np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))


def rank_reference(x, w):
    ranks = np.zeros(len(x))
    previous = 0.0
    for value in sorted(set(x)):
        positions = np.asarray(x) == value
        mass = w[positions].sum()
        count = np.count_nonzero(w[positions])
        ranks[positions] = previous + ((count + 1) * mass / (2 * count) if count else 0)
        previous += mass
    return ranks


def contingency_reference(x, y, w):
    table = (
        pd.crosstab(pd.Series(x), pd.Series(y), values=w, aggfunc="sum")
        .fillna(0)
        .to_numpy()
    )
    table = table[np.any(table > 0, axis=1)][:, np.any(table > 0, axis=0)]
    probabilities = table / table.sum()
    marginal = probabilities.sum(axis=1)
    entropy = -sum(p * np.log(p) for p in marginal)
    conditional = 0.0
    for i, j in zip(*np.nonzero(table)):
        conditional += probabilities[i, j] * np.log(
            probabilities[:, j].sum() / probabilities[i, j]
        )
    return conditional, (entropy - conditional) / entropy if entropy else 1.0


def ratio_reference(x, y, w):
    keep = w > 0
    x, y, w = np.asarray(x)[keep], np.asarray(y, dtype=float)[keep], w[keep]
    y = y - y[0]
    mean = np.average(y, weights=w)
    between, within = 0.0, 0.0
    for level in set(x):
        group = x == level
        group_mean = np.average(y[group], weights=w[group])
        between += w[group].sum() * (group_mean - mean) ** 2
        within += np.sum(w[group] * (y[group] - group_mean) ** 2)
    return np.sqrt(between / (between + within)) if between + within else 0.0


def mixed_frame():
    return pd.DataFrame(
        {
            "x": [1.0, 1, 2, 3, 4, 6, 8, 10],
            "y": [4.0, 3, 1, 2, 8, 5, 7, 6],
            "g": ["a", "a", "b", "b", "c", "c", "a", "ignored"],
            "h": ["u", "v", "u", "u", "v", "v", "u", "zero"],
        },
        index=[8, 8, 2, 4, 6, 6, 0, 0],
    )


import os
from functools import partial
from itertools import permutations

import joblib
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.stats import spearmanr

from arfs.feature_selection import CollinearityThreshold, MinRedundancyMaxRelevance


@pytest.mark.parametrize("scale", [0.1, 1.0, 1e6])
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_numeric_independent_references_all_paths(scale, method):
    X = mixed_frame()
    X["x"] += 1e12
    X["constant"] = 5.0
    w = np.array([0.5, 0, 1.25, 0.75, 2, 1, 3, 0]) * scale
    method_fn = partial(a.weighted_corr, method=method)
    for target in ["x", "y"]:
        series = a.wcorr_series(X, target, w, method=method)
        mixed = a.association_series(
            X, target, sample_weight=w, num_num_assoc=method_fn
        )
        for label in ["x", "y", "constant"]:
            x, y = X[label].to_numpy(), X[target].to_numpy()
            if method == "spearman":
                assert_allclose(a.wrank(x, w), rank_reference(x, w))
                x, y = rank_reference(x, w), rank_reference(y, w)
            expected = pearson_reference(x, y, w)
            actual = a.weighted_corr(X[label], X[target], w, method=method)
            assert_allclose(actual, expected, atol=1e-14)
            assert_allclose(
                series[label], expected if np.isfinite(expected) else 0, atol=1e-14
            )
            assert_allclose(mixed[label], series[label], atol=1e-14)
    for matrix in [
        a.wcorr_matrix(X, w, method=method),
        a.association_matrix(X, w, num_num_assoc=method_fn),
    ]:
        for left, right, value in matrix.itertuples(index=False, name=None):
            if left in ["x", "y", "constant"] and right in ["x", "y", "constant"]:
                assert_allclose(
                    value,
                    a.weighted_corr(X[left], X[right], w, method=method),
                    atol=1e-14,
                )


def test_equal_weights_match_scipy_spearman():
    X = mixed_frame()[["x", "y"]]
    expected = spearmanr(X.x, X.y).statistic
    assert_allclose(a.weighted_corr(X.x, X.y), expected)
    assert_allclose(a.wcorr_matrix(X).val, expected)
    assert_allclose(a.wcorr_series(X, "x")["y"], expected)


@pytest.mark.parametrize("scale", [0.1, 1.0, 1e6])
@pytest.mark.parametrize("offset", [0.0, 1e12])
def test_categorical_independent_references(scale, offset):
    X = mixed_frame()
    X[["x", "y"]] += offset
    w = np.array([0.5, 0, 1.25, 0.75, 2, 1, 3, 0]) * scale
    conditional, u = contingency_reference(X.g.to_numpy(), X.h.to_numpy(), w)
    assert_allclose(
        a.weighted_conditional_entropy(X.g, X.h, w), conditional, atol=1e-14
    )
    assert_allclose(a.weighted_theils_u(X.g, X.h, w), u, atol=1e-14)
    for target in ["g", "h"]:
        result = a.theils_u_series(X, target, w)
        for label in ["g", "h"]:
            assert_allclose(
                result[label],
                contingency_reference(X[label].to_numpy(), X[target].to_numpy(), w)[1],
                atol=1e-14,
            )
    for target in X.columns:
        series = a.correlation_ratio_series(X, target, w)
        for label, value in series.items():
            cat, num = (target, label) if target in ["g", "h"] else (label, target)
            expected = ratio_reference(X[cat], X[num], w)
            assert_allclose(value, expected, atol=1e-14)
            assert_allclose(
                a.correlation_ratio(X[cat], X[num], w), expected, atol=1e-14
            )
    for left, right, value in a.theils_u_matrix(X, w).itertuples(
        index=False, name=None
    ):
        assert_allclose(
            value,
            contingency_reference(X[left].to_numpy(), X[right].to_numpy(), w)[1],
            atol=1e-14,
        )
    for left, right, value in a.correlation_ratio_matrix(X, w).itertuples(
        index=False, name=None
    ):
        cat, num = (left, right) if left in ["g", "h"] else (right, left)
        assert_allclose(value, ratio_reference(X[cat], X[num], w), atol=1e-14)


@pytest.mark.parametrize("backend", ["loky", "threading", "sequential"])
def test_prepared_categorical_backends_and_mixed_block_order(backend):
    X = mixed_frame()
    w = np.array([0.5, 0, 1.25, 0.75, 2, 1, 3, 0])
    expected = a.association_matrix(X, w)
    with joblib.parallel_backend(backend):
        assert_frame_equal(a.association_matrix(X, w, n_jobs=2), expected)
        for target in X.columns:
            assert_series_equal(
                a.association_series(X, target, sample_weight=w, n_jobs=2),
                a.association_series(X, target, sample_weight=w),
            )
    block = pd.concat(
        [
            a.wcorr_matrix(X, w),
            a.correlation_ratio_matrix(X, w),
            a.theils_u_matrix(X, w),
        ],
        ignore_index=True,
    )
    assert_frame_equal(expected, block)


def callback_pid(x, y, sample_weight, as_frame):
    pid = float(os.getpid())
    if as_frame:
        return pd.DataFrame({"row": [x.name], "col": [y.name], "val": [pid]})
    return pid


def test_custom_path_uses_two_real_processes():
    X = pd.DataFrame(np.arange(60).reshape(10, 6))
    with joblib.parallel_backend("loky"):
        result = a.association_matrix(X, num_num_assoc=callback_pid, n_jobs=2)
    assert len(set(result.val)) == 2
    assert os.getpid() not in set(result.val)


@pytest.mark.parametrize("backend", ["loky", "threading", "sequential"])
def test_callback_exceptions_propagate(backend):
    def broken(x, y, sample_weight, as_frame):
        raise RuntimeError("callback boundary")

    with (
        joblib.parallel_backend(backend),
        pytest.raises(RuntimeError, match="callback boundary"),
    ):
        a.association_matrix(mixed_frame(), num_num_assoc=broken, n_jobs=2)


def test_wrappers_keep_raw_callback_inputs_and_whole_frame_policy():
    X = mixed_frame()
    X.loc[2, "g"] = None
    w = np.arange(1.0, 9)
    seen = []

    def capture(x, y, sample_weight, as_frame):
        seen.append((x.copy(), y.copy(), sample_weight.copy(), as_frame))
        return 0.25

    result = a.association_series(
        X,
        "x",
        sample_weight=w,
        handle_na=None,
        num_num_assoc=capture,
        nom_num_assoc=capture,
    )
    assert (result == 0.25).all()
    assert any(x.isna().any() for x, _, _, _ in seen)
    for x, y, weights, as_frame in seen:
        assert_series_equal(x, X[x.name])
        assert_series_equal(y, X.x)
        assert_allclose(weights, w)
        assert as_frame is False
    seen.clear()
    a.association_series(
        X,
        "x",
        sample_weight=w,
        handle_na="drop",
        num_num_assoc=capture,
        nom_num_assoc=capture,
    )
    assert all(len(x) == 7 for x, _, _, _ in seen)
    for fn in [a.wcorr_matrix, a.theils_u_matrix, a.correlation_ratio_matrix]:
        assert_frame_equal(
            fn(X, w, handle_na="drop"), fn(X.dropna(), w[X.notna().all(axis=1)])
        )
        filled = X.fillna(0)
        assert_frame_equal(fn(X, w, handle_na="fill"), fn(filled, w))
        with pytest.raises(ValueError, match="finite|nonmissing"):
            # Numeric-only kernels validate only the evaluated numeric block.
            fn(
                X.assign(x=np.nan), w, handle_na=None
            ) if fn is not a.theils_u_matrix else fn(X, w, handle_na=None)


@pytest.mark.parametrize(
    "labels",
    [
        [1, "b", ("c", 2), "d"],
        pd.Index([("a", 1), ("b", 2), ("c", 3), ("d", 4)], tupleize_cols=False),
        pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3), ("d", 4)], names=["name", "id"]
        ),
    ],
)
def test_prepared_labels_and_permutation_ties(labels):
    X = mixed_frame()
    X.columns = labels
    for fn in [a.theils_u_matrix, a.correlation_ratio_matrix, a.wcorr_matrix]:
        assert_frame_equal(fn(X), fn(X, n_jobs=2))
    for target in X.columns:
        assert_series_equal(
            a.association_series(X, target), a.association_series(X, target, n_jobs=2)
        )
    cats = pd.DataFrame({"a": ["a"] * 4, "b": ["b"] * 4, "c": ["c"] * 4})
    result = a.theils_u_matrix(cats)
    assert list(zip(result.row, result.col)) == list(permutations(cats.columns, 2))


@pytest.mark.parametrize("jobs", [0, True, 1.5, "2", None])
@pytest.mark.parametrize(
    "fn",
    [
        a.wcorr_matrix,
        a.theils_u_matrix,
        a.correlation_ratio_matrix,
        a.association_matrix,
    ],
)
def test_native_paths_validate_workers(fn, jobs):
    with pytest.raises(ValueError, match="n_jobs.*nonzero integer"):
        fn(mixed_frame(), n_jobs=jobs)


def test_series_never_allocates_a_full_numeric_matrix_and_ranks_once(monkeypatch):
    X = mixed_frame()[["x", "y"]]
    X["z"] = X.x * -1
    rank, product = a._weighted_rank, a._numeric_correlation
    calls = []

    def count(values, weights):
        calls.append(len(values))
        return rank(values, weights)

    def vector_only(centered, weights, target=None):
        assert target is not None
        return product(centered, weights, target)

    monkeypatch.setattr(a, "_weighted_rank", count)
    monkeypatch.setattr(a, "_numeric_correlation", vector_only)
    for fn in [a.wcorr_series, a.association_series]:
        calls.clear()
        fn(X, "x")
        assert calls == [len(X)] * X.shape[1]


def test_factorization_once_and_one_joint_per_pair(monkeypatch):
    X = mixed_frame()
    factorize, joint = pd.factorize, a._conditional_entropies
    factors, joints = [], []

    def count_factorize(values, *args, **kwargs):
        factors.append(values.name)
        return factorize(values, *args, **kwargs)

    def count_joint(x, y, weights):
        joints.append(True)
        return joint(x, y, weights)

    monkeypatch.setattr(pd, "factorize", count_factorize)
    monkeypatch.setattr(a, "_conditional_entropies", count_joint)
    a.theils_u_matrix(X)
    assert factors == ["g", "h"]
    assert len(joints) == 1
    factors.clear()
    a.correlation_ratio_matrix(X)
    assert factors == ["g", "h"]
    factors.clear()
    a.theils_u_series(X, "g")
    assert factors == ["g", "h"]


def test_selector_relevance_agrees_with_independent_pearson():
    X = mixed_frame()[["x", "y"]]
    weights = np.array([0.5, 0, 1.25, 0.75, 2, 1, 3, 0])
    y = pd.Series([1.0, 2, 4, 1, 6, 7, 8, 3])
    scores = a.f_cont_regression_parallel(X, y, weights)
    for label in X:
        r2 = pearson_reference(X[label], y, weights) ** 2
        assert_allclose(scores[label], r2 / (1 - r2) * (weights.sum() - 2))
    assert MinRedundancyMaxRelevance(1, show_progress=False).fit(
        X, y, sample_weight=weights
    ).selected_features_ == [scores.idxmax()]
    selector = CollinearityThreshold(threshold=0.8).fit(X, sample_weight=weights)
    assert selector.n_features_in_ == X.shape[1]


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_spearman_ranks_large_integers_before_float_conversion(dtype):
    X = pd.DataFrame({"x": pd.Series([2**60 + i for i in [0, 1, 2, 3]], dtype=dtype), "y": [2., 1., 4., 3.]})
    expected = spearmanr([0, 1, 2, 3], X.y).statistic
    assert_allclose(a.weighted_corr(X.x, X.y), expected)
    assert_allclose(a.wcorr_matrix(X).val, expected)
    assert_allclose(a.wcorr_series(X, "x")["y"], expected)


@pytest.mark.parametrize("fn", [a.wcorr_series, a.theils_u_series, a.correlation_ratio_series, a.association_series])
@pytest.mark.parametrize("jobs", [0, True, 1.5, "2", None])
def test_native_series_validate_workers(fn, jobs):
    target = "g" if fn is a.theils_u_series else "x"
    with pytest.raises(ValueError, match="n_jobs.*nonzero integer"):
        fn(mixed_frame(), target, n_jobs=jobs)


@pytest.mark.parametrize("dtype", [object, "string", "category"])
def test_direct_spearman_keeps_comparable_nonnumeric_inputs(dtype):
    x = pd.Series(["a", "b", "c", "d"], dtype=dtype)
    y = pd.Series(["b", "a", "d", "c"], dtype=dtype)
    expected = spearmanr([0, 1, 2, 3], [1, 0, 3, 2]).statistic
    assert_allclose(a.weighted_corr(x, y), expected)
    assert_allclose(a.wspearman(x, y, None), expected)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("weighted", [False, True])
def test_exact_duplicates_keep_unit_correlation_at_realistic_sizes(method, weighted):
    rng = np.random.default_rng(8)
    x = rng.normal(size=110)[10:]
    X = pd.DataFrame({"x": x, "copy": x, "negative": -x})
    weights = rng.uniform(.2, 2, len(X)) if weighted else None
    assert a.weighted_corr(X.x, X["copy"], weights, method=method) == 1.0
    series = a.wcorr_series(X, "x", weights, method=method)
    assert series["x"] == series["copy"] == 1.0
    matrix = a.wcorr_matrix(X, weights, method=method)
    duplicate = matrix.loc[((matrix.row == "x") & (matrix.col == "copy")) | ((matrix.row == "copy") & (matrix.col == "x")), "val"]
    assert (duplicate == 1.0).all()
    if method == "pearson" or not weighted:
        assert series["negative"] == -1.0
    assert a.weighted_correlation_1cpu(X, weights).loc["x", "copy"] == 1.0
