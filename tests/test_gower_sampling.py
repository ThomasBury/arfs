"""Condensed Gower sampling against the previous dense clustering path."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering

from arfs import sampling


def dense_sample(frame, n):
    """Reference the original sampler, including cluster IDs and mode ties."""
    X = frame.copy()
    numeric = list(X.select_dtypes(include=[np.number]))
    nominal = [col for col in X if col not in numeric]
    if nominal:
        X[nominal] = X[nominal].fillna(X[nominal].mode().iloc[0])
    if numeric:
        X[numeric] = X[numeric].astype(float).fillna(X[numeric].mean())
    distances = sampling.gower_matrix(X, cat_features=X.columns.isin(nominal))
    labels = AgglomerativeClustering(
        n_clusters=n, metric="precomputed", linkage="complete"
    ).fit_predict(distances)
    labels = [f"clus_{label}" for label in labels]
    result = X.groupby(labels)[numeric].mean()
    if nominal:
        result = result.join(X.groupby(labels)[nominal].agg(sampling.get_most_common))
    result.index.name = "label"
    return result.reindex(frame.columns, axis=1)


@pytest.fixture(params=["numeric", "nominal", "mixed", "constant", "duplicates"])
def frame(request):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(12, 4)), columns=[10, 20, 30, 40])
    if request.param == "nominal":
        X = pd.DataFrame(rng.choice(["a", "b", "c"], size=(12, 7)))
    elif request.param == "mixed":
        X[20] = pd.Series(["a", None, "b"] * 4, dtype="category")
        X[40] = pd.Series([True, False, None] * 4, dtype="boolean")
        X.loc[1, 10] = np.nan
    elif request.param == "constant":
        X[:] = 7.0
        X[20] = "a"
    elif request.param == "duplicates":
        X = pd.DataFrame({10: [0, 0, 1, 1] * 3, 20: ["a", "a", "b", "b"] * 3})
    X.index = [8, 8, 90] * 4
    return X


@pytest.mark.parametrize("n", [1, 2, 5, 11, 12])
def test_sample_matches_dense_reference_including_cluster_ids(frame, n):
    before = frame.copy(deep=True)
    assert_frame_equal(sampling.sample(frame, n=n), dense_sample(frame, n))
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("n_rows", [2, 7])
def test_condensed_numeric_distances_preserve_constants_and_small_ranges(n_rows):
    X = pd.DataFrame(
        {
            "constant": np.ones(n_rows),
            "tiny": np.linspace(-1e-300, 1e-300, n_rows),
            "large": 1e12 + np.arange(n_rows),
        }
    )
    before = X.copy(deep=True)
    actual = sampling._gower_condensed(X, list(X.columns), [])
    assert actual.dtype == np.float64
    assert actual.shape == (n_rows * (n_rows - 1) // 2,)
    assert_allclose(actual, squareform(sampling.gower_matrix(X)))
    assert_frame_equal(X, before)


@pytest.mark.parametrize("n_features", [1, 3, 7, 11])
def test_condensed_nominal_distances_preserve_exact_mismatch_counts(n_features):
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.choice(["a", "b", "c"], size=(20, n_features)))
    actual = sampling._gower_condensed(X, [], list(X.columns))
    assert_array_equal(actual, squareform(sampling.gower_matrix(X)))


@pytest.mark.parametrize("dtype", [object, "category", "string", "boolean"])
def test_sample_matches_dense_with_nullable_nominal_columns(dtype):
    values = [True, False, None, True] if dtype == "boolean" else ["a", "b", None, "a"]
    X = pd.DataFrame(
        {
            10: pd.Series([0, 3, None, 6], dtype="Int64"),
            50: pd.Series(values, dtype=dtype),
        }
    )
    for n in (1, 2, len(X)):
        assert_frame_equal(sampling.sample(X, n=n), dense_sample(X, n))


def test_nominal_encoding_preserves_original_equality():
    X = pd.DataFrame({"c": pd.Series([1, "1", True, 0, False, ("a", 1)], dtype=object)})
    assert_array_equal(
        sampling._gower_condensed(X, [], ["c"]),
        squareform(sampling.gower_matrix(X)),
    )


@pytest.mark.parametrize("n_features", [3, 10, 100])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_interleaved_mixed_distances_match_dense(n_features, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(20, n_features)))
    nominal = list(range(0, n_features, 3))
    numeric = [col for col in X if col not in nominal]
    for col in nominal:
        X[col] = rng.choice(["a", "b", "c"], size=len(X))
    assert_allclose(
        sampling._gower_condensed(X, numeric, nominal),
        squareform(sampling.gower_matrix(X, cat_features=X.columns.isin(nominal))),
        rtol=1e-14,
        atol=1e-15,
    )


@pytest.mark.parametrize("values", [[0.0, np.inf], [-np.inf, 0.0], [-1e308, 1e308]])
def test_sampler_retains_numeric_validation(values):
    with pytest.raises(ValueError, match="finite|float64"):
        sampling.sample(pd.DataFrame({"x": values}), n=1)


def test_sampler_passes_only_unique_pairs_to_linkage(monkeypatch):
    X = pd.DataFrame({10: [0.0, 2.0, 4.0, 8.0], 50: ["a", "a", "b", "b"]})
    expected = squareform(sampling.gower_matrix(X))
    original = sampling.linkage
    seen = []

    def capture(distances, **kwargs):
        seen.append(distances.shape)
        assert distances.dtype == np.float64
        assert distances.nbytes == 8 * len(X) * (len(X) - 1) // 2
        assert_allclose(distances, expected)
        return original(distances, **kwargs)

    def reject(*args, **kwargs):
        pytest.fail("sampling must not construct a square matrix or triangular indices")

    monkeypatch.setattr(sampling, "linkage", capture)
    monkeypatch.setattr(sampling, "gower_matrix", reject)
    monkeypatch.setattr(np, "triu_indices", reject)
    result = sampling.sample(X, n=2)
    assert seen == [(6,)]
    assert len(result) == 2


def test_complete_linkage_cut_matches_dense_for_every_cluster_count():
    X = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": ["a", "b"] * 3})
    distances = sampling.gower_matrix(X)
    for n in range(1, len(X) + 1):
        expected = AgglomerativeClustering(
            n_clusters=n, metric="precomputed", linkage="complete"
        ).fit_predict(distances)
        actual = sampling._complete_linkage_labels(squareform(distances), n)
        assert_array_equal(actual, expected)
