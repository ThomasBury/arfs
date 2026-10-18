"""This module provide methods for sampling large datasets for reducing the running time"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import ks_2samp
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_random_state

from .utils import _is_categorical_dtype, validate_sample_weight


def sample(df, n=1000, sample_weight=None, method="gower", random_state=None):
    """Sampling rows from a dataframe when random sampling is not
    enough for reducing the number of rows.
    The strategies are either using hierarchical clustering
    based on the Gower distance or using isolation forest for identifying
    the most similar elements.
    For the clustering algorithm, clusters are determined using the Gower distance
    (mixed type data) and the dataset is shrunk from n_samples to n_clusters.
    Representatives are synthetic column means/modes, with cluster IDs as their
    index and the original feature order. Numeric missing values use each column's
    mean; categorical missing values use its first mode. Entirely missing columns
    are rejected. Numeric representative dtypes may differ from the input.

    Isolation sampling tries random subsets of 5%, 10%, ..., 95%, rounded up
    and capped below the input row count, with 20 attempts per distinct size.
    The first subset with an unweighted two-sample KS p-value >0.95 is returned;
    exhaustion returns the full original data. Empty/single-row inputs return
    unchanged. The subset overlaps the full score population, so this screen is
    a heuristic, not a 95% representativeness guarantee or outlier removal.

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to sample, with or without the target
    n : int, optional
        the number of clusters if method is ``"gower"``, by default 1000.
        Must be between 1 and len(df), with at least two input rows.
        Ignored for ``"isoforest"``.
    sample_weight : pd.Series or np.array, optional
        Positional observation weights for the isolation forest fit only.
        Zero-weight rows are omitted from its category vocabulary and fit;
        unseen score-only categories use code -1.
        Its subset draws and KS screen are unweighted. Gower representatives
        reject row weights; direct gower_matrix has separate feature weights.
    method : str, optional
        the strategy to use for sampling the rows. Either ``"gower"`` or ``"isoforest"``, by default 'gower'
    random_state : int, RandomState instance or None, default=None
        Controls isolation forest fitting and subset draws. An integer replays
        each call; a RandomState instance advances; None uses NumPy's global
        RNG. Ignored by deterministic Gower clustering.

    Returns
    -------
    pd.DataFrame
        the sampled dataframe

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be a DataFrame")
    if not df.columns.is_unique or df.shape[1] == 0:
        raise ValueError("df must have unique columns and at least one feature")
    X = df.copy()
    num_cols = list(X.select_dtypes(include=[np.number]))
    non_num_cols = [col for col in X.columns if col not in num_cols]

    if method == "gower":
        if sample_weight is not None:
            raise ValueError(
                "Gower representatives do not support sample_weight (row weights)"
            )
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise ValueError("n must be an integer between 1 and len(df)")
        if len(X) < 2 or not 1 <= n <= len(X):
            raise ValueError(
                "Gower sampling requires at least two rows and 1 <= n <= len(df)"
            )
        if X.isna().all().any():
            raise ValueError(
                "Gower representatives require an observed value in every column"
            )
        # basic imputation
        if non_num_cols:
            X[non_num_cols] = X[non_num_cols].fillna(X[non_num_cols].mode().iloc[0])
        if num_cols:
            X[num_cols] = X[num_cols].astype(float).fillna(X[num_cols].mean())

        # no need for scaling, it is built-in the computation of the Gower distance
        gd = gower_matrix(X, cat_features=X.columns.isin(non_num_cols))

        labels = AgglomerativeClustering(
            n_clusters=n, metric="precomputed", linkage="complete"
        ).fit_predict(gd)
        labels = np.array([f"clus_{label}" for label in labels])
        X_num = X.groupby(labels)[num_cols].agg("mean")
        if non_num_cols:
            X_nonnum = X.groupby(labels)[non_num_cols].agg(get_most_common)
            X_sampled = X_num.join(X_nonnum)
        else:
            X_sampled = X_num
        X_sampled.index.name = "label"
        X_sampled = X_sampled.reindex(df.columns, axis=1)
        return X_sampled
    elif method == "isoforest":
        sample_weight = validate_sample_weight(sample_weight, len(X))
        if len(X) < 2:
            return df.copy()
        fit_rows = slice(None) if sample_weight is None else sample_weight > 0
        for col in non_num_cols:
            values = X[col].astype("str")
            categories = values.iloc[fit_rows].astype("category").cat.categories
            X[col] = categories.get_indexer(values)
        idx = isof_find_sample(
            X, sample_weight=sample_weight, random_state=random_state
        )
        return df.iloc[idx, :].copy()
    else:
        raise NotImplementedError(f"{method} not implemented")


def get_most_common(srs):
    x = list(srs)
    my_counter = Counter(x)
    return my_counter.most_common(1)[0][0]


def gower_matrix(data_x, data_y=None, weight=None, cat_features="auto"):
    """Compute pairwise Gower dissimilarities for numeric and nominal features.

    Parameters
    ----------
    data_x : ndarray or pandas.DataFrame, shape (n_x, n_features)
        Input rows. DataFrame columns must be unique.
    data_y : ndarray, pandas.DataFrame or pandas.Series, optional
        Reference rows with the same feature order and width. A Series is one
        named reference row. Named inputs must have matching columns in order.
        If omitted, compare data_x with itself.
    weight : array-like, shape (n_features,), optional
        Feature weights, not row/sample weights. Must be finite, nonnegative
        and contain positive mass; zero-weight features do not contribute.
    cat_features : "auto" or sequence of str, bool or int, optional
        Infer nominal columns using pandas dtype rules (strings, objects,
        categories and booleans). Arrays use soft per-column dtype inference.
        Otherwise use DataFrame column names, a full Boolean mask, or zero-based
        column positions. An empty sequence means all features are numeric.
        Integer entries always mean positions, including on integer-named frames.

    Returns
    -------
    ndarray of float64, shape (n_x, n_y)
        Weighted mean dissimilarities in [0, 1]. Missing features are omitted
        separately for each pair. A pair with no comparable positive-weight
        features returns NaN, including the diagonal of an entirely missing row.

    Notes
    -----
    Numeric ranges pool observed values from both inputs; observed constant
    numeric features contribute zero but retain their weight in the denominator.
    Numeric values are converted to float64 and must be finite or missing, with
    a representable max-minus-min range. Nominal values use equality, without
    string conversion. Ordinal and asymmetric-binary scoring are not implemented.
    The dense output requires quadratic storage for a self-comparison.

    Raises
    ------
    TypeError
        For sparse/unsupported input, non-matrix shape, mismatched named schemas,
        unsupported categorical masks, or nonnumeric values in numeric columns.
    ValueError
        For mismatched widths, duplicate/no columns, invalid masks or feature
        weights, infinite numeric values, or an overflowing numeric range.
    """
    if issparse(data_x) or issparse(data_y):
        raise TypeError("Sparse matrices are not supported!")
    self_comparison = data_y is None or data_y is data_x
    X = data_x
    Y = data_x if data_y is None else data_y
    if isinstance(Y, pd.Series):
        Y = Y.to_frame().T
    for matrix in (X, Y):
        if not isinstance(matrix, (np.ndarray, pd.DataFrame)) or matrix.ndim != 2:
            raise TypeError("X and Y must be two-dimensional arrays or DataFrames.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features.")
    if (
        isinstance(X, pd.DataFrame)
        and isinstance(Y, pd.DataFrame)
        and not X.columns.equals(Y.columns)
    ):
        raise TypeError("X and Y must have same columns in order!")
    for matrix in (X, Y):
        if isinstance(matrix, pd.DataFrame) and not matrix.columns.is_unique:
            raise ValueError("Feature columns must be unique.")
    n_features = X.shape[1]
    if n_features == 0:
        raise ValueError("At least one feature is required.")
    named = isinstance(X, pd.DataFrame)
    if not named:
        X = pd.DataFrame(X)
        X = X.where(X.notna(), np.nan).infer_objects()
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y, columns=X.columns)

    if isinstance(cat_features, str) and cat_features == "auto":
        categorical = np.array([_is_categorical_dtype(dtype) for dtype in X.dtypes])
    else:
        mask = np.asarray(cat_features, dtype=object)
        if mask.ndim != 1:
            raise TypeError("cat_features must be auto or a one-dimensional sequence.")
        categorical = np.zeros(n_features, dtype=bool)
        if mask.size == 0:
            pass
        elif all(isinstance(v, (bool, np.bool_)) for v in mask):
            if len(mask) != n_features:
                raise ValueError(
                    "Boolean cat_features must have one entry per feature."
                )
            categorical = mask.astype(bool)
        elif all(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
            for v in mask
        ):
            if any(v < 0 or v >= n_features for v in mask):
                raise ValueError("Categorical column positions are out of range.")
            categorical[mask.astype(int)] = True
        elif all(isinstance(v, str) for v in mask):
            if not named:
                raise TypeError("Named cat_features require a DataFrame.")
            if not set(mask).issubset(X.columns):
                raise ValueError("cat_features contains unknown column names.")
            categorical = X.columns.isin(mask)
        else:
            raise TypeError(
                "cat_features must contain only names, positions or Booleans."
            )

    if np.iscomplexobj(weight):
        raise ValueError("Feature weights must be real numbers.")
    weights = np.ones(n_features) if weight is None else np.asarray(weight, dtype=float)
    if weights.ndim != 1 or len(weights) != n_features:
        raise ValueError("weight must have one entry per feature.")
    if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError(
            "Feature weights must be finite, nonnegative and have positive mass."
        )

    values = np.concatenate((X.to_numpy(dtype=object), Y.to_numpy(dtype=object)))
    available = ~pd.isna(values)
    values[~available] = 0
    try:
        numeric = values[:, ~categorical].astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Non-categorical features must be numeric.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric features must be finite or missing.")
    for col, pos in enumerate(np.flatnonzero(~categorical)):
        observed = numeric[available[:, pos], col]
        if observed.size:
            minimum = observed.min()
            with np.errstate(over="ignore"):
                span = observed.max() - minimum
            if not np.isfinite(span):
                raise ValueError("Numeric feature range exceeds float64 capacity.")
            numeric[:, col] = np.where(available[:, pos], numeric[:, col], minimum)
            numeric[:, col] = (numeric[:, col] - minimum) / span if span else 0.0
    nominal = values[:, categorical]
    n_x, n_y = len(X), len(Y)
    out = np.empty((n_x, n_y), dtype=float)
    for i in range(n_x):
        start = i if self_comparison else 0
        res = _gower_distance_row(
            nominal[i],
            numeric[i],
            nominal[n_x + start :],
            numeric[n_x + start :],
            available[i] & available[n_x + start :],
            weights,
            categorical,
        )
        out[i, start:] = res
        if self_comparison:
            out[start:, i] = res
    return out


def _gower_distance_row(xi_cat, xi_num, xj_cat, xj_num, available, weight, categorical):
    """Weight each pair by its available features, using only row-sized scratch."""
    delta = np.empty(available.shape, dtype=float)
    delta[:, categorical] = xi_cat != xj_cat
    delta[:, ~categorical] = np.abs(xi_num - xj_num)
    pair_weights = np.where(available, weight, 0.0)
    # Scale within each pair so large weights cannot overflow the weighted mean.
    maximum = pair_weights.max(axis=1, keepdims=True)
    np.divide(pair_weights, maximum, out=pair_weights, where=maximum > 0)
    denominator = pair_weights.sum(axis=1)
    return np.divide(
        (delta * pair_weights).sum(axis=1),
        denominator,
        out=np.full(len(xj_num), np.nan),
        where=denominator > 0,
    )


def smallest_indices(ary, n):
    """Return the n smallest finite values and their flattened positions.

    n must be a nonnegative integer no larger than the finite-value count.
    Zero returns empty arrays. NaN and infinities are excluded; ties keep input
    order. Values retain their numeric dtype, and the input is not modified.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    flat = np.asarray(ary).ravel()
    if flat.dtype.kind not in "biuf":
        raise TypeError("ary must contain real numeric values")
    indices = np.flatnonzero(np.isfinite(flat))
    if n > len(indices):
        raise ValueError("n exceeds the number of finite values")
    if n:
        indices = indices[np.argsort(flat[indices], kind="stable")][:n]
    else:
        indices = indices[:0]
    return {"index": indices, "values": flat[indices]}


def gower_topn(
    data_x,
    data_y=None,
    weight=None,
    cat_features="auto",
    n=5,
    key=None,
):
    """Return the n nearest eligible candidate rows to an explicit one-row query.

    Parameters
    ----------
    data_x : ndarray or pandas.DataFrame
        Candidate rows. Returned indices are positions, not DataFrame labels.
    data_y : ndarray, pandas.DataFrame or pandas.Series
        Required one-row query with the same feature schema. A Series is a named
        reference row. Omission/None raises ValueError; other row counts fail.
    weight : array-like, optional
        Feature weights passed to gower_matrix, not observation weights. With a
        key, these and categorical masks describe features after removing it.
    cat_features : "auto" or sequence of str, bool or int, optional
        The gower_matrix category specification for the scored features.
        Automatic feature roles follow candidate dtypes, including Series queries.
    n : int, optional
        Number of neighbors, by default 5. Must be a nonnegative integer; zero
        returns empty arrays. Too few eligible neighbors raises ValueError.
    key : column label, optional
        Requires named candidate/query DataFrames (or a query Series). Exclude
        this column from distance computation and all rows matching the query's
        identifier. Query identifiers must be nonmissing; missing candidate IDs
        are ineligible. Keep the nearest finite-distance row per remaining ID,
        with pandas membership/duplicate semantics and input order breaking ties.
        Without a key, all finite-distance rows are eligible, including self.

    Returns
    -------
    dict
        "index" contains integer candidate positions, "values" their distances
        in ascending order. NaN/infinite distances are excluded, not replaced.

    Notes
    -----
    One candidate-by-query distance vector is allocated, not a self-distance
    matrix. Gower ranges still pool the full candidates and query, before key
    exclusion. Ranking sorts once (O(N log N)); distance storage is O(N).
    Inputs are not modified.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if data_y is None:
        raise ValueError("data_y must be provided as an explicit 1 row query")
    query = data_y.to_frame().T if isinstance(data_y, pd.Series) else data_y
    if not isinstance(query, (pd.DataFrame, np.ndarray)) or query.ndim != 2:
        raise TypeError("data_y must be a two-dimensional 1 row query or a Series")
    if len(query) != 1:
        raise TypeError("data_y must contain exactly 1 row")
    candidates = data_x
    if key is not None:
        if not isinstance(data_x, pd.DataFrame) or not isinstance(query, pd.DataFrame):
            raise TypeError("key requires named candidate and query DataFrames")
        if not data_x.columns.is_unique or not query.columns.is_unique:
            raise ValueError("Feature columns must be unique")
        if key not in data_x.columns or key not in query.columns:
            raise ValueError("key must be a column in both candidates and query")
        if query[key].isna().iloc[0]:
            raise ValueError("The query key must be nonmissing")
        eligible = data_x[key].notna() & ~data_x[key].isin([query[key].iloc[0]])
        candidates = data_x.drop(columns=[key])
        query = query.drop(columns=[key])
    # shortcut: candidate-first Gower keeps dtype inference; vectorize row calls if profiling requires it.
    distances = gower_matrix(candidates, query, weight, cat_features)[:, 0]
    if key is not None:
        distances[~eligible.to_numpy()] = np.nan
    if n == 0:
        return smallest_indices(distances, 0)
    indices = smallest_indices(distances, np.isfinite(distances).sum())["index"]
    if key is not None:
        indices = indices[~data_x[key].iloc[indices].duplicated().to_numpy()]
    if n > len(indices):
        raise ValueError("n exceeds the number of eligible finite-distance neighbors")
    indices = indices[:n]
    return {"index": indices, "values": distances[indices]}


def get_5_percent_splits(length):
    """Return distinct proper-subset sizes for 5%, 10%, ..., 95%.

    Parameters
    ----------
    length : int
        Positive integer array length.

    Returns
    -------
    array
        Sorted sizes, rounded up to at least one row and capped at length - 1.
        A single-row input has no proper subset, so its result is empty.
    """

    if (
        isinstance(length, (bool, np.bool_))
        or not isinstance(length, (int, np.integer))
        or length < 1
    ):
        raise ValueError("length must be a positive integer")
    if length == 1:
        return np.array([], dtype=int)
    length = int(length)
    return np.unique(
        [
            min(length - 1, (length * percent + 99) // 100)
            for percent in range(5, 100, 5)
        ]
    )


def isolation_forest(X, sample_weight=None, random_state=None):
    """fits isolation forest to the dataset and gives an anomaly score to every sample

    Parameters
    ----------
    X : pd.DataFrame or np.array
        the predictors matrix
    sample_weight : pd.Series or np.array, optional
        Positional fit weights. Zero-weight rows are omitted from fitting;
        all original rows are scored, without scoring weights.
    random_state : int, RandomState instance or None, default=None
        Controls forest fitting. An integer replays each call; None uses
        NumPy's global RNG.
    """
    sample_weight = validate_sample_weight(sample_weight, len(X))
    clf = IsolationForest(random_state=random_state)
    if sample_weight is None:
        clf.fit(X)
    else:
        positive = sample_weight > 0
        clf.fit(X[positive], sample_weight=sample_weight[positive])
    return clf.score_samples(X)


def isof_find_sample(X, sample_weight=None, random_state=None):
    """Return row positions using a bounded anomaly-score similarity screen.

    Try 20 random draws without replacement per distinct size from
    get_5_percent_splits, starting at 5%. Accept the first KS p-value >0.95;
    after at most 380 draws return all positions in original order. Empty and
    single-row inputs return all positions without fitting. Observation weights
    apply only to forest fitting; subset draws and KS comparisons are unweighted.

    A large p-value does not prove representativeness. KS assumes independent
    samples from continuous distributions, whereas each candidate here overlaps
    the full score population. Repeated screening and one-dimensional score
    matching do not guarantee feature/target or minority-group preservation,
    and the random subset is not an outlier-removal rule.

    References
    ----------
    Sampling method taken from boruta_shap, author: https://github.com/Ekeany

    Parameters
    ----------
    X : pd.DataFrame
        Two-dimensional DataFrame or ndarray predictor matrix.
    sample_weight : pd.Series or np.array, optional
        Positional weights for the forest fit only, by default None.
    random_state : int, RandomState instance or None, default=None
        Controls forest fitting and subset draws. An integer replays each call;
        a RandomState instance advances; None uses NumPy's global RNG.

    Returns
    -------
    array
        Selected positional row indices, or all indices on exhaustion/tiny input.
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)) or X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a two-dimensional predictor matrix with features")
    sample_weight = validate_sample_weight(sample_weight, len(X))
    if len(X) < 2:
        return np.arange(len(X))
    rng = check_random_state(random_state)
    preds = isolation_forest(X, sample_weight, random_state=rng)
    for size in get_5_percent_splits(len(X)):
        for _ in range(20):
            sample_indices = rng.choice(len(X), size=size, replace=False)
            if ks_2samp(preds, preds[sample_indices]).pvalue > 0.95:
                return sample_indices
    return np.arange(len(X))
