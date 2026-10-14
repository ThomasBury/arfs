"""This module provide methods for sampling large datasets for reducing the running time"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import ks_2samp
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest

from .utils import _is_categorical_dtype, validate_sample_weight


def sample(df, n=1000, sample_weight=None, method="gower"):
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
        Its subset draws and KS screen are unweighted. Gower representatives
        reject row weights; direct gower_matrix has separate feature weights.
    method : str, optional
        the strategy to use for sampling the rows. Either ``"gower"`` or ``"isoforest"``, by default 'gower'

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
        X[non_num_cols] = X[non_num_cols].astype("str").astype("category")
        for col in non_num_cols:
            X[col] = X[col].astype("category").cat.codes
        idx = isof_find_sample(X, sample_weight=sample_weight)
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
    """Returns the n largest indices from a numpy array.

    Parameters
    ----------
    ary : np.array
        the array for which to return largest indices
    n : int
        the number of indices to return

    Returns
    -------
    dict
        the dictionary of indices and values of the largest elements
    """
    # n += 1
    flat = np.nan_to_num(ary.flatten(), nan=999)
    indices = np.argpartition(-flat, -n)[-n:]
    indices = indices[np.argsort(flat[indices])]
    # indices = np.delete(indices,0,0)
    values = flat[indices]
    return {"index": indices, "values": values}


def gower_topn(
    data_x,
    data_y=None,
    weight=None,
    cat_features="auto",
    n=5,
    key=None,
):
    """Get the n most similar elements

    Parameters
    ----------
    data_x : np.array or pd.DataFrame
        The data for the look up
    data_y : np.array or pd.DataFrame or pd.Series, optional
        elements for which to return the most similar elements, should be a single row
    weight : np.array or pd.Series, optional
        sample weight, by default None
    cat_features : list of str or bool or int, optional
        auto detection of cat features or a list of strings, booleans or integers, by default 'auto'
    n : int, optional
        the number of neighbors/similar rows to find, by default 5
    key : str, optional
        identifier key. If several rows refer to the same id, this column
        will be used for finding the nearest neighbors with a
        different id, by default None

    Returns
    -------
    dict
        the dictionary of indices and values of the closest elements

    Raises
    ------
    TypeError
        if the reference element is not a single row
    """

    if data_y.shape[0] >= 2:
        raise TypeError("Only support `data_y` of 1 row. ")
    if key is None:
        dm = gower_matrix(data_y, data_x, weight, cat_features)
    else:
        X = data_x.drop(key, axis=1)
        Y = data_x.drop(key, axis=1)
        dm = gower_matrix(Y, X, weight, cat_features)

    if key is not None:
        idx = smallest_indices(np.nan_to_num(dm[0], nan=1), n)["index"]
        val = smallest_indices(np.nan_to_num(dm[0], nan=1), n)["values"]
        unique_id = data_x.iloc[idx, :]
        unique_id = unique_id[key]
        nunique_id = unique_id.nunique()
        mul = 1
        # continue looking for the closest n unique records with a different id
        while nunique_id < n:
            idx = smallest_indices(np.nan_to_num(dm[0], nan=1), mul * n)["index"]
            val = smallest_indices(np.nan_to_num(dm[0], nan=1), mul * n)["values"]
            unique_id = data_x.iloc[idx, :].reset_index()
            unique_id = unique_id[key]
            nunique_id = unique_id.nunique()
            mul += 1

        # find the indices of the unique id
        _, idx_n = np.unique(unique_id, return_index=True)
        # select only the rows corresponding to unique id
        val = val[idx_n]
        idx = idx[idx_n]
        # sort them from the closest to the farthest, according to the Gower metrics
        idx_n = np.argsort(val)
        # return the n closest records, with a different id
        return {"index": idx[idx_n[:n]], "values": val[idx_n[:n]]}
    else:
        return smallest_indices(np.nan_to_num(dm[0], nan=1), n)


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


def isolation_forest(X, sample_weight=None):
    """fits isolation forest to the dataset and gives an anomaly score to every sample

    Parameters
    ----------
    X : pd.DataFrame or np.array
        the predictors matrix
    sample_weight : pd.Series or np.array, optional
        Positional fit weights. Zero-weight rows are omitted from fitting;
        all original rows are scored, without scoring weights.
    """
    sample_weight = validate_sample_weight(sample_weight, len(X))
    clf = IsolationForest()
    if sample_weight is None:
        clf.fit(X)
    else:
        positive = sample_weight > 0
        clf.fit(X[positive], sample_weight=sample_weight[positive])
    return clf.score_samples(X)


def isof_find_sample(X, sample_weight=None):
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
    preds = isolation_forest(X, sample_weight)
    for size in get_5_percent_splits(len(X)):
        for _ in range(20):
            sample_indices = np.random.choice(len(X), size=size, replace=False)
            if ks_2samp(preds, preds[sample_indices]).pvalue > 0.95:
                return sample_indices
    return np.arange(len(X))
