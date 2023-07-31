"""This module provide methods for sampling large datasets for reducing the running time
"""
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp
from .utils import is_list_of_str, is_list_of_bool, is_list_of_int


def sample(df, n=1000, sample_weight=None, method="gower"):
    """Sampling rows from a dataframe when random sampling is not
    enough for reducing the number of rows.
    The strategies are either using hierarchical clustering
    based on the Gower distance or using isolation forest for identifying
    the most similar elements.
    For the clustering algorithm, clusters are determined using the Gower distance
    (mixed type data) and the dataset is shrunk from n_samples to n_clusters.

    For the isolation forest algorithm, samples are added till a suffisant 2-samples
    KS statistics is reached or if the number iteration reached the max number (20)

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to sample, with or without the target
    n : int, optional
        the number of clusters if method is ``"gower"``, by default 100
    sample_weight : pd.Series or np.array, optional
        sample weights, by default None
    method : str, optional
        the strategy to use for sampling the rows. Either ``"gower"`` or ``"isoforest"``, by default 'gower'

    Returns
    -------
    pd.DataFrame
        the sampled dataframe

    """
    assert isinstance(df, pd.DataFrame), "X should be a DataFrame"
    X = df.copy()
    num_cols = list(X.select_dtypes(include=[np.number]))
    non_num_cols = list(set(list(X.columns)) - set(num_cols))

    if method == "gower":
        # basic imputation
        if non_num_cols:
            X[non_num_cols] = X[non_num_cols].fillna(X[non_num_cols].mode().iloc[0])
        if num_cols:
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean().iloc[0])

        # no need for scaling, it is built-in the computation of the Gower distance
        gd = gower_matrix(X, cat_features=non_num_cols, weight=sample_weight)

        labels = AgglomerativeClustering(
            n_clusters=n, affinity="precomputed", linkage="complete"
        ).fit_predict(gd)
        X["label"] = labels
        X["label"] = "clus_" + X["label"].astype(str)
        X_num = X.groupby("label")[num_cols].agg("mean")
        if non_num_cols:
            X_nonnum = X.groupby("label")[non_num_cols].agg(get_most_common)
            X_sampled = X_num.join(X_nonnum)
        else:
            X_sampled = X_num
        X_sampled = X_sampled.reindex(X.columns, axis=1)
        return X_sampled
    elif method == "isoforest":
        X[non_num_cols] = X[non_num_cols].astype("str").astype("category")
        for col in non_num_cols:
            X[col] = X[col].astype("category").cat.codes
        idx = isof_find_sample(X, sample_weight=None)
        return X.iloc[idx, :]
    else:
        NotImplementedError(f"{method} not implemented")


def get_most_common(srs):
    x = list(srs)
    my_counter = Counter(x)
    return my_counter.most_common(1)[0][0]


def gower_matrix(
    data_x,
    data_y=None,
    weight=None,
    cat_features="auto",
):
    """Computes the gower distances between X and Y

    Gower is a similarity measure for categorical, boolean and numerical mixed
    data.

    Parameters
    ----------
    data_x : np.array or pd.DataFrame
        The data for computing the Gower distance
    data_y : np.array or pd.DataFrame or pd.Series, optional
        The reference matrix or vector to compare with, optional
    weight : np.array or pd.Series, optional
        sample weight, optional
    cat_features : list of str or bool or int, optional
        auto-detect cat features or a list of cat features, by default 'auto'

    Returns
    -------
    np.array
        The Gower distance matrix, shape (n_samples, n_samples)

    Notes
    -----
    The non-numeric features, and numeric feature ranges are determined from X and not Y.

    Raises
    ------
    TypeError
        If two dataframes are passed but have different number of columns
    TypeError
        If two arrays are passed but have different number of columns
    TypeError
        Sparse matrices are not supported
    TypeError
        if a list of categorical columns is passed, it should be a list of strings or integers or boolean values
    """
    # function checks
    X = data_x
    if data_y is None:
        Y = data_x
    else:
        Y = data_y
    if not isinstance(X, np.ndarray):
        y_col = Y.columns if isinstance(Y, pd.DataFrame) else Y.index
        if not np.array_equal(X.columns, y_col):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")
    if issparse(X) or issparse(Y):
        raise TypeError("Sparse matrices are not supported!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    if cat_features == "auto":
        if not isinstance(X, np.ndarray):
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col] = True
    else:
        # force categorical columns (if integer encoded for instance)
        if is_list_of_str(cat_features):
            cat_feat = [True if c in cat_features else False for c in X.columns]
            cat_features = np.array(cat_feat)
        elif is_list_of_bool(cat_features):
            cat_features = np.array(cat_features)
        elif is_list_of_int(cat_features):
            cat_feat = [
                True if c in cat_features else False for c in range(len(X.columns))
            ]
            cat_features = np.array(cat_feat)
        else:
            raise TypeError(
                "If not 'auto' cat_features should be a list of strings, integers or Booleans"
            )

    # print(cat_features)

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    Z = np.concatenate((X, Y))

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z[:, np.logical_not(cat_features)]

    num_cols = Z_num.shape[1]
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)

    for col in range(num_cols):
        col_array = Z_num[:, col].astype(np.float32)
        max_ = np.nanmax(col_array)
        min_ = np.nanmin(col_array)

        if np.isnan(max_):
            max_ = 0.0
        if np.isnan(min_):
            min_ = 0.0
        num_max[col] = max_
        num_ranges[col] = (1 - min_ / max_) if (max_ != 0) else 0.0

    # This is to normalize the numeric values between 0 and 1.
    Z_num = np.divide(
        Z_num.astype(float),
        num_max.astype(float),
        out=np.zeros_like(Z_num).astype(float),
        where=num_max != 0,
    )
    Z_cat = Z[:, cat_features]

    if weight is None:
        weight = np.ones(Z.shape[1])

    # print(weight)

    weight_cat = weight[cat_features]
    weight_num = weight[np.logical_not(cat_features)]

    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    weight_sum = weight.sum()

    X_cat = Z_cat[x_index,]
    X_num = Z_num[x_index,]
    Y_cat = Z_cat[y_index,]
    Y_num = Z_num[y_index,]

    # print(X_cat,X_num,Y_cat,Y_num)

    for i in range(x_n_rows):
        j_start = i
        if x_n_rows != y_n_rows:
            j_start = 0
        # call the main function
        res = _gower_distance_row(
            X_cat[i, :],
            X_num[i, :],
            Y_cat[j_start:y_n_rows, :],
            Y_num[j_start:y_n_rows, :],
            weight_cat,
            weight_num,
            weight_sum,
            num_ranges,
        )
        # print(res)
        out[i, j_start:] = res
        if x_n_rows == y_n_rows:
            out[i:, j_start] = res

    return out


def _gower_distance_row(
    xi_cat,
    xi_num,
    xj_cat,
    xj_num,
    feature_weight_cat,
    feature_weight_num,
    feature_weight_sum,
    ranges_of_numeric,
):
    """Compute a row of the Gower matrix

    Parameters
    ----------
    xi_cat : np.array
        categorical row of the X matrix
    xi_num : np.array
        numerical row of the X matrix
    xj_cat : np.array
        categorical row of the X matrix
    xj_num : np.array
        numerical row of the X matrix
    feature_weight_cat : np.array
        weight vector for the categorical features
    feature_weight_num : np.array
        weight vector for the numerical features
    feature_weight_sum : float
        The sum of the wieghts
    ranges_of_numeric : np.array
        range of the scaled numerical features (between 0 and 1)

    Returns
    -------
    np.array : array
        a row vector of the Gower distance
    """
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat, np.zeros_like(xi_cat), np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # numerical columns
    abs_delta = np.absolute(xi_num - xj_num)
    sij_num = np.divide(
        abs_delta,
        ranges_of_numeric,
        out=np.zeros_like(abs_delta),
        where=ranges_of_numeric != 0,
    )

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)

    return sum_sij


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
    """splits dataframe into 5% intervals

    Parameters
    ----------
    length : int
        array length

    Returns
    -------
    array
        vector of sizes
    """

    five_percent = round(5 / 100 * length)
    return np.arange(five_percent, length, five_percent)


def isolation_forest(X, sample_weight=None):
    """fits isloation forest to the dataset and gives an anomally score to every sample

    Parameters
    ----------
    X : pd.DataFrame or np.array
        the predictors matrix
    sample_weight : pd.Series or np.array, optional
        the sample weights, if any, by default None
    """
    clf = IsolationForest().fit(X, sample_weight=sample_weight)
    return clf.score_samples(X)


def isof_find_sample(X, sample_weight=None):
    """Finds a sample by comparing the distributions of the anomally scores between the sample and the original
    distribution using the KS-test. Starts of a 5% howver will increase to 10% and then 15% etc. if a significant sample can not be found

    References
    ----------
    Sampling method taken from boruta_shap, author: https://github.com/Ekeany

    Parameters
    ----------
    X : pd.DataFrame
        the predictors matrix
    sample_weight : pd.Series or np.array, optional
        the sample weights, if any, by default None

    Returns
    -------
    array
        the indices for reducing the shadow predictors matrix
    """
    loop = True
    iteration = 0
    size = get_5_percent_splits(length=X.shape[0])
    element = 1
    preds = isolation_forest(X, sample_weight)
    while loop:
        sample_indices = np.random.choice(
            np.arange(preds.size), size=size[element], replace=False
        )
        sample = np.take(preds, sample_indices)
        if ks_2samp(preds, sample).pvalue > 0.95:
            break
        if iteration == 20:
            element += 1
            iteration = 0
    return sample_indices
