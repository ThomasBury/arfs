"""Parallelized Association and Correlation matrix

This module provides parallelized methods for computing associations.
Namely, correlation, correlation ratio, Theil's U, Cramer's V

They are the basis of the MRmr feature selection.

Observation weights are positional, finite and nonnegative, with positive
mass after the frame's missing-value policy. Weights are not normalized.
Direct pairwise statistics require complete one-dimensional observations.
Grouped statistics omit zero-mass groups and contingency margins; ANOVA and
Cramer's V reject inputs with fewer than two usable groups/categories.
Relevance functions reject missing/nonfinite responses before applying any
predictor missing-value policy. handle_na=None passes all rows to custom
association callbacks unchanged.

Relevance scores use frequency weights: N is retained weight mass, not stored
row count. Regression/information scores require N > 2; grouped ANOVA requires
N > the number of positive-mass groups. Integer counts reproduce row replication.
Fractional frequency mass is accepted for scoring, but does not represent literal
replication or establish an F null distribution. Relative/exposure weights are
supported by association coefficients, not given scale-invariant relevance.
The categorical classification U-to-F transform is a ranking heuristic, not an
ANOVA test. Mixed score normalization and ranking policy remain separate.

Parallel association helpers preserve original column labels. n_jobs must be a
nonzero integer: joblib resolves negative counts using the active backend and
available CPUs; workers are capped to nonempty pair/column chunks. Results sort
descending with stable ties in callback order. Matrix results have a fresh
RangeIndex; no-pair matrices have row/col/val columns with float values.
"""

import math
import warnings
import matplotlib
import numpy as np
import gc
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.stats as ss


from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import safe_mask

from itertools import combinations, permutations, product
from scipy.stats import rankdata
from scipy.spatial.distance import pdist
from functools import partial

from .parallel import (
    _effective_n_jobs,
    parallel_matrix_entries,
    parallel_df,
    _compute_series,
    _compute_matrix_entries,
)
from .utils import (
    _is_categorical_dtype,
    create_dtype_dict,
    validate_sample_weight,
    validate_target,
)

_PRECISION = 1e-13

########################
# Redundancy measures
########################

# For computing the redundancy of all the columns with a given series
# R(y, x_i) for i=1,..., N --> a series (y is a fixed, chosen column)
# the main functions are:
# - the series-series computation (two columns)
# - the closure for applying the latter function to all columns of a dataframe
# - the "series" version, using the closure for computing the redundancy with all the cols of the DF
#
# For computing the redundancy matrix all the cols combinations of columns
# R(x_i, x_j) for i, j=1,..., N --> a data frame (either TRIUL if the measure is symmetric
# or the full matrix if asymmetric)
# - the series-series computation (two columns), same as for series case
# - the function looping over a chunk of combinations
# - the parallelization (sending different chunks to different cores and applying the latter function)


def symmetric_function(func):
    func.is_symmetric = True
    return func


def asymmetric_function(func):
    func.is_symmetric = False
    return func


def create_col_combinations(func, selected_cols):
    """
    Create column combinations or permutations based on the symmetry of the function.

    This function checks if `func` is symmetric. If it is, it creates combinations of `selected_cols`;
    otherwise, it creates permutations.

    Parameters
    ----------
    func : callable
        The function to check for symmetry. Should be decorated with `@symmetric_function`.
    selected_cols : list
        The columns to be combined or permuted.

    Returns
    -------
    list of tuples
        A list of tuples representing column combinations or permutations.
        If `func` is symmetric, combinations of `selected_cols` are returned;
        otherwise, permutations are returned.
    """

    if getattr(func, "is_symmetric", False):
        # If the function is symmetric, use combinations
        return list(combinations(selected_cols, 2)) if selected_cols else []
    else:
        # If the function is not symmetric, use permutations
        return list(permutations(selected_cols, 2)) if selected_cols else []


##################
# CAT-CAT
##################


def weighted_conditional_entropy(x, y, sample_weight=None):
    """
    Computes the weighted conditional entropy between two categorical predictors.

    Parameters
    ----------
    x : pd.Series of shape (n_samples,)
        The predictor vector.
    y : pd.Series of shape (n_samples,)
        The target vector.
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None.

    Returns
    -------
    float
        Weighted conditional entropy.
    """

    x, y, sample_weight = _check_association_pair(x, y, sample_weight)

    # Integer encoding for categorical data
    y_encoded, _ = pd.factorize(y)
    x_encoded, _ = pd.factorize(x)

    # Total weight
    tot_weight = np.sum(sample_weight)
    if tot_weight == 0:
        return 0

    # Grouped weights for y and (x, y)
    y_weights = np.bincount(
        y_encoded, weights=sample_weight, minlength=len(np.unique(y_encoded))
    )
    xy_weights = {
        level: np.bincount(
            y_encoded[x_encoded == level],
            weights=sample_weight[x_encoded == level],
            minlength=len(np.unique(y_encoded)),
        )
        for level in np.unique(x_encoded)
    }

    # Conditional entropy calculation
    h_xy = 0.0
    for level in xy_weights:
        for y_index, xy_weight in enumerate(xy_weights[level]):
            p_xy = xy_weight / tot_weight
            p_y = y_weights[y_index] / tot_weight

            if p_xy != 0:
                h_xy += p_xy * math.log(p_y / p_xy, math.e)

    return h_xy


@asymmetric_function
def weighted_theils_u(x, y, sample_weight=None, as_frame=False):
    """
    Computes the weighted Theil's U statistic between two categorical predictors.

    Parameters
    ----------
    x : pd.Series of shape (n_samples,)
        The predictor vector.
    y : pd.Series of shape (n_samples,)
        The target vector.
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None.
    as_frame : bool
        Return output as a dataframe or a float.

    Returns
    -------
    pd.DataFrame or float
        Predictor names and value of the Theil's U statistic.

    Notes
    -----
    U(x, y) is one when x has zero entropy, including a constant x. This
    association convention does not make a constant predictor relevant.
    """

    x, y, sample_weight = _check_association_pair(x, y, sample_weight)

    tot_weight = np.sum(sample_weight)

    # Integer encoding
    y_encoded, y_unique = pd.factorize(y)
    x_encoded, x_unique = pd.factorize(x)

    # Extend bincount to cover all categories
    y_weights = np.bincount(y_encoded, weights=sample_weight, minlength=len(y_unique))
    x_weights = np.bincount(x_encoded, weights=sample_weight, minlength=len(x_unique))

    # Entropy calculations
    p_x = x_weights / tot_weight
    h_x = ss.entropy(p_x)

    h_xy = 0.0
    for unique_x in np.unique(x_encoded):
        x_mask = x_encoded == unique_x
        y_sub_weights = np.bincount(
            y_encoded[x_mask], weights=sample_weight[x_mask], minlength=len(y_unique)
        )
        p_xy = y_sub_weights / tot_weight
        p_y = y_weights / tot_weight
        # Avoid division by zero in log calculation
        valid_mask = (p_xy != 0) & (p_y != 0)
        h_xy += np.sum(p_xy[valid_mask] * np.log(p_y[valid_mask] / p_xy[valid_mask]))

    # A constant first argument has no uncertainty left to explain.
    u = 1.0 if h_x == 0 else (h_x - h_xy) / h_x

    # Check for floating point precision issues
    if h_x != 0 and (abs(u) < _PRECISION or abs(u - 1.0) < _PRECISION):
        rounded_u = round(u)
        warnings.warn(
            f"Rounded U = {u} to {rounded_u}. This is probably due to floating point precision issues.",
            RuntimeWarning,
        )
        u = rounded_u

    # Return as DataFrame or float
    if as_frame:
        return pd.DataFrame({"row": [x.name], "col": [y.name], "val": [u]})
    else:
        return u


def theils_u_matrix(X, sample_weight=None, n_jobs=1, handle_na="drop"):
    """theils_u_matrix theils_u_matrix computes the weighted Theil's U statistic for
    categorical-categorical association. This is an asymmetric coefficient: U(x,y) != U(y,x)
    U(x, y) means the uncertainty of x given y: value is on the range of [0,1] -
    where 0 means y provides no information about x, and 1 means y provides full information about x

    The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format.

    Parameters
    ----------
    X :  array-like of shape (n_samples, n_features)
        predictor dataframe
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.DataFrame
        The Theil's U matrix in a tidy (long) format.
    """

    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)

    # Cramer's V only for categorical columns
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")
    cat_cols = dtypes_dic["cat"]

    if cat_cols and (len(cat_cols) >= 2):
        # explicitly store the unique 2-permutation of column names
        # permutations and not combinations because U is asymmetric
        comb_list = [comb for comb in permutations(cat_cols, 2)]
        # parallelize jobs
        theil_u_matrix_entries = partial(
            _compute_matrix_entries, func_xyw=weighted_theils_u
        )
        lst = parallel_matrix_entries(
            func=theil_u_matrix_entries,
            df=X,
            comb_list=comb_list,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )
        return lst
    else:
        return None


def theils_u_series(X, target, sample_weight=None, n_jobs=1, handle_na="drop"):
    """theils_u_series computes the weighted Theil's U statistic for
    categorical-categorical association. This is an asymmetric coefficient: U(x,y) != U(y,x)
    U(x, y) means the uncertainty of x given y: value is on the range of [0,1] -
    where 0 means y provides no information about x, and 1 means y provides full information about x

    The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    target : str or int
        the predictor name or index with which to compute association
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.Series
        The Theil's U series.
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    if col_dtypes_dic[target] != "cat":
        raise TypeError("the target column is not categorical")

    # Cramer's V only for categorical columns
    cat_cols = dtypes_dic["cat"]

    if cat_cols:
        # parallelize jobs
        _theil_u = partial(_compute_series, func_xyw=weighted_theils_u)
        lst = parallel_df(
            func=_theil_u,
            df=X[cat_cols],
            series=X[target],
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )
        return lst
    else:
        return None


@symmetric_function
def cramer_v(x, y, sample_weight=None, as_frame=False):
    """
    Computes the weighted V statistic of two categorical predictors.

    Parameters
    ----------
    x : pd.Series of shape (n_samples,)
        The first categorical predictor.
    y : pd.Series of shape (n_samples,)
        The second categorical predictor, order doesn't matter, symmetrical association.
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None.
    as_frame : bool
        Return output as a DataFrame or a float.

    Returns
    -------
    pd.DataFrame or float
        Single row DataFrame with the predictor names and the statistic value, or the statistic as a float.
    """

    x, y, sample_weight = _check_association_pair(x, y, sample_weight)

    weighted_tab = pd.crosstab(
        x.to_numpy(), y.to_numpy(), sample_weight, aggfunc="sum"
    ).fillna(0)
    weighted_tab = weighted_tab.loc[
        weighted_tab.sum(axis=1) > 0, weighted_tab.sum(axis=0) > 0
    ]
    if min(weighted_tab.shape) < 2:
        raise ValueError(
            "Cramer's V requires at least two positive-mass categories per variable."
        )
    chi2 = ss.chi2_contingency(weighted_tab, correction=False)[0]
    tot_weight = sample_weight.sum()
    phi2 = chi2 / tot_weight
    r, k = weighted_tab.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (tot_weight - 1))
    rcorr = r - ((r - 1) ** 2) / (tot_weight - 1)
    kcorr = k - ((k - 1) ** 2) / (tot_weight - 1)
    v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    if as_frame:
        x_name = x.name if x.name else "var1"
        y_name = y.name if y.name else "var2"
        return pd.DataFrame(
            {"row": [x_name, y_name], "col": [y_name, x_name], "val": [v, v]}
        )
    else:
        return v


def cramer_v_matrix(X, sample_weight=None, n_jobs=1, handle_na="drop"):
    """cramer_v_matrix computes the weighted Cramer's V statistic for
    categorical-categorical association. This is a symmetric coefficient: V(x,y) = V(y,x)

    It uses the corrected Cramer's V statistics, itself based on the chi2 contingency
    table. The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format.

    Parameters
    ----------
    X :  array-like of shape (n_samples, n_features)
        predictor dataframe
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default 1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.DataFrame
        The Cramer's V matrix (lower triangular) in a tidy (long) format.
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    # Cramer's V only for categorical columns
    # in GLM supposed to be all the columns
    cat_cols = dtypes_dic["cat"]

    if cat_cols and (len(cat_cols) >= 2):
        # explicitly store the unique 2-combinations of column names
        comb_list = [comb for comb in combinations(cat_cols, 2)]
        _cramer_v_matrix_entries = partial(_compute_matrix_entries, func_xyw=cramer_v)
        lst = parallel_matrix_entries(
            func=_cramer_v_matrix_entries,
            df=X,
            comb_list=comb_list,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )
        return lst
    else:
        return None


def cramer_v_series(X, target, sample_weight=None, n_jobs=1, handle_na="drop"):
    """cramer_v_series computes the weighted Cramer's V statistic for
    categorical-categorical association. This is a symmetric coefficient: V(x,y) = V(y,x)

    It uses the corrected Cramer's V statistics, itself based on the chi2 contingency
    table. The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    target : str or int
        the predictor name or index with which to compute association
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default 1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.Series
        The Cramer's V series
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    if col_dtypes_dic[target] != "cat":
        raise TypeError("the target column is not categorical")

    # Cramer's V only for categorical columns
    cat_cols = dtypes_dic["cat"]

    if cat_cols:
        X = X[cat_cols]
        # parallelize jobs
        _cramer_v = partial(_compute_series, func_xyw=cramer_v)
        lst = parallel_df(
            func=_cramer_v,
            df=X[cat_cols],
            series=X[target],
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )
        # concatenate the results
        # v_df_list = list(chain(*v_df_list))
        return lst  # pd.concat(lst)
    else:
        return None


def _weighted_correlation_ratio(*args):
    """Calculates the Correlation Ratio (sometimes marked by the greek letter Eta)
    for categorical-continuous association.
    Answers the question - given a continuous value of a measurement, is it
    possible to know which category is it associated with?
    Value is in the range [0,1], where 0 means a category cannot be determined
    by a continuous measurement, and 1 means a category can be determined with
    absolute certainty.

    Based on the scikit-learn implementation of the unweighted version.

    Returns
    -------
    float
        value of the correlation ratio
    """
    # Convert to float array and compute weights
    args = _check_weighted_groups(args)
    ssbn, sswn = _weighted_group_sums(args)
    sstot = ssbn + sswn
    return np.array([np.sqrt(ssbn / sstot) if sstot != 0 else 0.0])


@symmetric_function
def correlation_ratio(x, y, sample_weight=None, as_frame=False):
    """Compute the weighted correlation ratio. The association between a continuous predictor (y)
    and a categorical predictor (x). It can be weighted.

    Parameters
    ----------
    x : pd.Series of shape (n_samples,)
        The categorical predictor vector
    y : pd.Series of shape (n_samples,)
        The continuous predictor
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    as_frame: bool
        return output as a dataframe or a float

    Returns
    -------
    float
        value of the correlation ratio
    """
    if not (isinstance(x, pd.Series) and isinstance(y, pd.Series)):
        raise TypeError(
            f"Both x and y must be pandas Series. The type of x: {type(x)} and The type of y: {type(y)}"
        )

    x, y, sample_weight = _check_association_pair(x, y, sample_weight)

    # Determine the categorical and continuous variables
    if _is_categorical_dtype(x.dtype):
        categorical, continuous = x, y
    elif _is_categorical_dtype(y.dtype):
        categorical, continuous = y, x
    else:
        raise TypeError(
            "One of the series must be categorical and the other numerical."
        )

    # Prepare arguments for the weighted correlation ratio calculation
    unique_categories = pd.unique(categorical)
    args = [
        (continuous.to_numpy()[categorical == category], sample_weight[categorical == category])
        for category in unique_categories
    ]

    # Compute the weighted correlation ratio
    v = _weighted_correlation_ratio(*args)[0]

    # Format the result
    if as_frame:
        x_name = x.name if x.name else "var1"
        y_name = y.name if y.name else "var2"
        return pd.DataFrame(
            {"row": [x_name, y_name], "col": [y_name, x_name], "val": [v, v]}
        )
    else:
        return v


def correlation_ratio_matrix(X, sample_weight=None, n_jobs=1, handle_na="drop"):
    """correlation_ratio_matrix computes the weighted Correlation Ratio for
    categorical-numerical association. This is a symmetric coefficient: CR(x,y) = CR(y,x)

    The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format.

    Parameters
    ----------
    X :  array-like of shape (n_samples, n_features)
        predictor dataframe
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default 1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.DataFrame
        The correlation ratio matrix (lower triangular) in a tidy (long) format.
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)

    # Cramer's V only for categorical columns
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")
    cat_cols = dtypes_dic["cat"]
    num_cols = dtypes_dic["num"]

    if cat_cols and num_cols:
        # explicitly store the unique 2-combinations of column names
        # the first one should be the categorical predictor
        comb_list = list(product(cat_cols, num_cols))
        # parallelize jobs
        corr_ratio_matrix_entries = partial(
            _compute_matrix_entries, func_xyw=correlation_ratio
        )
        lst = parallel_matrix_entries(
            func=corr_ratio_matrix_entries,
            df=X,
            comb_list=comb_list,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )
        return lst
    else:
        return None


def correlation_ratio_series(X, target, sample_weight=None, n_jobs=1, handle_na="drop"):
    """correlation_ratio_series computes the weighted correlation ration for
    categorical-numerical association. This is a symmetric coefficient: CR(x,y) = CR(y,x)

    The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format, a series.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    target : str or int
        the predictor name or index with which to compute association
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.Series
        The Correlation ratio series (lower triangular) in a tidy (long) format.
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    if col_dtypes_dic[target] == "cat":
        # if the target is categorical, pick only num predictors
        pred_list = dtypes_dic["num"]
    else:
        # if the target is numerical, the 2nd pred should be categorical
        pred_list = dtypes_dic["cat"]

    if pred_list:
        # parallelize jobs
        _corr_ratio = partial(_compute_series, func_xyw=correlation_ratio)
        lst = parallel_df(
            func=_corr_ratio,
            df=X[pred_list],
            series=X[target],
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )
        return lst
    else:
        return None


def wm(x, w):
    """wm computes the weighted mean

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        the target array
    w : array-like of shape (n_samples,)
        the sample weights array

    Returns
    -------
    float
        weighted mean
    """
    x = validate_target(x, np.size(x)).to_numpy()
    w = validate_sample_weight(w, len(x))
    if w is None:
        w = np.ones(len(x))
    return np.sum(x * w) / np.sum(w)


def wcov(x, y, w):
    """wcov computes the weighted covariance

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        the predictor 1 array
    y : array-like of shape (n_samples,)
        the predictor 2 array
    w : array-like of shape (n_samples,)
        the sample weights array

    Returns
    -------
    float
        weighted covariance
    """
    x, y, w = _check_association_pair(x, y, w)
    x, y = x.to_numpy(), y.to_numpy()
    return np.sum(w * (x - wm(x, w)) * (y - wm(y, w))) / np.sum(w)


def wcorr(x, y, w):
    """wcov computes the weighted Pearson correlation coefficient

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        the predictor 1 array
    y : array-like of shape (n_samples,)
        the predictor 2 array
    w : array-like of shape (n_samples,)
        the sample weights array

    Returns
    -------
    float
        weighted correlation coefficient
    """
    return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))


def wrank(x, w):
    """wrank computes the weighted rank

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        the target array
    w : array-like of shape (n_samples,)
        the sample weights array

    Returns
    -------
    float
        weighted rank
    """
    x = validate_target(x, np.size(x)).to_numpy()
    w = validate_sample_weight(w, len(x))
    if w is None:
        w = np.ones(len(x))
    (unique, arr_inv, counts) = np.unique(
        rankdata(x), return_counts=True, return_inverse=True
    )
    a = np.bincount(arr_inv, w)
    return (np.cumsum(a) - a)[arr_inv] + ((counts + 1) / 2 * (a / counts))[arr_inv]


def wspearman(x, y, w):
    """wcov computes the weighted Spearman correlation coefficient

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        the perdictor 1 array
    y : array-like of shape (n_samples,)
        the perdictor 2 array
    w : array-like of shape (n_samples,)
        the sample weights array

    Returns
    -------
    float
        Spearman weighted correlation coefficient
    """
    return wcorr(wrank(x, w), wrank(y, w), w)


@symmetric_function
def weighted_corr(x, y, sample_weight=None, as_frame=False, method="spearman"):
    """weighted_corr computes the weighted correlation coefficient (Pearson or Spearman)

    Parameters
    ----------
    x : pd.Series of shape (n_samples,)
        The categorical predictor vector
    y : pd.Series of shape (n_samples,)
        The continuous predictor
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    as_frame: bool
        return output as a dataframe or a float
    method : str
        either "spearman" or "pearson", by default "pearson"

    Returns
    -------
    float or pd.DataFrame
        weighted correlation coefficient
    """
    x, y, sample_weight = _check_association_pair(x, y, sample_weight)

    if method == "pearson":
        c = wcorr(x, y, sample_weight)
    else:
        c = wspearman(x, y, sample_weight)

    if as_frame:
        x_name = x.name if isinstance(x, pd.Series) else "var"
        y_name = y.name if isinstance(y, pd.Series) else "target"
        return pd.DataFrame(
            {"row": [x_name, y_name], "col": [y_name, x_name], "val": [c, c]}
        )
    else:
        return c


def wcorr_series(
    X, target, sample_weight=None, n_jobs=1, handle_na="drop", method="spearman"
):
    """wcorr_series computes the weighted correlation coefficient (Pearson or Spearman) for
    continuous-continuous association. This is an symmetric coefficient: corr(x,y) = corr(y,x)

    The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    target : str or int
        the predictor name or index with which to compute association
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default 1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"
    method : str
        either "spearman" or "pearson", by default "spearman"

    Returns
    -------
    pd.Series
        The weighted correlation series.
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    if col_dtypes_dic[target] == "cat":
        raise TypeError("the target column is categorical")

    num_cols = dtypes_dic["num"]
    y = X[target]

    if num_cols:
        _wcorr_method = partial(weighted_corr, method=method)
        # parallelize jobs
        _wcorr_method_series = partial(_compute_series, func_xyw=_wcorr_method)
        return parallel_df(
            func=_wcorr_method_series,
            df=X[num_cols],
            series=y,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )
    else:
        return None


def wcorr_matrix(X, sample_weight=None, n_jobs=1, handle_na="drop", method="spearman"):
    """wcorr_matrix computes the weighted correlation statistic for
    (Pearson or Spearman) for continuous-continuous association.
    This is an symmetric coefficient: corr(x,y) = corr(y,x)

    The computation is embarrassingly parallel and is distributed on available cores.
    Moreover, the statistic is computed for the unique combinations only and returned in a
    tidy (long) format.

    Parameters
    ----------
    X :  array-like of shape (n_samples, n_features)
        predictor dataframe
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"
    method : str
        either "spearman" or "pearson"

    Returns
    -------
    pd.DataFrame
        Both directions of each distinct numeric pair, sorted by val descending
        with stable pair-order ties. No diagonal entries are included. A single
        numeric column returns an empty row/col/val frame.
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    num_cols = dtypes_dic["num"]
    if num_cols:
        # explicitely store the unique 2-combinations of column names
        comb_list = [comb for comb in combinations(num_cols, 2)]
        n_jobs = _effective_n_jobs(n_jobs, len(comb_list))

        if (n_jobs > 1) or (method != "pearson") or not comb_list:
            _wcorr_method = partial(weighted_corr, method=method)
            _wcorr_method_entries = partial(
                _compute_matrix_entries, func_xyw=_wcorr_method
            )
            lst = parallel_matrix_entries(
                func=_wcorr_method_entries,
                df=X,
                comb_list=comb_list,
                sample_weight=sample_weight,
                n_jobs=n_jobs,
            )
            return lst
        else:
            corr = weighted_correlation_1cpu(X, sample_weight, handle_na)
            rows = []
            for left, right in comb_list:
                value = corr.at[left, right]
                rows.extend([(left, right, value), (right, left, value)])
            return (
                pd.DataFrame(rows, columns=["row", "col", "val"])
                .sort_values("val", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
    else:
        return None


def weighted_correlation_1cpu(X, sample_weight=None, handle_na="drop"):
    """weighted_correlation computes the lower triangular weighted correlation matrix
    using a single CPU, therefore using common numpy linear algebra

    Parameters
    ----------
    X :  array-like of shape (n_samples, n_features)
        predictor dataframe
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.DataFrame
        the lower triangular weighted correlation matrix in long format
    """
    # sanity checks
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    numeric_cols = dtypes_dic["num"]

    if numeric_cols:
        data = X[numeric_cols]
        col_idx = data.columns
        data = data.values
        sum_weights = sample_weight.sum()
        weighted_sum = np.dot(data.T, sample_weight)
        weighted_mean = weighted_sum / sum_weights
        demeaned = data - weighted_mean
        sum_of_squares = np.dot((demeaned**2).T, sample_weight)
        weighted_std = np.sqrt(sum_of_squares)
        weighted_cov = np.dot(sample_weight * demeaned.T, demeaned)
        weighted_corcoef = pd.DataFrame(
            weighted_cov / weighted_std / weighted_std[:, None],
            index=col_idx,
            columns=col_idx,
        )
        return weighted_corcoef
    else:
        return None


#################################
# association
# cat-cat + cat-cont + cont-cont
#################################


def association_series(
    X,
    target,
    features=None,
    sample_weight=None,
    nom_nom_assoc=weighted_theils_u,
    num_num_assoc=partial(weighted_corr, method="spearman"),
    nom_num_assoc=correlation_ratio,
    normalize=False,
    n_jobs=1,
    handle_na="drop",
):
    """
    Computes the association series for different types of predictors.

    This function calculates the association between the specified `target` and other predictors in `X`.
    It supports different types of associations: nominal-nominal, numerical-numerical, and nominal-numerical.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Predictor dataframe.
    target : str or int
        The predictor name or index with which to compute the association.
    features : list of str, optional
        List of features with which to compute the association. If None, all features in X are used.
    sample_weight : array-like, shape (n_samples,), optional
        The weight vector, by default None.
    nom_nom_assoc : callable
        Function to compute the nominal-nominal (categorical-categorical) association.
        It should take two pd.Series and an optional weight array, and return a single number.
    num_num_assoc : callable
        Function to compute the numerical-numerical association.
        It should take two pd.Series and return a single number.
    nom_num_assoc : callable
        Function to compute the nominal-numerical association.
        It should take two pd.Series and return a single number.
    normalize : bool, optional
        Whether to normalize the scores or not. If True, scores are normalized to the range [0, 1].
    n_jobs : int, optional
        The number of cores to use for the computation. The default, -1, uses all available cores.
    handle_na : str, optional
        How to handle NA values. Options are 'drop', 'fill', and None. The default, 'drop', drops rows with NA values.

    Returns
    -------
    pd.Series
        A series with all the association values with the target column, sorted in descending order.

    Raises
    ------
    TypeError
        If `features` is provided but is not a list of strings.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> association_series(X, 'sepal length (cm)', num_num_assoc=my_num_num_function)

    Notes
    -----
    The function dynamically selects the appropriate association method based on the data types
    of the target and other predictors. For numerical-numerical associations,
    it uses `num_num_assoc`; for nominal-nominal, `nom_nom_assoc`; and for nominal-numerical, `nom_num_assoc`.
    """
    # Input validation and preprocessing
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    if features is not None:
        if not all(isinstance(f, str) for f in features):
            raise TypeError("Features must be a list of strings.")
        data = X[features + [target]]
    else:
        data = X.copy()

    # Determine the data types
    dtypes_dic = create_dtype_dict(data, dic_keys="dtypes")

    # Compute associations based on data types
    if len(dtypes_dic["num"]) == len(data.columns):
        assoc_series = _callable_association_series_fn(
            num_num_assoc, data, target, sample_weight, n_jobs, "num-num"
        )
    elif len(dtypes_dic["cat"]) == len(data.columns):
        assoc_series = _callable_association_series_fn(
            nom_nom_assoc, data, target, sample_weight, n_jobs, "nom-nom"
        )
    else:
        assoc_series = _callable_association_series_fn(
            nom_num_assoc, data, target, sample_weight, n_jobs, "nom-num"
        )

        # Additional association for target-specific types
        if target in dtypes_dic["num"]:
            assoc_series_complement = _callable_association_series_fn(
                num_num_assoc, data, target, sample_weight, n_jobs, "num-num"
            )
            assoc_series = pd.concat([assoc_series, assoc_series_complement])
        elif target in dtypes_dic["cat"]:
            assoc_series_complement = _callable_association_series_fn(
                nom_nom_assoc, data, target, sample_weight, n_jobs, "nom-nom"
            )
            assoc_series = pd.concat([assoc_series, assoc_series_complement])

    # Normalize if required
    if normalize:
        assoc_series = (assoc_series - assoc_series.min()) / np.ptp(assoc_series)

    return assoc_series.sort_values(ascending=False, kind="stable")


def association_matrix(
    X,
    sample_weight=None,
    nom_nom_assoc=weighted_theils_u,
    num_num_assoc=weighted_corr,
    nom_num_assoc=correlation_ratio,
    n_jobs=1,
    handle_na="drop",
):
    """
    Computes the association matrix for continuous-continuous, categorical-continuous,
    and categorical-categorical predictors using specified callable functions.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor dataframe.
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None.
    nom_nom_assoc : callable
        Function to compute the categorical-categorical association.
    num_num_assoc : callable
        Function to compute the numerical-numerical association.
    nom_num_assoc : callable
        Function to compute the categorical-numerical association.
    n_jobs : int, optional
        The number of cores to use for the computation, by default 1.
    handle_na : str, optional
        How to handle NA values ('drop', 'fill', or None), by default "drop".

    Returns
    -------
    pd.DataFrame
        The association matrix.
    """
    # Input validation and preprocessing
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    n_cat_cols = len(dtypes_dic["cat"])
    n_num_cols = len(dtypes_dic["num"])

    df_to_concat = []

    # Numerical-Numerical Associations
    if n_num_cols >= 2:
        w_num_num = _callable_association_matrix_fn(
            assoc_fn=num_num_assoc,
            X=X,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
            kind="num-num",
        )
        df_to_concat.append(w_num_num)

    # Categorical-Numerical Associations
    if n_num_cols >= 1 and n_cat_cols >= 1:
        w_nom_num = _callable_association_matrix_fn(
            assoc_fn=nom_num_assoc,
            X=X,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
            kind="nom-num",
        )
        df_to_concat.append(w_nom_num)

    # Categorical-Categorical Associations
    if n_cat_cols >= 2:
        w_nom_nom = _callable_association_matrix_fn(
            assoc_fn=nom_nom_assoc,
            X=X,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
            kind="nom-nom",
        )
        df_to_concat.append(w_nom_nom)

    if not df_to_concat:
        return parallel_matrix_entries(
            _compute_matrix_entries, X, [], sample_weight, n_jobs
        )
    return pd.concat(df_to_concat, ignore_index=True)


def _callable_association_series_fn(
    assoc_fn, X, target, sample_weight=None, n_jobs=1, kind="nom-nom"
):
    """_callable_association_series_fn private function, utility for computing association series
    for a callable custom association

    Parameters
    ----------
    assoc_fn : callable
        a function which receives two `pd.Series` (and optionally a weight array) and returns a single number
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    target : str or int
        the predictor name or index with which to compute association
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default 1
    kind : str
        kind of association, either 'num-num' or 'nom-nom' or 'nom-num'

    Returns
    -------
    pd.Series
        the association series

    Raises
    ------
    ValueError
        if kind is not 'num-num' or 'nom-nom' or 'nom-num'
    """
    # X and weights were validated by association_series with its row policy.
    # Validate 'kind' parameter
    valid_kinds = ["num-num", "nom-nom", "nom-num"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}")

    # Create dtype dictionaries
    col_dtypes_dic = create_dtype_dict(X)
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    # Determine predictor list based on 'kind'
    if kind in ["nom-nom", "nom-num"]:
        if kind == "nom-nom" and col_dtypes_dic[target] != "cat":
            raise TypeError(
                "Target column is not categorical for 'nom-nom' association"
            )
        pred_list = (
            dtypes_dic["cat"]
            if kind == "nom-nom"
            else dtypes_dic["num"]
            if col_dtypes_dic[target] == "cat"
            else dtypes_dic["cat"]
        )
    else:  # kind == 'num-num'
        pred_list = dtypes_dic["num"]

    # Return None if no predictors are available
    if not pred_list:
        return None

    # Setup parallel computation
    _assoc_fn = partial(_compute_series, func_xyw=assoc_fn)
    return parallel_df(
        func=_assoc_fn,
        df=X[pred_list],
        series=X[target],
        sample_weight=sample_weight,
        n_jobs=n_jobs,
    )


def _callable_association_matrix_fn(
    assoc_fn, X, sample_weight=None, n_jobs=1, kind="nom-nom", cols_comb=None
):
    """_callable_association_matrix_fn private function, utility for computing association matrix
    for a callable custom association

    Parameters
    ----------
    assoc_fn : callable
        a function which receives two `pd.Series` (and optionally a weight array) and returns a single number
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1
    kind : str
        kind of association, either 'num-num' or 'nom-nom' or 'nom-num'
    cols_comb : list of 2-uple of str, optional
        combination of column names (list of 2-uples of strings)

    Returns
    -------
    pd.DataFrame
        the association matrix
    """
    # Validate 'kind' parameter
    valid_kinds = ["num-num", "nom-nom", "nom-num"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}")

    # Create dtype dictionaries
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    # Determine column combinations based on 'kind' and 'cols_comb'
    if cols_comb is None:
        if kind == "num-num":
            selected_cols = dtypes_dic["num"]
            cols_comb = create_col_combinations(assoc_fn, selected_cols)
        elif kind == "nom-nom":
            selected_cols = dtypes_dic["cat"]
            cols_comb = create_col_combinations(assoc_fn, selected_cols)
        elif kind == "nom-num":
            # cols_comb = create_col_combinations(assoc_fn, selected_cols)
            cols_comb = list(product(dtypes_dic["cat"], dtypes_dic["num"]))

    # Return None if no column combinations are available
    if not cols_comb:
        return None

    # Setup parallel computation
    _assoc_fn = partial(_compute_matrix_entries, func_xyw=assoc_fn)
    return parallel_matrix_entries(
        func=_assoc_fn,
        df=X,
        comb_list=cols_comb,
        sample_weight=sample_weight,
        n_jobs=n_jobs,
    )


################################
# Association predictor-target
################################


def f_oneway_weighted(*args):
    """
    Calculate the weighted F-statistic for one-way ANOVA (continuous target, categorical predictor).

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        The predictor dataframe.
    y : array-like of shape (n_samples,)
        The target vector.
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None.

    Returns
    -------
    float
        The value of the F-statistic.

    Notes
    -----
    Weights are frequency mass, not normalized proportions. N is the retained
    total weight and must exceed K, the number of positive-mass groups. Integer
    weights agree with literal row replication; fractional mass only defines
    a score. One group is rejected. Constant observations return NaN; distinct
    constant group levels return infinity. Both outputs have shape (1,).

    The F-statistic is calculated as:

    .. math::
        F = \\frac{SS_B / (K-1)}{SS_W / (N-K)}

    SS_B sums group mass times squared group-mean deviations from the global
    mean. SS_W sums frequency-weighted squared deviations from each group mean.
    """
    args = _check_weighted_groups(args)
    n_classes = len(args)
    if n_classes < 2:
        raise ValueError("ANOVA requires at least two positive-mass groups.")
    # compute the total weight per level
    weight_per_class = np.array([a[1].sum() for a in args])
    # total weight
    tot_weight = np.sum(weight_per_class)
    if tot_weight <= n_classes:
        raise ValueError("Frequency weight total must exceed the number of ANOVA groups.")
    ssbn, sswn = _weighted_group_sums(args)
    dfbn = n_classes - 1
    dfwn = tot_weight - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    if msw == 0.0 and msb == 0.0:
        warnings.warn("Features [0] are constant.", UserWarning)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    return f


def f_cat_regression(x, y, sample_weight=None, as_frame=False):
    """f_cat_regression computes the weighted ANOVA F-value for the provided sample.
    (continuous target, categorical predictor)

    Parameters
    ----------
    x : pd.Series of shape (n_samples,)
        The predictor vector, the first categorical predictor
    y : pd.Series of shape (n_samples,)
        second categorical predictor, order doesn't matter, symmetrical association
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    as_frame: bool
        return output as a dataframe or a float

    Returns
    -------
    float
        value of the F-statistic
    """
    x, y, sample_weight = _check_association_pair(x, y, sample_weight)

    # one 2-uple per level of the categorical feature x
    args = [
        (
            y[safe_mask(y, x == k)],
            sample_weight[safe_mask(sample_weight, x == k)],
        )
        for k in pd.unique(x)
    ]

    if as_frame:
        x_name = x.name if isinstance(x, pd.Series) else "var"
        y_name = y.name if isinstance(y, pd.Series) else "target"
        return pd.DataFrame(
            {"row": x_name, "col": y_name, "val": f_oneway_weighted(*args)[0]},
            index=[0],
        )
    else:
        return f_oneway_weighted(*args)[0]


def f_cat_regression_parallel(X, y, sample_weight=None, n_jobs=1, handle_na="drop"):
    """f_cat_regression_parallel computes the weighted ANOVA F-value for the provided categorical predictors
    using parallelization of the code (continuous target, categorical predictor).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    y : array-like of shape (n_samples,)
        The target vector
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default 1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.Series
        the value of the F-statistic for each predictor

    Notes
    -----
    Uses the frequency-weighted ANOVA contract of f_oneway_weighted. Predictors
    constant on retained positive-weight rows score zero in this relevance
    wrapper; the direct ANOVA API still rejects fewer than two usable groups.
    """

    # Cramer's V only for categorical columns
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    cat_cols = dtypes_dic["cat"]

    X, y, sample_weight = _check_relevance_input(X, y, sample_weight, handle_na)
    target = y.name

    X = X.drop(columns=[target])

    # parallelize jobs
    _f_stat_cat = partial(_compute_series, func_xyw=f_cat_regression)
    varying_cols = [c for c in cat_cols if X.loc[sample_weight > 0, c].nunique() > 1]
    constants = pd.Series(0.0, index=[c for c in cat_cols if c not in varying_cols])
    result = parallel_df(
        func=_f_stat_cat,
        df=X[varying_cols],
        series=y,
        sample_weight=sample_weight,
        n_jobs=n_jobs,
    )
    return pd.concat([result, constants]).sort_values(ascending=False, kind="stable")


def f_cont_regression_parallel(
    X, y, sample_weight=None, n_jobs=-1, force_finite=True, handle_na="drop"
):
    """Univariate linear regression tests returning F-statistic.

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.
    This is done in 2 steps:
    1. The cross-correlation between each regressor and the target is computed using:
           E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))
    2. It is converted to an F score ranks features in the same order if all the features
       are positively correlated with the target.
    Note that it is therefore recommended as a feature selection criterion to identify
    potentially predictive features for a downstream classifier, irrespective of
    the sign of the association with the target variable.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The predictor dataframe.
    y : array-like of shape (n_samples,)
        The target vector.
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None.
    n_jobs : int, optional
        The number of cores to use for the computation, by default -1.
    handle_na : str, optional
        Either drop rows with NaN, fill NaN with 0, or do nothing, by default "drop".
    force_finite : bool, optional
        Whether or not to force the F-statistics and associated p-values to
        be finite. There are two cases where the F-statistic is expected to not
        be finite:
        - when the target `y` or some features in `X` are constant. In this
          case, the Pearson's R correlation is not defined leading to obtain
          `np.nan` values in the F-statistic and p-value. When
          `force_finite=True`, the F-statistic is set to `0.0` and the
          associated p-value is set to `1.0`.
        - when a feature in `X` is perfectly correlated (or
          anti-correlated) with the target `y`. In this case, the F-statistic
          is expected to be `np.inf`. When `force_finite=True`, the F-statistic
          is set to `np.finfo(dtype).max`.

    Returns
    -------
    f_statistic : array-like of shape (n_features,)
        F-statistic for each feature.

    Notes
    -----
    Uses Pearson correlation and retained frequency mass N, with N - 2 degrees
    of freedom; N <= 2 is rejected. Integer weights reproduce row replication.
    Weights are never normalized. Fractional mass defines a score only.
    """
    X, y, sample_weight = _check_relevance_input(X, y, sample_weight, handle_na)
    target = y.name

    deg_of_freedom = sample_weight.sum() - 2
    if deg_of_freedom <= 0:
        raise ValueError("Frequency weight total must exceed 2 for regression relevance.")
    correlation_coefficient = wcorr_series(
        X, target, sample_weight, n_jobs, handle_na, method="pearson"
    )
    constant = X.loc[sample_weight > 0].nunique() <= 1
    constant_cols = constant[constant].index.intersection(correlation_coefficient.index)
    correlation_coefficient.loc[constant_cols] = np.nan
    if constant[target]:
        correlation_coefficient[:] = np.nan

    corr_coef_squared = (correlation_coefficient**2).clip(upper=1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom

    if force_finite and not np.isfinite(f_statistic).all():
        # case where there is a perfect (anti-)correlation
        # f-statistics can be set to the maximum and p-values to zero
        mask_inf = np.isinf(f_statistic)
        f_statistic[mask_inf] = np.finfo(f_statistic.dtype).max
        # case where the target or some features are constant
        # f-statistics would be minimum and thus p-values large
        mask_nan = np.isnan(f_statistic)
        f_statistic[mask_nan] = 0.0

    return f_statistic.drop(labels=[target]).sort_values(ascending=False)


def f_stat_regression_parallel(
    X, y, sample_weight=None, n_jobs=-1, force_finite=True, handle_na="drop"
):
    """Bounded mixed regression relevance for ranking.

    Applies F / (1 + F) to both numeric and categorical F-style scores, without
    centering or rescaling by the other predictors. Returns a labelled Series
    on [0, 1]: zero means no relevance and perfect dependence maps to one.
    This shared bounded scale is a ranking heuristic, not a common effect size
    or calibrated mixed-type significance. Large F values may round to one.

    X is a predictor DataFrame; y and sample_weight are positional array-like
    vectors. Weights retain the frequency-mass contract of the underlying
    relevance functions, including their degrees-of-freedom checks. n_jobs
    controls workers; handle_na is 'drop', 'fill', or None.

    With force_finite=True (default), undefined constant scores map to zero.
    With False, undefined scores remain NaN; infinity maps to one in either
    case. The primitive f_cont_* and f_cat_* APIs still return raw scores.
    Underlying validation is preserved, including one-class ANOVA rejection.
    """
    f_stat_cont_series = f_cont_regression_parallel(
        X,
        y,
        sample_weight,
        n_jobs,
        force_finite=force_finite,
        handle_na=handle_na,
    )
    f_stat_cat_series = f_cat_regression_parallel(
        X, y, sample_weight, n_jobs, handle_na=handle_na
    )

    scores = pd.concat([f_stat_cont_series, f_stat_cat_series]).replace(
        np.inf, np.finfo(float).max
    )
    if force_finite:
        scores = scores.fillna(0.0)
    relevance = scores / (1 + scores)
    return relevance.sort_values(ascending=False, kind="stable")


def f_cont_classification(x, y, sample_weight=None, as_frame=False):
    """f_cont_classification computes the weighted ANOVA F-value for the provided sample.
    Categorical target, continuous predictor.

    Parameters
    ----------
    x : pd.Series of shape (n_samples,)
        The predictor vector, the first categorical predictor
    y : pd.Series of shape (n_samples,)
        second categorical predictor, order doesn't matter, symmetrical association
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    as_frame: bool
        return output as a dataframe or a float

    Returns
    -------
    float :
        value of the F-statistic
    """
    x, y, sample_weight = _check_association_pair(x, y, sample_weight)

    # one 2-uple per level of the categorical target y, continuous predictor x
    args = [
        (
            x[safe_mask(x, y == k)],
            sample_weight[safe_mask(sample_weight, y == k)],
        )
        for k in pd.unique(y)
    ]

    if as_frame:
        x_name = x.name if isinstance(x, pd.Series) else "var"
        y_name = y.name if isinstance(y, pd.Series) else "target"
        return pd.DataFrame(
            {"row": x_name, "col": y_name, "val": f_oneway_weighted(*args)[0]},
            index=[0],
        )
    else:
        return f_oneway_weighted(*args)[0]


def f_cont_classification_parallel(
    X, y, sample_weight=None, n_jobs=-1, handle_na="drop"
):
    """f_cont_classification_parallel computes the weighted ANOVA F-value
    for the provided categorical predictors using parallelization of the code.
    Categorical target, continuous predictor.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The set of regressors that will be tested sequentially
    y : array-like of shape (n_samples,)
        The target vector
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    pd.Series
        the value of the F-statistic for each predictor
    """
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")

    num_cols = dtypes_dic["num"]

    X, y, sample_weight = _check_relevance_input(X, y, sample_weight, handle_na)
    target = y.name
    X = X.drop(target, axis=1)

    # parallelize jobs
    _f_stat_cont_clf = partial(_compute_series, func_xyw=f_cont_classification)
    return parallel_df(
        func=_f_stat_cont_clf,
        df=X[num_cols],
        series=y,
        sample_weight=sample_weight,
        n_jobs=n_jobs,
    )


def f_cat_classification_parallel(
    X,
    y,
    sample_weight=None,
    n_jobs=-1,
    force_finite=True,
    handle_na="drop",
):
    """
    Univariate information dependence, expressed as a ranking heuristic.

    It ranks features in the same order if all the features are positively correlated with the target.
    Note that it is therefore recommended as a feature selection criterion to identify
    potentially predictive features for a downstream classifier, irrespective of
    the sign of the association with the target variable.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The predictor dataframe.
    y : array-like of shape (n_samples,)
        The target vector.
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None.
    n_jobs : int, optional
        The number of cores to use for the computation, by default -1.
    handle_na : str, optional
        Either drop rows with NaN, fill NaN with 0, or do nothing, by default "drop".
    force_finite : bool, optional
        Whether or not to force the F-statistics and associated p-values to
        be finite. There are two cases where the F-statistic is expected to not
        be finite:
            - when the target `y` or some features in `X` are constant. In this
              case, the Pearson's R correlation is not defined leading to obtain
              `np.nan` values in the F-statistic and p-value. When
              `force_finite=True`, the F-statistic is set to `0.0` and the
              associated p-value is set to `1.0`.
            - when a feature in `X` is perfectly correlated (or
              anti-correlated) with the target `y`. In this case, the F-statistic
              is expected to be `np.inf`. When `force_finite=True`, the F-statistic
              is set to `np.finfo(dtype).max`.

    Returns
    -------
    f_statistic : array-like of shape (n_features,)
        U-to-F ranking score for each feature; it has no asserted F null law.

    Notes
    -----
    Uses U(x, y)**2 / (1 - U(x, y)**2) * (N - 2), where N is retained frequency
    mass and must exceed 2. Integer counts reproduce replication. Fractional
    mass defines a score only; relative weights are not normalized. Predictors
    constant on positive-weight retained rows, or a constant response, score
    zero regardless of the association API's zero-entropy convention. Perfect
    nonconstant dependence returns infinity, or maximum float with force_finite.
    """
    X, y, sample_weight = _check_relevance_input(X, y, sample_weight, handle_na)
    target = y.name

    deg_of_freedom = sample_weight.sum() - 2
    if deg_of_freedom <= 0:
        raise ValueError("Frequency weight total must exceed 2 for categorical relevance.")

    theils_u_coef = theils_u_series(X, target, sample_weight, n_jobs, handle_na)
    theils_u_coef_squared = theils_u_coef**2

    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic = (
            theils_u_coef_squared / (1 - theils_u_coef_squared) * deg_of_freedom
        )

    if force_finite and not np.isfinite(f_statistic).all():
        # case where there is a perfect (anti-)correlation
        # f-statistics can be set to the maximum and p-values to zero
        mask_inf = np.isinf(f_statistic)
        f_statistic[mask_inf] = np.finfo(f_statistic.dtype).max
        # case where the target or some features are constant
        # f-statistics would be minimum and thus p-values large
        mask_nan = np.isnan(f_statistic)
        f_statistic[mask_nan] = 0.0

    # The entropy convention U(constant, y)=1 is not predictive information.
    constant = X.loc[sample_weight > 0].nunique() <= 1
    constant_cols = constant[constant].index.intersection(f_statistic.index)
    f_statistic.loc[constant_cols] = 0.0
    if constant[target]:
        f_statistic[:] = 0.0

    f_statistic = f_statistic.reindex(X.columns[X.columns.isin(f_statistic.index)])
    return f_statistic.drop(labels=[target]).sort_values(ascending=False)


def f_stat_classification_parallel(
    X, y, sample_weight=None, n_jobs=1, force_finite=True, handle_na="drop"
):
    """Bounded mixed classification relevance for ranking.

    Applies F / (1 + F) to both numeric and categorical F-style scores, without
    centering or rescaling by the other predictors. Returns a labelled Series
    on [0, 1]: zero means no relevance and perfect dependence maps to one.
    This shared bounded scale is a ranking heuristic, not a common effect size
    or calibrated mixed-type significance. Large F values may round to one.

    X is a predictor DataFrame; y and sample_weight are positional array-like
    vectors. Weights retain the frequency-mass contract of the underlying
    relevance functions, including their degrees-of-freedom checks. n_jobs
    controls workers; handle_na is 'drop', 'fill', or None.

    With force_finite=True (default), undefined constant scores map to zero.
    With False, undefined scores remain NaN; infinity maps to one in either
    case. The primitive f_cont_* and f_cat_* APIs still return raw scores.
    Underlying validation is preserved, including one-class ANOVA rejection.
    """
    f_stat_cont_series = f_cont_classification_parallel(
        X, y, sample_weight, n_jobs, handle_na=handle_na
    )
    f_stat_cat_series = f_cat_classification_parallel(
        X,
        y,
        sample_weight,
        n_jobs,
        handle_na=handle_na,
        force_finite=force_finite,
    )

    scores = pd.concat([f_stat_cont_series, f_stat_cat_series]).replace(
        np.inf, np.finfo(float).max
    )
    if force_finite:
        scores = scores.fillna(0.0)
    relevance = scores / (1 + scores)
    return relevance.sort_values(ascending=False, kind="stable")


############
# Utilities
############


def _check_relevance_input(X, y, sample_weight, handle_na):
    """Apply one row policy without overwriting a predictor with the response."""
    y = validate_target(y, len(X))
    target = y.name if y.name is not None else "target"
    while target in X.columns:
        target = f"{target}_"
    X = X.copy()
    X[target] = y.array
    X, sample_weight = _check_association_input(X, sample_weight, handle_na)
    return X, X[target].copy(), sample_weight


def _weighted_group_sums(args):
    """Center before squaring, and accumulate within-group residuals directly."""
    origin = args[0][0][0]
    centered = [(v - origin, w) for v, w in args]
    masses = np.array([w.sum() for _, w in centered])
    means = np.array([np.average(v, weights=w) for v, w in centered])
    mean = np.average(means, weights=masses)
    between = np.sum(masses * (means - mean) ** 2)
    within = sum(np.sum(w * (v - m) ** 2) for (v, w), m in zip(centered, means))
    return between, within


def _check_association_pair(x, y, sample_weight):
    """Direct statistics require complete inputs; zero-weight rows add no groups."""
    x = validate_target(x, np.size(x))
    y = validate_target(y, len(x))
    weights = validate_sample_weight(sample_weight, len(x))
    if weights is None:
        weights = np.ones(len(x))
    return x, y, weights


def _check_weighted_groups(args):
    """Validate every group, then omit groups without positive weight mass."""
    groups = []
    for values, weights in args:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
            raise ValueError(
                "Each group must contain aligned one-dimensional observations and weights."
            )
        if not np.isfinite(values).all():
            raise ValueError("Group observations must be finite.")
        groups.append((values, weights))
    validate_sample_weight(np.concatenate([w for _, w in groups]) if groups else [])
    return [(v[w > 0], w[w > 0]) for v, w in groups if (w > 0).any()]


def _check_association_input(X, sample_weight=None, handle_na="drop"):
    """_check_association_input private function. Check the inputs,
    convert X to a pd.DataFrame if needed, adds column names if non are provided.
    Check if the sample_weight is None or of the right dimensionality and handle NA
    according to the chosen methods (drop, fill, None).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        predictor dataframe
    sample_weight : array-like of shape (n_samples,), optional
        The weight vector, by default None
    handle_na : str, optional
        either drop rows with na, fill na with 0 or do nothing, by default "drop"

    Returns
    -------
    tuple
        the dataframe and the sample weights

    Raises
    ------
    ValueError
        if sample_weight contains NA
    """

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"pred_{i}" for i in range(X.shape[1])])

    sample_weight = validate_sample_weight(sample_weight, len(X))
    if sample_weight is None:
        sample_weight = np.ones(len(X))

    # handle nans
    if handle_na is None:
        pass
    elif handle_na == "drop":
        # mask the na
        na_mask = (~X.isnull().any(axis=1)).values
        X, sample_weight = X.loc[na_mask, :], sample_weight[na_mask]
    else:
        X = X.fillna(0)
    if len(X) == 0:
        raise ValueError("No observations remain after the missing-value policy.")
    sample_weight = validate_sample_weight(sample_weight, len(X))
    return X, sample_weight


def is_list_of_str(str_list):
    """Raise an exception if ``str_list`` is not a list of strings
    Parameters
    ----------
    str_list : list
        to list to be tested

    Raises
    ------
    TypeError
        if ``str_list`` is not a ``list[str]``
    """
    if str_list is not None:
        if not (
            isinstance(str_list, list) and all(isinstance(s, str) for s in str_list)
        ):
            return False
        else:
            return True


def matrix_to_xy(df, columns=None, reset_index=False):
    """matrix_to_xy wide to long format of the association matrix

    Parameters
    ----------
    df : pd.DataFrame
        the wide format of the association matrix
    columns : list of str, optional
        list of column names, by default None
    reset_index : bool, optional
        wether to reset_index or not, by default False

    Returns
    -------
    pd.DataFrame
        the long format of the association matrix
    """
    bool_index = np.tril(np.ones(df.shape), 0).astype(bool)
    xy = df.where(bool_index).stack().dropna()
    if reset_index:
        xy = xy.reset_index()
        xy.columns = columns or ["row", "col", "val"]
    return xy


def xy_to_matrix(xy):
    """xy_to_matrix long to wide format of the association matrix

    Parameters
    ----------
    xy : pd.DataFrame
        the long format of the association matrix, 3 columns.

    Returns
    -------
    pd.DataFrame

    """
    xy = xy.pivot(index="row", columns="col").fillna(0)
    xy.columns = xy.columns.droplevel(0)
    return xy.rename_axis(None, axis=1).rename_axis(None, axis=0)


###############
# visualization
###############


def cluster_sq_matrix(sq_matrix, method="ward"):
    """
    Apply agglomerative clustering to sort a square correlation matrix.

    Parameters
    ----------
    sq_matrix : pd.DataFrame
        A nonempty square correlation matrix. A singleton is returned as a copy
        without clustering, preserving its value even when undefined.
    method : str, optional
        The linkage method, by default "ward".

    Returns
    -------
    pd.DataFrame
        A sorted square matrix.

    Example
    -------
    >>> from some_module import association_matrix, cluster_sq_matrix

    >>> assoc = association_matrix(iris_df, plot=False)
    >>> assoc_clustered = cluster_sq_matrix(assoc, method="complete")
    """
    if sq_matrix.shape[0] == 0 or sq_matrix.shape[0] != sq_matrix.shape[1]:
        raise ValueError("sq_matrix must be a nonempty square matrix.")
    if len(sq_matrix) == 1:
        return sq_matrix.copy()
    d = pdist(sq_matrix.values)
    L = sch.linkage(d, method=method)
    ind = sch.fcluster(L, 0.5 * d.max(), "distance")
    columns = [sq_matrix.columns.tolist()[i] for i in list((np.argsort(ind)))]
    sq_matrix = sq_matrix.reindex(columns, axis=1)
    sq_matrix = sq_matrix.reindex(columns, axis=0)
    return sq_matrix


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """heatmap Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data : array-like of shape (M, N)
        matrix to plot
    row_labels : array-like of shape (M,)
        labels for the rows
    col_labels : array-like of shape (N,)
        labels for the columns
    ax : matplotlib.axes.Axes, optional
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one, by default None
    cbar_kw : dict, optional
         A dictionary with arguments to `matplotlib.Figure.colorbar`, by default None
    cbarlabel : str, optional
        The label for the colorbar, by default ""
    kwargs : dict, optional
        All other arguments are forwarded to `imshow`.

    Returns
    -------
    tuple
        imgshow and cbar objects
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)

    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    cbar = ax.figure.colorbar(im, cax=ax_cb, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """annotate_heatmap annotates a heatmap

    Parameters
    ----------
    im : matplotlib.axes.Axes
        The AxesImage to be labeled
    data : array-like of shape (M, N), optional
        data to illustrate, if none is provided the function retrieves
        the array of the mlp obkect, by default None
    valfmt : str, optional
        annotation formating, by default "{x:.2f}"
    textcolors : tuple, optional
        A pair of colors.  The first is used for values below a threshold,
        the second for those above, by default ("black", "white")
    threshold : float, optional
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation, by default None
    textkw : dict, optional
        All other arguments are forwarded to mpl annotation.

    Returns
    -------
    _type_
        _description_
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_association_matrix(
    assoc_mat,
    suffix_dic=None,
    ax=None,
    cmap="PuOr",
    cbarlabel=None,
    figsize=None,
    show=True,
    cbar_kw=None,
    imgshow_kw=None,
    annotate=False,
):
    """plot_association_matrix renders the sorted associations/correlation matrix.
    The sorting is done using hierarchical clustering,
    very like in seaborn or other packages.
    Categorical(nom): uncertainty coefficient & correlation ratio from 0 to 1.
    The uncertainty coefficient is asymmetrical, (approximating how much the elements on the
    left PROVIDE INFORMATION on elements in the row). Continuous(con): symmetrical numerical
    correlations (Spearman's) from -1 to 1

    Parameters
    ----------
    assoc_mat : pd.DataFrame
        the square association frame
    suffix_dic : Dict[str, str], optional
        dictionary of data type for adding suffixes to column names
        in the plotting utility for association matrix, by default None
    ax : matplotlib.axes.Axes, optional
        _description_, by default None
    cmap : str, optional
        the colormap. Please use a scientific colormap. See the ``scicomap`` package, by default "PuOr"
    cbarlabel : str, optional
        the colorbar label, by default None
    figsize : Tuple[float, float], optional
        figure size in inches, by default None
    show : bool, optional
        Whether or not to display the figure, by default True
    cbar_kw : Dict, optional
        colorbar kwargs, by default None
    imgshow_kw : Dict, optional
        imgshow kwargs, by default None
    annotate : bool
        Whether to annotate or not the colormap

    Returns
    -------
    matplotlib.figure and matplotlib.axes.Axes
        the figure and the axes
    """
    assoc_mat = cluster_sq_matrix(assoc_mat)

    # default size if None
    if figsize is None:
        ncol = len(assoc_mat)
        figsize = (ncol / 2.5, ncol / 2.5)

    # provide default to the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # provide default to imshow
    if imgshow_kw is None:
        imgshow_kw = {"vmin": -1, "vmax": 1}

    # provide default to the colorbar
    if cbar_kw is None:
        cbar_kw = {"ticks": [-1, -0.5, 0, 0.5, 1]}

    # rename the columns for keeping track of num vs cat columns
    if suffix_dic is not None:
        rename_dic = {c: f"{c}_{suffix_dic[c]}" for c in assoc_mat.columns}
        assoc_mat = assoc_mat.rename(columns=rename_dic)
        assoc_mat = assoc_mat.rename(index=rename_dic)

    im, cbar = heatmap(
        assoc_mat.values,
        assoc_mat.columns,
        assoc_mat.columns,
        ax=ax,
        cmap=cmap,
        cbarlabel=cbarlabel,
        cbar_kw=cbar_kw,
        **imgshow_kw,
    )

    if annotate:
        texts = annotate_heatmap(im, valfmt="{x:.1f}", textcolors=("white", "black"))

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_association_matrix_int(
    assoc_mat, suffix_dic=None, cmap="PuOr", figsize=(800, 600), cluster_matrix=True
):
    """Plot the interactive sorted associations/correlation matrix.
    The sorting is done using hierarchical clustering,
    very like in seaborn or other packages.
    Categorical(nom): uncertainty coefficient & correlation ratio from 0 to 1.
    The uncertainty coefficient is assymmetrical, (approximating how much the elements on the
    left PROVIDE INFORMATION on elements in the row). Continuous(con): symmetrical numerical
    correlations (Spearman's) from -1 to 1

    Parameters
    ----------
    assoc_mat : pd.DataFrame
        the square association frame
    suffix_dic : Dict[str, str], optional
        dictionary of data type for adding suffixes to column names
        in the plotting utility for association matrix, by default None
    cmap : str, optional
        the colormap. Please use a scientific colormap. See the ``scicomap`` package, by default "PuOr"
    figsize : Tuple[float, float], optional
        figure size in inches, by default None
    cluster_matrix : bool
        whether or not to cluster the square matrix, by default True

    Returns
    -------
    panel.Column
        the panel object
    """
    try:
        import holoviews as hv
    except ImportError:
        raise ImportError(
            "Holoviews is not installed. Please install it using 'pip install holoviews'."
        )

    try:
        import panel as pn
    except ImportError:
        raise ImportError(
            "Panel is not installed. Please install it using 'pip install panel'."
        )

    cmap = cmap if cmap is not None else "coolwarm"

    # rename the columns for keeping track of num vs cat columns
    if suffix_dic is not None:
        rename_dic = {c: f"{c}_{suffix_dic[c]}" for c in assoc_mat.columns}
        assoc_mat = assoc_mat.rename(columns=rename_dic)
        assoc_mat = assoc_mat.rename(index=rename_dic)

    if cluster_matrix:
        assoc_mat = cluster_sq_matrix(assoc_mat)

    heatmap = hv.HeatMap((assoc_mat.columns, assoc_mat.index, assoc_mat)).redim.range(
        z=(-1, 1)
    )

    heatmap.opts(
        tools=["tap", "hover"],
        height=figsize[1],
        width=figsize[0],
        toolbar="left",
        colorbar=True,
        cmap=cmap,
        fontsize={"title": 12, "ticks": 12, "minor_ticks": 12},
        xrotation=90,
        invert_xaxis=False,
        invert_yaxis=True,
        xlabel="",
        ylabel="",
    )
    title_str = "**Continuous (con) and Categorical (nom) Associations **"
    sub_title_str = (
        "*Categorical(nom): uncertainty coefficient & correlation ratio from 0 to 1. The uncertainty "
        "coefficient is asymmetrical, (approximating how much the elements on the "
        "left PROVIDE INFORMATION on elements in the row). Continuous(con): symmetrical numerical "
        "correlations (Spearman's) from -1 to 1*"
    )
    panel_layout = pn.Column(
        pn.pane.Markdown(title_str, align="start", style={"color": "#575757"}),  # bold
        pn.pane.Markdown(
            sub_title_str, align="start", style={"color": "#575757"}
        ),  # italic
        heatmap,
        background="#ebebeb",
    )

    gc.enable()
    del assoc_mat
    gc.collect()
    return panel_layout
