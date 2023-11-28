"""Parallelize Pandas

This module provides utilities for parallelizing operations on pd.DataFrame

Module Structure:
-----------------
- ``parallel_matrix_entries`` for parallelizing operations returning a matrix (2D) (apply on pairs of columns)
- ``parallel_df`` for parallelizing operations returning a series (1D) (apply on a single column at a time)
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from itertools import chain


def parallel_matrix_entries(func, df, comb_list, sample_weight=None, n_jobs=-1):
    """parallel_matrix_entries applies a function to each chunk of
    combinaison of columns of the dataframe, distributed by cores.
    This is similar to https://github.com/smazzanti/mrmr/mrmr/pandas.py


    Parameters
    ----------
    func : callable
        function to be applied to each column
    df : pd.DataFrame
        the dataframe on which to apply the function
    comb_list : list of tuples of str
        Pairs of column names corresponding to the entries
    sample_weight : pd.Series or np.array, optional
        The weight vector, if any, of shape (n_samples,), by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1

    Returns
    -------
    pd.DataFrame
        concatenated results into a single pandas DF
    """
    # Determining the number of jobs
    n_jobs = cpu_count() if n_jobs == -1 else min(cpu_count(), n_jobs)

    if n_jobs == 1:
        lst = func(X=df, sample_weight=sample_weight, comb_list=comb_list)
        return pd.concat(lst, ignore_index=True).sort_values("val", ascending=False)

    comb_chunks = np.array_split(comb_list, n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(X=df, sample_weight=sample_weight, comb_list=comb_chunk)
        for comb_chunk in comb_chunks
    )
    # Directly return the single DataFrame if lst contains only one element
    if len(lst) == 1:
        return lst[0]
    else:
        return pd.concat(list(chain(*lst)), ignore_index=True)


def parallel_df(func, df, series, sample_weight=None, n_jobs=-1):
    """parallel_df apply a function to each column of the dataframe, distributed by cores.
    This is similar to https://github.com/smazzanti/mrmr/mrmr/pandas.py

    Parameters
    ----------
    func : callable
        function to be applied to each column
    df : pd.DataFrame
        the dataframe on which to apply the function
    series : pd.Series
        series (target) used by the function
    sample_weight : pd.Series or np.array, optional
        The weight vector, if any, of shape (n_samples,), by default None
    n_jobs : int, optional
        the number of cores to use for the computation, by default -1

    Returns
    -------
    pd.DataFrame
        concatenated results into a single pandas DF
    """
    # Determining the number of jobs
    n_jobs = cpu_count() if n_jobs == -1 else min(cpu_count(), n_jobs)

    if n_jobs == 1:
        lst = func(df, series, sample_weight).sort_values(ascending=False)

        return (
            pd.concat(lst, ignore_index=True).sort_values("val", ascending=False)
            if isinstance(lst, list)
            else lst
        )
    else:
        col_chunks = np.array_split(range(len(df.columns)), n_jobs)
        lst = Parallel(n_jobs=n_jobs)(
            delayed(func)(df.iloc[:, col_chunk], series, sample_weight)
            for col_chunk in col_chunks
        )

        return pd.concat(lst).sort_values(ascending=False)


def _compute_series(
    X,
    y,
    sample_weight=None,
    func_xyw=None,
):
    """_compute_series is a utility function for computing the series
    resulting of the ``apply``

    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        The set of regressors that will be tested sequentially
    y : pd.Series or np.array, of shape (n_samples,)
        The target vector
    sample_weight : pd.Series or np.array, of shape (n_samples,), optional
        The weight vector, if any, by default None
    func_xyw : callable, optional
        callable (function) for computing the individual elements of the series
        takes two mandatory inputs (x and y) and an optional input w, sample_weights
    """

    def _closure_compute_series(x, y, sample_weight):
        x_not_na = ~x.isna()
        if x_not_na.sum() == 0:
            return 0
        return func_xyw(
            x=x[x_not_na],
            y=y[x_not_na],
            sample_weight=sample_weight[x_not_na],
            as_frame=False,
        )

    return X.apply(
        lambda col: _closure_compute_series(x=col, y=y, sample_weight=sample_weight)
    ).fillna(0.0)


def _compute_matrix_entries(
    X,
    comb_list,
    sample_weight=None,
    func_xyw=None,
):
    """base closure for computing matrix entries appling a function to each chunk of
    combinaison of columns of the dataframe, distributed by cores.
    This is similar to https://github.com/smazzanti/mrmr/mrmr/pandas.py

    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        The set of regressors that will be tested sequentially
    sample_weight : pd.Series or np.array, of shape (n_samples,), optional
        The weight vector, if any, by default None
    func_xyw : callable, optional
        callable (function) for computing the individual elements of the matrix
        takes two mandatory inputs (x and y) and an optional input w, sample_weights
    comb_list : list of 2-uple of str
        Pairs of column names corresponding to the entries

    Returns
    -------
    pd.DataFrame
        concatenated results into a single pandas DF
    """
    v_df_list = [
        func_xyw(x=X[comb[0]], y=X[comb[1]], sample_weight=sample_weight, as_frame=True)
        for comb in comb_list
    ]

    return v_df_list
