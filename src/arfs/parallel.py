"""Parallelize Pandas

This module provides utilities for parallelizing operations on pd.DataFrame

Module Structure:
-----------------
- ``parallel_matrix_entries`` for parallelizing operations returning a matrix (2D) (apply on pairs of columns)
- ``parallel_df`` for parallelizing operations returning a series (1D) (apply on a single column at a time)
"""

from itertools import chain
from numbers import Integral

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs


def _effective_n_jobs(n_jobs, work_size):
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    return min(effective_n_jobs(n_jobs), max(1, work_size))


def parallel_matrix_entries(func, df, comb_list, sample_weight=None, n_jobs=-1):
    """parallel_matrix_entries applies a function to each chunk of
    combination of columns of the dataframe, distributed by cores.
    This is similar to https://github.com/smazzanti/mrmr/mrmr/pandas.py


    Parameters
    ----------
    func : callable
        function to be applied to each pair of columns in comb_list
    df : pd.DataFrame
        the dataframe on which to apply the function
    comb_list : list of pairs of column labels
        Original column labels corresponding to the entries, including tuples.
    sample_weight : pd.Series or np.array, optional
        The weight vector, if any, of shape (n_samples,), by default None
    n_jobs : int, optional
        Maximum workers, by default -1. Joblib resolves negative counts (-1
        uses all available CPUs, -2 all but one). Workers are capped to pair
        count; zero, Boolean and noninteger counts are rejected.

    Returns
    -------
    pd.DataFrame
        Results sorted by val descending, with ties in pair/callback order and
        a fresh RangeIndex, regardless of worker count. Empty work/results
        return columns row, col (object) and val (float). Empty work skips callbacks.
    """
    n_jobs = _effective_n_jobs(n_jobs, len(comb_list))

    if not comb_list:
        lst = []
    elif n_jobs == 1:
        lst = func(X=df, sample_weight=sample_weight, comb_list=comb_list)
    else:
        comb_chunks = [
            [comb_list[i] for i in positions]
            for positions in np.array_split(range(len(comb_list)), n_jobs)
        ]
        lst = list(
            chain.from_iterable(
                Parallel(n_jobs=n_jobs)(
                    delayed(func)(X=df, sample_weight=sample_weight, comb_list=chunk)
                    for chunk in comb_chunks
                )
            )
        )
    if not lst:
        return pd.DataFrame(columns=["row", "col", "val"]).astype({"val": float})
    return (
        pd.concat(lst, ignore_index=True)
        .sort_values("val", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


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
        Maximum workers, by default -1. Joblib resolves negative counts (-1
        uses all available CPUs, -2 all but one). Workers are capped to column
        count; zero, Boolean and noninteger counts are rejected.

    Returns
    -------
    pd.Series
        Results sorted descending, with stable callback/column-order ties and
        original labels. Empty column work returns a float Series without
        calling the callback or dispatching workers.
    """
    n_jobs = _effective_n_jobs(n_jobs, len(df.columns))
    if df.shape[1] == 0:
        return pd.Series(dtype=float)

    if n_jobs == 1:
        return func(df, series, sample_weight).sort_values(
            ascending=False, kind="stable"
        )
    else:
        col_chunks = np.array_split(range(len(df.columns)), n_jobs)
        lst = Parallel(n_jobs=n_jobs)(
            delayed(func)(df.iloc[:, col_chunk], series, sample_weight)
            for col_chunk in col_chunks
        )

        return pd.concat(lst).sort_values(ascending=False, kind="stable")


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

    # Empty column work still needs a Series result; pandas cannot infer it.
    if X.shape[1] == 0:
        return pd.Series(dtype=float)
    # The association boundary owns the missing-value policy and aligned rows.
    return X.apply(
        lambda col: func_xyw(x=col, y=y, sample_weight=sample_weight, as_frame=False)
    ).fillna(0.0)


def _compute_matrix_entries(
    X,
    comb_list,
    sample_weight=None,
    func_xyw=None,
):
    """base closure for computing matrix entries applying a function to each chunk of
    combination of columns of the dataframe, distributed by cores.
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
    comb_list : list of 2-tuple of str
        Pairs of column names corresponding to the entries

    Returns
    -------
    List[pd.DataFrame]
        a list of partial dfs to be concatenated
    """
    v_df_list = [
        func_xyw(x=X[comb[0]], y=X[comb[1]], sample_weight=sample_weight, as_frame=True)
        for comb in comb_list
    ]

    return v_df_list
