"""
This module provides utilities for parallelizing operations on pd.DataFrame

**The module structure is the following:**

- The ``parallel_matrix_entries`` for parallelizing operations returning a matrix (2D) (apply on pairs of columns)
- The ``parallel_df`` for parallelizing operations returning a series (1D) (apply on a single column at a time)
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional, Dict, Callable
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from itertools import chain


def parallel_matrix_entries(
    func: callable,
    df: pd.DataFrame,
    comb_list: List[Tuple[str]],
    sample_weight: Optional[Union[pd.Series, np.array]] = None,
    n_jobs: int = -1,
):
    """parallel_matrix_entries apply a function to each chunk of
    combinaison of columns of the dataframe, distributed by cores.
    This is similar to https://github.com/smazzanti/mrmr/mrmr/pandas.py

    Parameters
    ----------
    func :
        function to be applied to each column
    df :
        the dataframe on which to apply the function
    comb_list :
        a list of 2-uple of strings. Pairs of column names corresponding to the entries
    sample_weight :
        The weight vector, if any, of shape (n_samples,)
    n_jobs :
        the number of cores to use for the computation

    Returns
    -------
    pd.DataFrame
        concatenated results into a single pandas DF
    """
    
    if n_jobs == 1:
        return func(X=df, sample_weight=sample_weight, comb_list=comb_list)
    
    n_jobs = (
        min(cpu_count(), len(df.columns))
        if n_jobs == -1
        else min(cpu_count(), n_jobs)
    )
    comb_chunks = np.array_split(comb_list, n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(X=df, sample_weight=sample_weight, comb_list=comb_chunk)
        for comb_chunk in comb_chunks
    )
    # return flatten list of pandas DF
    return pd.concat(list(chain(*lst)), ignore_index=True)


def parallel_df(
    func: callable,
    df: pd.DataFrame,
    series: Union[pd.Series, np.array],
    sample_weight: Optional[Union[pd.Series, np.array]] = None,
    n_jobs: int = -1,
):
    """parallel_df apply a function to each column of the dataframe, distributed by cores.
    This is similar to https://github.com/smazzanti/mrmr/mrmr/pandas.py


    Parameters
    ----------
    func :
        function to be applied to each column
    df :
        the dataframe on which to apply the function
    series :
        series (target) used by the function
    sample_weight :
        The weight vector, if any, of shape (n_samples,)
    n_jobs :
        the number of cores to use for the computation

    Returns
    -------
    pd.DataFrame
        concatenated results into a single pandas DF
    """
    
    if n_jobs == 1:
        return func(df, series, sample_weight).sort_values(ascending=False)
        
    
    n_jobs = (
        min(cpu_count(), len(df.columns))
        if n_jobs == -1
        else min(cpu_count(), n_jobs)
    )
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series, sample_weight)
        for col_chunk in col_chunks
    )
    return pd.concat(lst).sort_values(ascending=False)


def _compute_series(
    X: pd.DataFrame,
    y: Union[pd.Series, np.array],
    sample_weight: Optional[Union[pd.Series, np.array]] = None,
    func_xyw: Callable = None,
):
    """base closure for parallelizing the computation

    apply the Cramer V computation with the target for all columns using a closure

    Parameters
    ----------
    X :
        The set of regressors that will be tested sequentially, of shape (n_samples, n_features)
    y :
        The target vector of shape (n_samples,)
    sample_weight :
        The weight vector, if any, of shape (n_samples,)
    func_xyw :
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
        lambda col: _closure_compute_series(
            x=col, y=y, sample_weight=sample_weight
        )
    ).fillna(0.0)


def _compute_matrix_entries(
    X: pd.DataFrame,
    comb_list: List[Tuple[str]],
    sample_weight: Optional[Union[pd.Series, np.array]] = None,
    func_xyw: Callable = None,
):
    """base closure for computing matrix entries appling a function to each chunk of
    combinaison of columns of the dataframe, distributed by cores.
    This is similar to https://github.com/smazzanti/mrmr/mrmr/pandas.py

    Parameters
    ----------
    X :
        The set of regressors that will be tested sequentially, of shape (n_samples, n_features)
    sample_weight :
        The weight vector, if any, of shape (n_samples,)
    func_xyw :
        callable (function) for computing the individual elements of the series
        takes two mandatory inputs (x and y) and an optional input w, sample_weights
    comb_list :
        a list of 2-uple of strings. Pairs of column names corresponding to the entries

    Returns
    -------
    pd.DataFrame
        concatenated results into a single pandas DF
    """
    v_df_list = []
    for comb in comb_list:
        v_df_list.append(
            func_xyw(
                x=X[comb[0]],
                y=X[comb[1]],
                sample_weight=sample_weight,
                as_frame=True,
            )
        )

    return v_df_list
