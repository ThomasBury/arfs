"""Unsupervised Feature Selection

This module provides selectors using unsupervised statistics and a threshold

Module Structure:
-----------------
- ``MissingValueThreshold``: child class of the ``BaseThresholdSelector``, filter out columns with too many missing values
- ``UniqueValuesThreshold`` child of the ``BaseThresholdSelector``, filter out columns with zero variance
- ``CardinalityThreshold`` child of the ``BaseThresholdSelector``, filter out categorical columns with too many levels
- ``CollinearityThreshold`` child of the ``BaseThresholdSelector``, filter out collinear columns
"""

from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from ..association import (
    association_matrix,
    correlation_ratio,
    plot_association_matrix,
    weighted_corr,
    weighted_theils_u,
    xy_to_matrix,
)
from ..preprocessing import OrdinalEncoderPandas
from ..utils import create_dtype_dict
from .base import BaseThresholdSelector, _validate_dataframe


def _missing_ratio(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be a pandas DataFrame")
    numeric_columns = df.select_dtypes(np.number).columns
    n_samples = len(df)

    missing_counts = {}
    for column in df.columns:
        if column in numeric_columns:
            missing_counts[column] = (
                df[column].isnull().sum() + np.isinf(df[column]).sum()
            ) / n_samples
        else:
            missing_counts[column] = df[column].isnull().sum() / n_samples
    return pd.Series(missing_counts, dtype=float)


class MissingValueThreshold(BaseThresholdSelector):
    """Feature selector that removes all high missing percentage features.
    This feature selection algorithm looks only at the features (X),
    not the desired outputs (y), and can thus be used for unsupervised learning.


    Parameters
    ----------
    threshold: float, default = .05
        Finite cutoff in [0, 1]. Retain features with a missing fraction strictly
        below it; equality is rejected. Zero rejects all features, including complete
        ones; one rejects fully missing features. NaN and numeric infinities count as missing.

    Returns
    -------
    selected_features: list of str
        List of selected features.

    Attributes
    ----------
    n_features_in_ : int
        number of input predictors
    support_ : list of bool
        the list of the selected X-columns
    selected_features_ : list of str
        the list of names of selected features
    not_selected_features_ : list of str
        the list of names of rejected features

    Example
    -------
    >>> from sklearn.datasets import make_classification, make_regression
    >>> X, y = make_regression(n_samples = 1000, n_features = 50, n_informative = 5, shuffle=False) # , n_redundant = 5
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> pred_name = [f"pred_{i}" for i in range(X.shape[1])]
    >>> X.columns = pred_name
    >>> selector = MissingValueThreshold(0.05)
    >>> selector.fit_transform(X)
    """

    def __init__(self, threshold=0.05):
        super().__init__(
            threshold=threshold,
            statistic_fn=_missing_ratio,
            greater_than_threshold=False,
        )

    def _validate_params(self):
        super()._validate_params()
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")


def _pandas_count_unique_values(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X should be a pandas DataFrame")
    return X.nunique()


class UniqueValuesThreshold(BaseThresholdSelector):
    """Feature selector that removes all features with zero variance (single unique values)
    or with at most threshold unique nonmissing values.
    This feature selection algorithm looks only at the features (X),
    not the desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold: int, default = 1
        Integer >= 1. Retain features with strictly more unique nonmissing
        values than this cutoff; equality is rejected.

    Returns
    -------
    selected_features: list of str
        List of selected features.

    Attributes
    ----------
    n_features_in_ : int
        number of input predictors
    support_ : list of bool
        the list of the selected X-columns
    selected_features_ : list of str
        the list of names of selected features
    not_selected_features_ : list of str
        the list of names of rejected features

    Example
    -------
    >>> from sklearn.datasets import make_classification, make_regression
    >>> X, y = make_regression(n_samples = 1000, n_features = 50, n_informative = 5, shuffle=False) # , n_redundant = 5
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> pred_name = [f"pred_{i}" for i in range(X.shape[1])]
    >>> X.columns = pred_name
    >>> selector = UniqueValuesThreshold(1)
    >>> selector.fit_transform(X)
    """

    def __init__(self, threshold=1):
        super().__init__(
            threshold=threshold,
            statistic_fn=_pandas_count_unique_values,
            greater_than_threshold=True,
        )

    def _validate_params(self):
        super()._validate_params()
        if not isinstance(self.threshold, Integral) or self.threshold < 1:
            raise ValueError("threshold must be an integer >= 1.")


def _pandas_count_unique_values_cat_features(X):
    """
    Counts the number of unique values in categorical features of a pandas DataFrame.

    Parameters
    ----------
    X : pandas DataFrame
        The input data.

    Returns
    -------
    pandas Series
        The number of unique values in each categorical feature.

    Raises
    ------
    TypeError
        If the input data is not a pandas DataFrame.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X should be a pandas DataFrame")
    count_series = pd.Series(data=0, index=X.columns)
    dtype_dic = create_dtype_dict(X, dic_keys="dtypes")
    for c in dtype_dic["cat"]:
        count_series[c] = X[c].nunique()
    return count_series


class CardinalityThreshold(BaseThresholdSelector):
    """Feature selector that removes categorical features with at least threshold unique values.
    This feature selection algorithm looks only at the features (X),
    not the desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold: int, default = 1000
        Integer >= 1. Retain features with a categorical nonmissing cardinality
        strictly below this cutoff; equality is rejected. Numeric features count as zero.

    Returns
    -------
    selected_features: list of str
        List of selected features.

    Attributes
    ----------
    n_features_in_ : int
        number of input predictors
    support_ : list of bool
        the list of the selected X-columns
    selected_features_ : list of str
        the list of names of selected features
    not_selected_features_ : list of str
        the list of names of rejected features

    Example
    -------
    >>> from sklearn.datasets import make_classification, make_regression
    >>> X, y = make_regression(n_samples = 1000, n_features = 50, n_informative = 5, shuffle=False) # , n_redundant = 5
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> pred_name = [f"pred_{i}" for i in range(X.shape[1])]
    >>> X.columns = pred_name
    >>> selector = CardinalityThreshold(100)
    >>> selector.fit_transform(X)
    """

    def __init__(self, threshold=1000):
        super().__init__(
            threshold=threshold,
            statistic_fn=_pandas_count_unique_values_cat_features,
            greater_than_threshold=False,
        )

    def _validate_params(self):
        super()._validate_params()
        if not isinstance(self.threshold, Integral) or self.threshold < 1:
            raise ValueError("threshold must be an integer >= 1.")


class CollinearityThreshold(SelectorMixin, BaseEstimator):
    """Feature selector that removes collinear features.
    This feature selection algorithm looks only at the features (X),
    not the desired outputs (y), and can thus be used for unsupervised learning.
    It computes the association between features (continuous or categorical),
    store the pairs of collinear features and remove one of them for all pairs having
    an association value above the threshold.

    The association measures are the Spearman correlation coefficient, correlation ratio
    and Theil's U. The association matrix is not necessarily symmetrical.

    By changing the method to "correlation", data will be encoded as integer
    and the Spearman correlation coefficient will be used instead. Faster but not
    a best practice because the categorical variables are considered as numeric.

    Parameters
    ----------
    threshold : float, default = .8
        Finite cutoff in [0, 1]. Remove associations strictly above it in absolute
        value; equality is retained. Equal connection totals drop the first feature
        in fitted input order. Zero/one-feature inputs have no pairs and retain all columns.
    method : str, default = "association"
        method for computing the association matrix. Either "association" or "correlation".
        Correlation leads to encoding of categorical variables as numeric
    n_jobs : int, default = -1
        the number of threads, -1 uses all the threads for computating the association matrix
    nom_nom_assoc : str or callable, default = "theil"
        the categorical-categorical association measure, by default Theil's U, not symmetrical!
    num_num_assoc : str or callable, default = "spearman"
        the numeric-numeric association measure
    nom_num_assoc : str or callable, default = "correlation_ratio"
        the numeric-categorical association measure

    Notes
    -----
    X must have unique column names. Transform projects selected names in fitted
    order; extra/reordered columns and omitted rejected columns are accepted.
    Missing selected columns raise KeyError. Empty selections preserve the row index.

    Returns
    -------
    selected_features: list of str
        List of selected features.

    Attributes
    ----------
    n_features_in_ : int
        number of input predictors
    assoc_matrix_ : pd.DataFrame
        the square association matrix
    collinearity_summary_ : pd.DataFrame
        the pairs of collinear features and the association values
    support_ : list of bool
        the list of the selected X-columns
    selected_features_ : list of str
        the list of names of selected features
    not_selected_features_ : list of str
        the list of names of rejected features

    Example
    -------
    >>> from sklearn.datasets import make_classification, make_regression
    >>> X, y = make_regression(n_samples = 1000, n_features = 50, n_informative = 5, shuffle=False) # , n_redundant = 5
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> pred_name = [f"pred_{i}" for i in range(X.shape[1])]
    >>> X.columns = pred_name
    >>> selector = CollinearityThreshold(threshold=0.75)
    >>> selector.fit_transform(X)
    """

    def __init__(
        self,
        threshold=0.80,
        method="association",
        n_jobs=1,
        nom_nom_assoc=weighted_theils_u,
        num_num_assoc=weighted_corr,
        nom_num_assoc=correlation_ratio,
    ):
        self.threshold = threshold
        self.method = method
        self.n_jobs = n_jobs
        self.nom_nom_assoc = nom_nom_assoc
        self.num_num_assoc = num_num_assoc
        self.nom_num_assoc = nom_num_assoc

    def fit(self, X, y=None, sample_weight=None):
        """Learn empirical associtions from X.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.
        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        sample_weight : pd.Series, optional, shape (n_samples,)
            weights for computing the statistics (e.g. weighted average)

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        _validate_dataframe(X)
        if (
            isinstance(self.threshold, (bool, np.bool_))
            or not isinstance(self.threshold, Real)
            or not np.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("threshold must be a finite real number in [0, 1].")
        if self.method not in ["association", "correlation"]:
            raise ValueError("method should be 'association' or 'correlation'.")
        suffix_dic = create_dtype_dict(X)
        feature_names = X.columns.to_numpy(copy=True)

        if self.method == "correlation" and X.shape[1] >= 2:
            encoder = OrdinalEncoderPandas()
            X = encoder.fit_transform(X)
            del encoder

        assoc_matrix = association_matrix(
            X=X,
            sample_weight=sample_weight,
            n_jobs=self.n_jobs,
            nom_nom_assoc=self.nom_nom_assoc,
            num_num_assoc=self.num_num_assoc,
            nom_num_assoc=self.nom_num_assoc,
        )
        self.assoc_matrix_ = (
            pd.DataFrame(0.0, index=X.columns, columns=X.columns)
            if assoc_matrix.empty
            else xy_to_matrix(assoc_matrix).reindex(
                index=X.columns, columns=X.columns, fill_value=0
            )
        )

        to_drop = _recursive_collinear_elimination(self.assoc_matrix_, self.threshold)

        self.feature_names_in_ = feature_names
        self.n_features_in_ = X.shape[1]
        self.suffix_dic = suffix_dic
        self.support_ = np.asarray([c not in to_drop for c in X.columns], dtype=bool)
        self.selected_features_ = self.feature_names_in_[self.support_]
        self.not_selected_features_ = self.feature_names_in_[~self.support_]

        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")

        return self.support_

    def transform(self, X):
        check_is_fitted(self, "support_")
        _validate_dataframe(X)
        return X[self.selected_features_]

    def _more_tags(self):
        return {"allow_nan": True}

    def plot_association(
        self, ax=None, cmap="PuOr", figsize=None, cbar_kw=None, imgshow_kw=None
    ):
        """plot_association plots the association matrix

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            the mpl axes if the figure object exists already, by default None
        cmap : str, optional
            colormap name, by default "PuOr"
        figsize : tuple of float, optional
            figure size, by default None
        cbar_kw : dict, optional
            colorbar kwargs, by default None
        imgshow_kw : dict, optional
            imgshow kwargs, by default None
        """

        if figsize is None:
            figsize = (self.assoc_matrix_.shape[0] / 3, self.assoc_matrix_.shape[0] / 3)

        f, ax = plot_association_matrix(
            assoc_mat=self.assoc_matrix_,
            suffix_dic=self.suffix_dic,
            ax=ax,
            cmap=cmap,
            cbarlabel="association value",
            figsize=figsize,
            show=True,
            cbar_kw=cbar_kw,
            imgshow_kw=imgshow_kw,
        )

        return f


def _most_collinear(association_matrix_abs, threshold):
    cols_to_drop = association_matrix_abs.loc[
        :, (association_matrix_abs > threshold).any(axis=0)
    ].columns.values
    rows_to_drop = association_matrix_abs.loc[
        (association_matrix_abs > threshold).any(axis=1), :
    ].index.values
    to_drop = association_matrix_abs.columns[
        association_matrix_abs.columns.isin(cols_to_drop)
        | association_matrix_abs.columns.isin(rows_to_drop)
    ].tolist()
    if not to_drop:
        return None, None
    # for features in `to_drop` sum up their column and row values to find
    # the most collinear feature
    most_collinear_series = association_matrix_abs.loc[:, to_drop].sum(axis=0)
    most_collinear_series += association_matrix_abs.loc[to_drop, :].sum(axis=1)
    # not necessarily but avoids exceeding 1
    most_collinear_series /= 2
    return most_collinear_series.idxmax(), to_drop


def _recursive_collinear_elimination(association_matrix, threshold):
    dum = association_matrix.abs()
    most_collinear_features = []

    while True:
        most_collinear_feature, to_drop = _most_collinear(dum, threshold)

        # Break if no more features to drop
        if not to_drop:
            break
        # the if statement below can probably also be removed since we can only
        # remove features we have left in dum
        if most_collinear_feature not in most_collinear_features:
            most_collinear_features.append(most_collinear_feature)
            dum = dum.drop(columns=most_collinear_feature, index=most_collinear_feature)

    return most_collinear_features
