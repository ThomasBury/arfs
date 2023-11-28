"""Unsupervised Feature Selection

This module provides selectors using unsupervised statistics and a threshold

Module Structure:
-----------------
- ``MissingValueThreshold``: child class of the ``BaseThresholdSelector``, filter out columns with too many missing values
- ``UniqueValuesThreshold`` child of the ``BaseThresholdSelector``, filter out columns with zero variance
- ``CardinalityThreshold`` child of the ``BaseThresholdSelector``, filter out categorical columns with too many levels  
- ``CollinearityThreshold`` child of the ``BaseThresholdSelector``, filter out collinear columns
"""

from __future__ import print_function
from tqdm.auto import trange

# pandas
import pandas as pd

# numpy
import numpy as np

# sklearn
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin

# ARFS
from .base import BaseThresholdSelector
from ..utils import create_dtype_dict
from ..association import (
    association_matrix,
    xy_to_matrix,
    plot_association_matrix,
    weighted_theils_u,
    weighted_corr,
    correlation_ratio,
)
from ..preprocessing import OrdinalEncoderPandas


# fix random seed for reproducibility
np.random.seed(7)


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
    return pd.Series(missing_counts)


class MissingValueThreshold(BaseThresholdSelector):
    """Feature selector that removes all high missing percentage features.
    This feature selection algorithm looks only at the features (X),
    not the desired outputs (y), and can thus be used for unsupervised learning.


    Parameters
    ----------
    threshold: float, default = .05
        Features with a training-set missing larger than this threshold will be removed.

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


def _pandas_count_unique_values(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X should be a pandas DataFrame")
    return X.nunique()


class UniqueValuesThreshold(BaseThresholdSelector):
    """Feature selector that removes all features with zero variance (single unique values)
    or remove columns with less unique values than threshold
    This feature selection algorithm looks only at the features (X),
    not the desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold: int, default = 1
        Features with a training-set missing larger than this threshold will be removed.
        The thresold should be >= 1

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
    """Feature selector that removes all categorical features with more unique values than threshold
    This feature selection algorithm looks only at the features (X),
    not the desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold: int, default = 1000
        Features with a training-set missing larger than this threshold will be removed.
        The thresold should be >= 1

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
        Features with a training-set missing larger than this threshold will be removed
        The thresold should be > 0 and =< 1
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

        if self.method not in ["association", "correlation"]:
            raise ValueError("``method`` should be 'association' or 'correlation'")

        if (self.threshold > 1.0) or (self.threshold < 0.0):
            raise ValueError("``threshold`` should be larger than 0 and smaller than 1")

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

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            raise TypeError("X is not a dataframe")

        self.suffix_dic = create_dtype_dict(X)

        if self.method == "correlation":
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
        self.assoc_matrix_ = xy_to_matrix(assoc_matrix)

        to_drop = _recursive_collinear_elimination(self.assoc_matrix_, self.threshold)

        self.support_ = np.asarray(
            [True if c not in to_drop else False for c in X.columns]
        )
        self.selected_features_ = self.feature_names_in_[self.support_]
        self.not_selected_features_ = self.feature_names_in_[~self.support_]

        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.support_

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
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


def _most_collinear(association_matrix, threshold):
    cols_to_drop = [
        column
        for column in association_matrix.columns
        if any(association_matrix.loc[:, column].abs() > threshold)
    ]
    rows_to_drop = [
        row
        for row in association_matrix.index
        if any(association_matrix.loc[row, :].abs() > threshold)
    ]
    to_drop = list(set(cols_to_drop).union(set(rows_to_drop)))
    most_collinear_series = (
        association_matrix[to_drop].abs().sum(axis=1).sort_values(ascending=False)
    )
    most_collinear_series += (
        association_matrix[to_drop].abs().sum(axis=0).sort_values(ascending=False)
    )
    most_collinear_series /= 2
    return most_collinear_series.index[0], to_drop


def _recursive_collinear_elimination(association_matrix, threshold):
    dum = association_matrix.copy()
    most_collinear_features = []

    while True:
        most_collinear_feature, to_drop = _most_collinear(dum, threshold)

        # Break if no more features to drop
        if not to_drop:
            break

        if most_collinear_feature not in most_collinear_features:
            most_collinear_features.append(most_collinear_feature)
            dum = dum.drop(columns=most_collinear_feature, index=most_collinear_feature)

    return most_collinear_features
