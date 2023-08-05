"""Base Submodule

This module provides a base class for selector using a statistic and a threshold

Module Structure:
-----------------
- ``BaseThresholdSelector``: parent class for the "treshold-based" selectors

"""

# Settings and libraries
from __future__ import print_function

# pandas
import pandas as pd

# numpy
import numpy as np

# sklearn

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin


# fix random seed for reproducibility
np.random.seed(7)


class BaseThresholdSelector(SelectorMixin, BaseEstimator):
    """Base class for threshold-based feature selection

    Parameters
    ----------
    threshold : float, .05
        Features with a training-set missing greater/lower (geq/leq) than this threshold will be removed
    statistic_fn : callable, optional
        The function for computing the statistic series. The index should be the column names and the
        the values the computed statistic
    greater_than_threshold : bool, False
        Whether or not to reject the features if lower or greater than threshold

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

    """

    def __init__(
        self,
        threshold=0.05,
        statistic_fn=None,
        greater_than_threshold=False,
    ):
        self.threshold = threshold
        self.statistic_fn = statistic_fn
        self.greater_than_threshold = greater_than_threshold

    def fit(self, X, y=None, sample_weight=None):
        """Learn empirical statistics from X.

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

        # Calculate the fraction of missing in each column

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            raise TypeError("X is not a dataframe")

        self.statistic_series_ = self.statistic_fn(X)
        self.statistic_df_ = pd.DataFrame(self.statistic_series_).rename(
            columns={"index": "feature", 0: "statistic"}
        )

        # Sort with highest number of missing values on top
        self.statistic_df_ = self.statistic_df_.sort_values(
            "statistic", ascending=False
        )
        if self.greater_than_threshold:
            self.support_ = self.statistic_series_.values > self.threshold
        else:
            self.support_ = self.statistic_series_.values < self.threshold

        self.selected_features_ = self.feature_names_in_[self.support_]
        self.not_selected_features_ = self.feature_names_in_[~self.support_]

        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.support_

    def transform(self, X):
        """
        Transform the data, returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        return X[self.selected_features_]

    def fit_transform(self, X, y=None, sample_weight=None, **fit_params):
        """
        Fit to data, then transform it.
        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        sample_weight :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            sample weight values.
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X=X, y=y, sample_weight=sample_weight, **fit_params).transform(
            X
        )

    def _more_tags(self):
        return {"allow_nan": True}
