"""Base Submodule

This module provides a base class for selector using a statistic and a threshold

Module Structure:
-----------------
- ``BaseThresholdSelector``: parent class for the "treshold-based" selectors

"""

from numbers import Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted


def _validate_dataframe(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X is not a dataframe")
    if not X.columns.is_unique:
        raise ValueError("X must have unique feature names.")


class BaseThresholdSelector(SelectorMixin, BaseEstimator):
    """Base class for threshold-based feature selection

    Parameters
    ----------
    threshold : float, .05
        Finite cutoff. Retain statistics strictly above it when
        greater_than_threshold=True, otherwise strictly below it. Equality is rejected.
    statistic_fn : callable, optional
        The function for computing the statistic series. The index should be the column names and the
        the values the computed statistic
    greater_than_threshold : bool, False
        Whether to retain statistics strictly above rather than below the cutoff.

    Notes
    -----
    X must be a DataFrame with unique column names. Transform projects selected
    names in fitted order: reordered/extra columns and omitted rejected columns
    are accepted; missing selected columns raise KeyError. Empty selections are valid.

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

    def _validate_params(self):
        if (
            isinstance(self.threshold, (bool, np.bool_))
            or not isinstance(self.threshold, Real)
            or not np.isfinite(self.threshold)
        ):
            raise ValueError("threshold must be a finite real number.")
        if not callable(self.statistic_fn):
            raise TypeError("statistic_fn must be callable.")
        if not isinstance(self.greater_than_threshold, (bool, np.bool_)):
            raise TypeError("greater_than_threshold must be boolean.")

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
            Ignored. The callback receives X only; built-in missing/unique/cardinality
            statistics count rows or distinct values without observation weights.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        _validate_dataframe(X)
        self._validate_params()
        if len(X) == 0:
            raise ValueError("X must contain observations.")
        statistics = self.statistic_fn(X)
        if (
            not isinstance(statistics, pd.Series)
            or not statistics.index.is_unique
            or len(statistics) != X.shape[1]
            or not X.columns.isin(statistics.index).all()
        ):
            raise ValueError(
                "statistic_fn must return a Series indexed by each input feature exactly once."
            )
        statistics = statistics.reindex(X.columns)
        if (
            statistics.isna().any()
            or not pd.api.types.is_numeric_dtype(statistics.dtype)
            or pd.api.types.is_complex_dtype(statistics.dtype)
            or not np.isfinite(statistics.to_numpy(dtype=float)).all()
        ):
            raise ValueError("statistic_fn must return finite numeric statistics.")

        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = X.shape[1]
        self.statistic_series_ = statistics
        self.statistic_df_ = statistics.to_frame(name="statistic")

        # Sort with highest number of missing values on top
        self.statistic_df_ = self.statistic_df_.sort_values(
            "statistic", ascending=False, kind="stable"
        )
        if self.greater_than_threshold:
            self.support_ = self.statistic_series_.values > self.threshold
        else:
            self.support_ = self.statistic_series_.values < self.threshold

        self.selected_features_ = self.feature_names_in_[self.support_]
        self.not_selected_features_ = self.feature_names_in_[~self.support_]

        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")

        return self.support_

    def transform(self, X):
        """
        Transform the data, returns a transformed version of `X`.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples with unique names and all selected columns.

        Returns
        -------
        X_new : pd.DataFrame of shape (n_samples, n_features_new)
            Selected columns in fitted order, preserving row index and dtypes.
        """
        check_is_fitted(self, "support_")
        _validate_dataframe(X)
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
