"""Supervised Feature Selection

This module provides selectors using supervised statistics and a threshold, using SHAP, permutation importance or impurity (Gini) importance.

Module Structure:
-----------------
- ``VariableImportance`` main class for identifying non-important features
"""

from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from ..gbm import GradientBoosting
from ..preprocessing import OrdinalEncoderPandas
from ..utils import reset_plot
from .allrelevant import _compute_importance
from .base import _validate_dataframe


class VariableImportance(SelectorMixin, BaseEstimator):
    """Feature selector that removes predictors with zero or low variable importance.

    Fit one LightGBM with early stopping, then rank predictors by mean absolute
    validation-row contributions, summed over model outputs. Retain the smallest
    positive-importance prefix reaching the requested cumulative share, including
    the crossing feature. Equal importances follow input order.

    Parameters
    ----------
    task : string
        The machine learning task, either 'classification' or 'regression' or 'multiclass',
        be sure to use a consistent objective function
    encode : boolean, default = True
        Whether or not to encode the predictors
    threshold : float, default = .99
        Finite share in [0, 1]. Zero selects nothing; one selects every predictor
        with positive importance. An all-zero total selects nothing at any cutoff.
    lgb_kwargs : dict or None, default=None
        LightGBM parameters with an objective. None uses objective='rmse' and
        zero_as_missing=False. Caller dictionaries are preserved.
    encoder_kwargs : dictionary of keyword arguments, optional
        dictionary of the :class:`OrdinalEncoderPandas` parameters
    fastshap : bool, default=False
        Use optional fasttreeshap; a missing backend raises ImportError before training.
    verbose : int, default=-1
        Training metric log frequency; nonpositive values disable metric logging.

    Notes
    -----
    X must have unique column names. Transform projects selected names in fitted
    order; extra/reordered columns and omitted rejected columns are accepted.
    Missing selected columns raise KeyError; duplicate columns are rejected.
    Observation weights affect training and validation loss. Global importance
    averages validation rows equally, even with weights; it describes the observed
    validation rows rather than a weight-adjusted population. The split is fixed
    by the training wrapper. One fit does not estimate resampling variability.

    Returns
    -------
    selected_features: list of str
        List of selected features.

    Attributes
    ----------
    n_features_in_ : int
        number of input predictors
    feature_importances_summary_ : pd.DataFrame
        Ranked finite importance, normalized share and cumulative share. Both
        shares are zero when total importance is zero.
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
    >>> selector = VariableImportance(threshold=0.75)
    >>> selector.fit_transform(X, y)
    """

    def __init__(
        self,
        task="regression",
        encode=True,
        threshold=0.99,
        lgb_kwargs=None,
        encoder_kwargs=None,
        fastshap=False,
        verbose=-1,
    ):
        self.task = task
        self.encode = encode
        self.threshold = threshold
        self.lgb_kwargs = lgb_kwargs
        self.encoder_kwargs = encoder_kwargs
        self.verbose = verbose
        self.fastshap = fastshap

    def fit(self, X, y, sample_weight=None):
        """Learn variable importance from X and y, supervised learning.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.
        y : array-like, shape (n_samples,)
            Finite, nonmissing responses, aligned by row position.
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
        feature_names = X.columns.to_numpy(copy=True)

        feature_importances = _compute_varimp_lgb(
            X=X,
            y=y,
            sample_weight=sample_weight,
            encode=self.encode,
            task=self.task,
            verbose=self.verbose,
            encoder_kwargs=self.encoder_kwargs,
            lgb_kwargs=self.lgb_kwargs,
            fastshap=self.fastshap,
        )

        self.feature_importances_summary_ = feature_importances

        keep = (feature_importances["importance"] > 0) & (
            (self.threshold == 1)
            | (
                feature_importances["cumulative_importance"].shift(fill_value=0)
                < self.threshold
            )
        )
        selected = feature_importances.loc[keep, "feature"].tolist()

        self.support_ = np.asarray([c in selected for c in feature_names], dtype=bool)
        self.feature_names_in_ = feature_names
        self.n_features_in_ = X.shape[1]
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
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X : ndarray array of shape (n_samples, n_features_new)
            Transformed array.

        Raises
        ------
        TypeError
            if the input is not a pd.DataFrame
        """

        check_is_fitted(self, "support_")
        _validate_dataframe(X)
        return X[self.selected_features_]

    def fit_transform(self, X, y=None, sample_weight=None):
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
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X=X, y=y, sample_weight=sample_weight).transform(X)

    def _more_tags(self):
        return {"allow_nan": True}

    def plot_importance(
        self, figsize=None, plot_n=50, n_feat_per_inch=3, log=True, style=None
    ):
        """Plot the highest-ranked features and their full-data cumulative shares.

        Parameters
        ----------
        plot_n : int, default=50
            Positive maximum number of features displayed in each panel.
        n_feat_per_inch : float, default=3
            Positive number of features per inch; height is at least two inches.
        figsize : tuple of float, optional
            Figure width and height in inches. Overrides automatic sizing.
        log : bool, default=True
            Use a log scale for positive importance values. Feature positions
            and cumulative shares remain linear; all-zero importance stays linear.
        style : str or None, default=None
            Matplotlib style name; None resets the default plotting style.

        Returns
        -------
        matplotlib.figure.Figure
            Caller-owned figure. The title reports the full fitted selection
            count. Cutoff markers use its last selected feature and are omitted
            if none is selected or the cutoff lies beyond the displayed prefix.
        """
        check_is_fitted(self, "support_")
        if (
            isinstance(plot_n, (bool, np.bool_))
            or not isinstance(plot_n, Integral)
            or plot_n < 1
        ):
            raise ValueError("plot_n must be a positive integer.")
        if (
            isinstance(n_feat_per_inch, (bool, np.bool_))
            or not isinstance(n_feat_per_inch, Real)
            or not np.isfinite(n_feat_per_inch)
            or n_feat_per_inch <= 0
        ):
            raise ValueError("n_feat_per_inch must be a finite positive number.")

        df = self.feature_importances_summary_.iloc[:plot_n]
        n_selected = len(self.selected_features_)
        importance_index = n_selected - 1
        positions = np.arange(len(df))

        if style:
            plt.style.use(style)
        else:
            reset_plot()

        if figsize is None:
            figsize = (8, max(2, len(df) / n_feat_per_inch))
        fig = plt.figure(tight_layout=True, figsize=figsize)
        gs = gridspec.GridSpec(3, 3)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.scatter(df.normalized_importance, positions)
        ax1.set_yticks(positions, labels=df.feature)
        ax1.set_xlabel("normalized importance")
        ax1.xaxis.set_label_position("top")
        ax1.invert_yaxis()
        if log and (df.normalized_importance > 0).any():
            ax1.set_xscale("log")
        ax1.grid()
        ax1.set(frame_on=False)

        ax2 = fig.add_subplot(gs[:, 1:])
        ax2.scatter(positions, df.cumulative_importance)
        ax2.set_xticks(positions, labels=df.feature)
        ax2.set_ylabel("cumulative importance")
        ax2.tick_params(axis="x", labelrotation=90)
        ax2.set_title(f"{n_selected} features selected")
        ax2.axhline(y=self.threshold, linestyle="dashed", color="r")
        if 0 <= importance_index < len(df):
            ax1.axvline(
                x=df.normalized_importance.iloc[importance_index],
                linestyle="dashed",
                color="r",
            )
            ax2.axvline(x=importance_index, linestyle="dashed", color="r")
        ax2.set_ylim(0, 1.05)
        ax2.grid()
        ax2.set(frame_on=False)

        fig.align_labels()
        return fig


def _compute_varimp_lgb(
    X,
    y,
    sample_weight=None,
    encode=False,
    task="regression",
    verbose=-1,
    fastshap=True,
    encoder_kwargs=None,
    lgb_kwargs=None,
):
    """One weighted fit; equal-row absolute importance, summed over outputs."""
    _validate_dataframe(X)
    if task not in ["regression", "classification", "multiclass"]:
        raise ValueError('Task must be "classification", "regression" or "multiclass".')
    if y is None:
        raise ValueError("No training labels provided.")
    if fastshap:
        try:
            from fasttreeshap import TreeExplainer as FastTreeExplainer
        except ImportError as exc:
            raise ImportError(
                "fasttreeshap is not installed; install it or use fastshap=False."
            ) from exc

    if encode:
        encoder = (
            OrdinalEncoderPandas(**encoder_kwargs)
            if encoder_kwargs is not None
            else OrdinalEncoderPandas()
        )
        X = encoder.fit_transform(X)
    params = (
        {"objective": "rmse", "zero_as_missing": False}
        if lgb_kwargs is None
        else lgb_kwargs
    )
    gbm_model = GradientBoosting(
        cat_feat="auto",
        stratified=False,
        params=params,
        show_learning_curve=False,
        return_valid_features=True,
        verbose_eval=verbose,
    )
    gbm_model.fit(X=X, y=y, sample_weight=sample_weight)
    if fastshap:
        explainer = FastTreeExplainer(
            gbm_model.model,
            algorithm="auto",
            shortcut=False,
            feature_perturbation="tree_path_dependent",
        )
        shap_matrix = explainer.shap_values(gbm_model.valid_features)
    else:
        shap_matrix = gbm_model.model.predict(
            gbm_model.valid_features, pred_contrib=True
        )
    values = (
        np.stack(shap_matrix, axis=-1)
        if isinstance(shap_matrix, list)
        else np.asarray(shap_matrix)
    )
    if not np.isfinite(values).all():
        raise ValueError("SHAP contributions must be finite.")
    if not fastshap:
        n_rows, n_features = gbm_model.valid_features.shape
        block_size = n_features + 1
        if (
            values.ndim != 2
            or values.shape[0] != n_rows
            or values.shape[1] == 0
            or values.shape[1] % block_size
        ):
            raise ValueError(
                "Native SHAP contributions must have shape (rows, (features + 1) * outputs)."
            )
        # Native output has a bias column in every output block; TreeExplainer does not.
        values = values.reshape(n_rows, -1, block_size)[:, :, :-1].transpose(0, 2, 1)
    importance = dict(
        _compute_importance(
            gbm_model.valid_features, values, params, params["objective"], fastshap
        )
    )
    feature_importances = pd.DataFrame(
        {"feature": X.columns, "importance": [importance[c] for c in X.columns]}
    )
    feature_importances = feature_importances.sort_values(
        "importance", ascending=False, kind="stable"
    ).reset_index(drop=True)
    total = feature_importances["importance"].sum()
    if not np.isfinite(total):
        raise ValueError("Total feature importance must be finite.")
    feature_importances["normalized_importance"] = (
        feature_importances["importance"] / total if total > 0 else 0.0
    )
    feature_importances["cumulative_importance"] = (
        feature_importances["importance"].cumsum() / total if total > 0 else 0.0
    )
    return feature_importances
