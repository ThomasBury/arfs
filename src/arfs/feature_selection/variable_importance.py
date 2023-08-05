"""Supervised Feature Selection

This module provides selectors using supervised statistics and a threshold, using SHAP, permutation importance or impurity (Gini) importance.

Module Structure:
-----------------
- ``VariableImportance`` main class for identifying non-important features
"""

from __future__ import print_function
from tqdm.auto import trange

# pandas
import pandas as pd

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# sklearn
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin

# ARFS
from ..utils import reset_plot
from ..gbm import GradientBoosting
from ..preprocessing import OrdinalEncoderPandas


class VariableImportance(SelectorMixin, BaseEstimator):
    """Feature selector that removes predictors with zero or low variable importance.

    Identify the features with zero/low importance according to SHAP values of a lightgbm.
    The gbm can be trained with early stopping using a utils set to prevent overfitting.
    The feature importances are averaged over `n_iterations` to reduce the variance.
    The predictors are then ranked from the most important to the least important and the
    cumulative variable importance is computed. All the predictors not contributing (VI=0) or
    contributing to less than the threshold to the cumulative importance are removed.

    Parameters
    ----------
    task : string
        The machine learning task, either 'classification' or 'regression' or 'multiclass',
        be sure to use a consistent objective function
    encode : boolean, default = True
        Whether or not to encode the predictors
    n_iterations : int, default = 10
        Number of iterations, the more iterations, the smaller the variance
    threshold : float, default = .99
        The selector computes the cumulative feature importance and ranks
        the predictors from the most important to the least important.
        All the predictors contributing to less than this value are rejected.
    lgb_kwargs : dictionary of keyword arguments
        dictionary of lightgbm estimators parameters with at least the objective function {'objective':'rmse'}
    encoder_kwargs : dictionary of keyword arguments, optional
        dictionary of the :class:`OrdinalEncoderPandas` parameters


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
    fastshap : boolean
        enable or not the fasttreeshap implementation
    verbose : int, default = -1
        controls the progress bar, > 1 print out progress

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
        n_iterations=10,
        threshold=0.99,
        lgb_kwargs={"objective": "rmse", "zero_as_missing": False},
        encoder_kwargs=None,
        fastshap=True,
        verbose=-1,
    ):
        self.task = task
        self.encode = encode
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.lgb_kwargs = lgb_kwargs
        self.encoder_kwargs = encoder_kwargs
        self.verbose = verbose
        self.fastshap = fastshap

        if (self.threshold > 1.0) or (self.threshold < 0.0):
            raise ValueError("``threshold`` should be larger than 0 and smaller than 1")

    def fit(self, X, y, sample_weight=None):
        """Learn variable importance from X and y, supervised learning.

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

        feature_importances = _compute_varimp_lgb(
            X=X,
            y=y,
            sample_weight=sample_weight,
            encode=self.encode,
            task=self.task,
            n_iterations=self.n_iterations,
            verbose=self.verbose,
            encoder_kwargs=self.encoder_kwargs,
            lgb_kwargs=self.lgb_kwargs,
            fastshap=self.fastshap,
        )

        self.feature_importances_summary_ = feature_importances

        support_ordered = (
            self.feature_importances_summary_["cumulative_importance"] >= self.threshold
        )
        to_drop = list(
            self.feature_importances_summary_.loc[support_ordered, "feature"]
        )

        self.support_ = np.asarray(
            [False if c in to_drop else True for c in self.feature_names_in_]
        )
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
        X : ndarray array of shape (n_samples, n_features_new)
            Transformed array.

        Raises
        ------
        TypeError
            if the input is not a pd.DataFrame
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
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
        """Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold`
        cumulative importance.

        Parameters
        ----------
        plot_n : int, default = 50
            Number of most important features to plot. Defaults to 15 or the maximum
            number of features whichever is smaller
        n_feat_per_inch : int
            number of features per inch, the larger the less space between labels
        figsize : tuple of float, optional
            The rendered size as a percentage size
        log : bool, default=True
            Whether or not render variable importance on a log scale
        style : bool, default=False
            set arfs style or not

        Returns
        -------
        hv.plot
            the feature importances holoviews object

        """
        if style:
            plt.style.use(style)
        else:
            reset_plot()

        if plot_n > self.feature_importances_summary_.shape[0]:
            plot_n = self.feature_importances_summary_.shape[0] - 1

        df = self.feature_importances_summary_
        importance_index = np.min(
            np.where(df["cumulative_importance"] > self.threshold)
        )
        non_cum_threshold = df.iloc[importance_index, 2]
        max_norm_importance = 0.99 * df.normalized_importance.max()

        if plot_n > df.shape[0]:
            plot_n = df.shape[0] - 1

        if figsize is None:
            figsize = (8, plot_n / n_feat_per_inch)
        fig = plt.figure(tight_layout=True, figsize=figsize)
        gs = gridspec.GridSpec(3, 3)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.scatter(df.normalized_importance, df.feature)
        # ax.set_ylabel('YLabel0')
        ax1.set_xlabel("normalized importance")
        ax1.xaxis.set_label_position("top")
        ax1.invert_yaxis()
        ax1.axvline(x=non_cum_threshold, linestyle="dashed", color="r")
        if log:
            ax1.set_xscale("log")
        ax1.grid()
        ax1.set(frame_on=False)

        ax2 = fig.add_subplot(gs[:, 1:])
        ax2.scatter(df.feature, df.cumulative_importance)
        # ax.set_ylabel('YLabel0')
        ax2.set_ylabel("cumulative importance")
        ax2.tick_params(axis="x", labelrotation=90)

        importance_min_value_on_axis = max_norm_importance if log else 0
        x_vert, y_vert = [importance_index, importance_index], [
            importance_min_value_on_axis,
            self.threshold,
        ]
        x_horiz, y_horiz = [importance_min_value_on_axis, importance_index], [
            self.threshold,
            self.threshold,
        ]

        ax2.plot(x_vert, y_vert, linestyle="dashed", color="r")
        ax2.plot(x_horiz, y_horiz, linestyle="dashed", color="r")
        ax2.set_ylim(max_norm_importance, 1.0)
        if log:
            ax2.set_xscale("log")
        ax2.grid()
        ax2.set(frame_on=False)

        fig.align_labels()
        plt.show()


def _compute_varimp_lgb(
    X,
    y,
    sample_weight=None,
    encode=False,
    task="regression",
    n_iterations=10,
    verbose=-1,
    fastshap=True,
    encoder_kwargs=None,
    lgb_kwargs={"objective": "rmse", "zero_as_missing": False},
):
    if task not in ["regression", "classification", "multiclass"]:
        raise ValueError('Task must be either "classification" or "regression"')

    if y is None:
        raise ValueError("No training labels provided.")

    if encode:
        encoder = (
            OrdinalEncoderPandas(**encoder_kwargs)
            if encoder_kwargs is not None
            else OrdinalEncoderPandas()
        )
        X = encoder.fit(X).transform(X)
        del encoder
    # Extract feature names
    feature_names = list(X.columns)
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    progress_bar = trange(n_iterations) if verbose > 1 else range(n_iterations)

    # Iterate through each fold
    for _ in progress_bar:
        if verbose > 1:
            progress_bar.set_description("Iteration nb: {0:<3}".format(_))

        # lgb_kwargs['verbose'] = -1
        gbm_model = GradientBoosting(
            cat_feat="auto",
            stratified=False,
            params=lgb_kwargs,
            show_learning_curve=False,
            return_valid_features=True,
            verbose_eval=0,
        )

        gbm_model.fit(X=X, y=y, sample_weight=sample_weight)

        # pimp cool but too slow
        # perm_imp =  permutation_importance(
        # model, valid_features, valid_labels, n_repeats=10, random_state=42, n_jobs=-1
        # )
        # perm_imp = perm_imp.importances_mean
        if fastshap:
            try:
                from fasttreeshap import TreeExplainer as FastTreeExplainer
            except ImportError:
                ImportError("fasttreeshap is not installed")
            
            explainer = FastTreeExplainer(
                gbm_model.model,
                algorithm="auto",
                shortcut=False,
                feature_perturbation="tree_path_dependent",
            )
            shap_matrix = explainer.shap_values(gbm_model.valid_features)
            if isinstance(shap_matrix, list):
                # For LightGBM classifier, RF, in sklearn API, SHAP returns a list of arrays
                # https://github.com/slundberg/shap/issues/526
                shap_imp = np.mean([np.abs(sv).mean(0) for sv in shap_matrix], axis=0)
            else:
                shap_imp = np.abs(shap_matrix).mean(0)
        else:
            shap_matrix = gbm_model.model.predict(
                gbm_model.valid_features, pred_contrib=True
            )
            # the dim changed in lightGBM >= 3.0.0
            if task == "multiclass":
                # X_SHAP_values (array-like of shape = [n_samples, n_features + 1]
                # or shape = [n_samples, (n_features + 1) * n_classes])
                # index starts from 0
                n_feat = gbm_model.valid_features.shape[1]
                y_freq_table = pd.Series(y.fillna(0)).value_counts(normalize=True)
                n_classes = y_freq_table.size
                shap_matrix = np.delete(
                    shap_matrix,
                    list(range(n_feat, (n_feat + 1) * n_classes, n_feat + 1)),
                    axis=1,
                )
                shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
            else:
                # for binary, only one class is returned, for regression a single column added as well
                shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)

        # Record the feature importances
        feature_importance_values += (
            shap_imp / n_iterations
        )  # model.feature_importances_ / n_iterations
    feature_importances = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance_values}
    )
    # Sort features according to importance
    feature_importances = feature_importances.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    # Normalize the feature importances to add up to one
    feature_importances["normalized_importance"] = (
        feature_importances["importance"] / feature_importances["importance"].sum()
    )
    feature_importances["cumulative_importance"] = np.cumsum(
        feature_importances["normalized_importance"]
    )
    # Extract the features with zero importance
    # record_zero_importance = feature_importances[
    #     feature_importances["importance"] == 0.0
    # ]
    return feature_importances
