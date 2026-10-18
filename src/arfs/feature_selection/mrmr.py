"""MRMR Feature Selection Module

This module provides MinRedundancyMaxRelevance (MRMR) feature selection for classification or regression tasks.
In a classification task, the target should be of object or pandas category dtype, while in a regression task,
the target should be numeric. The predictors can be categorical or numerical without requiring encoding,
as the appropriate method (correlation, correlation ratio, or Theil's U) will be automatically selected based on the data type.

Module Structure:
-----------------
- ``MinRedundancyMaxRelevance``: MRMR feature selection class for classification or regression tasks.
"""

import functools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from ..association import (
    association_series,
    f_stat_classification_parallel,
    f_stat_regression_parallel,
)
from ..utils import validate_sample_weight, validate_target

FLOOR = 0.001


class MinRedundancyMaxRelevance(SelectorMixin, BaseEstimator):
    """MRMR feature selection for a classification or a regression task
    For a classification task, the target should be of object or pandas category
    dtype. For a regression task, the target should be numeric.
    The predictors can be categorical or numerical, there is no encoding required.
    The dtype will be automatically detected and the right method applied (either
    correlation, correlation ration or Theil's U)


    Parameters
    ----------
    n_features_to_select: int
        Number of features to select.
    relevance_func: callable, optional
        relevance function having arguments "X", "y", "sample_weight" and returning a pd.Series
        containing finite nonnegative scores indexed by predictor names.
        Missing/NaN scores make a predictor unavailable. Defaults use F/(1+F)
        on [0, 1], a mixed-type ranking heuristic without significance claims.
    redundancy_func: callable, optional
        Redundancy method.
        If callable, it should take "X", "sample_weight" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator_func: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    task: str
        either "regression" or "classification"
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    n_jobs: int (optional, default=1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    sample_size : int or None, default=None
        Optional maximum number of rows used to estimate both relevance and
        redundancy. Draw without replacement only when the input exceeds this
        positive integer. Classification draws are stratified by target;
        regression draws are uniform. None uses all rows. Weights are subset
        positionally, not used as sampling probabilities or rescaled.
    random_state : int, RandomState instance or None, default=None
        Controls row sampling. An integer replays each fit; a RandomState
        instance advances; None uses NumPy's global RNG when sampling is needed.

    Attributes
    ----------
    selected_features_ : list
        Selected feature names, in selection order. ``fit`` always returns self.
    relevance_ : pd.Series
        Relevance scores of available predictors.
    redundancy_ : pd.Series
        Aggregate redundancy at each feature's selection step.
    relevance_func_, redundancy_func_ : callable
        Callbacks resolved from the current parameters during fit.
    n_features_in_ : int
        number of input predictors
    ranking_ : pd.DataFrame
        name and scores for the selected features
    support_ : list of bool
        the list of the selected X-columns
    sample_indices_ : ndarray or None
        Original row positions used for estimation, in input order. None means
        all rows; empty selections do not draw a sample.
    n_samples_fit_ : int
        Number of rows supplied to the relevance and redundancy callbacks,
        before their missing-value handling.
    Notes
    -----
    Ties follow the input column order. Zero-relevance predictors remain eligible:
    MRMR fills the requested count rather than testing significance. Missing
    custom scores or zero input columns yield an empty fitted selection; a zero
    requested count does too. The default denominator uses absolute unnormalized
    associations, floored at 0.001. Aggregate redundancy exactly one gives a
    zero quotient score, so exact duplicates are postponed, but remain eligible.
    The first selected feature keeps the historical ranking convention of an
    infinite MRMR score and zero recorded redundancy.

    Sampling changes scores and can change selected features. It does not reduce
    the square redundancy table's storage. Rare predictor levels can disappear;
    a draw losing any positive-weight target class is rejected. Default row
    handling and weight-mass requirements still apply after sampling. In
    particular, frequency-weight relevance uses the smaller retained mass.
    Check selection stability and held-out utility before choosing a row cap.

    Example
    -------
    >>> from sklearn.datasets import make_classification, make_regression
    >>> X, y = make_regression(n_samples = 1000, n_features = 50, n_informative = 5, shuffle=False) # , n_redundant = 5
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> pred_name = [f"pred_{i}" for i in range(X.shape[1])]
    >>> X.columns = pred_name
    >>> y.name = "target"
    >>> fs_mrmr = MinRedundancyMaxRelevance(
    >>>                  n_features_to_select=5,
    >>>                  relevance_func=None,
    >>>                  redundancy_func=None,
    >>>                  task="regression", #"classification",
    >>>                  denominator_func=np.mean,
    >>>                  only_same_domain=False,
    >>>                  show_progress=True)
    >>> #fs_mrmr.fit(X=X, y=y.astype(str), sample_weight=None)
    >>> fs_mrmr.fit(X=X, y=y, sample_weight=None)
    """

    def __init__(
        self,
        n_features_to_select,
        relevance_func=None,
        redundancy_func=None,
        task="regression",
        denominator_func=np.mean,
        only_same_domain=False,
        n_jobs=1,
        show_progress=True,
        sample_size=None,
        random_state=None,
    ):
        self.n_features_to_select = n_features_to_select
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func
        self.denominator_func = denominator_func
        self.only_same_domain = only_same_domain
        self.show_progress = show_progress
        self.n_jobs = n_jobs
        self.task = task
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit the selector and return self; scores are stored in fitted attributes.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Predictor data.
        y : array-like or pd.Series of shape (n_samples,)
            Target vector. Must be numeric for regression or categorical for classification.
        sample_weight : pd.Series, optional, shape (n_samples,)
            Default relevance uses frequency weight mass for degrees of freedom,
            without normalizing weights. The F/(1+F) relevance map preserves
            integer row replication; fractional mass defines scores only.
            Custom relevance defines its own weight meaning. Redundancy
            coefficients accept relative weights.

        Returns
        -------
        self : object
            Fitted selector.
        """
        self.relevance_func_ = self.relevance_func
        self.redundancy_func_ = self.redundancy_func
        if self.relevance_func_ is None:
            if self.task == "regression":
                self.relevance_func_ = functools.partial(
                    f_stat_regression_parallel, n_jobs=self.n_jobs
                )
            else:
                self.relevance_func_ = functools.partial(
                    f_stat_classification_parallel, n_jobs=self.n_jobs
                )

        if self.redundancy_func_ is None:
            self.redundancy_func_ = functools.partial(
                association_series, n_jobs=self.n_jobs, normalize=False
            )

        if (
            isinstance(self.n_features_to_select, (bool, np.bool_))
            or not isinstance(self.n_features_to_select, (int, np.integer))
            or self.n_features_to_select < 0
        ):
            raise ValueError("n_features_to_select must be a nonnegative integer.")

        if self.sample_size is not None and (
            isinstance(self.sample_size, (bool, np.bool_))
            or not isinstance(self.sample_size, (int, np.integer))
            or self.sample_size <= 0
        ):
            raise ValueError("sample_size must be a positive integer or None.")

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            raise TypeError("X is not a pd.DataFrame")

        target = validate_target(y, len(X)).rename("target")
        sample_weight = validate_sample_weight(sample_weight, len(X))
        if self.task == "classification":
            target = target.astype("category")

        self.sample_indices_ = None
        if (
            self.sample_size is not None
            and len(X) > self.sample_size
            and self.n_features_to_select > 0
            and X.shape[1] > 0
        ):
            self.sample_indices_ = np.sort(
                resample(
                    np.arange(len(X)),
                    replace=False,
                    n_samples=self.sample_size,
                    random_state=self.random_state,
                    stratify=target.cat.codes
                    if self.task == "classification"
                    else None,
                )
            )
            sampled_target = target.iloc[self.sample_indices_]
            sampled_weight = (
                None if sample_weight is None else sample_weight[self.sample_indices_]
            )
            sampled_weight = validate_sample_weight(sampled_weight, self.sample_size)
            if self.task == "classification":
                full_classes = (
                    target[sample_weight > 0] if sample_weight is not None else target
                )
                sampled_classes = (
                    sampled_target[sampled_weight > 0]
                    if sampled_weight is not None
                    else sampled_target
                )
                if sampled_classes.nunique() != full_classes.nunique():
                    raise ValueError(
                        "Sampling lost a positive-weight target class; increase "
                        "sample_size or use sample_size=None."
                    )
            X = X.iloc[self.sample_indices_]
            target, sample_weight = sampled_target, sampled_weight
        self.n_samples_fit_ = len(X)

        self.relevance_args = {"X": X, "y": target, "sample_weight": sample_weight}
        self.redundancy_args = {"X": X, "sample_weight": sample_weight}

        if self.n_features_to_select == 0 or X.shape[1] == 0:
            self.relevance = pd.Series(dtype=float)
        else:
            self.relevance = self.relevance_func_(**self.relevance_args)
        if (
            not isinstance(self.relevance, pd.Series)
            or not self.relevance.index.is_unique
        ):
            raise ValueError("Relevance must be a Series with unique predictor labels.")
        if not self.relevance.index.isin(X.columns).all():
            raise ValueError("Relevance contains unknown predictor labels.")
        available = self.relevance.dropna()
        if not np.isfinite(available).all() or (available < 0).any():
            raise ValueError(
                "Available relevance scores must be finite and nonnegative."
            )
        self.relevance = self.relevance.reindex(X.columns)
        self.features = self.relevance[~self.relevance.isna()].index.to_list()
        self.relevance = self.relevance.loc[self.features]
        self.redundancy = pd.DataFrame(
            FLOOR, index=self.features, columns=self.features
        )
        self.n_features_to_select_ = min(self.n_features_to_select, len(self.features))
        self.n_features_in_ = X.shape[1]

        self.selected_features = []
        self.not_selected_features = self.features.copy()
        self.ranking_ = pd.Series(
            dtype="float64"
        )  # pd.DataFrame(columns=['var_name', 'mrmr', 'relevancy', 'redundancy'])
        self.redundancy_ = pd.Series(dtype="float64")
        self.run_feature_selection()

        # store the output in the sklearn flavour
        self.relevance_ = self.relevance
        self.ranking_ = pd.concat(
            [self.ranking_, self.relevance_, self.redundancy_], axis=1
        )
        self.ranking_.columns = ["mrmr", "relevance", "redundancy"]
        self.ranking_ = self.ranking_.loc[self.selected_features]

        # Set back the mrmr score to Inf for the first selected feature to avoid dividing by zero
        if self.selected_features:
            self.ranking_.iloc[0, 0] = float("Inf")

        self.selected_features_ = self.selected_features
        self.support_ = np.asarray(
            [x in self.selected_features for x in self.feature_names_in_], dtype=bool
        )
        self.not_selected_features_ = self.not_selected_features

        return self

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

    def fit_transform(self, X, y, sample_weight=None, **fit_params):
        """
        Fit to data, then transform it.
        Fits transformer to `X` and `y` and optionally sample_weight
        with optional parameters `fit_params`
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
        return self.fit(X=X, y=y, sample_weight=sample_weight).transform(X)

    def _get_support_mask(self):
        check_is_fitted(self, "support_")

        return self.support_

    def _more_tags(self):
        return {"allow_nan": True}

    def select_next_feature(
        self, not_selected_features, selected_features, relevance, redundancy
    ):
        score_numerator = relevance.loc[not_selected_features]

        if len(selected_features) > 0:
            last_selected_feature = selected_features[-1]

            if self.only_same_domain:
                not_selected_features_sub = [
                    c
                    for c in not_selected_features
                    if c.split("_")[0] == last_selected_feature.split("_")[0]
                ]
            else:
                not_selected_features_sub = not_selected_features

            if not_selected_features_sub:
                redundancy.loc[not_selected_features_sub, last_selected_feature] = (
                    self.redundancy_func_(
                        target=last_selected_feature,
                        features=not_selected_features_sub,
                        **self.redundancy_args,
                    )
                    .fillna(FLOOR)
                    .abs()
                    .clip(FLOOR)
                )
                score_denominator = (
                    redundancy.loc[not_selected_features, selected_features]
                    .apply(self.denominator_func, axis=1)
                    .replace(1.0, float("Inf"))
                )

            else:
                score_denominator = pd.Series(1, index=self.features)

        else:
            score_denominator = pd.Series(1, index=self.features)

        score = score_numerator / score_denominator
        score = score.sort_values(ascending=False, kind="stable")
        best_feature = score.index[score.argmax()]

        return best_feature, score, score_denominator

    def update_ranks(self, best_feature, score, score_denominator):
        self.ranking_ = pd.concat(
            [
                self.ranking_,
                pd.Series({best_feature: score.loc[best_feature]}, dtype="float64"),
            ]
        )
        self.redundancy_ = pd.concat(
            [
                self.redundancy_,
                pd.Series(
                    {best_feature: score_denominator.loc[best_feature]},
                    dtype="float64",
                ),
            ]
        )
        # the first selected feature has a default denominator (redundancy) = 1 to avoid dividing by zero
        # I set it back to zero
        self.redundancy_ = self.redundancy_.replace(1.0, 0.0)
        self.selected_features.append(best_feature)
        self.not_selected_features.remove(best_feature)

    def run_feature_selection(self):
        for i in tqdm(
            range(self.n_features_to_select_), disable=not self.show_progress
        ):
            best_feature, score, score_denominator = self.select_next_feature(
                self.not_selected_features,
                self.selected_features,
                self.relevance,
                self.redundancy,
            )
            self.update_ranks(best_feature, score, score_denominator)
