"""MRMR Feature Selection Module

This module provides MinRedundancyMaxRelevance (MRMR) feature selection for classification or regression tasks. 
In a classification task, the target should be of object or pandas category dtype, while in a regression task, 
the target should be of numpy categorical dtype. The predictors can be categorical or numerical without requiring encoding, 
as the appropriate method (correlation, correlation ratio, or Theil's U) will be automatically selected based on the data type.

Module Structure:
-----------------
- ``MinRedundancyMaxRelevance``: MRMR feature selection class for classification or regression tasks.
"""

import functools
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from sklearn.feature_selection._base import SelectorMixin
from ..association import (
    f_stat_classification_parallel,
    f_stat_regression_parallel,
    association_series,
)

FLOOR = 0.001


class MinRedundancyMaxRelevance(SelectorMixin, BaseEstimator):
    """MRMR feature selection for a classification or a regression task
    For a classification task, the target should be of object or pandas category
    dtype. For a regression task, the target should be of numpy categorical dtype.
    The predictors can be categorical or numerical, there is no encoding required.
    The dtype will be automatically detected and the right method applied (either
    correlation, correlation ration or Theil's U)


    Parameters
    ----------
    n_features_to_select: int
        Number of features to select.
    relevance_func: callable, optional
        relevance function having arguments "X", "y", "sample_weight" and returning a pd.Series
        containing a score of relevance for each feature
    redundancy: callable, optional
        Redundancy method.
        If callable, it should take "X", "sample_weight" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    task: str
        either "regression" or "classifiction"
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.
    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.

    Attributes
    ----------
    n_features_in_ : int
        number of input predictors
    ranking_ : pd.DataFrame
        name and scores for the selected features
    support_ : list of bool
        the list of the selected X-columns
    Example
    -------
    >>> from sklearn.datasets import make_classification, make_regression
    >>> X, y = make_regression(n_samples = 1000, n_features = 50, n_informative = 5, shuffle=False) # , n_redundant = 5
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> pred_name = [f"pred_{i}" for i in range(X.shape[1])]
    >>> X.columns = pred_name
    >>> y.name = "target"
    >>> fs_mrmr = MinRedundancyMaxRelevance(n_features_to_select=5,
    >>>                  relevance_func=None,
    >>>                  redundancy_func=None,
    >>>                  task= "regression",#"classification",
    >>>                  denominator_func=np.mean,
    >>>                  only_same_domain=False,
    >>>                  return_scores=False,
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
        return_scores=False,
        n_jobs=1,
        show_progress=True,
    ):
        self.n_features_to_select = n_features_to_select
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func
        self.denominator_func = denominator_func
        self.only_same_domain = only_same_domain
        self.return_scores = return_scores
        self.show_progress = show_progress
        self.n_jobs = n_jobs
        self.task = task

        if self.relevance_func is None:
            if self.task == "regression":
                self.relevance_func = functools.partial(
                    f_stat_regression_parallel, n_jobs=self.n_jobs
                )
            else:
                self.relevance_func = functools.partial(
                    f_stat_classification_parallel, n_jobs=self.n_jobs
                )

        if self.redundancy_func is None:
            self.redundancy_func = functools.partial(
                association_series, n_jobs=self.n_jobs, normalize=True
            )

    def fit(self, X, y, sample_weight=None):
        """fit the MRmr selector by learning the associations

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
            raise TypeError("X is not a pd.DataFrame")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        y.name = "target"

        target = y.copy()
        if self.task == "classification":
            target = target.astype("category")

        self.relevance_args = {"X": X, "y": target, "sample_weight": sample_weight}
        self.redundancy_args = {"X": X, "sample_weight": sample_weight}

        self.relevance = self.relevance_func(**self.relevance_args)
        self.features = self.relevance[~self.relevance.isna()].index.to_list()
        self.relevance = self.relevance.loc[self.features]
        self.redundancy = pd.DataFrame(
            FLOOR, index=self.features, columns=self.features
        )
        self.n_features_to_select = min(self.n_features_to_select, len(self.features))

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()

        self.n_features_in_ = len(self.features)

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
        self.ranking_ = self.ranking_.iloc[: self.n_features_to_select, :]

        # Set back the mrmr score to Inf for the first selected feature to avoid dividing by zero
        self.ranking_.iloc[0, 0] = float("Inf")

        self.selected_features_ = self.selected_features
        self.support_ = np.asarray(
            [x in self.selected_features for x in self.feature_names_in_]
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

    def fit_transform(self, X, y, sample_weight=None):
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
        check_is_fitted(self)

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
                    self.redundancy_func(
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
        score = score.sort_values(ascending=False)
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
        for i in tqdm(range(self.n_features_to_select), disable=not self.show_progress):
            best_feature, score, score_denominator = self.select_next_feature(
                self.not_selected_features,
                self.selected_features,
                self.relevance,
                self.redundancy,
            )
            self.update_ranks(best_feature, score, score_denominator)
