"""This module provides 3 different methods to perform 'all relevant feature selection'


Reference:
----------
NILSSON, Roland, PEÑA, José M., BJÖRKEGREN, Johan, et al.
Consistent feature selection for pattern recognition in polynomial time.
Journal of Machine Learning Research, 2007, vol. 8, no Mar, p. 589-612.

KURSA, Miron B., RUDNICKI, Witold R., et al.
Feature selection with the Boruta package.
J Stat Softw, 2010, vol. 36, no 11, p. 1-13.

https://github.com/chasedehan/BoostARoota

The module structure
--------------------
- The ``Leshy`` class, a heavy re-work of ``BorutaPy`` class
  itself a modified version of Boruta, the pull request I submitted and still pending:
  https://github.com/scikit-learn-contrib/boruta_py/pull/100

- The ``BoostAGroota`` class, a modified version of BoostARoota, PR still to be submitted
  https://github.com/chasedehan/BoostARoota

- The ``GrootCV`` class for a new method for all relevant feature selection using a lightgGBM model,
  cross-validated SHAP importances and shadowing.

Original BorutaPy version
-------------------------
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
Modified by Thomas Bury, pull request:
https://github.com/scikit-learn-contrib/boruta_py/pull/100
Waiting for merging

https://github.com/scikit-learn-contrib/boruta_py/pull/100
is a new PR based on #77 making all the changes optional. Waiting for merge

Leshy is a re-work of the PR I submitted.

License: BSD 3 clause

"""

from __future__ import print_function, division
import operator
import warnings
import time
import shap
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp

from typing import Tuple
from tqdm.auto import tqdm
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import BaseEstimator, is_regressor, is_classifier, clone

from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection._base import SelectorMixin
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import _check_sample_weight
from matplotlib.lines import Line2D
from lightgbm import early_stopping


from ..utils import (
    check_if_tree_based,
    is_lightgbm,
    is_catboost,
    create_dtype_dict,
    get_pandas_cat_codes,
    validate_sample_weight,
)

########################################################################################
#
# Main Classes and Methods
# Provide a fit, transform and fit_transform method
#
########################################################################################
# !/usr/bin/env python
# -*- coding: utf-8 -*-

NO_FEATURE_SELECTED_WARNINGS = "No feature selected - No data to plot"
ARFS_COLOR_LIST = [
    "#000000",
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    "#008695",
    "#CF1C90",
    "#F97B72",
]
BLUE = "#2590fa"
YELLOW = "#f0be00"
RED = "#b51204"
BCKGRD_COLOR = "#f5f5f5"
PLT_PARAMS = {
    "axes.prop_cycle": plt.cycler(color=ARFS_COLOR_LIST),
    "axes.facecolor": BCKGRD_COLOR,
    "patch.edgecolor": BCKGRD_COLOR,
    "figure.facecolor": BCKGRD_COLOR,
    "axes.edgecolor": BCKGRD_COLOR,
    "savefig.edgecolor": BCKGRD_COLOR,
    "savefig.facecolor": BCKGRD_COLOR,
    "grid.color": "#d2d2d2",
    "lines.linewidth": 1.5,
}
PLOT_KWARGS = dict(
    kind="box",
    boxprops=dict(linestyle="-", linewidth=1.5, color="gray", facecolor="gray"),
    flierprops=dict(linestyle="-", linewidth=1.5, color="gray"),
    medianprops=dict(linestyle="-", linewidth=1.5, color="#000000"),
    whiskerprops=dict(linestyle="-", linewidth=1.5, color="gray"),
    capprops=dict(linestyle="-", linewidth=1.5, color="gray"),
    showfliers=False,
    grid=True,
    rot=0,
    vert=False,
    patch_artist=True,
    fontsize=9,
)


class Leshy(SelectorMixin, BaseEstimator):
    """This is an improved version of BorutaPy which itself is an
    improved Python implementation of the Boruta R package.
    Boruta is an all relevant feature selection method, while most other are
    minimal optimal; this means it tries to find all features carrying
    information usable for prediction, rather than finding a possibly compact
    subset of features on which some estimator has a minimal error.
    Why bother with all relevant feature selection?
    When you try to understand the phenomenon that made your data, you should
    care about all factors that contribute to it, not just the bluntest signs
    of it in context of your methodology (minimal optimal set of features
    by definition depends on your estimator choice).

    Parameters
    ----------
    estimator : object
        A supervised learning estimator, with a 'fit' method that returns the
        ``feature_importances_`` attribute. Important features must correspond to
        high absolute values in the ``feature_importances_``
    n_estimators : int or string, default = 1000
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.
    perc : int, default = 100
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        The default is essentially the vanilla Boruta corresponding to the max.
    alpha : float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.
    importance : str, default = 'shap'
        The kind of variable importance used to compare and discriminate original
        vs shadow predictors. Note that the builtin tree importance (gini/impurity based
        importance) is biased towards numerical and large cardinality predictors, even
        if they are random. Shapley values and permutation imp. are robust w.r.t those predictors.
        Possible values: 'shap' (Shapley values), 'fastshap' (FastTreeShap implementation),
        'pimp' (permutation importance) and 'native' (Gini/impurity)
    two_step : Boolean, default = True
        If you want to use the original implementation of Boruta with Bonferroni
        correction only set this to False.
    max_iter : int, default = 100
        The number of maximum iterations to perform.
    random_state : int, RandomState instance or None; default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, default=0
        Controls verbosity of output. 0: no output, 1: displays iteration number,
        2: which features have been selected already


    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features - only confirmed ones are True.
    support_weak_ : array of shape [n_features]
        The mask of selected tentative features, which haven't gained enough
        support during the max_iter number of iterations.
    selected_features_ : list of str
        the list of columns to keep
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank one and tentative features are assigned
        rank 2.
    ranking_absolutes_ : array of shape [n_features]
        The absolute feature ranking as ordered by selection process. It does not guarantee
        that this order is correct for all models. For a model agnostic ranking, see the
        the attribute ``ranking``
    cat_name : list of str
        the name of the categorical columns
    cat_idx : list of int
        the index of the categorical columns
    imp_real_hist : array
        array of the historical feature importance of the real predictors
    sha_max : float
        the maximum feature importance of the shadow predictors
    col_names : list of str
        the names of the real predictors


    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from boruta import BorutaPy
    >>>
    >>> # load X and y
    >>> # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    >>> X = pd.read_csv('examples/test_X.csv', index_col=0).values
    >>> y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    >>> y = y.ravel()
    >>>
    >>> # define random forest classifier, with utilising all cores and
    >>> # sampling in proportion to y labels
    >>> rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    >>>
    >>> # define Boruta feature selection method
    >>> feat_selector = Leshy(rf, n_estimators='auto', verbose=2, random_state=1)
    >>>
    >>> # find all relevant features - 5 features should be selected
    >>> feat_selector.fit(X, y)
    >>>
    >>> # check selected features - first 5 features are selected
    >>> feat_selector.selected_features_
    >>>
    >>> # check ranking of features
    >>> feat_selector.ranking_
    >>>
    >>> # call transform() on X to filter it down to selected features
    >>> X_filtered = feat_selector.transform(X)

    References
    ----------
    See the original paper [1]_ for more details.

    ..[1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
        Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010

    """

    def __init__(
        self,
        estimator,
        n_estimators=1000,
        perc=90,
        alpha=0.05,
        importance="shap",
        two_step=True,
        max_iter=100,
        random_state=None,
        verbose=0,
        keep_weak=False,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.random_state_instance = None
        self.verbose = verbose
        self.keep_weak = keep_weak
        self.importance = importance
        self.cat_name = None
        self.cat_idx = None
        # Catboost doesn't allow to change random seed after fitting
        self.is_cat = is_catboost(estimator)
        self.is_lgb = is_lightgbm(estimator)
        # plotting
        self.imp_real_hist = None
        self.sha_max = None
        self.n_features_ = 0
        self.support_ = None
        self.support_weak_ = None

    def fit(self, X, y, sample_weight=None):
        """Fits the Boruta feature selection with the provided estimator.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample

        Returns
        -------
        self : object
            Nothing but attributes

        """
        if self.importance == "fastshap":
            try:
                from fasttreeshap import TreeExplainer as FastTreeExplainer
            except ImportError:
                warnings.warn("fasttreeshap is not installed. Fallback to shap.")
                self.importance = "shap"

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            raise TypeError("X is not a dataframe")

        self.imp_real_hist = np.empty((0, X.shape[1]), float)
        self._fit(X, y, sample_weight=sample_weight)
        self.selected_features_ = self.feature_names_in_[self.support_]
        self.not_selected_features_ = self.feature_names_in_[~self.support_]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        return X[self.selected_features_]

    @mpl.rc_context(PLT_PARAMS)
    def plot_importance(self, n_feat_per_inch=5):
        """Boxplot of the variable importance, ordered by magnitude
        The max shadow variable importance illustrated by the dashed line.
        Requires to apply the fit method first.

        Parameters
        ----------
        n_feat_per_inch : int, default=5
            number of features to plot per inch (for scaling the figure)

        Returns
        -------
        fig : plt.figure
            the matplotlib figure object containing the boxplot
        """

        if self.imp_real_hist is None:
            raise ValueError("Use the fit method first to compute the var.imp")

        vimp_df = pd.DataFrame(self.imp_real_hist, columns=self.feature_names_in_)
        vimp_df = vimp_df.reindex(
            vimp_df.mean().sort_values(ascending=True).index, axis=1
        )

        if vimp_df.dropna().empty:
            warnings.warn(NO_FEATURE_SELECTED_WARNINGS)
            return None
        else:
            fig, ax = plt.subplots(figsize=(16, vimp_df.shape[1] / n_feat_per_inch))
            bp = vimp_df.plot(**PLOT_KWARGS, ax=ax)

            n_strong = sum(self.support_)
            n_weak = np.sum(self.support_weak_)
            n_discarded = np.sum(~(self.support_ | self.support_weak_))
            box_face_col = (
                [BLUE] * n_strong + [YELLOW] * n_weak + ["gray"] * n_discarded
            )
            for c in range(len(box_face_col)):
                bp.findobj(mpl.patches.Patch)[len(self.support_) - c - 1].set_facecolor(
                    box_face_col[c]
                )
                bp.findobj(mpl.patches.Patch)[len(self.support_) - c - 1].set_color(
                    box_face_col[c]
                )

            xrange = vimp_df.max(skipna=True).max(skipna=True) - vimp_df.min(
                skipna=True
            ).min(skipna=True)
            bp.set_xlim(left=vimp_df.min(skipna=True).min(skipna=True) - 0.10 * xrange)

            custom_lines = [
                Line2D([0], [0], color=BLUE, lw=5),
                Line2D([0], [0], color=YELLOW, lw=5),
                Line2D([0], [0], color="gray", lw=5),
                Line2D([0], [0], linestyle="--", color=RED, lw=2),
            ]
            bp.legend(
                custom_lines,
                ["confirmed", "tentative", "rejected", "sha. max"],
                loc="lower right",
            )
            ax.axvline(x=self.sha_max, linestyle="--", color=RED)
            ax.set_title("Leshy importance and selected predictors")
            return fig

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.support_

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}

    def _fit(self, X_raw, y, sample_weight=None):
        """Private method. See the methods overview in the documentation
        for explanation of the process

        Parameters
        ----------
        X_raw : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample

        Returns
        -------
        self : object
            Nothing but attributes
        """

        start_time = time.time()
        # the basic cat features encoding
        # is performed when getting importances
        # because the columns are dynamically created/rejected
        X = X_raw

        # only sklearn requires to fillna data
        # modern GBM implementations can handle this
        # X = X.fillna(0)
        y = pd.Series(y).fillna(0) if not isinstance(y, pd.Series) else y.fillna(0)

        # check input params
        self._check_params(X, y)
        sample_weight = validate_sample_weight(sample_weight)
        self.random_state = check_random_state(self.random_state)

        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != "auto":
            self.estimator.set_params(n_estimators=self.n_estimators)

        dec_reg, sha_max_history, imp_history, imp_sha_max = self.select_features(
            X=X, y=y, sample_weight=sample_weight
        )
        confirmed, tentative = _get_confirmed_and_tentative(dec_reg)
        tentative = _select_tentative(tentative, imp_history, sha_max_history)
        self._calculate_support(confirmed, tentative, n_feat)

        # for plotting
        self.imp_real_hist = imp_history
        self.sha_max = imp_sha_max

        # absolute and relative ranking
        self._calculate_absolute_ranking()
        self._calculate_relative_ranking(
            n_feat=n_feat,
            tentative=tentative,
            confirmed=confirmed,
            imp_history=imp_history,
        )
        self._print_result(dec_reg, _iter, start_time)
        return self

    def _get_tree_num(self, n_feat):
        """private method, get a good estimated for the number of trees
           given the number of features

        Parameters
        ----------
        n_feat : int
            The number of features

        Returns
        -------
        n_estimators : int
            the number of trees
        """
        depth = (
            self.estimator.get_params()["max_depth"]
            if not self.is_cat
            else self.estimator.get_param("max_depth")
        )
        if depth is None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        # n_feat * 2 because the training matrix is extended with n shadow features
        multi = (n_feat * 2) / (np.sqrt(n_feat * 2) * depth)
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _add_shadows_get_imps(self, X, y, sample_weight, dec_reg):
        """Add a shuffled copy of the columns (shadows) and get the feature
        importance of the augmented data set

        Parameters
        ----------
        X: pd.DataFrame of shape [n_samples, n_features]
            predictor matrix
        y: pd.series of shape [n_samples]
            target
        sample_weight: array-like, shape = [n_samples], default=None
            Individual weights for each sample
        dec_reg: array
            holds the decision about each feature 1, 0, -1 (accepted, undecided, rejected)
        Returns
        -------
         imp_real: array
            feature importance of the real predictors
         imp_sha: array
            feature importance of the shadow predictors
        """
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = X.iloc[:, x_cur_ind].copy()
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = x_cur.copy()
        # make sure there's at least 5 columns in the shadow matrix for
        while x_sha.shape[1] < 5:
            x_sha = pd.concat([x_sha, x_sha], axis=1)
        # shuffle xSha
        x_sha = x_sha.apply(self.random_state.permutation, axis=0)
        x_sha.columns = [f"Shadow_{i}" for i in range(x_sha.shape[1])]
        # get importance of the merged matrix
        if self.importance == "shap":
            imp = _get_shap_imp(
                self.estimator, pd.concat([x_cur, x_sha], axis=1), y, sample_weight
            )
        elif self.importance == "fastshap":
            imp = _get_shap_imp_fast(
                self.estimator, pd.concat([x_cur, x_sha], axis=1), y, sample_weight
            )
        elif self.importance == "pimp":
            imp = _get_perm_imp(
                self.estimator, pd.concat([x_cur, x_sha], axis=1), y, sample_weight
            )
        else:
            imp = _get_imp(
                self.estimator, pd.concat([x_cur, x_sha], axis=1), y, sample_weight
            )

        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]

        return imp_real, imp_sha

    @staticmethod
    def _assign_hits(hit_reg, cur_imp, imp_sha_max):
        """count how many times a given feature was more important than
        the best of the shadow features

        Parameters
        ----------
        hit_reg: array
            count how many times a given feature was more important than the
            best of the shadow features
        cur_imp: array
            current importance
        imp_sha_max: array
            importance of the best shadow predictor
        Returns
        -------
        hit_reg : array
            the how many times a given feature was more important than the
            best of the shadow features
        """
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp[0]
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        """Private method, Perform the rest if the feature should be tagget as relevant (confirmed), not relevant (rejected)
        or undecided. The test is performed by considering the binomial tentatives over several attempts.
        I.e. count how many times a given feature was more important than the best of the shadow features
        and test if the associated probability to the z-score is below, between or above the rejection or
        acceptance threshold.

        Parameters
        ----------
        dec_reg : array
            holds the decision about each feature 1, 0, -1 (accepted, undecided, rejected)
        hit_reg : array
            counts how many times a given feature was more important than the best of the shadow features
        _iter : int
            iteration number
        Returns
        -------
        dec_reg : array
            holds the decision about each feature 1, 0, -1 (accepted, undecided, rejected)

        """
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, 0.5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, 0.5).flatten()

        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = self._fdrcorrection(to_accept_ps, alpha=self.alpha)[0]
            to_reject = self._fdrcorrection(to_reject_ps, alpha=self.alpha)[0]

            # second we correct for testing the same feature over and over again
            # using bonferroni
            to_accept2 = to_accept_ps <= self.alpha / float(_iter)
            to_reject2 = to_reject_ps <= self.alpha / float(_iter)

            # combine the two multi corrections, and get indexes
            to_accept *= to_accept2
            to_reject *= to_reject2
        else:
            # as in th original Boruta, we simply do bonferroni correction
            # with the total n_feat in each iteration
            to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
            to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    @staticmethod
    def _fdrcorrection(pvals, alpha=0.05):
        """Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.

        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate
        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_

    @staticmethod
    def _nanrankdata(X, axis=1):
        """Replaces bottleneck's nanrankdata with scipy and numpy alternative.

        Parameters
        ----------
        X : array or pd.DataFrame
            the data array
        axis : int, optional
            row-wise (0) or column-wise (1), by default 1

        Returns
        -------
        ranks : array
            the ranked array
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """Private method, Check hyperparameters as well as X and y before proceeding with fit.

        Parameters
        ----------
        X : pd.DataFrame
            predictor matrix
        y : pd.series
            target series

        Raises
        ------
        ValueError
            [description]
        ValueError
            [description]
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y, dtype=None, force_all_finite=False)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError("The percentile should be between 0 and 100.")

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("Alpha should be between 0 and 1.")

    def _print_results(self, dec_reg, _iter, flag):
        """Private method, printing the result

        Parameters
        ----------
        dec_reg: array
            if the feature as been tagged as relevant (confirmed),
            not relevant (rejected) or undecided
        _iter: int
            the iteration number
        flag: int
            is still in the feature selection process or not
        Returns
        -------
         output: str
            the output to be printed out
        """
        n_iter = str(_iter) + " / " + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ["Iteration: ", "Confirmed: ", "Tentative: ", "Rejected: "]

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = "\n".join([x[0] + "\t" + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            n_rejected = np.sum(~(self.support_ | self.support_weak_))
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = "\n".join([x[0] + "\t" + x[1] for x in zip(cols, content)])
            if self.importance in ["shap", "pimp"]:
                vimp = str(self.importance)
            else:
                vimp = "native"
            output = (
                "\n\nLeshy finished running using " + vimp + " var. imp.\n\n" + result
            )
        print(output)

    def _update_estimator(self):
        """
        Update the estimator with a new random state, if applicable.

        If the dataset is not categorical, the estimator's `random_state` parameter is updated
        with a new random state generated by the `random_state` attribute of the Leshy object.
        If the estimator is a LightGBM model, the random state value is generated between 0 and 10000.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.is_cat is False:
            if self.is_lgb:
                self.estimator.set_params(
                    random_state=self.random_state.randint(0, 10000)
                )
            else:
                self.estimator.set_params(random_state=self.random_state)

    def _update_tree_num(self, dec_reg):
        """Update the number of trees in the estimator based on the number of selected features.

        Parameters
        ----------
        dec_reg : array-like of shape (n_features,)
            The decision rule for each feature, where negative values indicate that the feature should be rejected
            and non-negative values indicate that the feature should be selected.

        Returns
        -------
        None

        Notes
        -----
        This function updates the `n_estimators` parameter of the estimator if it is set to "auto". The number of trees is
        determined based on the number of selected features. Specifically, the number of trees is set to the value returned
        by the `_get_tree_num` method, which takes as input the number of selected features that are not rejected.

        If `n_estimators` is not set to "auto", this function does nothing.
        """
        if self.n_estimators == "auto":
            # number of features that aren't rejected
            not_rejected = np.where(dec_reg >= 0)[0].shape[0]
            n_tree = self._get_tree_num(not_rejected)
            self.estimator.set_params(n_estimators=n_tree)

    def _run_iteration(
        self, X, y, sample_weight, dec_reg, sha_max_history, imp_history, hit_reg, _iter
    ):
        """
        Run an iteration of the Gradient Boosting algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        dec_reg : array-like of shape (n_samples,)
            Decision function of the estimator.

        sha_max_history : list of floats
            List of the maximum shadow importance value at each iteration.

        imp_history : array-like of shape (n_iterations, n_features)
            Matrix of feature importances at each iteration.

        hit_reg : array-like of shape (n_samples,)
            Array of hit counts for each sample.

        _iter : int
            The current iteration number.

        Returns
        -------
        dec_reg : array-like of shape (n_samples,)
            Updated decision function of the estimator.

        sha_max_history : list of floats
            List of the maximum shadow importance value at each iteration.

        imp_history : array-like of shape (n_iterations, n_features)
            Matrix of feature importances at each iteration.

        hit_reg : array-like of shape (n_samples,)
            Array of hit counts for each sample.

        imp_sha_max : float
            The maximum shadow importance value for this iteration.
        """
        cur_imp = self._add_shadows_get_imps(X, y, sample_weight, dec_reg)
        imp_sha_max = np.percentile(cur_imp[1], self.perc)
        sha_max_history.append(imp_sha_max)
        imp_history = np.vstack((imp_history, cur_imp[0]))
        hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)
        dec_reg = self._do_tests(dec_reg, hit_reg, _iter)
        return dec_reg, sha_max_history, imp_history, hit_reg, imp_sha_max

    def select_features(self, X, y, sample_weight=None):
        """
        Select features using the Leshy algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        dec_reg : ndarray of shape (n_features,)
            The decision rule. 1 means the feature is selected, 0 means the feature is not selected.
        sha_max_history : list
            List of the maximum shadow importances per iteration.
        imp_history : ndarray of shape (n_iterations, n_features)
            Array containing the feature importances per iteration.
        imp_sha_max : float
            Maximum shadow importance value.
        """
        pbar = tqdm(total=self.max_iter, desc="Leshy iteration")
        dec_reg = np.zeros(X.shape[1])
        hit_reg = np.zeros(X.shape[1])
        sha_max_history = []
        imp_history = np.empty((0, X.shape[1]))

        for _iter in range(1, self.max_iter):
            if not np.any(dec_reg == 0):
                break
            self._update_tree_num(dec_reg)
            self._update_estimator()
            (
                dec_reg,
                sha_max_history,
                imp_history,
                hit_reg,
                imp_sha_max,
            ) = self._run_iteration(
                X,
                y,
                sample_weight,
                dec_reg,
                sha_max_history,
                imp_history,
                hit_reg,
                _iter,
            )
            _iter += 1
            pbar.update(1)

        pbar.close()
        return dec_reg, sha_max_history, imp_history, imp_sha_max

    def _calculate_support(self, confirmed, tentative, n_feat):
        """
        Calculate the feature support arrays.

        Parameters
        ----------
        confirmed : array-like of shape (n_confirmed,)
            Indices of confirmed features.
        tentative : array-like of shape (n_tentative,)
            Indices of tentative features.
        n_feat : int
            Total number of features.

        Returns
        -------
        None
            The function populates the following class attributes:
            - n_features_ : int
                Number of selected features.
            - support_ : ndarray of shape (n_feat,)
                Boolean array indicating the selected features.
            - support_weak_ : ndarray of shape (n_feat,)
                Boolean array indicating the tentatively selected features.
        """
        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=bool)
        self.support_weak_ = np.zeros(n_feat, dtype=bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=bool)
        self.support_weak_[tentative] = 1
        if self.keep_weak:
            self.support_[tentative] = 1

    def _calculate_absolute_ranking(self):
        """
        Compute feature importance scores using SHAP values.

        Parameters
        ----------
        new_x_tr : numpy.ndarray
            The training dataset after being processed.
        shap_matrix : numpy.ndarray
            The matrix containing SHAP values computed by a LightGBM model.
        param : dict
            A dictionary containing the parameters for a LightGBM model.
        objective : str
            The objective function of the LightGBM model.

        Returns
        -------
        list
            A list of tuples containing feature names and their corresponding importance scores.
        """
        vimp_df = pd.DataFrame(self.imp_real_hist, columns=self.feature_names_in_)
        self.ranking_absolutes_ = list(
            vimp_df.mean().sort_values(ascending=False).index
        )

    def _calculate_relative_ranking(self, n_feat, tentative, confirmed, imp_history):
        """
        Calculates the relative ranking of features based on their importance history.

        Parameters
        ----------
        n_feat : int
            The total number of features.
        tentative : ndarray of shape (n_tentative_features,)
            An array containing the indices of tentative features.
        confirmed : ndarray of shape (n_confirmed_features,)
            An array containing the indices of confirmed features.
        imp_history : ndarray of shape (n_iterations + 1, n_features)
            An array containing the feature importances for each iteration.

        Returns
        -------
        None

        """
        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)

            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=bool)

    def _print_result(self, dec_reg, _iter, start_time):
        """
        Print the results of feature selection.

        Parameters
        ----------
        dec_reg : bool
            Decision on whether to proceed with another round of feature selection.
        _iter : int
            Current iteration number.
        start_time : float
            Time when the feature selection process started.

        Returns
        -------
        None
            The function prints the relevant results and running time.
        """
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        self.running_time = time.time() - start_time
        hours, rem = divmod(self.running_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "All relevant predictors selected in {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )


def _get_confirmed_and_tentative(dec_reg):
    """Extracts the confirmed and tentative features from dec_reg."""
    confirmed = np.where(dec_reg == 1)[0]
    tentative = np.where(dec_reg == 0)[0]
    return confirmed, tentative


def _select_tentative(tentative, imp_history, sha_max_history):
    """
    Select tentative features based on median importance values.

    Parameters
    ----------
    tentative: array-like of shape (n_tentative,)
        Array of indices representing tentative features.
    imp_history: array-like of shape (n_iterations + 1, n_features)
        Importance values for each feature in each iteration.
    sha_max_history: array-like of shape (n_iterations + 1,)
        The history of the highest stability scores.

    Returns
    -------
    tentative: array-like of shape (n_tentative_confirmed,)
        The confirmed tentative features based on their median importance values.
    """
    # ignore the first row of zeros
    tentative_median = np.median(imp_history[1:, tentative], axis=0)
    # which tentative to keep
    tentative_confirmed = np.where(tentative_median > np.median(sha_max_history))[0]
    tentative = tentative[tentative_confirmed]
    return tentative


def _split_fit_estimator(estimator, X, y, sample_weight=None, cat_feature=None):
    """Private function, split the train, test and fit the model

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    y : pd.series of shape [n_samples]
        target
    sample_weight : array-like, shape = [n_samples], default=None
        Individual weights for each sample
    cat_feature : list of int or None
        the list of integers, cols loc, of the categrocial predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.
    Returns
    -------
     model :
        fitted model
     X_tt : array [n_samples, n_features]
        the test split, predictors
     y_tt : array [n_samples]
        the test split, target
    """
    if cat_feature is None:
        # detect, store and encode categorical predictors
        X, _, cat_idx = get_pandas_cat_codes(X)
    else:
        cat_idx = cat_feature

    if sample_weight is not None:
        w = sample_weight
        if is_regressor(estimator):
            X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(
                X, y, w, random_state=42
            )
        else:
            X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(
                X, y, w, stratify=y, random_state=42
            )
    else:
        if is_regressor(estimator):
            X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, random_state=42)
        else:
            X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, stratify=y, random_state=42)
        w_tr, w_tt = None, None

    if check_if_tree_based(estimator):
        try:
            # handle cat features if supported by the fit method
            if is_catboost(estimator) or (
                "cat_feature" in estimator.fit.__code__.co_varnames
            ):
                model = estimator.fit(
                    X_tr, y_tr, sample_weight=w_tr, cat_features=cat_idx
                )
            else:
                model = estimator.fit(X_tr, y_tr, sample_weight=w_tr)

        except Exception as e:
            raise ValueError(
                "Please check your X and y variable. The provided "
                "estimator cannot be fitted to your data.\n" + str(e)
            )
    else:
        raise ValueError("Not a tree based model")

    return model, X_tt, y_tt, w_tt


def _get_shap_imp(estimator, X, y, sample_weight=None, cat_feature=None):
    """Get the SHAP feature importance

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and `predict` methods.
    X : pd.DataFrame of shape [n_samples, n_features]
        Predictor matrix.
    y : pd.Series of shape [n_samples]
        Target variable.
    sample_weight : array-like, shape = [n_samples], default=None
        Individual weights for each sample.
    cat_feature : list of int or None, default=None
        The list of integers, columns loc, of the categorical predictors. Avoids detecting and encoding
        each iteration if the exact same columns are passed to the selection methods.

    Returns
    -------
    shap_imp : array
        The SHAP importance array.
    """
    # Clone the estimator to avoid modifying the original one
    estimator = clone(estimator)

    # Split the data into train and test sets and fit the model
    model, X_tt, y_tt, w_tt = _split_fit_estimator(
        estimator, X, y, sample_weight=sample_weight, cat_feature=cat_feature
    )
    # Compute the SHAP values
    if is_lightgbm(estimator):
        # For LightGBM models use the built-in SHAP method
        shap_matrix = model.predict(X_tt, pred_contrib=True)

        # The shape of the shap_matrix depends on whether the estimator is a classifier or a regressor
        if is_classifier(estimator) and (len(np.unique(y_tt)) > 2):
            # For multi-class classifiers, reshape the shap_matrix
            n_features = X_tt.shape[1]
            shap_matrix = np.delete(
                shap_matrix,
                list(range(n_features, shap_matrix.shape[1], n_features + 1)),
                axis=1,
            )
            shap_imp = np.mean(np.abs(shap_matrix), axis=0)
        else:
            shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
    else:
        # For other tree-based models, use the shap.TreeExplainer method to compute SHAP values
        explainer = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        )
        shap_values = explainer.shap_values(X_tt)
        # For multi-class classifiers, reshape the shap_values
        if is_classifier(estimator):
            if isinstance(shap_values, list):
                # For LightGBM classifier in sklearn API, SHAP returns a list of arrays
                # https://github.com/slundberg/shap/issues/526
                shap_imp = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
            else:
                shap_imp = np.abs(shap_values).mean(0)
        else:
            shap_imp = np.abs(shap_values).mean(0)
    return shap_imp


def _get_shap_imp_fast(estimator, X, y, sample_weight=None, cat_feature=None):
    """Get the SHAP feature importance using the fasttreeshap implementation

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and `predict` methods.
    X : pd.DataFrame of shape [n_samples, n_features]
        Predictor matrix.
    y : pd.Series of shape [n_samples]
        Target variable.
    sample_weight : array-like, shape = [n_samples], default=None
        Individual weights for each sample.
    cat_feature : list of int or None, default=None
        The list of integers, columns loc, of the categorical predictors. Avoids detecting and encoding
        each iteration if the exact same columns are passed to the selection methods.

    Returns
    -------
    shap_imp : array
        The SHAP importance array.
    """
    try:
        from fasttreeshap import TreeExplainer as FastTreeExplainer
    except ImportError:
        ImportError("fasttreeshap is not installed")

    # Clone the estimator to avoid modifying the original one
    estimator = clone(estimator)

    # Split the data into train and test sets and fit the model
    model, X_tt, y_tt, w_tt = _split_fit_estimator(
        estimator, X, y, sample_weight=sample_weight, cat_feature=cat_feature
    )
    explainer = FastTreeExplainer(
        model,
        algorithm="auto",
        shortcut=False,
        feature_perturbation="tree_path_dependent",
    )
    shap_matrix = explainer.shap_values(X_tt)
    # multiclass returns a list
    # for binary and for some models, shap is still returning a list
    if is_classifier(estimator):
        if isinstance(shap_matrix, list):
            # For LightGBM classifier, RF, in sklearn API, SHAP returns a list of arrays
            # https://github.com/slundberg/shap/issues/526
            shap_imp = np.mean([np.abs(sv).mean(0) for sv in shap_matrix], axis=0)
        else:
            shap_imp = np.abs(shap_matrix).mean(0)
    else:
        shap_imp = np.abs(shap_matrix).mean(0)
    return shap_imp


def _get_perm_imp(estimator, X, y, sample_weight, cat_feature=None):
    """Private function, Get the SHAP feature importance

    Parameters
    ----------
    estimator: sklearn estimator
    X : pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    y : pd.series of shape [n_samples]
        target
    sample_weight : array-like, shape = [n_samples], default=None
        Individual weights for each sample
    cat_feature : list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.
    Returns
    -------
    imp : array
        the permutation importance array
    """
    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    model, X_tt, y_tt, w_tt = _split_fit_estimator(
        estimator, X, y, sample_weight=sample_weight, cat_feature=cat_feature
    )
    perm_imp = permutation_importance(
        model, X_tt, y_tt, n_repeats=5, random_state=42, n_jobs=-1
    )
    imp = perm_imp.importances_mean.ravel()
    return imp


def _get_imp(estimator, X, y, sample_weight=None, cat_feature=None):
    """Private function, Get the native feature importance (impurity based for instance)

    Notes
    -----
    This is know to return biased and uninformative results.
    e.g.
    https://scikit-learn.org/stable/auto_examples/inspection/
    plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

    or

    https://explained.ai/rf-importance/

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        The training input samples.
    y : array-like, shape = [n_samples]
        The target values.
    sample_weight : array-like, shape = [n_samples], default=None
        Individual weights for each sample
    cat_feature: list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.
    Returns
    -------
    imp : array
        the permutation importance array
    """
    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    try:
        if cat_feature is None:
            X, _, cat_idx = get_pandas_cat_codes(X)
        else:
            cat_idx = cat_feature

        # handle catboost and cat features
        if is_catboost(estimator) or (
            "cat_feature" in estimator.fit.__code__.co_varnames
        ):
            X = pd.DataFrame(X)
            estimator.fit(X, y, sample_weight=sample_weight, cat_features=cat_idx)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)

    except Exception as e:
        raise ValueError(
            "Please check your X and y variable. The provided "
            "estimator cannot be fitted to your data.\n" + str(e)
        )
    try:
        imp = estimator.feature_importances_
    except Exception:
        raise ValueError(
            "Only methods with feature_importance_ attribute "
            "are currently supported in BorutaPy."
        )
    return imp


###################################
#
# BoostAGroota
#
###################################
class BoostAGroota(SelectorMixin, BaseEstimator):  # (object):
    """
    BoostAGroota is an all-relevant feature selection method, while most others are minimal optimal.
    It tries to find all features carrying information usable for prediction, rather than finding a possibly compact
    subset of features on which some estimator has a minimal error.

    Why bother with all-relevant feature selection?
    When you try to understand the phenomenon that made your data, you should
    care about all factors that contribute to it, not just the bluntest signs
    of it in the context of your methodology (minimal optimal set of features
    by definition depends on your estimator choice).

    Parameters
    ----------
    estimator : scikit-learn estimator
        The model to train, lightGBM recommended, see the reduce lightgbm method.
    cutoff : float
        The value by which the max of shadow imp is divided, to compare to real importance.
    iters : int (>0)
        The number of iterations to average for the feature importance (on the same split),
        to reduce the variance.
    max_rounds : int (>0)
        The number of times the core BoostAGroota algorithm will run.
        Each round eliminates more and more features.
    delta : float (0 < delta <= 1)
        Stopping criteria for whether another round is started.
    silent : bool
        Set to True if you don't want to see the BoostAGroota output printed.
    importance : str, default='shap'
        The kind of feature importance to use. Possible values: 'shap' (Shapley values),
        'pimp' (permutation importance), and 'native' (Gini/impurity).

    Attributes
    ----------
    selected_features_ : list of str
        The list of columns to keep.
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1, and tentative features are assigned
        rank 2.
    ranking_absolutes_ : array of shape [n_features]
        The absolute feature ranking as ordered by the selection process. It does not guarantee
        that this order is correct for all models. For a model-agnostic ranking, see the
        attribute ``ranking``.
    sha_cutoff_df : dataframe
        Feature importance of the real+shadow predictors over iterations.
    mean_shadow : float
        The threshold below which the predictors are rejected.

    Examples
    --------
    >>> X = df[filtered_features].copy()
    >>> y = df['target'].copy()
    >>> w = df['weight'].copy()
    >>> model = LGBMRegressor(n_jobs=-1, n_estimators=100, objective='rmse', random_state=42, verbose=0)
    >>> feat_selector = BoostAGroota(estimator=model, cutoff=1, iters=10, max_rounds=10, delta=0.1, importance='shap')
    >>> feat_selector.fit(X, y, sample_weight=None)
    >>> print(feat_selector.selected_features_)
    >>> feat_selector.plot_importance(n_feat_per_inch=5)
    """

    def __init__(
        self,
        estimator=None,
        cutoff=4,
        iters=10,
        max_rounds=500,
        delta=0.1,
        silent=True,
        importance="shap",
    ):
        self.estimator = estimator
        self.cutoff = cutoff
        self.iters = iters
        self.max_rounds = max_rounds
        self.delta = delta
        self.silent = silent
        self.importance = importance
        self.sha_cutoff_df = None
        self.mean_shadow = None
        self.ranking_absolutes_ = None
        self.ranking_ = None

        # Throw errors if the inputted parameters don't meet the necessary criteria
        if cutoff <= 0:
            raise ValueError(
                "cutoff should be greater than 0. You entered" + str(cutoff)
            )
        if iters <= 0:
            raise ValueError("iters should be greater than 0. You entered" + str(iters))
        if (delta <= 0) | (delta > 1):
            raise ValueError("delta should be between 0 and 1, was " + str(delta))

        # Issue warnings for parameters to still let it run
        if delta < 0.02:
            warnings.warn(
                "WARNING: Setting a delta below 0.02 may not converge on a solution."
            )
        if max_rounds < 1:
            warnings.warn(
                "WARNING: Setting max_rounds below 1 will automatically be set to 1."
            )

        if importance == "native":
            warnings.warn(
                "[BoostAGroota]: using native variable importance might break the FS"
            )

    def __repr__(self):
        s = (
            "BoostARoota(est={est}, \n"
            "                cutoff={cutoff},\n"
            "                iters={iters},\n"
            "                max_rounds={mr},\n"
            "                delta={delta},\n"
            "                silent={silent}, \n"
            '                importance="{importance}")'.format(
                estimator=self.estimator,
                cutoff=self.cutoff,
                iters=self.iters,
                mr=self.max_rounds,
                delta=self.delta,
                silent=self.silent,
                importance=self.importance,
            )
        )
        return s

    def fit(self, X, y, sample_weight=None):
        """Fit the BoostAGroota transformer with the provided estimator.
        Parameters
        ----------
        X : pd.DataFrame
            the predictors matrix
        y : pd.Series
            the target
        sample_weight : pd.series
            sample_weight, if any
        """
        if self.importance == "fastshap":
            try:
                from fasttreeshap import TreeExplainer as FastTreeExplainer
            except ImportError:
                warnings.warn("fasttreeshap is not installed. Fallback to shap.")
                self.importance = "shap"

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            raise TypeError("X is not a dataframe")

        if sample_weight is not None:
            sample_weight = pd.Series(_check_sample_weight(sample_weight, X))

        # crit, keep_vars, df_vimp, mean_shadow
        _, self.selected_features_, self.sha_cutoff_df, self.mean_shadow = _boostaroota(
            X,
            y,
            # metric=self.metric,
            estimator=self.estimator,
            cutoff=self.cutoff,
            iters=self.iters,
            max_rounds=self.max_rounds,
            delta=self.delta,
            silent=self.silent,
            weight=sample_weight,
            imp=self.importance,
        )
        self.selected_features_ = self.selected_features_.values
        self.support_ = np.asarray(
            [c in self.selected_features_ for c in self.feature_names_in_]
        )
        self.ranking_absolutes_ = list(
            self.sha_cutoff_df.iloc[:, : int(self.sha_cutoff_df.shape[1] / 2)]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        self.ranking_ = np.where(self.support_, 2, 1)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.support_

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        return X[self.selected_features_]

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}

    @mpl.rc_context(PLT_PARAMS)
    def plot_importance(self, n_feat_per_inch=5):
        """
        Boxplot of the variable importance, ordered by magnitude.
        The max shadow variable importance illustrated by the dashed line.
        Requires to apply the fit method first.

        Parameters
        ----------
        n_feat_per_inch : int, default=5
            Number of features to plot per inch (for scaling the figure).

        Returns
        -------
        fig : plt.figure or None
            The matplotlib figure object containing the boxplot, or None if there are no selected features.
        """
        if self.mean_shadow is None:
            raise ValueError("Apply fit method first")

        b_df = self.sha_cutoff_df
        real_df = b_df.iloc[:, : int(b_df.shape[1] / 2)].copy()

        real_df = real_df.reindex(
            real_df.mean().sort_values(ascending=True).index, axis=1
        )

        if real_df.dropna().empty:
            warnings.warn(NO_FEATURE_SELECTED_WARNINGS)
            return None

        fig, ax = plt.subplots(figsize=(16, real_df.shape[1] / n_feat_per_inch))
        bp = real_df.plot(**PLOT_KWARGS, ax=ax)
        bp.set_xlim(left=real_df.min(skipna=True).min(skipna=True) - 0.025)

        for c in range(len(self.selected_features_)):
            patch = bp.findobj(mpl.patches.Patch)[real_df.shape[1] - c - 1]
            patch.set_facecolor(BLUE)
            patch.set_color(BLUE)

        custom_lines = [
            Line2D([0], [0], color=BLUE, lw=5),
            Line2D([0], [0], color="gray", lw=5),
            Line2D([0], [0], linestyle="--", color=RED, lw=2),
        ]
        bp.legend(
            custom_lines, ["confirmed", "rejected", "sha. max"], loc="lower right"
        )

        ax.axvline(x=self.mean_shadow, linestyle="--", color=RED)
        ax.set_title("BoostAGroota importance of selected predictors")

        return fig


############################################
# Helper Functions to do the Heavy Lifting
############################################


def _create_shadow(X_train):
    """Create shadow features by making copies of all X variables and randomly shuffling them.

    Parameters
    ----------
    X_train : pd.DataFrame
        The dataframe to create shadow features on.

    Returns
    -------
    pd.DataFrame
        A dataframe that is twice the width of X_train and contains the shadow features, along with a list of the shadow feature names.
    """
    X_shadow = X_train.copy()
    for c in X_shadow.columns:
        np.random.shuffle(X_shadow[c].values)
    # Rename the shadow variables
    shadow_names = [f"ShadowVar{i+1}" for i in range(X_train.shape[1])]
    X_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_x = pd.concat([X_train, X_shadow], axis=1)
    return new_x, shadow_names


########################################################################################
# BoostARoota. In principle, you cannot/don't need to access those methods (reason of
# the _ in front of the function name, they're internal functions)
########################################################################################


def _reduce_vars_sklearn(
    X,
    y,
    estimator,
    this_round,
    cutoff,
    n_iterations,
    delta,
    silent,
    weight,
    imp_kind,
    cat_feature,
):
    """
    Private function, reduce the number of predictors using a sklearn estimator

    Parameters
    ----------
    x : pd.DataFrame
        the dataframe to create shadow features on
    y : pd.Series
        the target
    estimator : sklearn estimator
        the model to train, lightGBM recommended
    this_round : int
        The number of times the core BoostARoota algorithm will run.
        Each round eliminates more and more features
    cutoff : float
        the value by which the max of shadow imp is divided, to compare to real importance
    n_iterations : int
        The number of iterations to average for the feature importance (on the same split),
        to reduce the variance
    delta : float (0 < delta <= 1)
        Stopping criteria for whether another round is started
    silent : bool
        Set to True if don't want to see the BoostARoota output printed.
        Will still show any errors or warnings that may occur
    weight : pd.series
        sample_weight, if any
    imp_kind : str
        whether if native, shap, fastshap or permutation importance should be used
    cat_feature : list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.

    Returns
    -------
    criteria : bool
        if the criteria has been reached or not
    real_vars['feature'] : pd.dataframe
        feature importance of the real predictors over iter
    df : pd.DataFrame
        feature importance of the real+shadow predictors over iter
    mean_shadow : float
        the feature importance threshold, to reject or not the predictors

    Raises
    ------
    ValueError
        error if the feature importance type is not
    """
    # Set up the parameters for running the model in XGBoost - split is on multi log loss
    for i in range(1, n_iterations + 1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(X)
        imp_func = {
            "shap": _get_shap_imp,
            "fastshap": _get_shap_imp,
            "pimp": _get_perm_imp,
            "native": _get_imp,
        }
        importance = imp_func[imp_kind](
            estimator, new_x, y, sample_weight=weight, cat_feature=cat_feature
        )

        # Create a dataframe to store the feature importances
        if i == 1:
            df = pd.DataFrame({"feature": new_x.columns})
            df2 = df.copy()

        # Store the feature importances in df2
        try:
            # Normalize the feature importances
            df2["fscore" + str(i)] = importance / importance.sum()
        except ValueError:
            print("Only Sklearn tree based methods allowed")

        # Merge the current feature importances with the existing ones in df
        df = pd.merge(
            df, df2, on="feature", how="outer", suffixes=("_x" + str(i), "_y" + str(i))
        )

        # Print the iteration number if not silent
        if not silent:
            print("Round: ", this_round, " iteration: ", i)

    df["Mean"] = df.select_dtypes(include=[np.number]).mean(axis=1, skipna=True)
    # Split them back out
    real_vars = df.loc[~df["feature"].isin(shadow_names)]
    shadow_vars = df.loc[df["feature"].isin(shadow_names)]

    # Get mean value from the shadows (max, like in Boruta, median to mitigate variance)
    mean_shadow = (
        shadow_vars.select_dtypes(include=[np.number])
        .max(skipna=True)
        .mean(skipna=True)
        / cutoff
    )
    real_vars = real_vars[(real_vars.Mean >= mean_shadow)]

    # Check for the stopping criteria
    # Compute the fraction of features selected in this round
    selected_frac = len(real_vars) / len(X.columns)

    # Check if we should stop feature selection
    # make sure we are removing at least delta % of the variables
    criteria = len(X.columns) == 0 or selected_frac == 0 or selected_frac > 1 - delta

    return criteria, real_vars["feature"], df, mean_shadow


def _boostaroota(
    X, y, estimator, cutoff, iters, max_rounds, delta, silent, weight, imp
):
    """
    Private function, reduces the number of predictors using a sklearn estimator.

    Parameters
    ----------
    x : pd.DataFrame
        The dataframe to create shadow features on.
    y : pd.Series
        The target.
    estimator : scikit-learn estimator
        The model to train, lightGBM recommended, see the reduce lightgbm method.
    cutoff : float
        The value by which the max of shadow imp is divided, to compare to real importance.
    iters : int (>0)
        The number of iterations to average for the feature importances (on the same split),
        to reduce the variance.
    max_rounds : int (>0)
        The number of times the core BoostARoota algorithm will run.
        Each round eliminates more and more features.
    delta : float (0 < delta <= 1)
        Stopping criteria for whether another round is started.
    silent : bool
        Set to True if you don't want to see the BoostARoota output printed.
        Will still show any errors or warnings that may occur.
    weight : pd.Series, optional
        Sample weights, if any.

    Returns
    -------
    crit : bool
        If the criteria have been reached or not.
    keep_vars : pd.DataFrame
        Feature importance of the real predictors over iterations.
    df_vimp : pd.DataFrame
        Feature importance of the real+shadow predictors over iterations.
    mean_shadow : float
        The feature importance threshold to reject or not the predictors.
    """
    start_time = time.time()
    new_x = X.copy()

    # extract the categorical names for the first time, store it for next iterations
    # In the below while loop this list will be update only once some of the predictors
    # are removed. This way the encoding is done only every predictors update and not
    # every iteration. The code will then be much faster since the encoding is done only once.
    new_x, obj_feat, cat_idx = get_pandas_cat_codes(X)

    # Run through loop until "crit" changes
    i = 0
    imp_dic = {}
    with tqdm(total=max_rounds, desc="BoostaGRoota round") as pbar:
        while True:
            # Inside this loop we reduce the dataset on each iteration exiting with keep_vars
            i += 1
            crit, keep_vars, df_vimp, mean_shadow = _reduce_vars_sklearn(
                new_x,
                y,
                estimator=estimator,
                this_round=i,
                cutoff=cutoff,
                n_iterations=iters,
                delta=delta,
                silent=silent,
                weight=weight,
                imp_kind=imp,
                cat_feature=cat_idx,
            )

            b_df = df_vimp.T.iloc[1:-1].astype(float)
            b_df.columns = df_vimp.T.iloc[0].values
            imp_dic.update(b_df.to_dict())

            if crit or i >= max_rounds or len(keep_vars) == 0:
                break
            else:
                new_x = new_x[keep_vars].copy()
                _, _, cat_idx = get_pandas_cat_codes(new_x)

                pbar.update(1)

    elapsed = (time.time() - start_time) / 60
    if not silent:
        print(f"BoostARoota ran successfully! Algorithm went through {i} rounds.")
        print(f"The feature selection BoostARoota running time is {elapsed:8.2f} min")

    df_vimp = pd.DataFrame.from_dict(imp_dic)

    return crit, keep_vars, df_vimp, mean_shadow


###################################
#
# GrootCV
#
###################################


class GrootCV(SelectorMixin, BaseEstimator):
    """
    GrootCV is a feature selection method based on cross-validation with lightGBM.

    A shuffled copy of the predictors matrix is added (shadows) to the original set of predictors.
    The lightGBM is fitted using repeated cross-validation, the feature importance
    is extracted each time and averaged to smooth out the noise.
    If the feature importance is larger than the average shadow feature importance then the predictors are rejected, the others are kept.
        - Cross-validated feature importance to smooth out the noise, based on lightGBM only
          (which is, most of the time, the fastest and more accurate Boosting).
        - the feature importance is derived using SHAP importance
        - Taking the max of median of the shadow var. imp over folds otherwise not enough conservative and
          it improves the convergence (needs less evaluation to find a threshold)
        - Not based on a given percentage of cols needed to be deleted
        - Plot method for var. imp

    Parameters
    ----------
    objective : str or callable, default=None
        The objective function to use in lightGBM. If None, it uses the objective specified in `lgbm_params`.
    cutoff : float, default=1
        The value by which the max of shadow imp is divided, to compare to real importance.
    n_folds : int, default=5
        The number of folds for cross-validation.
    n_iter : int, default=5
        The number of iterations to average for the feature importance (on the same split), to reduce variance.
    silent : bool, default=True
        Set to True if you don't want to see the GrootCV output printed.
    rf : bool, default=False
        If True, use random forest for calculating feature importances; otherwise, use lightGBM.
    fastshap : bool, default=False
        If True, use fastSHAP for calculating feature importances; otherwise, use SHAP.
    n_jobs : int, default=0
        The number of jobs to run in parallel. If 0, no parallelism is used.
    lgbm_params : dict, default=None
        The parameters for the lightGBM model.

    Attributes
    ----------
    selected_features_ : ndarray
        The list of columns to keep as selected features.
    cv_df : pd.DataFrame
        DataFrame containing feature importance values for each fold and iteration.
    sha_cutoff : float
        The threshold below which the predictors are rejected.
    ranking_absolutes_ : list
        The absolute feature ranking as ordered by the selection process.
    ranking_ : ndarray
        The feature ranking, where 2 corresponds to selected features and 1 to tentative features.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the GrootCV on the input data.
    transform(X)
        Apply the fitted GrootCV on new data.
    plot_importance(n_feat_per_inch=5)
        Plot the feature importance of the fitted GrootCV.

    Warnings
    --------
    If `sha_cutoff` is None, you should apply the fit method first.
    Examples
    -------
    >>> X = df[filtered_features].copy()
    >>> y = df['target'].copy()
    >>> w = df['weight'].copy()
    >>> feat_selector = arfsgroot.GrootCV(objective='rmse', cutoff = 1, n_folds=5, n_iter=5)
    >>> feat_selector.fit(X, y, sample_weight=None)
    >>> feat_selector.plot_importance(n_feat_per_inch=5)
    """

    def __init__(
        self,
        objective=None,
        cutoff=1,
        n_folds=5,
        n_iter=5,
        silent=True,
        rf=False,
        fastshap=False,
        n_jobs=0,
        lgbm_params=None,
    ):
        self.objective = objective
        self.cutoff = cutoff
        self.n_folds = n_folds
        self.n_iter = n_iter
        self.silent = silent
        self.rf = rf
        self.fastshap = fastshap
        self.cv_df = None
        self.sha_cutoff = None
        self.ranking_absolutes_ = None
        self.ranking_ = None
        self.lgbm_params = lgbm_params
        self.n_jobs = n_jobs

        # Throw errors if the inputted parameters don't meet the necessary criteria
        # Ensure parameters meet necessary criteria
        if cutoff <= 0 or n_iter <= 0 or n_folds <= 0:
            raise ValueError("cutoff, n_iter, and n_folds should be greater than 0.")

    def fit(self, X, y, sample_weight=None):
        """
        Fit the GrootCV on the input data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The predictor dataframe.
        y : array-like of shape (n_samples,)
            The target vector.
        sample_weight : array-like of shape (n_samples,), optional
            The weight vector, by default None.

        Returns
        -------
        self : object
            Returns self.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a pandas DataFrame")

        self.feature_names_in_ = X.columns.to_numpy()
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        if sample_weight is not None:
            sample_weight = pd.Series(_check_sample_weight(sample_weight, X))

        # internal encoding (ordinal encoding)
        X, obj_feat, cat_idx = get_pandas_cat_codes(X)

        self.selected_features_, self.cv_df, self.sha_cutoff = _reduce_vars_lgb_cv(
            X,
            y,
            objective=self.objective,
            cutoff=self.cutoff,
            n_folds=self.n_folds,
            n_iter=self.n_iter,
            silent=self.silent,
            weight=sample_weight,
            rf=self.rf,
            fastshap=self.fastshap,
            lgbm_params=self.lgbm_params,
            n_jobs=self.n_jobs,
        )

        self.selected_features_ = self.selected_features_.values
        self.support_ = np.asarray(
            [c in self.selected_features_ for c in self.feature_names_in_]
        )
        self.ranking_ = np.where(self.support_, 2, 1)

        b_df = self.cv_df.T.copy()
        b_df.columns = b_df.iloc[0]
        b_df = b_df.drop(b_df.index[0])
        b_df = b_df.drop(b_df.index[-1])
        b_df = b_df.convert_dtypes()
        real_df = b_df.iloc[:, : int(b_df.shape[1] / 2)].copy()
        self.ranking_absolutes_ = list(
            real_df.mean().sort_values(ascending=False).index
        )
        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.support_

    def transform(self, X):
        """
        Apply the fitted GrootCV on new data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The predictor dataframe.

        Returns
        -------
        X_selected : pd.DataFrame of shape (n_samples, n_selected_features)
            The selected features from the input dataframe.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        return X[self.selected_features_]

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}

    @mpl.rc_context(PLT_PARAMS)
    def plot_importance(self, n_feat_per_inch=5):
        """
        Plot the feature importance of the fitted GrootCV.

        Parameters
        ----------
        n_feat_per_inch : int, default=5
            The number of features per inch in the plot.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The matplotlib figure containing the plot or None if no feature is selected.
        """

        if self.sha_cutoff is None:
            raise ValueError("Apply fit method first")

        b_df = self.cv_df.T.copy()
        b_df.columns = b_df.iloc[0]
        b_df = b_df.drop(b_df.index[0])
        b_df = b_df.drop(b_df.index[-1])
        b_df = b_df.convert_dtypes()
        real_df = b_df.iloc[:, : int(b_df.shape[1] / 2)].copy()
        sha_df = b_df.iloc[:, int(b_df.shape[1] / 2) :].copy()

        real_df = real_df.reindex(
            real_df.select_dtypes(include=[np.number])
            .mean()
            .sort_values(ascending=True)
            .index,
            axis=1,
        )

        if real_df.dropna().empty:
            warnings.warn(NO_FEATURE_SELECTED_WARNINGS)
            return None
        else:
            fig, ax = plt.subplots(figsize=(16, real_df.shape[1] / n_feat_per_inch))
            bp = real_df.plot(**PLOT_KWARGS, ax=ax)
            col_idx = np.argwhere(real_df.columns.isin(self.selected_features_)).ravel()

            for c in range(real_df.shape[1]):
                bp.findobj(mpl.patches.Patch)[c].set_facecolor("gray")
                bp.findobj(mpl.patches.Patch)[c].set_color("gray")

            for c in col_idx:
                bp.findobj(mpl.patches.Patch)[c].set_facecolor(BLUE)
                bp.findobj(mpl.patches.Patch)[c].set_color(BLUE)

            ax.axvline(x=self.sha_cutoff, linestyle="--", color=RED)
            bp.set_xlim(left=real_df.min(skipna=True).min(skipna=True) - 0.025)
            custom_lines = [
                Line2D([0], [0], color=BLUE, lw=5),
                Line2D([0], [0], color="gray", lw=5),
                Line2D([0], [0], linestyle="--", color=RED, lw=2),
            ]
            bp.legend(
                custom_lines, ["confirmed", "rejected", "threshold"], loc="lower right"
            )
            ax.set_title("Groot CV importance and selected predictors")

            return fig


########################################################################################
#
# GrootCV. In principle, you cannot/don't need to access those methods (reason of
# the _ in front of the function name, they're internal functions)
#
########################################################################################


def _reduce_vars_lgb_cv(
    X,
    y,
    objective,
    n_folds,
    cutoff,
    n_iter,
    silent,
    weight,
    rf,
    fastshap,
    lgbm_params=None,
    n_jobs=0,
):
    """
    Reduce the number of predictors using a lightgbm (python API)

    Parameters
    ----------
    X : pd.DataFrame
            the dataframe to create shadow features on
    y : pd.Series
            the target
    objective : str
            the lightGBM objective
    cutoff : float
            the value by which the max of shadow imp is divided, to compare to real importance
    n_iter : int
            The number of repetition of the cross-validation, smooth out the feature importance noise
    silent : bool
            Set to True if don't want to see the BoostARoota output printed.
            Will still show any errors or warnings that may occur
    weight : pd.series
            sample_weight, if any
    rf : bool, default=False
            the lightGBM implementation of the random forest
    fastshap : bool
            enable or not the fasttreeshap implementation
    lgbm_params : dict, optional
            dictionary of lightgbm parameters
    n_jobs: int, default 0
        0 means default number of threads in OpenMP
        for the best speed, set this to the number of real CPU cores, not the number of threads

    Returns
    -------
    real_vars['feature'] : pd.dataframe
            feature importance of the real predictors over iter
    df : pd.DataFrame
            feature importance of the real+shadow predictors over iter
    cutoff_shadow : float
            the feature importance threshold, to reject or not the predictors
    """

    params = _set_lgb_parameters(
        X=X,
        y=y,
        objective=objective,
        rf=rf,
        silent=silent,
        n_jobs=n_jobs,
        lgbm_params=lgbm_params,
    )

    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")
    category_cols = dtypes_dic["cat"] + dtypes_dic["time"] + dtypes_dic["unk"]
    cat_idx = [X.columns.get_loc(col) for col in category_cols]

    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_iter, random_state=2652124)
    iter = 0
    df = pd.DataFrame({"feature": X.columns})
    for tridx, validx in tqdm(
        rkf.split(X, y), total=rkf.get_n_splits(), desc="Repeated k-fold"
    ):
        X_train, X_val, y_train, y_val, weight_tr, weight_val = _split_data(
            X, y, tridx, validx, weight
        )

        # Create the shadow variables and run the model to obtain importances
        new_x_tr, shadow_names = _create_shadow(X_train)
        new_x_val, _ = _create_shadow(X_val)

        bst, shap_matrix, bst.best_iteration = _train_lgb_model(
            new_x_tr,
            y_train,
            weight_tr,
            new_x_val,
            y_val,
            weight_val,
            category_cols=category_cols,
            early_stopping_rounds=20,
            fastshap=fastshap,
            **params,
        )

        importance = _compute_importance(
            new_x_tr, shap_matrix, params, objective, fastshap
        )
        df = _merge_importance_df(
            df=df,
            importance=importance,
            iter=iter,
            n_folds=n_folds,
            column_names=new_x_tr.columns,
            silent=silent,
        )
        iter += 1

    df["Med"] = df.select_dtypes(include=[np.number]).mean(axis=1)
    # Split them back out
    real_vars = df[~df["feature"].isin(shadow_names)]
    shadow_vars = df[df["feature"].isin(shadow_names)]

    # Get median value from the shadows, comparing predictor by predictor. Not the same criteria
    # max().max() like in Boruta but max of the median to mitigate.
    # Otherwise too conservative (reject too often)
    cutoff_shadow = shadow_vars.select_dtypes(include=[np.number]).max().mean() / cutoff
    real_vars = real_vars[(real_vars.Med.values >= cutoff_shadow)]

    return real_vars["feature"], df, cutoff_shadow


def _set_lgb_parameters(
    X: np.ndarray,
    y: np.ndarray,
    objective: str,
    rf: bool,
    silent: bool,
    n_jobs: int = 0,
    lgbm_params: dict = None,
) -> dict:
    """Set parameters for a LightGBM model based on the input features and the objective.

    Parameters
    ----------
    X : numpy array or pandas DataFrame
        The feature matrix of the training data.
    y : numpy array or pandas Series
        The target variable of the training data.
    objective : str
        The objective function to optimize during training.
    rf : bool, default False
        Whether to use random forest boosting.
    silent : bool, default True
        Whether to print messages during parameter setting.
    n_jobs: int, default 0
        0 means default number of threads in OpenMP
        for the best speed, set this to the number of real CPU cores, not the number of threads

    Returns
    -------
    dict
        The dictionary of LightGBM parameters.

    """

    n_feat = X.shape[1]

    params = lgbm_params if lgbm_params is not None else {}

    params["objective"] = objective
    params["verbosity"] = -1
    if objective == "softmax":
        params["num_class"] = len(np.unique(y))

    if rf:
        feat_frac = (
            np.sqrt(n_feat) / n_feat
            if objective in ["softmax", "binary"]
            else n_feat / (3 * n_feat)
        )
        params.update(
            {
                "boosting_type": "rf",
                "bagging_fraction": 0.7,
                "feature_fraction": feat_frac,
                "bagging_freq": 1,
            }
        )

    clf_losses = [
        "binary",
        "softmax",
        "multi_logloss",
        "multiclassova",
        "multiclass",
        "multiclass_ova",
        "ova",
        "ovr",
        "binary_logloss",
    ]
    if objective in clf_losses:
        y = y.astype(int)
        y_freq_table = pd.Series(y.fillna(0)).value_counts(normalize=True)
        n_classes = y_freq_table.size
        if n_classes > 2 and objective != "softmax":
            params["objective"] = "softmax"
            params["num_class"] = len(np.unique(y))
            if not silent:
                print("Multi-class task, setting objective to softmax")
        main_class = y_freq_table[0]
        if not silent:
            print("GrootCV: classification with unbalance classes")
        if main_class > 0.8:
            params.update({"is_unbalance": True})

    params.update({"num_threads": n_jobs})

    # we are using early_stopping
    # we prevent the overridding of it by popping the n_iterations
    keys_to_pop = [
        "num_iterations",
        "num_iteration",
        "n_iter",
        "num_tree",
        "num_trees",
        "num_round",
        "num_rounds",
        "nrounds",
        "num_boost_round",
        "n_estimators",
        "max_iter",
    ]
    for key in keys_to_pop:
        params.pop(key, None)

    return params


def _split_data(X, y, tridx, validx, weight=None):
    """
    Split data into train and validation sets based on provided indices.

    Parameters
    ----------
    X : pandas.DataFrame
        Features.
    y : pandas.Series
        Target variable.
    tridx : list
        Indices to be used for training.
    validx : list
        Indices to be used for validation.
    weight : pandas.Series, optional
        Weights for each sample, by default None.

    Returns
    -------
    tuple of pandas.DataFrame and pandas.Series
        X_train, X_val, y_train, y_val, weight_tr, weight_val
    """
    if weight is not None:
        X_train, y_train, weight_tr = (
            X.iloc[tridx, :],
            y.iloc[tridx],
            weight.iloc[tridx],
        )
        X_val, y_val, weight_val = (
            X.iloc[validx, :],
            y.iloc[validx],
            weight.iloc[validx],
        )
    else:
        X_train, y_train = X.iloc[tridx, :], y.iloc[tridx]
        X_val, y_val = X.iloc[validx, :], y.iloc[validx]
        weight_tr = None
        weight_val = None
    return X_train, X_val, y_train, y_val, weight_tr, weight_val


def _train_lgb_model(
    X_train,
    y_train,
    weight_train,
    X_val,
    y_val,
    weight_val,
    category_cols=None,
    early_stopping_rounds=20,
    fastshap=True,
    **params,
):
    """
    Train a LightGBM model with the given training data and hyperparameters and return the trained model and its SHAP values.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        The input training data.
    y_train : array-like of shape (n_samples,)
        The target training data.
    weight_train : array-like of shape (n_samples,)
        The sample weights for training data.
    X_val : array-like of shape (n_val_samples, n_features)
        The input validation data.
    y_val : array-like of shape (n_val_samples,)
        The target validation data.
    weight_val : array-like of shape (n_val_samples,)
        The sample weights for validation data.
    category_cols : array-like or None, optional (default=None)
        The indices of categorical columns. If None, no categorical columns will be considered.
    early_stopping_rounds : int, optional (default=20)
        Activates early stopping. Validation metric needs to improve at least once in every early_stopping_rounds
        round(s) to continue training._train_lgb_model
    fastshap : bool
        enable or not fasttreeshap implementation
    **params : dict
        Other parameters passed to the LightGBM model.

    Returns
    -------
    tuple of (Booster, numpy.ndarray, int)
        The trained LightGBM model, its SHAP values for X_train, and the best iteration reached during training.
    """

    d_train = lgb.Dataset(
        X_train, label=y_train, weight=weight_train, categorical_feature=category_cols
    )
    d_valid = lgb.Dataset(
        X_val, label=y_val, weight=weight_val, categorical_feature=category_cols
    )
    watchlist = [d_train, d_valid]

    bst = lgb.train(
        params,
        train_set=d_train,
        num_boost_round=10000,
        valid_sets=watchlist,
        categorical_feature=category_cols,
        callbacks=[early_stopping(early_stopping_rounds, False, False)],
    )

    if fastshap:
        try:
            from fasttreeshap import TreeExplainer as FastTreeExplainer
        except ImportError:
            raise ImportError(
                "fasttreeshap is not installed. Please install it using 'pip/conda install fasttreeshap'."
            )

        explainer = FastTreeExplainer(
            bst,
            algorithm="auto",
            shortcut=False,
            feature_perturbation="tree_path_dependent",
        )
        shap_matrix = explainer.shap_values(X_train)
    else:
        shap_matrix = bst.predict(X_train, pred_contrib=True)

    return bst, shap_matrix, bst.best_iteration


def _compute_importance(new_x_tr, shap_matrix, param, objective, fastshap):
    """Compute feature importance scores using SHAP values.

    Parameters
    ----------
    new_x_tr : numpy.ndarray
        The training dataset after being processed.
    shap_matrix : numpy.ndarray
        The matrix containing SHAP values computed by a LightGBM model.
    param : dict
        A dictionary containing the parameters for a LightGBM model.
    objective : str
        The objective function of the LightGBM model.

    Returns
    -------
    list
        A list of tuples containing feature names and their corresponding importance scores.
    """
    if fastshap:
        if objective == "softmax":
            shap_matrix = np.abs(np.concatenate(shap_matrix, axis=1))
        shap_imp = np.mean(np.abs(shap_matrix), axis=0)
    else:
        if objective == "softmax":
            n_feat = new_x_tr.shape[1]
            shap_matrix = np.delete(
                shap_matrix,
                list(range(n_feat, (n_feat + 1) * param["num_class"], n_feat + 1)),
                axis=1,
            )
            shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
        else:
            shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
    importance = dict(zip(new_x_tr.columns, shap_imp))
    return sorted(importance.items(), key=operator.itemgetter(1))


def _merge_importance_df(df, importance, iter, n_folds, column_names, silent=True):
    """
    Merge the feature importance dataframe `df` with the importance
    information for the current iteration of a cross-validation loop.

    Parameters
    ----------
    df : pandas.DataFrame
        The current feature importance dataframe.
    importance : dict
        A dictionary with the feature importance information for
        the current iteration.
    i : int
        The index of the current iteration.
    n_folds : int
        The number of folds used in the cross-validation loop.
    silent : bool, optional
        If True, suppress output.

    Returns
    -------
    pandas.DataFrame
        The updated feature importance dataframe.
    """

    df2 = pd.DataFrame(importance, columns=["feature", "fscore" + str(iter)])
    df2["fscore" + str(iter)] = (
        df2["fscore" + str(iter)] / df2["fscore" + str(iter)].sum()
    )
    df = pd.merge(df, df2, on="feature", how="outer")
    nit = divmod(iter, n_folds)[0]
    nf = divmod(iter, n_folds)[1]
    if not silent:
        if nf == 0:
            print("Groot iteration: ", nit, " with " + str(n_folds) + " folds")
    return df
