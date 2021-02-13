"""
This module provides 3 different methods to perform 'all relevant feature selection'


Reference:
----------
NILSSON, Roland, PEÑA, José M., BJÖRKEGREN, Johan, et al.
Consistent feature selection for pattern recognition in polynomial time.
Journal of Machine Learning Research, 2007, vol. 8, no Mar, p. 589-612.

KURSA, Miron B., RUDNICKI, Witold R., et al.
Feature selection with the Boruta package.
J Stat Softw, 2010, vol. 36, no 11, p. 1-13.

https://github.com/chasedehan/BoostARoota

The module structure is the following:
---------------------------------------
- The ``Leshy`` class, a heavy re-work of ``BorutaPy`` class
  itself a modified version of Boruta, the pull request I submitted and still pending:
  https://github.com/scikit-learn-contrib/boruta_py/pull/77

- The ``BoostAGroota`` class, a modified version of BoostARoota, PR still to be submitted
  https://github.com/chasedehan/BoostARoota

- The ``GrootCV`` class for a new method for all relevant feature selection using a lgGBM model,
  cross-validated SHAP importances and shadowing.
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
# from tqdm.autonotebook import tqdm
from tqdm import tqdm
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator, is_regressor, is_classifier, clone
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import _check_sample_weight
from matplotlib.lines import Line2D

from arfs.utils import check_if_tree_based, is_lightgbm, is_catboost

########################################################################################
#
# Main Classes and Methods
# Provide a fit, transform and fit_transform method
#
########################################################################################
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
Modified by Thomas Bury, pull request:
https://github.com/scikit-learn-contrib/boruta_py/pull/77
Not going to be merge because of the dependencies introduced.
Those dependencies are critical to improve and speed up the All Relevant Feature Selection (arfs).
The native feature importance is biased and does not uncover the real impact of each feature (aka
"feature impact" in the literature). 

The reason is that scikit-contrib tries to be as "vanilla" as possible, but giving a large coverage
to biased methods is harmful to the community IMHO (native feature importance flaws are known for 
10 years or so). In the case where a small increase of complexity (lightGBM+SHAP) fixes known problem,
it's not too much a burden.

Leshy is actually a re-work of the PR I submitted.

License: BSD 3 clause
"""


class Leshy(BaseEstimator, TransformerMixin):
    """
    This is an improved version of BorutaPy which itself is an
    improved Python implementation of the Boruta R package.
    For chronological dev, see https://github.com/scikit-learn-contrib/boruta_py/pull/77

    Leshy vs BorutaPy:
    ------------------
    To summarize, this PR solves/enhances:
     - The categorical features (they are detected, encoded. The tree-based models are working
       better with integer encoding rather than with OHE, which leads to deep and unstable trees).
       If Catboost is used, then the cat.pred (if any) are set up
     - Work with Catboost sklearn API
     - Allow using sample_weight, for applications like Poisson regression or any requiring weights
     - 3 different feature importances: native, SHAP and permutation. Native being the least consistent
       (because of the imp. biased towards numerical and large cardinality categorical)
       but the fastest of the 3.
     - Using lightGBM as default speed up by an order of magnitude the running time
     - Visualization like in the R package

    BorutaPy vs Boruta R:
    ---------------------
    The improvements of this implementation include:
    - Faster run times:
        Thanks to scikit-learn's fast implementation of the ensemble methods.
    - Scikit-learn like interface:
        Use BorutaPy just like any other scikit learner: fit, fit_transform and
        transform are all implemented in a similar fashion.
    - Modularity:
        Any ensemble method could be used: random forest, extra trees
        classifier, even gradient boosted trees.
    - Two step correction:
        The original Boruta code corrects for multiple testing in an overly
        conservative way. In this implementation, the Benjamini Hochberg FDR is
        used to correct in each iteration across active features. This means
        only those features are included in the correction which are still in
        the selection process. Following this, each that passed goes through a
        regular Bonferroni correction to check for the repeated testing over
        the iterations.
    - Percentile:
        Instead of using the max values of the shadow features the user can
        specify which percentile to use. This gives a finer control over this
        crucial parameter. For more info, please read about the perc parameter.
    - Automatic tree number:
        Setting the n_estimator to 'auto' will calculate the number of trees
        in each iteration based on the number of features under investigation.
        This way more trees are used when the training data has many features
        and less when most of the features have been rejected.
    - Ranking of features:
        After fitting BorutaPy it provides the user with ranking of features.
        Confirmed ones are 1, Tentatives are 2, and the rejected are ranked
        starting from 3, based on their feature importance history through
        the iterations.
    - Using either the native variable importance, scikit permutation importance,
        SHAP importance.

    We highly recommend using pruned trees with a depth between 3-7.
    For more, see the docs of these functions, and the examples below.
    Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
    Boruta is an all relevant feature selection method, while most other are
    minimal optimal; this means it tries to find all features carrying
    information usable for prediction, rather than finding a possibly compact
    subset of features on which some classifier has a minimal error.
    Why bother with all relevant feature selection?
    When you try to understand the phenomenon that made your data, you should
    care about all factors that contribute to it, not just the bluntest signs
    of it in context of your methodology (yes, minimal optimal set of features
    by definition depends on your classifier choice).


    Summary
    -------
         - Loop over n_iter or until dec_reg == 0
         - add shadows
            o find features that are tentative
            o make sure that at least 5 columns are added
            o shuffle shadows
            o get feature importance
                * fit the estimator
                * extract feature importance (native, shap or permutation)
                * return feature importance
            o separate the importance of shadows and real

         - Calculate the maximum shadow importance and append to the previous run
         - Assign hits using the imp_sha_max of this run
            o find all the feat imp > imp_sha_max
            o tag them as hits
            o add +1 to the previous tag vector
         - Perform a test
            o select non rejected features yet
            o get a binomial p-values (nbr of times the feat has been tagged as important
            on the n_iter done so far) o reject or not according the (corrected) p-val


    Parameters
    ----------
    estimator : object
        A supervised learning estimator, with a 'fit' method that returns the
        feature_importances_ attribute. Important features must correspond to
        high absolute values in the feature_importances_.
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
        Possible values: 'shap' (Shapley values),
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
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already


    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features - only confirmed ones are True.
    support_weak_ : array of shape [n_features]
        The mask of selected tentative features, which haven't gained enough
        support during the max_iter number of iterations..
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1 and tentative features are assigned
        rank 2.
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
    tag_df : dataframe
        the df with the details (accepted or rejected) of the feature selection


    Examples
    --------

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy

    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    y = y.ravel()

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    # define Boruta feature selection method
    feat_selector = Leshy(rf, n_estimators='auto', verbose=2, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)

    # check selected features - first 5 features are selected
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------
    [1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
        Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
    """

    def __init__(self, estimator, n_estimators=1000, perc=90, alpha=0.05, importance='shap',
                 two_step=True, max_iter=100, random_state=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.random_state_instance = None
        self.verbose = verbose
        self.importance = importance
        self.cat_name = None
        self.cat_idx = None
        # Catboost doesn't allow to change random seed after fitting
        self.is_cat = is_catboost(estimator)
        self.is_lgb = is_lightgbm(estimator)
        # plotting
        self.imp_real_hist = None
        self.sha_max = None
        self.col_names = None
        self.tag_df = None

    def fit(self, X, y, sample_weight=None):
        """
        Fits the Boruta feature selection with the provided estimator.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample
        """
        self.imp_real_hist = np.empty((0, X.shape[1]), float)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.col_names = X.columns.to_list()

        return self._fit(X, y, sample_weight=sample_weight)

    def transform(self, X, weak=False, return_df=False):
        """
        Reduces the input X to the features selected by Boruta.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        return self._transform(X, weak, return_df)

    def fit_transform(self, X, y, sample_weight=None, weak=False, return_df=False):
        """
        Fits Boruta, then reduces the input X to the selected features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.

        Summary
        -------
         - Loop over n_iter or until dec_reg == 0
         - add shadows
            o find features that are tentative
            o make sure that at least 5 columns are added
            o shuffle shadows
            o get feature importance
                * fit the estimator
                * extract feature importance (native, shap or permutation)
                * return feature importance
            o separate the importance of shadows and real

         - Calculate the maximum shadow importance and append to the previous run
         - Assign hits using the imp_sha_max of this run
            o find all the feat imp > imp_sha_max
            o tag them as hits
            o add +1 to the previous tag vector
         - Perform a test
            o select non rejected features yet
            o get a binomial p-values (nbr of times the feat has been tagged as
            important on the n_iter done so far) o reject or not according the (corrected) p-val
        """

        self._fit(X, y, sample_weight=sample_weight)
        return self._transform(X, weak, return_df)

    def plot_importance(self, n_feat_per_inch=5):
        """
        Boxplot of the variable importance, ordered by magnitude
        The max shadow variable importance illustrated by the dashed line.
        Requires to apply the fit method first.
        :return: boxplot
        """
        # plt.style.use('fivethirtyeight')
        my_colors_list = ['#000000', '#7F3C8D', '#11A579', '#3969AC',
                          '#F2B701', '#E73F74', '#80BA5A', '#E68310',
                          '#008695', '#CF1C90', '#F97B72']
        bckgnd_color = "#f5f5f5"
        params = {"axes.prop_cycle": plt.cycler(color=my_colors_list),
                  "axes.facecolor": bckgnd_color, "patch.edgecolor": bckgnd_color,
                  "figure.facecolor": bckgnd_color,
                  "axes.edgecolor": bckgnd_color, "savefig.edgecolor": bckgnd_color,
                  "savefig.facecolor": bckgnd_color, "grid.color": "#d2d2d2",
                  'lines.linewidth': 1.5}  # plt.cycler(color=my_colors_list)
        mpl.rcParams.update(params)

        if self.imp_real_hist is None:
            raise ValueError("Use the fit method first to compute the var.imp")

        color = {'boxes': 'gray', 'whiskers': 'gray', 'medians': '#000000', 'caps': 'gray'}
        vimp_df = pd.DataFrame(self.imp_real_hist, columns=self.col_names)
        vimp_df = vimp_df.reindex(vimp_df.mean().sort_values(ascending=True).index, axis=1)
        bp = vimp_df.boxplot(color=color,
                             boxprops=dict(linestyle='-', linewidth=1.5),
                             flierprops=dict(linestyle='-', linewidth=1.5),
                             medianprops=dict(linestyle='-', linewidth=1.5, color='#000000'),
                             whiskerprops=dict(linestyle='-', linewidth=1.5),
                             capprops=dict(linestyle='-', linewidth=1.5),
                             showfliers=False, grid=True, rot=0, vert=False, patch_artist=True,
                             figsize=(16, vimp_df.shape[1] / n_feat_per_inch), fontsize=9
                             )
        blue_color = "#2590fa"
        yellow_color = "#f0be00"
        n_strong = sum(self.support_)
        n_weak = sum(self.support_weak_)
        n_discarded = len(self.col_names) - n_weak - n_strong
        box_face_col = [blue_color] * n_strong + [yellow_color] * n_weak + ['gray'] * n_discarded
        for c in range(len(box_face_col)):
            bp.findobj(mpl.patches.Patch)[len(self.support_) - c - 1].set_facecolor(box_face_col[c])
            bp.findobj(mpl.patches.Patch)[len(self.support_) - c - 1].set_color(box_face_col[c])

        xrange = vimp_df.max(skipna=True).max(skipna=True) - vimp_df.min(skipna=True).min(skipna=True)
        bp.set_xlim(left=vimp_df.min(skipna=True).min(skipna=True) - 0.10 * xrange)

        custom_lines = [Line2D([0], [0], color=blue_color, lw=5),
                        Line2D([0], [0], color=yellow_color, lw=5),
                        Line2D([0], [0], color="gray", lw=5),
                        Line2D([0], [0], linestyle='--', color="gray", lw=2)]
        bp.legend(custom_lines, ['confirmed', 'tentative', 'rejected', 'sha. max'], loc="lower right")
        plt.axvline(x=self.sha_max, linestyle='--', color='gray')
        fig = bp.get_figure()
        plt.title('Leshy importance and selected predictors')
        # fig.set_size_inches((10, 1.5 * np.rint(max(vimp_df.shape) / 10)))
        # plt.tight_layout()
        # plt.show()
        return fig

    @staticmethod
    def _validate_pandas_input(arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    def _fit(self, X_raw, y, sample_weight=None):
        """
        Private method.
        Chaining:
         - Loop over n_iter or until dec_reg == 0
         - add shadows
            o find features that are tentative
            o make sure that at least 5 columns are added
            o shuffle shadows
            o get feature importance
                * fit the estimator
                * extract feature importance (native, shap or permutation)
                * return feature importance
            o separate the importance of shadows and real

         - Calculate the maximum shadow importance and append to the previous run
         - Assign hits using the imp_sha_max of this run
            o find all the feat imp > imp_sha_max
            o tag them as hits
            o add +1 to the previous tag vector
         - Perform a test
            o select non rejected features yet
            o get a binomial p-values (nbr of times the feat has been tagged as
            important on the n_iter done so far) o reject or not according the (corrected) p-val

        Parameters
        ----------
        X_raw : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample

        :return:
         self : object
            Nothing but attributes
        """
        # self.is_cat = is_catboost(self.estimator)

        start_time = time.time()
        # basic cat features encoding
        # First, let's store "object" columns as categorical columns
        # obj_feat = X_raw.dtypes.loc[X_raw.dtypes == 'object'].index.tolist()
        obj_feat = list(set(list(X_raw.columns)) - set(list(X_raw.select_dtypes(include=[np.number]))))
        X = X_raw
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        # w = w.fillna(0)

        self.cat_name = obj_feat
        # self.cat_idx = cat_idx

        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X)

        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                sample_weight = self._validate_pandas_input(sample_weight)

        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        pbar = tqdm(total=self.max_iter, desc="Leshy iteration")
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            # Catboost doesn't allow to change random seed after fitting
            if self.is_cat is False:
                if self.is_lgb:
                    self.estimator.set_params(random_state=self.random_state.randint(0, 10000))
                else:
                    self.estimator.set_params(random_state=self.random_state)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, sample_weight, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            # if self.verbose > 0 and _iter < self.max_iter:
            #     self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1
                pbar.update(1)
        pbar.close()
        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1
        # for plotting
        self.imp_real_hist = imp_history
        self.sha_max = imp_sha_max

        if isinstance(X_raw, np.ndarray):
            X_raw = pd.DataFrame(X_raw)

        if isinstance(X_raw, pd.DataFrame):
            self.support_names_ = [X_raw.columns[i] for i, x in enumerate(self.support_) if x]
            self.tag_df = pd.DataFrame({'predictor': list(X_raw.columns)})
            self.tag_df['Boruta'] = 1
            self.tag_df['Boruta'] = np.where(self.tag_df['predictor'].isin(list(self.support_names_)), 1, 0)

        if isinstance(X_raw, pd.DataFrame):
            self.support_weak_names_ = [X_raw.columns[i] for i, x in enumerate(self.support_weak_) if x]
            self.tag_df['Boruta_weak_incl'] = np.where(self.tag_df['predictor'].isin(
                list(self.support_names_ + self.support_weak_names_)
            ), 1, 0)

            # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
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
            self.support_ = np.ones(n_feat, dtype=np.bool)

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        self.running_time = time.time() - start_time
        hours, rem = divmod(self.running_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("All relevant predictors selected in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                                                                  int(minutes),
                                                                                  seconds))
        return self

    def _transform(self, X, weak=False, return_df=False):
        """
        Private method

        transform the predictor matrix by dropping the rejected and
        (optional) the undecided predictors
        :param X: pd.DataFrame
            predictor matrix
        :param weak: bool
            whether to drop or not the undecided predictors
        :param return_df: bool
            return a pandas dataframe or not
        :return:
         X: np.array or pd.DataFrame
            the transformed predictors matrix
        """
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        if return_df:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]
        return X

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if depth is None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        # n_feat * 2 because the training matrix is extended with n shadow features
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _get_shuffle(self, seq):
        self.random_state.shuffle(seq)
        return seq

    def _add_shadows_get_imps(self, X, y, sample_weight, dec_reg):
        """
        Add a shuffled copy of the columns (shadows) and get the feature
        importance of the augmented data set

        :param X: pd.DataFrame of shape [n_samples, n_features]
            predictor matrix
        :param y: pd.series of shape [n_samples]
            target
        :param sample_weight: array-like, shape = [n_samples], default=None
            Individual weights for each sample
        :param dec_reg: array
            holds the decision about each feature 1, 0, -1 (accepted, undecided, rejected)
        :return:
         imp_real: array
            feature importance of the real predictors
         imp_sha: array
            feature importance of the shadow predictors
        """
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = np.copy(x_cur)
        # make sure there's at least 5 columns in the shadow matrix for
        while x_sha.shape[1] < 5:
            x_sha = np.hstack((x_sha, x_sha))
        # shuffle xSha
        x_sha = np.apply_along_axis(self._get_shuffle, 0, x_sha)
        # get importance of the merged matrix
        if self.importance == 'shap':
            imp = _get_shap_imp(self.estimator, np.hstack((x_cur, x_sha)), y, sample_weight)
        elif self.importance == 'pimp':
            imp = _get_perm_imp(self.estimator, np.hstack((x_cur, x_sha)), y, sample_weight)
        else:
            imp = _get_imp(self.estimator, np.hstack((x_cur, x_sha)), y, sample_weight)

        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]

        return imp_real, imp_sha

    @staticmethod
    def _assign_hits(hit_reg, cur_imp, imp_sha_max):
        """
        count how many times a given feature was more important than
        the best of the shadow features

        :param hit_reg: array
            count how many times a given feature was more important than the
            best of the shadow features
        :param cur_imp: array
            current importance
        :param imp_sha_max: array
            importance of the best shadow predictor
        :return:
        """
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp[0]
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        """
        Perform the rest if the feature should be tagget as relevant (confirmed), not relevant (rejected)
        or undecided. The test is performed by considering the binomial tentatives over several attempts.
        I.e. count how many times a given feature was more important than the best of the shadow features
        and test if the associated probability to the z-score is below, between or above the rejection or
        acceptance threshold.

        :param dec_reg: array
            holds the decision about each feature 1, 0, -1 (accepted, undecided, rejected)
        :param hit_reg: array
            counts how many times a given feature was more important than the best of the shadow features
        :param _iter:
        :return:
        """
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

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
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
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
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Private method
        Check hyperparameters as well as X and y before proceeding with fit.
        :param X: pd.DataFrame
            predictor matrix
        :param y: pd.series
            target
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y, dtype=None, force_all_finite=False)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _print_results(self, dec_reg, _iter, flag):
        """
        Private method
        printing the result
        :param dec_reg: array
            if the feature as been tagged as relevant (confirmed),
            not relevant (rejected) or undecided
        :param _iter: int
            the iteration number
        :param flag: int
            is still in the feature selection process or not
        :return:
         output: str
            the output to be printed out
        """
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            if self.importance in ['shap', 'pimp']:
                vimp = str(self.importance)
            else:
                vimp = 'native'
            output = "\n\nLeshy finished running using " + vimp + " var. imp.\n\n" + result
        print(output)


def _split_fit_estimator(estimator, X, y, sample_weight=None, cat_feature=None):
    """
    Private function
    split the train, test and fit the model

    :param estimator: sklearn estimator
    :param X: pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    :param y: pd.series of shape [n_samples]
        target
    :param sample_weight: array-like, shape = [n_samples], default=None
        Individual weights for each sample
    :param cat_feature: list of int or None
        the list of integers, cols loc, of the categrocial predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.


    :return:
     model
        fitted model
     X_tt: array [n_samples, n_features]
        the test split, predictors
     y_tt: array [n_samples]
        the test split, target
    """
    if cat_feature is None:
        # detect, store and encode categorical predictors
        X = pd.DataFrame(X)
        obj_feat = list(set(list(X.columns)) - set(list(X.select_dtypes(include=[np.number]))))
        if obj_feat:
            X[obj_feat] = X[obj_feat].astype('str').astype('category')
            for col in obj_feat:
                X[col] = X[col].astype('category').cat.codes
            cat_idx = [X.columns.get_loc(col) for col in obj_feat]
        else:
            obj_feat = None
            cat_idx = None
    else:
        cat_idx = cat_feature

    if sample_weight is not None:
        w = sample_weight
        if is_regressor(estimator):
            X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, random_state=42)
        else:
            X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, stratify=y, random_state=42)
    else:
        if is_regressor(estimator):
            X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, random_state=42)
        else:
            X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, stratify=y, random_state=42)
        w_tr, w_tt = None, None

    X_tr = pd.DataFrame(X_tr)
    X_tt = pd.DataFrame(X_tt)

    if check_if_tree_based(estimator):
        try:
            # handle cat features if supported by the fit method
            if is_catboost(estimator) or ('cat_feature' in estimator.fit.__code__.co_varnames):
                model = estimator.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=cat_idx)
            else:
                model = estimator.fit(X_tr, y_tr, sample_weight=w_tr)

        except Exception as e:
            raise ValueError('Please check your X and y variable. The provided '
                             'estimator cannot be fitted to your data.\n' + str(e))
    else:
        raise ValueError('Not a tree based model')

    return model, X_tt, y_tt, w_tt


def _get_shap_imp(estimator, X, y, sample_weight=None, cat_feature=None):
    """
    Private function
    Get the SHAP feature importance

    :param estimator: sklearn estimator
    :param X: pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    :param y: pd.series of shape [n_samples]
        target
    :param sample_weight: array-like, shape = [n_samples], default=None
        Individual weights for each sample
    :param cat_feature: list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.

    :return:
     shap_imp, array
        the SHAP importance array
    """

    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    model, X_tt, y_tt, w_tt = _split_fit_estimator(estimator, X, y,
                                                   sample_weight=sample_weight,
                                                   cat_feature=cat_feature)

    # Faster and safer to use the builtin lightGBM method
    # Note the xgboost and catboost have builtin shap as well
    # but it requires to use DMatrix or Pool respectively
    # for other tree-based models, no builtin SHAP
    if is_lightgbm(estimator):
        shap_matrix = model.predict(X_tt, pred_contrib=True)
        # the dim changed in lightGBM 3
        shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
    else:
        # build the explainer
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_tt)
        # flatten to 2D if classification and lightgbm
        if is_classifier(estimator):
            if isinstance(shap_values, list):
                # for lightgbm clf sklearn api, shap returns list of arrays
                # https://github.com/slundberg/shap/issues/526
                class_inds = range(len(shap_values))
                shap_imp = np.zeros(shap_values[0].shape[1])
                for i, ind in enumerate(class_inds):
                    shap_imp += np.abs(shap_values[ind]).mean(0)
                shap_imp /= len(shap_values)
            else:
                shap_imp = np.abs(shap_values).mean(0)
        else:
            shap_imp = np.abs(shap_values).mean(0)

    return shap_imp


def _get_perm_imp(estimator, X, y, sample_weight, cat_feature=None):
    """
    Private function
    Get the permutation feature importance

    :param estimator: sklearn estimator
    :param X: pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    :param y: pd.series of shape [n_samples]
        target
    :param sample_weight: array-like, shape = [n_samples], default=None
        Individual weights for each sample
    :param cat_feature: list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.

    :return:
     imp, array
        the permutation importance array
    """
    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    model, X_tt, y_tt, w_tt = _split_fit_estimator(estimator, X, y,
                                                   sample_weight=sample_weight,
                                                   cat_feature=cat_feature)
    perm_imp = permutation_importance(model, X_tt, y_tt, n_repeats=5, random_state=42, n_jobs=-1)
    imp = perm_imp.importances_mean.ravel()
    return imp


def _get_imp(estimator, X, y, sample_weight=None, cat_feature=None):
    """
        Get the native feature importance (impurity based for instance)
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
        """
    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    try:
        # handle categoricals
        X = pd.DataFrame(X)
        if cat_feature is None:
            obj_feat = list(set(list(X.columns)) - set(list(X.select_dtypes(include=[np.number]))))
            if obj_feat:
                X[obj_feat] = X[obj_feat].astype('str').astype('category')
                for col in obj_feat:
                    X[col] = X[col].cat.codes
                cat_idx = [X.columns.get_loc(col) for col in obj_feat]
            else:
                obj_feat = None
                cat_idx = None
        else:
            cat_idx = cat_feature

        # handle catboost and cat features
        if is_catboost(estimator) or ('cat_feature' in estimator.fit.__code__.co_varnames):
            X = pd.DataFrame(X)
            estimator.fit(X, y, sample_weight=sample_weight, cat_features=cat_idx)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)

    except Exception as e:
        raise ValueError('Please check your X and y variable. The provided '
                         'estimator cannot be fitted to your data.\n' + str(e))
    try:
        imp = estimator.feature_importances_
    except Exception:
        raise ValueError('Only methods with feature_importance_ attribute '
                         'are currently supported in BorutaPy.')
    return imp


###################################
#
# BoostAGroota
#
###################################

class BoostAGroota(BaseEstimator, TransformerMixin):  # (object):
    """
    BoostARoota becomes BoostAGroota, I'm Groot

    Original version of BoostARoota:
    * One-Hot-Encode the feature set
    * Double width of the data set, making a copy of all features in original dataset
    * Randomly shuffle the new features created in (2). These duplicated and shuffled features
      are referred to as "shadow features"
    * Run XGBoost classifier on the entire data set ten times. Running it ten times allows for
      random noise to be smoothed out, resulting in more robust estimates of importance.
      The number of repeats is a parameter than can be changed.
    * Obtain importance values for each feature. This is a simple importance metric that sums up
      how many times the particular feature was split on in the XGBoost algorithm.
    * Compute "cutoff": the average feature importance value for all shadow features and divide by a factor
      (parameter to deal with conservativeness). With values lower than this,
      features are removed at too high of a rate.
    * Remove features with average importance across the ten iterations that is less than
      the cutoff specified in (6)
    * Go back to (2) until the number of features removed is less than ten percent of the total.
    * Method returns the features remaining once completed.

    Modifications:
    - Replace XGBoost with LightGBM, you can still use tree-based scikitlearn models
    - Replace native var.imp by SHAP var.imp. Indeed, the impurity var.imp. are biased and
      sensitive to large cardinality
      (see [scikit demo](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.
      html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py)).
      Moreover, the native var. imp are computed on the train set, here the data are split (internally)
      in train and test, var. imp computed on the test set.
    - Handling categorical predictors. Cat. predictors should NOT be one hot encoded,
      it leads to deep unstable trees.
      Instead, it's better to use the native method of lightGBM or CatBoost.
      A preprocessing step is needed to encode
      (ligthGBM and CatBoost use integer encoding and reference to categorical columns.
      The splitting stratigies are different then, see official doc).
    - Work with sample_weight, for Poisson or any application requiring a weighting.


    Params:
    -------
    :param est: sklear estimator
        the model to train, lightGBM recommended, see the reduce lightgbm method
    :param cutoff: float
        the value by which the max of shadow imp is divided, to compare to real importance
    :param iters: int (>0)
        The number of iterations to average for the feature importance (on the same split),
        to reduce the variance
    :param max_rounds: int (>0)
        The number of times the core BoostARoota algorithm will run.
        Each round eliminates more and more features
    :param delta: float (0 < delta <= 1)
        Stopping criteria for whether another round is started
    :param silent: bool
        Set to True if don't want to see the BoostARoota output printed.
    :param importance: str, default='shap'
        the kind of feature importance to use. Possible values: 'shap' (Shapley values),
        'pimp' (permutation importance) and 'native' (Gini/impurity)


    Attributes:
    -----------
    support_names_: list of str
        the list of columns to keep
    tag_df: dataframe
        the df with the details (accepted or rejected) of the feature selection
    sha_cutoff_df: dataframe
        feature importance of the real+shadow predictors over iterations
    mean_shadow: float
        the threshold below which the predictors are rejected


    Example:
    --------
    X = df[filtered_features].copy()
    y = df['re_cl'].copy()
    w = df["exp_yr"].copy()
    y = y/w.replace(0,1)
    y = y.fillna(0)

    model = LGBMRegressor(n_jobs=-1, n_estimators=100, objective='poisson',
                          random_state=42, verbose=0)
    br = noglmgroot.BoostARoota(est=model, cutoff=1, iters=10, max_rounds=10,
                                delta=0.1, silent=False, weight=w)
    br.fit(X, y)
    br.plot_importance()
    """

    def __init__(self, est=None, cutoff=4, iters=10, max_rounds=500, delta=0.1,
                 silent=True, importance='shap'):

        self.est = est
        self.cutoff = cutoff
        self.iters = iters
        self.max_rounds = max_rounds
        self.delta = delta
        self.silent = silent
        self.importance = importance
        self.support_names_ = None
        self.tag_df = None
        self.sha_cutoff_df = None
        self.mean_shadow = None

        # Throw errors if the inputted parameters don't meet the necessary criteria
        if cutoff <= 0:
            raise ValueError('cutoff should be greater than 0. You entered' + str(cutoff))
        if iters <= 0:
            raise ValueError('iters should be greater than 0. You entered' + str(iters))
        if (delta <= 0) | (delta > 1):
            raise ValueError('delta should be between 0 and 1, was ' + str(delta))

        # Issue warnings for parameters to still let it run
        if delta < 0.02:
            warnings.warn("WARNING: Setting a delta below 0.02 may not converge on a solution.")
        if max_rounds < 1:
            warnings.warn("WARNING: Setting max_rounds below 1 will automatically be set to 1.")

    def __repr__(self):
        s = "BoostARoota(est={est}, \n" \
            "                cutoff={cutoff},\n" \
            "                iters={iters},\n" \
            "                max_rounds={mr},\n" \
            "                delta={delta},\n" \
            "                silent={silent}, \n" \
            "                importance=\"{importance}\")".format(est=self.est,
                                                                  cutoff=self.cutoff,
                                                                  iters=self.iters,
                                                                  mr=self.max_rounds,
                                                                  delta=self.delta,
                                                                  silent=self.silent,
                                                                  importance=self.importance)
        return s

    def fit(self, X, y, sample_weight=None):
        """
        Fit the transformer

        :param x: pd.DataFrame
            the predictors matrix
        :param y: pd.Series
            the target
        :param sample_weight: pd.series
            sample_weight, if any
        :return:
        """

        if isinstance(X, pd.DataFrame) is not True:
            X = pd.DataFrame(X)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # obj_feat = x.dtypes.loc[x.dtypes == 'object'].index.tolist()
        obj_feat = list(set(list(X.columns)) - set(list(X.select_dtypes(include=[np.number]))))
        if obj_feat:
            X[obj_feat] = X[obj_feat].astype('str').astype('category')

        # cat_idx = [X.columns.get_loc(c) for c in cat_feat if c in cat_feat]

        # crit, keep_vars, df_vimp, mean_shadow
        _, self.support_names_, self.sha_cutoff_df, self.mean_shadow = _BoostARoota(X, y,
                                                                                    # metric=self.metric,
                                                                                    est=self.est,
                                                                                    cutoff=self.cutoff,
                                                                                    iters=self.iters,
                                                                                    max_rounds=self.max_rounds,
                                                                                    delta=self.delta,
                                                                                    silent=self.silent,
                                                                                    weight=sample_weight,
                                                                                    imp=self.importance
                                                                                    )
        self.tag_df = pd.DataFrame({'predictor': list(X.columns)})
        self.tag_df['BoostAGroota'] = 1
        self.tag_df['BoostAGroota'] = np.where(self.tag_df['predictor'].isin(list(self.support_names_)), 1, 0)
        return self

    def transform(self, x):
        """
        Transform the predictors matrix by dropping the rejected columns
        :param x: pd.DataFrame
            the predictors matrix
        :return:
            the transformed predictors matrix
        """
        if self.support_names_ is None:
            raise ValueError("You need to fit the model first")
        return x[self.support_names_]

    def fit_transform(self, X, y=None, **fit_params):
        """
        chain fit and transform
        :param X: pd.DataFrame
            the predictors matrix
        :param y: pd.Series
            the target
        :param fit_params : dict
            Additional fit parameters: e.g. {'sample_weight': sample_weight}
        :return:
            the transformed predictors matrix
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def plot_importance(self, n_feat_per_inch=5):
        """
        Boxplot of the variable importance, ordered by magnitude
        The max shadow variable importance illustrated by the dashed line.

        :param n_feat_per_inch: int
            the number of features per inch in the figure, for readability

        :return: boxplot
        """
        # plt.style.use('fivethirtyeight')
        my_colors_list = ['#000000', '#7F3C8D', '#11A579', '#3969AC',
                          '#F2B701', '#E73F74', '#80BA5A', '#E68310',
                          '#008695', '#CF1C90', '#F97B72']
        bckgnd_color = "#f5f5f5"
        params = {"axes.prop_cycle": plt.cycler(color=my_colors_list),
                  "axes.facecolor": bckgnd_color, "patch.edgecolor": bckgnd_color,
                  "figure.facecolor": bckgnd_color,
                  "axes.edgecolor": bckgnd_color, "savefig.edgecolor": bckgnd_color,
                  "savefig.facecolor": bckgnd_color, "grid.color": "#d2d2d2",
                  'lines.linewidth': 1.5}  # plt.cycler(color=my_colors_list)
        mpl.rcParams.update(params)

        if self.mean_shadow is None:
            raise ValueError('Apply fit method first')

        b_df = self.sha_cutoff_df.T.copy()
        b_df.columns = b_df.iloc[0]
        b_df = b_df.drop(b_df.index[0])
        b_df = b_df.drop(b_df.index[-1])
        real_df = b_df.iloc[:, :int(b_df.shape[1] / 2)].copy()
        blue_color = "#2590fa"
        color = {'boxes': blue_color, 'whiskers': 'gray', 'medians': '#000000', 'caps': 'gray'}
        real_df = real_df.reindex(real_df.mean().sort_values(ascending=True).index, axis=1)
        bp = real_df.plot.box(  # kind='box',
            color=color,
            boxprops=dict(linestyle='-', linewidth=1.5, color=blue_color, facecolor=blue_color),
            flierprops=dict(linestyle='-', linewidth=1.5),
            medianprops=dict(linestyle='-', linewidth=1.5, color='#000000'),
            whiskerprops=dict(linestyle='-', linewidth=1.5),
            capprops=dict(linestyle='-', linewidth=1.5),
            showfliers=False, grid=True, rot=0, vert=False, patch_artist=True,
            figsize=(16, real_df.shape[1] / n_feat_per_inch), fontsize=9
        )
        # xrange = real_df.max(skipna=True).max(skipna=True)-real_df.min(skipna=True).min(skipna=True)
        bp.set_xlim(left=real_df.min(skipna=True).min(skipna=True) - 0.025)
        fig = bp.get_figure()
        plt.title('BoostAGroota importance of selected predictors')
        # plt.tight_layout()
        # plt.show()
        return fig


############################################
# Helper Functions to do the Heavy Lifting
############################################


def _create_shadow(x_train):
    """
    Take all X variables, creating copies and randomly shuffling them
    :param x_train: pd.DataFrame
        the dataframe to create shadow features on
    :return:
     new_x: pd.DataFrame
        dataframe 2x width and the names of the shadows for removing later
     shadow_names: list of str
        the name of the new columns
    """
    x_shadow = x_train.copy()
    for c in x_shadow.columns:
        np.random.shuffle(x_shadow[c].values)
    # rename the shadow
    shadow_names = ["ShadowVar" + str(i + 1) for i in range(x_train.shape[1])]
    x_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_x = pd.concat([x_train, x_shadow], axis=1)
    return new_x, shadow_names


########################################################################################
# BoostARoota. In principle, you cannot/don't need to access those methods (reason of
# the _ in front of the function name, they're internal functions)
########################################################################################

def _reduce_vars_sklearn(x, y, est, this_round, cutoff, n_iterations, delta, silent, weight, imp_kind, cat_feature):
    """
    Private function
    reduce the number of predictors using a sklearn estimator

    :param x: pd.DataFrame
        the dataframe to create shadow features on
    :param y: pd.Series
        the target
    :param est: sklear estimator
        the model to train, lightGBM recommended
    :param this_round: int
        The number of times the core BoostARoota algorithm will run.
        Each round eliminates more and more features
    :param cutoff: float
        the value by which the max of shadow imp is divided, to compare to real importance
    :param n_iterations: int
        The number of iterations to average for the feature importance (on the same split),
        to reduce the variance
    :param delta: float (0 < delta <= 1)
        Stopping criteria for whether another round is started
    :param silent: bool
        Set to True if don't want to see the BoostARoota output printed.
        Will still show any errors or warnings that may occur
    :param weight: pd.series
        sample_weight, if any
    :param imp_kind: str
        whether if native, shap or permutation importance should be used
    :param cat_feature: list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.

    :return:
     criteria: bool
        if the criteria has been reached or not
     real_vars['feature']: pd.dataframe
        feature importance of the real predictors over iter
     df: pd.DataFrame
        feature importance of the real+shadow predictors over iter
     mean_shadow: float
        the feature importance threshold, to reject or not the predictors
    """
    # Set up the parameters for running the model in XGBoost - split is on multi log loss

    for i in range(1, n_iterations + 1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(x)
        if imp_kind == 'shap':
            imp = _get_shap_imp(est, new_x, y, sample_weight=weight, cat_feature=cat_feature)
        elif imp_kind == 'pimp':
            imp = _get_perm_imp(est, new_x, y, sample_weight=weight, cat_feature=cat_feature)
        elif imp_kind == 'native':
            warnings.warn("[BoostAGroota]: using native variable importance might break the FS")
            imp = _get_imp(est, new_x, y, sample_weight=weight, cat_feature=cat_feature)
        else:
            raise ValueError("'imp' should be either 'native', 'shap' or 'pimp', "
                             "'native' is not recommended")

        if i == 1:
            df = pd.DataFrame({'feature': new_x.columns})
            df2 = df.copy()
            pass
        try:
            importance = imp  # est.feature_importances_
            df2['fscore' + str(i)] = importance
        except ValueError:
            print("this clf doesn't have the feature_importances_ method.  "
                  "Only Sklearn tree based methods allowed")

        # importance = sorted(importance.items(), key=operator.itemgetter(1))

        # df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
        df2['fscore' + str(i)] = df2['fscore' + str(i)] / df2['fscore' + str(i)].sum()
        df = pd.merge(df, df2, on='feature', how='outer')
        if not silent:
            print("Round: ", this_round, " iteration: ", i)

    df['Mean'] = df.mean(axis=1)
    # Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]

    # Get mean value from the shadows (max, like in Boruta, median to mitigate variance)
    mean_shadow = shadow_vars.select_dtypes(include=[np.number]).max().mean() / cutoff
    real_vars = real_vars[(real_vars.Mean >= mean_shadow)]

    # Check for the stopping criteria
    # Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if len(x.columns) == 0:
        criteria = True
    elif (len(real_vars['feature']) / len(x.columns)) > (1 - delta):
        criteria = True
    else:
        criteria = False

    return criteria, real_vars['feature'], df, mean_shadow


# Main function exposed to run the algorithm
def _BoostARoota(x, y, est, cutoff, iters, max_rounds, delta, silent, weight, imp):
    """

    Private function
    reduce the number of predictors using a sklearn estimator

    :param x: pd.DataFrame
        the dataframe to create shadow features on
    :param y: pd.Series
        the target
    :param est: sklear estimator
        the model to train, lightGBM recommended, see the reduce lightgbm method
    :param cutoff: float
        the value by which the max of shadow imp is divided, to compare to real importance
    :param iters: int (>0)
        The number of iterations to average for the feature importances (on the same split),
        to reduce the variance
    :param max_rounds: int (>0)
        The number of times the core BoostARoota algorithm will run.
        Each round eliminates more and more features
    :param delta: float (0 < delta <= 1)
        Stopping criteria for whether another round is started
    :param silent: bool
        Set to True if don't want to see the BoostARoota output printed.
        Will still show any errors or warnings that may occur
    :param weight: pd.series
        sample_weight, if any

    :return:
     crit: bool
        if the criteria has been reached or not
     keep_vars: pd.dataframe
        feature importance of the real predictors over iter
     df_vimp: pd.DataFrame
        feature importance of the real+shadow predictors over iter
     mean_shadow: float
        the feature importance threshold, to reject or not the predictors
    """
    t_boostaroota = time.time()
    new_x = x.copy()

    # extract the categorical names for the first time, store it for next iterations
    # In the below while loop this list will be update only once some of the predictors
    # are removed. This way the encoding is done only every predictors update and not
    # every iteration. The code will then be much faster since the encoding is done only once.
    obj_feat = list(set(list(x.columns)) - set(list(x.select_dtypes(include=[np.number]))))
    if obj_feat:
        new_x[obj_feat] = new_x[obj_feat].astype('str').astype('category')
        for col in obj_feat:
            new_x[col] = new_x[col].astype('category').cat.codes
    cat_idx = [new_x.columns.get_loc(c) for c in obj_feat if c in obj_feat]

    # Run through loop until "crit" changes
    i = 0
    pbar = tqdm(total=max_rounds, desc="BoostaGRoota round")
    while True:
        # Inside this loop we reduce the dataset on each iteration exiting with keep_vars
        i += 1
        crit, keep_vars, df_vimp, mean_shadow = _reduce_vars_sklearn(new_x,
                                                                     y,
                                                                     est=est,
                                                                     this_round=i,
                                                                     cutoff=cutoff,
                                                                     n_iterations=iters,
                                                                     delta=delta,
                                                                     silent=silent,
                                                                     weight=weight,
                                                                     imp_kind=imp,
                                                                     cat_feature=cat_idx
                                                                     )

        if crit | (i >= max_rounds):
            break  # exit and use keep_vars as final variables
        else:
            new_x = new_x[keep_vars].copy()
            cat_col = list(set(obj_feat).intersection(set(new_x.columns)))
            cat_idx = [new_x.columns.get_loc(c) for c in cat_col if c in cat_col]
            pbar.update(1)
    pbar.close()
    elapsed = time.time() - t_boostaroota
    elapsed = elapsed / 60
    if not silent:
        print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
        print("\nThe feature selection BoostARoota running time is {0:8.2f} min".format(elapsed))
    return crit, keep_vars, df_vimp, mean_shadow


###################################
#
# GrootCV
#
###################################

class GrootCV(BaseEstimator, TransformerMixin):
    """
    A shuffled copy of the predictors matrix is added (shadows) to the original set of predictors.
    The lightGBM is fitted using repeated cross-validation, the feature importance
    is extracted each time and averaged to smooth out the noise.
    If the feature importance is larger than the average shadow feature importance then the predictors
     are rejected, the others are kept.


    - Cross-validated feature importance to smooth out the noise, based on lightGBM only
      (which is, most of the time, the fastest and more accurate Boosting).
    - the feature importance is derived using SHAP importance
    - Taking the max of median of the shadow var. imp over folds otherwise not enough conservative and
      it improves the convergence (needs less evaluation to find a threshold)
    - Not based on a given percentage of cols needed to be deleted
    - Plot method for var. imp

    Params:
    -------
    :param objective: str
        the lightGBM objective
    :param cutoff: float
        the value by which the max of shadow imp is divided, to compare to real importance
    :param n_folds: int, default=5
        the number of folds for the cross-val
    :param n_iter: int, default=5
        the number of times the cross-validation is repeated
    :param silent: bool, default=False
        print out details or not
    :param rf: bool, default=False
        the lightGBM implementation of the random forest

    Attributes:
    -----------
    support_names_: list of str
        the list of columns to keep
    tag_df: dataframe
        the df with the details (accepted or rejected) of the feature selection
    cv_df: dataframe
        the statistics of the feature importance over the different repeats of the X-val
    sha_cutoff: float
        the threshold below which the predictors are rejected

    Example:
    --------
    X = df[filtered_features].copy()
    y = df['re_cl'].copy()
    w = df["exp_yr"].copy()
    y = y/w.replace(0,1)
    y = y.fillna(0)
    # The smaller the cutoff, the more aggresive the feature selection
    #br = noglmgroot.BoostARoota(objective = 'binary', cutoff = 1.1, weight=w, silent=True)
    br = noglmgroot.GrootCV(objective = 'poisson', cutoff = 1,
                            weight=w, silent=False, n_folds=3, n_iter=10)
    br.fit(X, y)
    br.plot_importance()
    """

    def __init__(self, objective=None, cutoff=1, n_folds=5, n_iter=5,
                 silent=True, rf=False):

        self.objective = objective
        self.cutoff = cutoff
        self.n_folds = n_folds
        self.n_iter = n_iter
        self.silent = silent
        self.support_names_ = None
        self.rf = rf
        self.tag_df = None
        self.cv_df = None
        self.sha_cutoff = None

        # Throw errors if the inputted parameters don't meet the necessary criteria
        if cutoff <= 0:
            raise ValueError('cutoff should be greater than 0. You entered' + str(cutoff))
        if n_iter <= 0:
            raise ValueError('n_iter should be greater than 0. You entered' + str(n_iter))
        if n_folds <= 0:
            raise ValueError('n_folds should be greater than 0. You entered' + str(n_folds))

    def fit(self, x, y, sample_weight=None):
        """
        Fit the transformer

        :param x: pd.DataFrame
            the predictors matrix
        :param y: pd.Series
            the target
        :param sample_weight: pd.series
            sample_weight, if any
        :return:
        """

        if isinstance(x, pd.DataFrame) is not True:
            x = pd.DataFrame(x)

        if isinstance(y, pd.Series) is not True:
            y = pd.Series(y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, x)
            sample_weight = pd.Series(sample_weight)

        # obj_feat = x.dtypes.loc[x.dtypes == 'object'].index.tolist()
        obj_feat = list(set(list(x.columns)) - set(list(x.select_dtypes(include=[np.number]))))
        if obj_feat:
            x[obj_feat] = x[obj_feat].astype('str').astype('category')
        cat_feat = x.dtypes.loc[x.dtypes == 'category'].index.tolist()
        cat_idx = [x.columns.get_loc(c) for c in cat_feat if c in cat_feat]
        # a way without loop but need to re-do astype

        if cat_feat:
            cat = x[cat_feat].stack().astype('category').cat.codes.unstack()
            X = pd.concat([x[x.columns.difference(cat_feat)], cat], axis=1)
        else:
            X = x

        self.support_names_, self.cv_df, self.sha_cutoff = _reduce_vars_lgb_cv(X,
                                                                               y,
                                                                               objective=self.objective,
                                                                               cutoff=self.cutoff,
                                                                               n_folds=self.n_folds,
                                                                               n_iter=self.n_iter,
                                                                               silent=self.silent,
                                                                               weight=sample_weight,
                                                                               rf=self.rf)
        self.tag_df = pd.DataFrame({'predictor': list(x.columns)})
        self.tag_df['GrootCV'] = 1
        self.tag_df['GrootCV'] = np.where(self.tag_df['predictor'].isin(list(self.support_names_)), 1, 0)
        return self

    def transform(self, x):
        """
        Transform the predictors matrix by dropping the rejected columns
        :param x: pd.DataFrame
            the predictors matrix
        :return:
            the transformed predictors matrix
        """
        if self.support_names_ is None:
            raise ValueError("You need to fit the model first")
        return x[self.support_names_]

    def fit_transform(self, X, y=None, **fit_params):
        """
        chain fit and transform
        :param X: pd.DataFrame
            the predictors matrix
        :param y: pd.Series
            the target
        :param fit_params : dict
            Additional fit parameters: e.g. {'sample_weight': sample_weight}
        :return:
            the transformed predictors matrix
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def plot_importance(self, n_feat_per_inch=5):
        """
        Boxplot of the variable importance, ordered by magnitude
        The max shadow variable importance illustrated by the dashed line.
        The statistics are computed over the different repetition and different folds
        of the repeated cross-validation.

        :param n_feat_per_inch: int
            the number of features per inch in the figure, for readability

        :return: boxplot
        """
        # plt.style.use('fivethirtyeight')
        my_colors_list = ['#000000', '#7F3C8D', '#11A579', '#3969AC',
                          '#F2B701', '#E73F74', '#80BA5A', '#E68310',
                          '#008695', '#CF1C90', '#F97B72']
        bckgnd_color = "#f5f5f5"
        params = {"axes.prop_cycle": plt.cycler(color=my_colors_list),
                  "axes.facecolor": bckgnd_color, "patch.edgecolor": bckgnd_color,
                  "figure.facecolor": bckgnd_color,
                  "axes.edgecolor": bckgnd_color, "savefig.edgecolor": bckgnd_color,
                  "savefig.facecolor": bckgnd_color, "grid.color": "#d2d2d2",
                  'lines.linewidth': 1.5}  # plt.cycler(color=my_colors_list)
        mpl.rcParams.update(params)

        if self.sha_cutoff is None:
            raise ValueError('Apply fit method first')

        b_df = self.cv_df.T.copy()
        b_df.columns = b_df.iloc[0]
        b_df = b_df.drop(b_df.index[0])
        b_df = b_df.drop(b_df.index[-1])
        real_df = b_df.iloc[:, :int(b_df.shape[1] / 2)].copy()
        sha_df = b_df.iloc[:, int(b_df.shape[1] / 2):].copy()

        color = {'boxes': 'gray', 'whiskers': 'gray', 'medians': '#000000', 'caps': 'gray'}
        real_df = real_df.reindex(real_df.mean().sort_values(ascending=True).index, axis=1)
        bp = real_df.plot(kind='box',
                          color=color,
                          boxprops=dict(linestyle='-', linewidth=1.5),
                          flierprops=dict(linestyle='-', linewidth=1.5),
                          medianprops=dict(linestyle='-', linewidth=1.5, color='#000000'),
                          whiskerprops=dict(linestyle='-', linewidth=1.5),
                          capprops=dict(linestyle='-', linewidth=1.5),
                          showfliers=False, grid=True, rot=0, vert=False, patch_artist=True,
                          figsize=(16, real_df.shape[1] / n_feat_per_inch), fontsize=9
                          )
        col_idx = np.argwhere(real_df.columns.isin(self.support_names_)).ravel()
        blue_color = "#2590fa"

        for c in range(real_df.shape[1]):
            bp.findobj(mpl.patches.Patch)[c].set_facecolor('gray')
            bp.findobj(mpl.patches.Patch)[c].set_color('gray')

        for c in col_idx:
            bp.findobj(mpl.patches.Patch)[c].set_facecolor(blue_color)
            bp.findobj(mpl.patches.Patch)[c].set_color(blue_color)

        plt.axvline(x=self.sha_cutoff, linestyle='--', color='gray')
        # xrange = real_df.max(skipna=True).max(skipna=True)-real_df.min(skipna=True).min(skipna=True)
        bp.set_xlim(left=real_df.min(skipna=True).min(skipna=True) - 0.025)
        custom_lines = [Line2D([0], [0], color=blue_color, lw=5),
                        Line2D([0], [0], color="gray", lw=5),
                        Line2D([0], [0], linestyle='--', color="gray", lw=2)]
        bp.legend(custom_lines, ['confirmed', 'rejected', 'threshold'], loc="lower right")
        fig = bp.get_figure()
        plt.title('Groot CV importance and selected predictors')
        # plt.tight_layout()
        # plt.show()
        return fig


########################################################################################
#
# BoostARoota. In principle, you cannot/don't need to access those methods (reason of
# the _ in front of the function name, they're internal functions)
#
########################################################################################

def _reduce_vars_lgb_cv(x, y, objective, n_folds, cutoff, n_iter, silent, weight, rf):
    """
    Private function
    reduce the number of predictors using a lightgbm (python API)

    :param x: pd.DataFrame
        the dataframe to create shadow features on
    :param y: pd.Series
        the target
    :param objective: str
        the lightGBM objective
    :param cutoff: float
        the value by which the max of shadow imp is divided, to compare to real importance
    :param n_iter: int
        The number of repetition of the cross-validation, smooth out the feature importance noise
    :param silent: bool
        Set to True if don't want to see the BoostARoota output printed.
        Will still show any errors or warnings that may occur
    :param weight: pd.series
        sample_weight, if any
    :param rf: bool, default=False
        the lightGBM implementation of the random forest

    :return:
     real_vars['feature']: pd.dataframe
        feature importance of the real predictors over iter
     df: pd.DataFrame
        feature importance of the real+shadow predictors over iter
     cutoff_shadow: float
        the feature importance threshold, to reject or not the predictors
    """
    # Set up the parameters for running the model in LGBM - split is on multi log loss

    n_feat = x.shape[1]

    if objective == 'softmax':
        param = {'objective': objective,
                 'eval_metric': 'mlogloss',
                 'num_class': len(np.unique(y))}
    else:
        param = {'objective': objective}

    param.update({'verbosity': -1})

    # best feature fraction according to "Elements of statistical learning"
    if objective in ['softmax', 'binary']:
        feat_frac = np.sqrt(n_feat) / n_feat
    else:
        feat_frac = n_feat / (3 * n_feat)

    if rf:
        param.update({'boosting_type': 'rf',
                      'bagging_fraction': '0.7',
                      'feature_fraction': feat_frac,
                      'bagging_freq': '1'
                      })

    if objective in ['softmax', 'binary']:
        y = y.astype(int)
        y_freq_table = pd.Series(y.fillna(0)).value_counts(normalize=True)
        n_classes = y_freq_table.size
        if n_classes > 2 and objective != 'softmax':
            param['objective'] = 'softmax'
            if not silent:
                print("Multi-class task, setting objective to softmax")

        main_class = y_freq_table[0]
        if not silent:
            print("GrootCV: classification with unbalance classes")

        if main_class > 0.8:
            param.update({'is_unbalance': True})

    param.update({'num_threads': 0})

    col_names = list(x)
    cat_var_index = [i for i, x in enumerate(x.dtypes.tolist()) if
                     isinstance(x, pd.CategoricalDtype) or x == 'object']
    category_cols = [x for i, x in enumerate(col_names) if i in cat_var_index]

    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_iter, random_state=2652124)
    i = 0
    for tridx, validx in tqdm(rkf.split(x, y), total=rkf.get_n_splits(), desc="Repeated k-fold"):

        if weight is not None:
            x_train, y_train, weight_tr = x.iloc[tridx, :], y.iloc[tridx], weight.iloc[tridx]
            x_val, y_val, weight_val = x.iloc[validx, :], y.iloc[validx], weight.iloc[validx]
        else:
            x_train, y_train = x.iloc[tridx, :], y.iloc[tridx]
            x_val, y_val = x.iloc[validx, :], y.iloc[validx]
            weight_tr = None
            weight_val = None

        # Create the shadow variables and run the model to obtain importances
        new_x_tr, shadow_names = _create_shadow(x_train)
        new_x_val, _ = _create_shadow(x_val)

        d_train = lgb.Dataset(new_x_tr, label=y_train, weight=weight_tr,
                              categorical_feature=category_cols)
        d_valid = lgb.Dataset(new_x_val, label=y_val, weight=weight_val,
                              categorical_feature=category_cols)
        watchlist = [d_train, d_valid]

        bst = lgb.train(param,
                        train_set=d_train,
                        num_boost_round=1000,
                        valid_sets=watchlist,
                        early_stopping_rounds=50,
                        verbose_eval=0,
                        categorical_feature=category_cols
                        )
        if i == 0:
            df = pd.DataFrame({'feature': new_x_tr.columns})
            pass

        shap_matrix = bst.predict(new_x_tr, pred_contrib=True)

        # the dim changed in lightGBM 3
        shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)

        # For LGBM version < 3
        # if objective in ['softmax', 'binary']:
        #     # X_SHAP_values (array-like of shape = [n_samples, n_features + 1]
        #     # or shape = [n_samples, (n_features + 1) * n_classes])
        #     n_feat = new_x_tr.shape[1]
        #     shap_matrix = np.delete(shap_matrix,
        #     list(range(n_feat + 1, 1 + (n_feat + 1) * n_classes, n_feat + 1)), axis=1)
        #     shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
        # else:
        #     shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)

        importance = dict(zip(new_x_tr.columns, shap_imp))  # bst.feature_importance() ))
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df2 = pd.DataFrame(importance, columns=['feature', 'fscore' + str(i)])
        df2['fscore' + str(i)] = df2['fscore' + str(i)] / df2['fscore' + str(i)].sum()
        df = pd.merge(df, df2, on='feature', how='outer')
        nit = divmod(i, n_folds)[0]
        nf = divmod(i, n_folds)[1]
        if not silent:
            if nf == 0:
                print("Groot iteration: ", nit, " with " + str(n_folds) + " folds")

        i += 1

    df['Med'] = df.mean(axis=1)
    # Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]

    # Get median value from the shadows, comparing predictor by predictor. Not the same criteria
    # max().max() like in Boruta but max of the median to mitigate.
    # Otherwise too conservative (reject too often)
    cutoff_shadow = shadow_vars.select_dtypes(include=[np.number]).max().mean() / cutoff
    real_vars = real_vars[(real_vars.Med.values >= cutoff_shadow)]

    return real_vars['feature'], df, cutoff_shadow
