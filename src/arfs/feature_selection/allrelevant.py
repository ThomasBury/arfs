"""This module provides 3 different methods to perform 'all relevant feature selection'

Selectors encode owned frames without changing the caller's values or dtypes.
Categorical shadows retain the original's nominal role and missing values;
categorical identities survive feature removal and fold splitting. Generated
shadow names are unique. Transform returns selected columns of the raw input.

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

import time
import warnings
from copy import deepcopy
from numbers import Integral, Real
from typing import Optional

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import shap
from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.feature_selection._base import SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from ..utils import (
    check_if_tree_based,
    get_pandas_cat_codes,
    is_catboost,
    is_lightgbm,
    validate_sample_weight,
    validate_target,
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
PLOT_KWARGS = {
    "kind": "box",
    "boxprops": {
        "linestyle": "-",
        "linewidth": 1.5,
        "color": "gray",
        "facecolor": "gray",
    },
    "flierprops": {"linestyle": "-", "linewidth": 1.5, "color": "gray"},
    "medianprops": {"linestyle": "-", "linewidth": 1.5, "color": "#000000"},
    "whiskerprops": {"linestyle": "-", "linewidth": 1.5, "color": "gray"},
    "capprops": {"linestyle": "-", "linewidth": 1.5, "color": "gray"},
    "showfliers": False,
    "grid": True,
    "rot": 0,
    "vert": False,
    "patch_artist": True,
    "fontsize": 9,
}


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
        The supplied estimator is preserved; fit configures an owned clone.
    n_estimators : int or string, default = 1000
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.
    perc : int, default = 90
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        Set perc=100 for the maximum-shadow threshold used by vanilla Boruta.
    alpha : float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.
    importance : str, default = 'shap'
        The kind of variable importance used to compare and discriminate original
        vs shadow predictors. Note that the builtin tree importance (gini/impurity based
        importance) is biased towards numerical and large cardinality predictors, even
        if they are random. Shapley values and permutation imp. are robust w.r.t those predictors.
        Possible values: 'shap' (Shapley values), 'pimp' (permutation importance)
        and 'native' (Gini/impurity). LightGBM uses native SHAP contributions;
        other estimators use TreeExplainer. The former 'fastshap' mode is removed.
        Permutation scoring uses validation sample weights and retains signed scores.
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
    keep_weak : bool, default=False
        Include median-retained tentative features in the selected mask. Their
        relative rank remains 2; confirmed features have rank 1.
    verbose : int, default=0
        Controls verbosity of output. 0: no output, 1: displays iteration number,
        2: which features have been selected already


    Attributes
    ----------
    estimator_ : estimator
        Owned learner configuration used by the importance helpers, which fit
        their own clones. Its tree count and integer seed may change per iteration.
    importance_ : str
        Importance method used by the fitted selector.
    random_state_ : RandomState
        Runtime RNG for shadows and learner-seed updates.
    n_features_in_ : int
        The number of input features.
    n_iter_ : int
        The number of completed selection iterations.
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of confirmed features, plus tentative features if keep_weak=True.
    support_weak_ : array of shape [n_features]
        The mask of selected tentative features, which haven't gained enough
        support during the max_iter number of iterations.
    selected_features_ : list of str
        the list of columns to keep
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Confirmed features have rank 1;
        retained tentative features have rank 2, including with keep_weak=True.
    ranking_absolutes_ : array of shape [n_features]
        The absolute feature ranking as ordered by selection process. It does not guarantee
        that this order is correct for all models.
    cat_name : list of str
        the name of the categorical columns
    cat_idx : list of int
        the index of the categorical columns
    imp_real_hist : array
        array of the historical feature importance of the real predictors
    sha_max : float
        The shadow percentile threshold from the final iteration.


    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from arfs.feature_selection import Leshy
    >>> X, y = make_classification(n_samples=120, n_features=4,
    ...                            n_informative=2, n_redundant=0, random_state=1)
    >>> X = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
    >>> rf = RandomForestClassifier(max_depth=3, random_state=1)
    >>> feat_selector = Leshy(rf, n_estimators=16, max_iter=5, random_state=1)
    >>> _ = feat_selector.fit(X, y)
    >>> X_filtered = feat_selector.transform(X)

    References
    ----------
    See the original Boruta paper in the documentation references for more details.

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
        self.verbose = verbose
        self.keep_weak = keep_weak
        self.importance = importance

    def fit(self, X, y, sample_weight=None):
        """Select features from a DataFrame, aligning targets and weights by position.

        The supplied estimator and input data are preserved. Use ``fit`` rather
        than the removed ``select_features`` orchestration method.
        """
        start_time = time.time()
        # A failed refit must not expose the previous selection as fitted state.
        self.__dict__.pop("support_", None)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        y = validate_target(y, len(X))
        self._check_params(X, y)
        sample_weight = validate_sample_weight(sample_weight, len(X))
        self.estimator_ = clone(self.estimator)
        self.importance_ = self.importance
        self._importance_getter = {
            "shap": _get_shap_imp,
            "pimp": _get_perm_imp,
            "native": _get_imp,
        }[self.importance_]
        self.random_state_ = check_random_state(self.random_state)
        self.is_cat = is_catboost(self.estimator)
        self.is_lgb = is_lightgbm(self.estimator)
        self.cat_name = self.cat_idx = None
        self.feature_names_in_ = X.columns.to_numpy()
        self.n_features_in_ = X.shape[1]
        if self.n_estimators != "auto":
            self.estimator_.set_params(n_estimators=self.n_estimators)
        decisions, thresholds, history, self.sha_max = self._select_features(
            X, y, sample_weight
        )
        self.n_iter_ = len(history)
        self.imp_real_hist = history
        confirmed, tentative = _get_confirmed_and_tentative(decisions)
        tentative = _select_tentative(tentative, history, thresholds)
        support, support_weak = self._calculate_support(
            confirmed, tentative, self.n_features_in_
        )
        self._calculate_relative_ranking(
            self.n_features_in_, tentative, confirmed, history
        )
        self.ranking_absolutes_ = list(
            pd.DataFrame(history, columns=X.columns)
            .mean()
            .sort_values(ascending=False)
            .index
        )
        self.selected_features_ = self.feature_names_in_[support]
        self.not_selected_features_ = self.feature_names_in_[~support]
        self.n_features_ = int(support.sum())
        self.support_weak_ = support_weak
        self.running_time = time.time() - start_time
        if self.verbose > 0:
            confirmed_count = np.count_nonzero(decisions == 1)
            weak_count = np.count_nonzero(support_weak)
            print(
                f"Leshy finished running using {self.importance_} var. imp.\n"
                f"Iteration: {self.n_iter_} / {self.max_iter}\n"
                f"Confirmed: {confirmed_count}\nTentative: {weak_count}\n"
                f"Rejected: {self.n_features_in_ - confirmed_count - weak_count}"
            )
        self.support_ = support
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "support_")

    def transform(self, X):
        check_is_fitted(self, "support_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        return X[self.selected_features_]

    @mpl.rc_context(PLT_PARAMS)
    def plot_importance(self, n_feat_per_inch=5):
        """Boxplot of the variable importance, ordered by magnitude
        The max shadow variable importance illustrated by the dashed line.
        Requires to apply the fit method first.
        Colors follow fitted membership; tentative features remain yellow when
        keep_weak includes them in the selection.

        Parameters
        ----------
        n_feat_per_inch : int, default=5
            number of features to plot per inch (for scaling the figure)

        Returns
        -------
        fig : plt.figure
            the matplotlib figure object containing the boxplot
        """

        check_is_fitted(self, "support_")

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

            box_face_col = pd.Series(
                np.where(
                    self.support_weak_, YELLOW, np.where(self.support_, BLUE, "gray")
                ),
                index=self.feature_names_in_,
            )
            for patch, color in zip(bp.patches, box_face_col.loc[vimp_df.columns]):
                patch.set_facecolor(color)
                patch.set_color(color)

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
        check_is_fitted(self, "support_")
        return self.support_

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}

    def _get_tree_num(self, n_feat):
        """Estimate tree count; unlimited depth uses the historical depth-10 fallback."""
        depth = (
            self.estimator_.get_params()["max_depth"]
            if not self.is_cat
            else self.estimator_.get_param("max_depth")
        )
        if depth is None or depth <= 0:
            depth = 10
        multi = (n_feat * 2) / (np.sqrt(n_feat * 2) * depth)
        return max(1, int(multi * 100))

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
        x_sha.columns = _make_shadow_names(
            X.columns, [f"Shadow_{i}" for i in range(x_sha.shape[1])]
        )
        for col in x_sha.columns:
            x_sha[col] = x_sha[col].array.take(
                self.random_state_.permutation(len(x_sha))
            )
        imp = self._importance_getter(
            self.estimator_, pd.concat([x_cur, x_sha], axis=1), y, sample_weight
        )

        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.full(X.shape[1], np.nan)
        imp_real[x_cur_ind] = imp[:x_cur_w]

        return imp_real, imp_sha

    @staticmethod
    def _assign_hits(hit_reg, cur_imp, imp_sha_max):
        """Count wins without changing the importance vector retained for history."""
        # Reproduce the historical zero comparison for absent (rejected) features.
        hit_reg[np.nan_to_num(cur_imp[0], nan=0.0) > imp_sha_max] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        """Apply the historical binomial acceptance and rejection corrections."""
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, 0.5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, 0.5).flatten()

        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = fdrcorrection(to_accept_ps, alpha=self.alpha)[0]
            to_reject = fdrcorrection(to_reject_ps, alpha=self.alpha)[0]

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
        """Validate the current parameters, including changes through set_params."""
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")
        if self.importance not in ("shap", "pimp", "native"):
            raise ValueError(
                "importance must be 'shap', 'pimp' or 'native'; "
                "use 'shap' instead of the removed 'fastshap' mode."
            )
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, Integral)
            or self.max_iter < 2
        ):
            raise ValueError("max_iter must be an integer greater than 1.")
        if self.n_estimators != "auto" and (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, Integral)
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer or 'auto'.")
        for name, maximum in (("perc", 100), ("alpha", 1)):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(f"{name} must be finite and in (0, {maximum}].")

    def _select_features(self, X, y, sample_weight):
        """Fit fresh shadows until decisions resolve or the iteration limit is reached."""
        decisions = np.zeros(X.shape[1], dtype=int)
        hits = np.zeros(X.shape[1], dtype=int)
        thresholds, history = [], []
        with tqdm(
            total=self.max_iter, desc="Leshy iteration", disable=self.verbose <= 0
        ) as progress:
            for iteration in range(1, self.max_iter + 1):
                if not np.any(decisions == 0):
                    break
                if self.n_estimators == "auto":
                    self.estimator_.set_params(
                        n_estimators=self._get_tree_num(
                            np.count_nonzero(decisions >= 0)
                        )
                    )
                if not self.is_cat:
                    self.estimator_.set_params(
                        random_state=int(self.random_state_.randint(0, 10000))
                    )
                real, shadow = self._add_shadows_get_imps(
                    X, y, sample_weight, decisions
                )
                threshold = np.percentile(shadow, self.perc)
                history.append(real)
                thresholds.append(threshold)
                self._assign_hits(hits, (real, shadow), threshold)
                self._do_tests(decisions, hits, iteration)
                progress.update(1)
        return decisions, thresholds, np.asarray(history), thresholds[-1]

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
        support, support_weak : tuple of boolean arrays
            The selected and tentative masks.
        """
        support = np.zeros(n_feat, dtype=bool)
        support_weak = np.zeros(n_feat, dtype=bool)
        support[confirmed] = True
        support_weak[tentative] = True
        if self.keep_weak:
            support[tentative] = True
        return support, support_weak

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
        imp_history : ndarray of shape (n_iterations, n_features)
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
        imp_history_rejected = imp_history[:, not_selected] * -1

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
    imp_history: array-like of shape (n_iterations, n_features)
        Importance values for each feature in each iteration.
    sha_max_history: array-like of shape (n_iterations,)
        The history of the highest stability scores.

    Returns
    -------
    tentative: array-like of shape (n_tentative_confirmed,)
        The confirmed tentative features based on their median importance values.
    """
    tentative_median = np.median(imp_history[:, tentative], axis=0)
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
    y = validate_target(y, len(X))
    sample_weight = validate_sample_weight(sample_weight, len(X))
    if cat_feature is None:
        # detect, store and encode categorical predictors
        X, _, cat_idx = get_pandas_cat_codes(X.copy())
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

    validate_sample_weight(w_tr, len(X_tr))
    validate_sample_weight(w_tt, len(X_tt))
    if check_if_tree_based(estimator):
        try:
            # Handle LightGBM categorical features at initialization
            if is_lightgbm(estimator):
                model = estimator.fit(
                    X_tr, y_tr, sample_weight=w_tr, categorical_feature=cat_idx
                )
            elif is_catboost(estimator) or (
                "cat_feature" in estimator.fit.__code__.co_varnames
            ):
                model = estimator.fit(
                    X_tr, y_tr, sample_weight=w_tr, cat_features=cat_idx
                )
            else:
                model = estimator.fit(X_tr, y_tr, sample_weight=w_tr)

        except Exception as e:  # noqa: BLE001 - normalize third-party fit failures.
            raise ValueError(
                "Please check your X and y variable. The provided "
                "estimator cannot be fitted to your data.\n" + str(e)
            )
    else:
        raise ValueError("Not a tree based model")

    return model, X_tt, y_tt, w_tt


def _get_shap_imp(estimator, X, y, sample_weight=None, cat_feature=None):
    """Get SHAP feature importance from list or array contributions.

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
        The list of integers, columns loc, of the categorical predictors.

    Returns
    -------
    shap_imp : array
        The SHAP importance array.
    """
    estimator = clone(estimator)
    model, X_tt, _y_tt, _w_tt = _split_fit_estimator(
        estimator, X, y, sample_weight=sample_weight, cat_feature=cat_feature
    )

    if is_lightgbm(model):
        return _compute_importance(
            X_tt, model.booster_.predict(X_tt, pred_contrib=True), native=True
        )

    shap_values = shap.TreeExplainer(model).shap_values(X_tt)
    if not isinstance(shap_values, (list, np.ndarray)):
        raise ValueError(f"Unsupported SHAP values format: {type(shap_values)}")  # noqa: TRY004 - preserve the backend-output error contract.
    if isinstance(shap_values, list):
        # Keep positive binary class and first regression output, as before.
        if is_classifier(estimator):
            if len(np.unique(y)) == 2:
                shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]
    return _compute_importance(X_tt, shap_values)


def _get_perm_imp(estimator, X, y, sample_weight, cat_feature=None):
    """Private function, get signed permutation importance with weighted scoring.

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
        model, X_tt, y_tt, sample_weight=w_tt, n_repeats=5, random_state=42, n_jobs=-1
    )
    imp = perm_imp.importances_mean.ravel()
    if not np.isfinite(imp).all():
        raise ValueError("Feature importance must be finite.")
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
    y = validate_target(y, len(X))
    sample_weight = validate_sample_weight(sample_weight, len(X))
    # be sure to use an non-fitted estimator
    estimator = clone(estimator)
    try:
        if cat_feature is None:
            X, _, cat_idx = get_pandas_cat_codes(X.copy())
        else:
            cat_idx = cat_feature

        if is_lightgbm(estimator):
            estimator.fit(
                X, y, sample_weight=sample_weight, categorical_feature=cat_idx
            )
        elif is_catboost(estimator) or (
            "cat_feature" in estimator.fit.__code__.co_varnames
        ):
            estimator.fit(X, y, sample_weight=sample_weight, cat_features=cat_idx)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)
    except Exception as e:  # noqa: BLE001 - normalize third-party fit failures.
        raise ValueError(f"Estimator fitting failed: {e!s}")

    if is_lightgbm(estimator):
        imp = estimator.booster_.feature_importance(
            importance_type="gain"
        )  # LightGBM 4.x compatible
    else:
        imp = estimator.feature_importances_
    if not np.isfinite(imp).all():
        raise ValueError("Feature importance must be finite.")
    return imp


###################################
#
# BoostAGroota
#
###################################
class BoostAGroota(SelectorMixin, BaseEstimator):
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
        Divide the maximum shadow mean importance by this value. In each round,
        normalize importances per iteration, then average each real and shadow
        feature across iterations. Keep real means at or above this threshold.
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
        Permutation scoring uses validation sample weights. Each iteration's
        signed scores are divided by their absolute total; an all-zero vector
        stays zero. Zero-total SHAP and native iterations also contribute zeros.
        Only positive mean scores can be selected. LightGBM uses native SHAP
        contributions; other estimators use TreeExplainer. The former
        'fastshap' mode is removed; use 'shap' instead.
    random_state : int, RandomState instance or None, default=None
        Controls shadow permutations. An integer replays each fit; None uses
        NumPy's global RNG. The supplied learner's seed is preserved; seed it
        separately for reproducible stochastic training.

    Attributes
    ----------
    importance_ : str
        Importance method used by the fitted selector.
    random_state_ : RandomState
        Runtime RNG for shadow permutations.
    n_features_in_ : int
        Number of input features.
    n_rounds_ : int
        Number of completed reduction rounds.
    selected_features_ : array
        The list of columns to keep.
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1, and rejected features are assigned
        rank 2.
    ranking_absolutes_ : list of length n_features
        All input feature names ordered by descending first-round mean importance.
        Ties retain input order. This model-dependent ordering does not combine
        importances normalized over different candidate sets in later rounds.
    sha_cutoff_df : dataframe
        Final-round normalized iteration importances only: real candidates first,
        then their shadows. Features eliminated in earlier rounds are absent.
        This table and the plot share the scale of mean_shadow.
    mean_shadow : float
        Maximum shadow mean divided by cutoff in the final round.

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
        random_state=None,
    ):
        self.estimator = estimator
        self.cutoff = cutoff
        self.iters = iters
        self.max_rounds = max_rounds
        self.delta = delta
        self.silent = silent
        self.importance = importance
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Select features without changing the supplied frame or learner.

        Targets and observation weights align by position. Each round averages
        normalized importances and keeps positive real means at or above the
        maximum shadow mean divided by ``cutoff``.
        """
        start_time = time.time()
        # A failed refit must not expose the previous selection as fitted state.
        self.__dict__.pop("support_", None)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        y = validate_target(y, len(X))
        self._check_params(X, y)
        sample_weight = validate_sample_weight(sample_weight, len(X))
        self.importance_ = self.importance
        self.random_state_ = check_random_state(self.random_state)
        get_importance = {
            "shap": _get_shap_imp,
            "pimp": _get_perm_imp,
            "native": _get_imp,
        }[self.importance_]
        new_x, cat_names, cat_idx = get_pandas_cat_codes(X.copy())

        with tqdm(
            total=self.max_rounds, desc="BoostAGroota round", disable=self.silent
        ) as pbar:
            for round_index in range(1, self.max_rounds + 1):
                scores, columns = self._run_round(
                    new_x, y, sample_weight, cat_idx, get_importance, round_index
                )
                means = scores.mean(axis=1)
                n_real = new_x.shape[1]
                threshold = means[n_real:].max() / self.cutoff
                keep = (means[:n_real] >= threshold) & (means[:n_real] > 0)
                keep_vars = new_x.columns[keep]
                if round_index == 1:
                    ranking = list(
                        new_x.columns[np.argsort(-means[:n_real], kind="stable")]
                    )
                pbar.update(1)
                if len(keep_vars) == 0 or len(keep_vars) / n_real > 1 - self.delta:
                    break
                if round_index < self.max_rounds:
                    new_x = new_x[keep_vars].copy()
                    cat_idx = [
                        new_x.columns.get_loc(col)
                        for col in (cat_names or [])
                        if col in new_x.columns
                    ]

        self.feature_names_in_ = X.columns.to_numpy()
        self.n_features_in_ = X.shape[1]
        self.n_rounds_ = round_index
        self.selected_features_ = keep_vars.to_numpy()
        self.ranking_absolutes_ = ranking
        self.sha_cutoff_df = pd.DataFrame(
            scores.T,
            index=[f"fscore{i + 1}" for i in range(self.iters)],
            columns=columns,
        )
        self.mean_shadow = threshold
        support = X.columns.isin(keep_vars)
        self.ranking_ = np.where(support, 1, 2)
        if not self.silent:
            elapsed = (time.time() - start_time) / 60
            print(
                f"BoostAGroota ran successfully! Algorithm went through {round_index} rounds."
            )
            print(
                f"The feature selection BoostAGroota running time is {elapsed:8.2f} min"
            )
        self.support_ = support
        return self

    def _check_params(self, X, y):
        """Validate current options, including changes through set_params."""
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")
        if self.estimator is None or not check_if_tree_based(self.estimator):
            raise ValueError(
                "estimator must be a compatible tree-based estimator, such as LGBMRegressor."
            )
        if self.importance not in ("shap", "pimp", "native"):
            raise ValueError(
                "importance must be 'shap', 'pimp' or 'native'; "
                "use 'shap' instead of the removed 'fastshap' mode."
            )
        for name in ("iters", "max_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("cutoff", "delta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
                or (name == "delta" and value > 1)
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
                    + (" and at most 1." if name == "delta" else ".")
                )
        if self.delta < 0.02:
            warnings.warn(
                "WARNING: Setting a delta below 0.02 may not converge on a solution."
            )
        if self.importance == "native":
            warnings.warn(
                "[BoostAGroota]: using native variable importance might break the FS"
            )

    def __sklearn_is_fitted__(self):
        return hasattr(self, "support_")

    def _run_round(self, X, y, weight, cat_idx, get_importance, round_index):
        """Collect one round's normalized vectors, with real features first."""
        n_real = X.shape[1]
        augmented_cat_idx = (
            cat_idx + [idx + n_real for idx in cat_idx] if cat_idx is not None else None
        )
        scores = np.empty((2 * n_real, self.iters), dtype=float)
        for i in range(self.iters):
            augmented, _ = _create_shadow(X, random_state=self.random_state_)
            importance = np.asarray(
                get_importance(
                    self.estimator,
                    augmented,
                    y,
                    sample_weight=weight,
                    cat_feature=augmented_cat_idx,
                ),
                dtype=float,
            )
            if importance.shape != (2 * n_real,):
                raise ValueError(
                    "Feature importance must have one value per real and shadow feature."
                )
            if not np.isfinite(importance).all():
                raise ValueError("Feature importance must be finite.")
            total = (
                np.abs(importance).sum()
                if self.importance_ == "pimp"
                else importance.sum()
            )
            if not np.isfinite(total):
                raise ValueError("Total feature importance must be finite.")
            scores[:, i] = importance / total if total > 0 else 0.0
            if not self.silent:
                print("Round: ", round_index, " iteration: ", i + 1)
        return scores, augmented.columns

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def transform(self, X):
        check_is_fitted(self, "support_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")
        return X[self.selected_features_]

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}

    @mpl.rc_context(PLT_PARAMS)
    def plot_importance(self, n_feat_per_inch=5):
        """
        Boxplot of final-round candidates, ordered by mean importance.
        The dashed line is that round's maximum shadow mean divided by cutoff.
        Colors identify selected and rejected features by name.
        Requires to apply the fit method first.

        Parameters
        ----------
        n_feat_per_inch : int, default=5
            Number of features to plot per inch (for scaling the figure).

        Returns
        -------
        fig : plt.figure or None
            The matplotlib figure object, or None if there is no candidate history.
        """
        check_is_fitted(self, "support_")

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

        for patch, name in zip(bp.patches, real_df.columns):
            color = BLUE if name in self.selected_features_ else "gray"
            patch.set_facecolor(color)
            patch.set_color(color)

        custom_lines = [
            Line2D([0], [0], color=BLUE, lw=5),
            Line2D([0], [0], color="gray", lw=5),
            Line2D([0], [0], linestyle="--", color=RED, lw=2),
        ]
        bp.legend(
            custom_lines, ["confirmed", "rejected", "sha. max"], loc="lower right"
        )

        ax.axvline(x=self.mean_shadow, linestyle="--", color=RED)
        ax.set_title("BoostAGroota final-round candidate importance")

        return fig


############################################
# Helper Functions to do the Heavy Lifting
############################################


def _make_shadow_names(columns, names):
    """Keep generated names distinct from real columns and other shadows."""
    used = set(columns)
    shadow_names = []
    for name in names:
        while name in used:
            name += "_"
        shadow_names.append(name)
        used.add(name)
    return shadow_names


def _create_shadow(X_train, random_state=None):
    """Create shadow features by making copies of all X variables and randomly shuffling them.

    Parameters
    ----------
    X_train : pd.DataFrame
        The dataframe to create shadow features on.
    random_state : int, RandomState instance or None, default=None
        Controls shadow permutations; None uses NumPy's global RNG.

    Returns
    -------
    pd.DataFrame
        A dataframe that is twice the width of X_train and contains the shadow features, along with a list of the shadow feature names.
    """
    rng = check_random_state(random_state)
    X_shadow = pd.DataFrame(
        {
            c: X_train[c].array.take(rng.permutation(len(X_train)))
            for c in X_train.columns
        },
        index=X_train.index,
    )
    # Rename the shadow variables
    shadow_names = _make_shadow_names(
        X_train.columns, [f"ShadowVar{i + 1}" for i in range(X_train.shape[1])]
    )
    X_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_x = pd.concat([X_train, X_shadow], axis=1)
    return new_x, shadow_names


###################################
#
# GrootCV
#
###################################


class GrootCV(SelectorMixin, BaseEstimator):
    """
    GrootCV is a feature selection method based on cross-validation with lightGBM.

    A shuffled copy of the predictors matrix is added (shadows) to the original set of predictors.
    LightGBM is fitted using repeated cross-validation. Each fold's SHAP
    importance is the row mean of absolute contributions, summed over outputs
    for multi-output models, then normalized across real and shadow features; a
    zero-total fold contributes zeros. Each feature's median is computed across
    all folds and repeats. Keep positive real medians at or above the maximum
    shadow median divided by cutoff.
    Explanations use training rows. Validation rows drive early stopping when
    supported by the boosting type.
    This empirical threshold does not establish false-discovery control.

    Parameters
    ----------
    objective : str or callable, default=None
        The objective function to use in lightGBM. If None, it uses the objective specified in `lgbm_params`.
        Recognized multiclass classification objectives resolve to softmax;
        binary is also promoted to softmax when the target has more than two classes.
    cutoff : float, default=1
        Divide the maximum of per-shadow fold/repeat medians by this value.
    n_folds : int, default=5
        The number of folds for cross-validation. Default folds are repeated
        stratified folds for recognized classification objectives and repeated
        ordinary folds otherwise. Default stratification requires at least two
        classes and at least n_folds observations per class; reduce n_folds or
        supply folds when this is infeasible.
    folds : Optional[Union[Iterable[Tuple[np.ndarray, np.ndarray]]
        (generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None))
        If generator or iterator, it should yield the train and test indices for each fold. If object, it should be one of the scikit-learn
        splitter classes (https://scikit-learn.org/stable/modules/classes.html#splitter-classes) and have split method.
        This argument has highest priority over other data split arguments.
        Iterators and generators are consumed by fit; use a list of index pairs
        or a splitter for refitting. Empty or exhausted folds raise ValueError.
    n_iter : int, default=5
        Number of cross-validation repeats; take each feature's median over
        all folds and repeats.
    silent : bool, default=True
        Set to True if you don't want to see the GrootCV output printed.
    rf : bool, default=False
        If True, use random forest for calculating feature importances; otherwise, use lightGBM.
    n_jobs : int, default=0
        LightGBM's number of OpenMP threads. If 0, use the OpenMP default.
    lgbm_params : dict, default=None
        The parameters for the LightGBM model. Boosting-round limits and their
        aliases follow LightGBM precedence; the fallback ceiling is 10,000
        rounds, with 20-round early stopping where supported. Set an explicit
        ``num_iterations`` for DART, which disables early stopping.
    random_state : int, RandomState instance or None, default=2652124
        Controls shadows and default repeated folds. An integer replays each
        fit; a RandomState instance advances across fits; None uses NumPy's
        global RNG. Supplied splitters retain their own random-state policy.
        LightGBM learner seeds remain controlled by lgbm_params.

    Attributes
    ----------
    random_state_ : RandomState
        Runtime RNG for shadow permutations and default folds.
    selected_features_ : ndarray
        The list of columns to keep as selected features.
    cv_df : pd.DataFrame
        Normalized importance per fold/repeat in fscore columns and each
        feature's median in Med. Summary columns never enter the aggregation.
    sha_cutoff : float
        The threshold below which the predictors are rejected.
    ranking_absolutes_ : list
        The absolute feature ranking as ordered by the selection process.
    ranking_ : ndarray
        The feature ranking, where 1 corresponds to selected features and 2 to rejected features.

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
        folds=None,
        n_iter=5,
        silent=True,
        rf=False,
        n_jobs=0,
        lgbm_params=None,
        random_state=2652124,
    ):
        self.objective = objective
        self.cutoff = cutoff
        self.n_folds = n_folds
        self.folds = folds
        self.n_iter = n_iter
        self.silent = silent
        self.rf = rf
        self.cv_df = None
        self.sha_cutoff = None
        self.ranking_absolutes_ = None
        self.ranking_ = None
        self.lgbm_params = lgbm_params
        self.n_jobs = n_jobs
        self.random_state = random_state

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

        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        y = validate_target(y, len(X))
        sample_weight = validate_sample_weight(sample_weight, len(X))
        if sample_weight is not None:
            sample_weight = pd.Series(sample_weight)

        random_state = check_random_state(self.random_state)
        selected_features, cv_df, sha_cutoff = _reduce_vars_lgb_cv(
            X,
            y,
            objective=self.objective,
            cutoff=self.cutoff,
            n_folds=self.n_folds,
            folds=self.folds,
            n_iter=self.n_iter,
            silent=self.silent,
            weight=sample_weight,
            rf=self.rf,
            lgbm_params=self.lgbm_params,
            n_jobs=self.n_jobs,
            random_state=random_state,
        )

        feature_names = X.columns.to_numpy()
        selected_features = selected_features.to_numpy()
        support = np.asarray([c in selected_features for c in feature_names])
        ranking_absolutes = list(
            cv_df.set_index("feature")["Med"]
            .loc[feature_names]
            .sort_values(ascending=False)
            .index
        )
        self.random_state_ = random_state
        self.feature_names_in_ = feature_names
        self.selected_features_ = selected_features
        self.cv_df = cv_df
        self.sha_cutoff = sha_cutoff
        self.ranking_ = np.where(support, 1, 2)
        self.ranking_absolutes_ = ranking_absolutes
        self.support_ = support
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "support_")

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
        check_is_fitted(self)
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
        real_df = b_df.loc[:, self.feature_names_in_].copy()

        real_df = real_df.reindex(
            real_df.select_dtypes(include=[np.number])
            .median()
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
    folds,
    n_folds,
    cutoff,
    n_iter,
    silent,
    weight,
    rf,
    lgbm_params=None,
    n_jobs=0,
    random_state=2652124,
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
    folds :
        (generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None))
        If generator or iterator, it should yield the train and test indices for each fold. If object, it should be one of the scikit-learn
        splitter classes (https://scikit-learn.org/stable/modules/classes.html#splitter-classes) and have split method.
        This argument has highest priority over other data split arguments.
    nfold : int
        Number of folds in CV.
    cutoff : float
            Divide the maximum of per-shadow fold/repeat medians by this value.
    n_iter : int
            The number of repetition of the cross-validation, smooth out the feature importance noise
    silent : bool
            Set to True if don't want to see the BoostARoota output printed.
            Will still show any errors or warnings that may occur
    weight : pd.series
            sample_weight, if any
    rf : bool, default=False
            the lightGBM implementation of the random forest
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

    X, category_cols, _ = get_pandas_cat_codes(X.copy())
    category_cols = category_cols or []
    params = _set_lgb_parameters(
        X=X,
        y=y,
        objective=objective,
        rf=rf,
        silent=silent,
        n_jobs=n_jobs,
        lgbm_params=lgbm_params,
    )

    rng = check_random_state(random_state)
    if folds is None:
        splitter = RepeatedKFold
        if params["objective"] in ("binary", "binary_logloss", "softmax"):
            counts = np.unique(y, return_counts=True)[1]
            if len(counts) < 2 or counts.min() < n_folds:
                raise ValueError(
                    "Default GrootCV stratification requires at least two classes "
                    "and at least n_folds observations per class. Reduce n_folds "
                    "or supply custom folds."
                )
            splitter = RepeatedStratifiedKFold
        # Keep lazy later repeats independent of shadow permutation counts.
        folds = splitter(n_splits=n_folds, n_repeats=n_iter, random_state=deepcopy(rng))

    splits = folds.split(X, y) if hasattr(folds, "split") else folds
    total = folds.get_n_splits(X, y) if hasattr(folds, "get_n_splits") else None
    fold_scores = []
    shadow_names = None
    for iter, (tridx, validx) in enumerate(
        tqdm(splits, total=total, desc="Cross Validation", disable=silent)
    ):
        X_train, X_val, y_train, y_val, weight_tr, weight_val = _split_data(
            X, y, tridx, validx, weight
        )

        # Create the shadow variables and run the model to obtain importances
        new_x_tr, shadow_names = _create_shadow(X_train, random_state=rng)
        new_x_val, _ = _create_shadow(X_val, random_state=rng)

        bst = _train_lgb_model(
            new_x_tr,
            y_train,
            weight_tr,
            new_x_val,
            y_val,
            weight_val,
            category_cols=category_cols
            + [shadow_names[X.columns.get_loc(col)] for col in category_cols],
            early_stopping_rounds=20,
            silent=silent,
            **params,
        )

        importance = _compute_importance(
            new_x_tr, bst.predict(new_x_tr, pred_contrib=True), native=True
        )
        importance_total = importance.sum()
        if not np.isfinite(importance_total):
            raise ValueError("Total feature importance must be finite.")
        fold_scores.append(
            importance / importance_total if importance_total > 0 else importance
        )
        if not silent and iter % n_folds == 0:
            print(
                "Groot iteration: ", iter // n_folds, " with " + str(n_folds) + " folds"
            )

    if shadow_names is None:
        raise ValueError(
            "folds must yield at least one train/validation split. "
            "For refitting, supply a reusable list of index pairs, a splitter, "
            "or a new iterator."
        )

    df = (
        pd.DataFrame(
            np.asarray(fold_scores).T,
            index=new_x_tr.columns,
            columns=[f"fscore{i}" for i in range(len(fold_scores))],
        )
        .rename_axis("feature")
        .reset_index()
    )
    df["Med"] = np.median(fold_scores, axis=0)
    # Retain the historical outer-merge row order, including mixed feature labels.
    df = pd.merge(pd.DataFrame({"feature": X.columns}), df, on="feature", how="outer")
    # Split them back out
    real_vars = df[~df["feature"].isin(shadow_names)]
    shadow_vars = df[df["feature"].isin(shadow_names)]

    cutoff_shadow = shadow_vars["Med"].max() / cutoff
    real_vars = real_vars[
        (real_vars.Med.values >= cutoff_shadow) & (real_vars.Med.values > 0)
    ]

    return real_vars["feature"], df, cutoff_shadow


def _set_lgb_parameters(
    X: np.ndarray,
    y: np.ndarray,
    objective: str,
    rf: bool,
    silent: bool,
    n_jobs: int = 0,
    lgbm_params: Optional[dict] = None,
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

    params = lgbm_params.copy() if lgbm_params is not None else {}
    if objective is None:
        objective = params.get("objective")

    params["objective"] = objective
    params["verbosity"] = -1
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
    if objective in clf_losses and objective not in ["binary", "binary_logloss"]:
        params["objective"] = "softmax"
        params["num_class"] = len(np.unique(y))

    if rf:
        feat_frac = (
            np.sqrt(n_feat) / n_feat
            if params["objective"] in ["softmax", "binary"]
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

    if objective in clf_losses:
        y = y.astype(int)
        y_freq_table = pd.Series(y.fillna(0)).value_counts(normalize=True)
        n_classes = y_freq_table.size
        if n_classes > 2 and params["objective"] != "softmax":
            params["objective"] = "softmax"
            params["num_class"] = len(np.unique(y))
            if not silent:
                print("Multi-class task, setting objective to softmax")
        main_class = y_freq_table.iloc[0]
        if not silent:
            print("GrootCV: classification with unbalance classes")
        if main_class > 0.8:
            params.update({"is_unbalance": True})

    params.update({"num_threads": n_jobs})

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
    y = validate_target(y, len(X))
    checked_weight = validate_sample_weight(weight, len(X))
    if weight is not None:
        if not isinstance(weight, pd.Series):
            weight = pd.Series(checked_weight)
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
    validate_target(y_train, len(X_train))
    validate_target(y_val, len(X_val))
    validate_sample_weight(weight_tr, len(X_train))
    validate_sample_weight(weight_val, len(X_val))
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
    silent=True,
    **params,
):
    """
    Train a LightGBM model with validation-based early stopping.

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
        round(s) to continue training.
    silent : bool, default=True
        Disable routine early-stopping messages; warnings remain visible.
    **params : dict
        Other parameters passed to the LightGBM model.

    Returns
    -------
    lightgbm.Booster
        The trained model; best_iteration records the early-stopping result.
    """
    d_train = lgb.Dataset(X_train, label=y_train, weight=weight_train)
    d_valid = lgb.Dataset(X_val, label=y_val, weight=weight_val)

    if category_cols:
        d_train.set_categorical_feature(category_cols)
        d_valid.set_categorical_feature(category_cols)

    return lgb.train(
        params,
        train_set=d_train,
        num_boost_round=10000,
        valid_sets=[d_train, d_valid],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=not silent)],
    )


def _compute_importance(new_x_tr, shap_matrix, *, native=False):
    """Compute feature importance scores using SHAP values.

    Parameters
    ----------
    new_x_tr : numpy.ndarray
        The training dataset after being processed.
    shap_matrix : numpy.ndarray
        TreeExplainer contributions with shape (rows, features) or
        (rows, features, outputs), or a list of per-output matrices.
        Native LightGBM contributions instead contain a flattened block of
        features plus a bias column per output.
    native : bool, default=False
        Whether contributions use the native LightGBM layout.

    Returns
    -------
    numpy.ndarray
        Importance scores in input-column order.
        Scores are row means of absolute contributions, summed over outputs
        for multi-output models. Layout, not version/objective/backend, sets
        the output axis.
    """
    values = (
        np.stack(shap_matrix, axis=-1)
        if isinstance(shap_matrix, list)
        else np.asarray(shap_matrix)
    )
    if not np.isfinite(values).all():
        raise ValueError("SHAP contributions must be finite.")
    if native:
        n_rows, n_features = new_x_tr.shape
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
        # Every output block has its own bias; TreeExplainer omits these columns.
        values = values.reshape(n_rows, -1, block_size)[:, :, :-1].transpose(0, 2, 1)
    if values.ndim not in (2, 3) or values.shape[:2] != new_x_tr.shape:
        raise ValueError("SHAP values must have shape (rows, features[, outputs]).")
    values = np.abs(values)
    if values.ndim == 3:
        values = values.sum(axis=2)
    shap_imp = values.mean(axis=0)
    if not np.isfinite(shap_imp).all():
        raise ValueError("Feature importance must be finite.")

    return shap_imp
