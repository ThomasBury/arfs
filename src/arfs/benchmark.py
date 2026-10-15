"""Benchmark Feature Selection

This module provides utilities for comparing and benchmarking feature selection methods

Module Structure:
-----------------
- ``sklearn_pimp_bench``: function for comparing using the sklearn permutation importance
- ``compare_varimp``: function for comparing using possible 4 kinds of variable importance
- ``highlight_tick``: function for highlighting specific (genuine or noise for instance) predictors in the importance chart
"""

import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .feature_selection.allrelevant import BoostAGroota, Leshy
from .preprocessing import OrdinalEncoderPandas
from .utils import validate_sample_weight, validate_target


def sklearn_pimp_bench(model, X, y, task="regression", sample_weight=None):
    """Benchmark using sklearn permutation importance, works for regression and classification.

    Parameters
    ----------
    model: object
        An estimator that has not been fitted, sklearn compatible.
    X : ndarray or DataFrame, shape (n_samples, n_features)
        Predictors, split before fitting the category encoder. Two-dimensional
        arrays receive names x0, x1, ...; DataFrame names are preserved.
        Held-out unknown categories use the encoder's default NaN code, so the
        estimator must support missing values when these occur.
    y : array-like, shape (n_samples,)
        Supervised targets.
    task : str, optional
        kind of task, either 'regression' or 'classification', by default 'regression'
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights, by default None

    Returns
    -------
    plt.figure
        the figure corresponding to the feature selection

    Raises
    ------
    ValueError
        if task is not 'regression' or 'classification'
    """

    if isinstance(X, np.ndarray) and X.ndim == 2:
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    elif not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame or a two-dimensional ndarray.")
    validate_target(y, len(X))
    sample_weight = validate_sample_weight(sample_weight, len(X))

    if task == "regression":
        stratify = None
    elif task == "classification":
        stratify = y
    else:
        raise ValueError("`task` should be either 'regression' or 'classification' ")

    if sample_weight is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weight, stratify=stratify, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, random_state=42
        )
        w_train, w_test = None, None

    validate_sample_weight(w_train, len(X_train))
    validate_sample_weight(w_test, len(X_test))
    encoder = OrdinalEncoderPandas().fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)
    model.fit(X_train, y_train, sample_weight=w_train)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=2,
        sample_weight=w_test,
    )

    sorted_idx = result.importances_mean.argsort()
    # Plot (5 predictors per inch)
    fig, ax = plt.subplots(figsize=(16, X.shape[1] / 5))
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        tick_labels=X_test.columns[sorted_idx],
    )
    ax.set_title("Permutation Importances (test set)")
    ax.tick_params(axis="both", which="major", labelsize=9)
    fig.tight_layout()
    highlight_tick("random", fig)
    highlight_tick("genuine", fig, color="green")
    plt.show()
    return fig


def compare_varimp(feat_selector, models, X, y, sample_weight=None):
    """Print selections and plot SHAP, permutation and native importance.

    Each model/method pair gets a fresh selector and estimator. Constructor
    parameters are copied from feat_selector; integer seeds are reused and
    RandomState instances are copied at their current state for each pair.
    None and BoostAGroota's global shadow RNG retain their usual unseeded
    behavior. The supplied selector and models are not fitted or modified.
    This is not a held-out predictive-performance comparison.

    Parameters
    ----------
    feat_selector : object
        An instance of Leshy or BoostAGroota. GrootCV uses its own LightGBM
        training and importance path and cannot compare these configurations.
    models : list of objects
        list of tree based scikit-learn estimators
    X : pd.DataFrame, shape (n_samples, n_features)
        the predictors frame
    y : pd.Series
        the target (same length as X)
    sample_weight : None or pd.Series, optional
        sample weights if any, by default None
    Raises
    ------
    TypeError
        If feat_selector is not Leshy or BoostAGroota.
    """

    if not isinstance(feat_selector, (Leshy, BoostAGroota)):
        raise TypeError(
            "feat_selector must be Leshy or BoostAGroota; GrootCV is unsupported."
        )

    varimp_list = ["shap", "pimp", "native"]
    for model, varimp in itertools.product(models, varimp_list):
        print(
            "=" * 20
            + " "
            + str(feat_selector.__class__.__name__)
            + f" - testing: {model.__class__.__name__!s:>25} for var.imp: {varimp:<15} "
            + "=" * 20
        )
        params = clone(feat_selector.get_params(deep=False), safe=False)
        params["importance"] = varimp
        params["estimator"] = clone(model)
        selector = type(feat_selector)(**params)
        # fit the feature selector
        selector.fit(X=X, y=y, sample_weight=sample_weight)
        # print the results
        print(selector.selected_features_)
        fig = selector.plot_importance(n_feat_per_inch=5)

        if fig is not None:
            # highlight synthetic random variable
            fig = highlight_tick(figure=fig, str_match="random")
            fig = highlight_tick(figure=fig, str_match="genuine", color="green")
            plt.show()


def highlight_tick(str_match, figure, color="red", axis="y"):
    """Highlight the x/y tick-labels if they contain a given string

    Parameters
    ----------
    str_match : str
        the substring to match
    figure : object
        the matplotlib figure
    color : str, optional
        the matplotlib color for highlighting tick-labels, by default 'red'
    axis : str, optional
        axis to use for highlighting, by default 'y'

    Returns
    -------
    plt.figure
        the modified matplotlib figure

    Raises
    ------
    ValueError
        if axis is not 'x' or 'y'
    """

    if axis == "y":
        labels = [item.get_text() for item in figure.gca().get_yticklabels()]
        indices = [i for i, s in enumerate(labels) if str_match in s]
        [figure.gca().get_yticklabels()[idx].set_color(color) for idx in indices]
    elif axis == "x":
        labels = [item.get_text() for item in figure.gca().get_xticklabels()]
        indices = [i for i, s in enumerate(labels) if str_match in s]
        [figure.gca().get_xticklabels()[idx].set_color(color) for idx in indices]
    else:
        raise ValueError("`axis` should be a string, either 'y' or 'x'")

    return figure
