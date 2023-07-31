"""Benchmark Feature Selection

This module provides utilities for comparing and benchmarking feature selection methods

Module Structure:
-----------------
- ``sklearn_pimp_bench``: function for comparing using the sklearn permutation importance
- ``compare_varimp``: function for comparing using 3 kinds of var.imp.
- ``highlight_tick``: function for highlighting specific (genuine or noise for instance) predictors in the importance chart
"""

from __future__ import print_function, division

import itertools
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from sklearn.base import clone

from .preprocessing import OrdinalEncoderPandas


def sklearn_pimp_bench(model, X, y, task="regression", sample_weight=None):
    """Benchmark using sklearn permutation importance, works for regression and classification.

    Parameters
    ----------
    model: object
        An estimator that has not been fitted, sklearn compatible.
    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.
    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or None for unsupervised.
    task : str, optional
        kind of task, either 'regression' or 'classification", by default 'regression'
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

    # for lightGBM cat feat as contiguous int
    # https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
    # same for Random Forest and XGBoost (OHE leads to deep and sparse trees).
    # For illustrations, see
    # https://towardsdatascience.com/one-hot-encoding-is-making-
    # your-tree-based-ensembles-worse-heres-why-d64b282b5769

    # X, cat_var_df, inv_mapper, mapper = cat_var(X)
    X = OrdinalEncoderPandas().fit_transform(X)

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

    # lightgbm faster and better than RF

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
        result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx]
    )
    ax.set_title("Permutation Importances (test set)")
    ax.tick_params(axis="both", which="major", labelsize=9)
    fig.tight_layout()
    indices = [i for i, s in enumerate(X_test.columns[sorted_idx]) if "random" in s]
    [fig.gca().get_yticklabels()[idx].set_color("red") for idx in indices]
    indices = [i for i, s in enumerate(X_test.columns[sorted_idx]) if "genuine" in s]
    [fig.gca().get_yticklabels()[idx].set_color("green") for idx in indices]
    plt.show()
    return fig


def compare_varimp(feat_selector, models, X, y, sample_weight=None):
    """Utility function to compare the results for the three possible kind of feature importance

    Parameters
    ----------
    feat_selector : object
        an instance of either Leshy, BoostaGRoota or GrootCV
    models : list of objects
        list of tree based scikit-learn estimators
    X : pd.DataFrame, shape (n_samples, n_features)
        the predictors frame
    y : pd.Series
        the target (same length as X)
    sample_weight : None or pd.Series, optional
        sample weights if any, by default None
    """

    varimp_list = ["shap", "fastshap", "pimp", "native"]
    for model, varimp in itertools.product(models, varimp_list):
        print(
            "=" * 20
            + " "
            + str(feat_selector.__class__.__name__)
            + " - testing: {mod:>25} for var.imp: {vimp:<15} ".format(
                mod=str(model.__class__.__name__), vimp=varimp
            )
            + "=" * 20
        )
        # change the varimp
        feat_selector.importance = varimp
        # change model
        mod_clone = clone(model, safe=True)
        feat_selector.estimator = mod_clone
        # fit the feature selector
        feat_selector.fit(X=X, y=y, sample_weight=sample_weight)
        # print the results
        print(feat_selector.selected_features_)
        fig = feat_selector.plot_importance(n_feat_per_inch=5)

        if fig is not None:
            # highlight synthetic random variable
            fig = highlight_tick(figure=fig, str_match="random")
            fig = highlight_tick(figure=fig, str_match="genuine", color="green")
            plt.show()


def highlight_tick(str_match, figure, color="red", axis="y"):
    """Highlight the x/y tick-labels if they contains a given string

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
