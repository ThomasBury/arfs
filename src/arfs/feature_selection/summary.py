"""Feature Selection Summary Module

This module provides a function for creating the summary report of a FS pipeline

Module Structure:
-----------------
- ``make_fs_summary`` main function for creating the summary
- ``highlight_discarded`` function for creating style for the pd.DataFrame
"""

import pandas as pd
import numpy as np


def highlight_discarded(s):
    """highlight X in red and V in green.

    Parameters
    ----------
    s : array-like of shape (n_features,)
        the boolean array for defining the style


    """
    is_X = s == 0
    return [
        "background-color: #ba0202" if v else "background-color: #0c8a30" for v in is_X
    ]


def make_fs_summary(selector_pipe):
    """make_fs_summary makes a summary dataframe highlighting at which step a
    given predictor has been rejected (if any).

    Parameters
    ----------
    selector_pipe : sklearn.pipeline.Pipeline
        the feature selector pipeline.

    Examples
    --------
    >>> groot_pipeline = Pipeline([
    ... ('missing', MissingValueThreshold()),
    ... ('unique', UniqueValuesThreshold()),
    ... ('cardinality', CardinalityThreshold()),
    ... ('collinearity', CollinearityThreshold(threshold=0.5)),
    ... ('lowimp', VariableImportance(eval_metric='poisson', objective='poisson', verbose=2)),
    ... ('grootcv', GrootCV(objective='poisson', cutoff=1, n_folds=3, n_iter=5))])
    >>> groot_pipeline.fit_transform(
        X=df[predictors],
        y=df[target],
        lowimp__sample_weight=df[weight],
        grootcv__sample_weight=df[weight])
    >>> fs_summary_df = make_fs_summary(groot_pipeline)
    """
    tag_df = pd.DataFrame({"predictor": selector_pipe[0].feature_names_in_})
    for selector_name, selector in selector_pipe.named_steps.items():
        if hasattr(selector, "support_"):
            feature_in = selector.feature_names_in_
            to_drop = list(set(feature_in) - set(selector.get_feature_names_out()))
            tag_df[selector_name] = np.where(
                tag_df["predictor"].isin(to_drop), 0, 1
            ) * np.where(tag_df["predictor"].isin(feature_in), 1, np.nan)
        else:
            tag_df[selector_name] = np.nan

    style = (
        tag_df.style.apply(highlight_discarded, subset=tag_df.columns[1:])
        .applymap(lambda x: "" if x == x else "background-color: #f57505")
        .format(precision=0)
    )

    return style
