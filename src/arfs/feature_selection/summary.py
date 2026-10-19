"""Feature Selection Summary Module

This module provides a function for creating the summary report of a FS pipeline

Module Structure:
-----------------
- ``make_fs_summary`` main function for creating the summary
- ``highlight_discarded`` function for creating style for the pd.DataFrame
"""

import numpy as np
import pandas as pd


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

    Returns
    -------
    pandas.io.formats.style.Styler
        The styled summary table. Jinja2 is a runtime dependency for this return type.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.pipeline import Pipeline
    >>> from arfs.feature_selection import (
    ...     MissingValueThreshold, VariableImportance, make_fs_summary)
    >>> X, y = make_regression(n_samples=80, n_features=4,
    ...                        n_informative=2, random_state=1)
    >>> X = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
    >>> pipeline = Pipeline([
    ...     ('missing', MissingValueThreshold()),
    ...     ('lowimp', VariableImportance(fastshap=False, verbose=-1,
    ...         lgb_kwargs={'objective': 'rmse', 'metric': 'rmse',
    ...                     'n_estimators': 10, 'num_threads': 1}))])
    >>> _ = pipeline.fit(X, y)
    >>> fs_summary_df = make_fs_summary(pipeline)

    Pass LightGBM objective and metric options through ``lgb_kwargs``; they
    are not top-level VariableImportance arguments. For count responses, for
    example, use ``lgb_kwargs={'objective': 'poisson', 'metric': 'poisson'}``.
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
        .map(lambda x: "background-color: #f57505" if pd.isna(x) else "")
        .format(precision=0)
    )

    return style
