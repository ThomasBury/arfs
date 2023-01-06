"""
This module provides a function for creating the summary report of a FS pipeline

**The module structure is the following:**

- The ``make_fs_summary`` main function for creating the summary
- The ``highlight_discarded`` function for creating style for the pd.DataFrame

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
    return ["background-color: #d65f5f" if v else "background-color: #33a654" for v in is_X]

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
    for selector_name in selector_pipe.named_steps.keys():
        if hasattr(selector_pipe.named_steps[selector_name], "support_"):
            feature_in = selector_pipe.named_steps[selector_name].feature_names_in_
            to_drop = list(set(selector_pipe.named_steps[selector_name].feature_names_in_) - set(selector_pipe.named_steps[selector_name].get_feature_names_out()))
            tag_df[selector_name] = np.where(tag_df["predictor"].isin(to_drop), 0, 1) * np.where(tag_df["predictor"].isin(feature_in), 1, np.nan)

    col_to_apply_style = tag_df.columns[1:]
    tag_df = tag_df.style.apply(highlight_discarded, subset=col_to_apply_style).applymap(lambda x: '' if x==x else 'background-color: #ffa500').format(precision=0)
    return tag_df
