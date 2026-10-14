import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.model_selection import KFold, check_cv

from arfs.feature_selection.lasso import (
    EnetGLM,
    grid_search_cv,
    weighted_cross_val_score,
)
from arfs.gbm import GradientBoosting

X = pd.DataFrame({"x": np.linspace(-2, 2, 60), "noise": np.sin(np.arange(60))})
y = 0.1 + 0.8 / (1 + np.exp(-X.x.to_numpy()))
weight = np.linspace(0.5, 2.0, len(X))
assert type(check_cv(5, y, classifier=True)) is KFold
alpha = 0.001
expected = []
for train, test in KFold(5).split(X, y):
    model = sm.GLM(
        y[train],
        sm.add_constant(X.iloc[train]),
        var_weights=weight[train],
        family=sm.families.Binomial(),
    ).fit_regularized(alpha=alpha, L1_wt=1, refit=True)
    mu = model.predict(sm.add_constant(X.iloc[test]))
    expected.append(
        -sm.families.Binomial().deviance(y[test], mu, var_weights=weight[test])
    )
for features in (X, X.to_numpy()):
    actual = weighted_cross_val_score(
        EnetGLM(family="binomial", alpha=alpha, L1_wt=1), features, y, weight, n_jobs=1
    )
    assert_allclose(actual, expected, atol=1e-10)
    selected = grid_search_cv(
        features,
        y,
        weight,
        family="binomial",
        score="deviance",
        n_iterations=3,
        n_jobs=1,
    )
    assert np.isfinite(selected.predict(features)).all()
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 5,
    "num_threads": 1,
    "verbosity": -1,
    "min_data_in_leaf": 2,
}
wrapper = GradientBoosting(params=params, show_learning_curve=False, verbose_eval=0)
wrapper.fit(X, (y > 0.5).astype(int), sample_weight=weight)
probability = wrapper.model.predict(X)
assert_allclose(wrapper.predict(X, predict_proba=True), probability)
assert_array_equal(wrapper.predict(X), (probability > 0.5).astype(int))
params["objective"] = "binary_logloss"
try:
    GradientBoosting(params=params, show_learning_curve=False, verbose_eval=0).fit(
        X, (y > 0.5).astype(int)
    )
except lgb.basic.LightGBMError as error:
    assert "Unknown objective type name: binary_logloss" in str(error)
else:
    raise AssertionError("Expected native rejection of the metric name as objective")
print(
    json.dumps(
        {
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "splitter": "KFold",
            "fractional_binomial_weighted_cv": "matches native GLM",
            "fractional_binomial_grid_frame_and_array": "passed",
            "binary_labels_and_probabilities": "passed",
            "binary_logloss_objective": "rejected before prediction",
        }
    )
)
