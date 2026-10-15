Introduction
============

All relevant feature selection means trying to find all features carrying information usable for prediction, 
rather than finding a possibly compact subset of features on which some particular model has a minimal error. 
This might include redundant predictors. The implemented selectors compare fitted model importances
with shuffled features. Their results depend on the estimator and importance method; they do not
establish calibrated false-discovery control for the whole procedure.
This package implements 3 different methods (Leshy is an evolution of Boruta, BoostAGroota is an evolution of BoostARoota and GrootCV is a new one). 
They expose sklearn-style fit/transform interfaces. Leshy and BoostAGroota require compatible tree
estimators supplying the requested importance method and weight support. GrootCV uses LightGBM.
Optional backend compatibility and speed require separate checks; see :doc:`workflows`.

It also provides a module for performing preprocessing and perform basic feature selection 
(autobinning, remove columns with too many missing values, zero variance, high-cardinality, highly correlated, etc.). 

MRmr provides a separate relevance/redundancy ranking heuristic. Its selected set need not be a subset
of an all-relevant selector's output. Lasso selects a penalized GLM design and then refits the surviving
columns without regularization. You can combine Lasso with the `TreeDiscretizer` for introducing
non-linearities into linear models and perform feature selection.
Please note that one limitation of the lasso is that it treats the levels of a categorical predictor individually. 
However, this issue can be addressed by utilizing the `TreeDiscretizer`, which automatically bins numerical variables and 
groups the levels of categorical variables.

