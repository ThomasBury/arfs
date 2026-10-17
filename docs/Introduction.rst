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
columns without regularization. Use ``TreeGrouper`` to group categorical predictors,
then ``TreeDiscretizer`` to bin numeric predictors. Each fits one LightGBM tree per
selected feature on all positive-weight training rows. Forward observation weights
to both steps and learn both only inside training folds. Nominal groups use native
categorical splits; ordered categoricals retain their declared order and form
contiguous groups. Missing and unseen levels follow fitted native missing routes.

Use ``num_bins_as_category=True`` and factor encoding for separate bin effects.
Lasso still selects individual encoded design columns; these transformations do
not implement group or fused Lasso or establish predictive gains. This API split
is breaking: refit old discretizers and replace mixed-data calls with sequential
grouping and numeric binning. See :doc:`arfs` for the interfaces.
