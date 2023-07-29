Introduction
============

All relevant feature selection means trying to find all features carrying information usable for prediction, 
rather than finding a possibly compact subset of features on which some particular model has a minimal error. 
This might include redundant predictors. All relevant feature selection is model agnostic in the sense that it 
doesn't optimize a scoring function for a *specific* model but rather tries to select all the predictors which are related to the response. 
This package implements 3 different methods (Leshy is an evolution of Boruta, BoostAGroota is an evolution of BoostARoota and GrootCV is a new one). 
They are sklearn compatible. See hereunder for details about those methods. You can use any sklearn compatible estimator 
with Leshy and BoostAGroota but I recommend lightGBM. It's fast, accurate and has SHAP values builtin. 

It also provides a module for performing preprocessing and perform basic feature selection 
(autobinning, remove columns with too many missing values, zero variance, high-cardinality, highly correlated, etc.). 

Moreover, as an alternative to the all relevant problem, the ARFS package provides a MRmr feature selection which, 
theoretically, returns a subset of the predictors selected by an arfs method. ARFS also provides a `LASSO` feature 
selection which works especially well for (G)LMs and GAMs. You can combine Lasso with the `TreeDiscretizer` for introducing 
non-linearities into linear models and perform feature selection.
Please note that one limitation of the lasso is that it treats the levels of a categorical predictor individually. 
However, this issue can be addressed by utilizing the `TreeDiscretizer`, which automatically bins numerical variables and 
groups the levels of categorical variables.

