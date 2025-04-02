# Changes

# 3.0.0

 - [ENHANCEMENT] Upgrade to newer SHAP and lightgbm version
 - [ENHANCEMENT] Migrate project management to `uv`

# 2.4.0
 - [BUG] Add a safety for the array size in the weighted correlation ratio
 - [DOC] Contribution for better documentation, typos and fixing docstrings

# 2.3.3
 - [BUG] Fix range, which should run from 1 to `max_iter`

# 2.3.2
 - [BUG] Fix errors generated when updating dependencies with different naming for arguments
  
# 2.3.1
 - [BUG] replace np.Inf by np.inf for compatibility purpose

# 2.3.0
 - [BUG] corrected the column names for the GrootCV scheme, setting the shadow var in last position to guarantee the real names are used
 - [ENHANCEMENT] support user defined cross-validation scheme for time series applications for GrootCV

# 2.2.6
 - [BUG] fix the calculation of the SHAP feature importance for multi-class
 - [ENHANCEMENT] Update pandas aggregation to get rid of the future deprecation warnings

# 2.2.5
 - [BUG] fix the calculation of the SHAP feature importance for multi-class
 - [ENHANCEMENT] return the feature for the importance

# 2.2.4
 - [BUG] add axis=1 to compute the max on the right dimension in _reduce_vars_sklearn
 - [BUG] remove merge causing duplication of the feature importance in _reduce_vars_sklearn

# 2.2.3
 - [BUG] change the default of the weighted correlation for consistency with existing doc
 - [ENHANCEMENTS] speedup the correlation feature selector
# 2.2.1
 - [BUG] add copy() to prevent modifying the input pandas DF in the mrmr when fitting the mrmr selector
# 2.2.0
 - [BUG] fix the collinearity feature elimination
 - [BUG] fix the feature importance if fasttreeshap not installed
 - [REFACTORING] refactor the association module for removing redundancy and faster computation
# 2.1.3
 - [BUG] fix the hardcoded threshold in collinearity elimination, closes #33
# 2.1.2
 - [BUG] fix a bug in computing the association matrix when a single column of a specific dtype is passed in the sub_matrix (nom-nom, num-num) calculators.
# 2.1.1
 - Refactor TreeDiscretizer
# 2.1.0
 - Add a mechanism to the TreeDiscretizer that restricts the length of combined strings for categorical columns, preventing excessively lengthy entries.
# 2.0.7
 - implement link for the lasso feature selection, e.g. log for ensuring positivity 
# 2.0.6
 - downgrade the lightgbm version to 3.3.1 for compatibility reasons (with optuna for instance)
## 2.0.5
 - Fix: strictly greater than threshold rather than geq in the base threshold transformer
 - Update: due to a change in the lightgbm train API (v4), update the code for GBM
## 2.0.4
 - Documentation: fix the format of some docstrings and remove old sphinx generated files
## 2.0.3
 - Fix: remove unnecessary `__all__` in the preprocessing module and improve the consistency of the module docstrings 
## 2.0.2
 - Fix: when the L1 == 0 in fit_regularized, statsmodels returns the regularized wrapper without refit, which breaks the class (statistics not available)
## 2.0.1
 - Build: remove explicit dependencies on holoviews and panel
## 2.0.0
 - Add fasttreeshap implementation as an option to compute shap importance (fasttreeshap does not work with XGBoost though)
 - New feature: lasso feature selection, especially useful for models without interactions (LM, GLM, GAM)
 - New feature: pass lightgbm parameters to GrootCV
 - Bug: fix sample weight shape in mrMR
 - Documentation: update and upgrade tuto NB
## 1.1.4
 - update the required python version >= 3.9
## 1.1.3
 - Change tqdm to auto for better rendering in NB for variable importance selector
 - User defined n_jobs for association matrix computation
## 1.1

 - Corrected an issue in Leshy that occurred when using categorical variables. The use of NumPy functions and methods instead of Pandas ones resulted in the modification of original data types.

## 1.0.7

 - Patch preventing zero division in the conditional entropy calculation
  
## 1.0.6

 - Return self in mrmr, fixing error when in scikit-learn pipeline

## 1.0.5

 - Patching classes where old unused argument was causing an error

## 1.0.2

 - Distribute a toy dataset for regression by modifying the Boston dataset adding noise and made up columns

## 1.0.1

 - Fix pkg data distribution
  
## 1.0.0

 - Parallelization of functions applied on pandas data frame
 - Faster and more modular association measures 
 - Removing dependencies (e.g. dython)
 - Better static and interactive visualization
 - Sklearn selectors rather than a big class
 - Discretization of continuous and categorical predictors
 - Minimal redundancy maximal relevance feature selection added (a subset of all relevant predictors), based on Uber's MRmr flavor
 - architecture closer to the scikit-learn one
  
## 0.3.8

 - Fix bug when compute shap importance for classifier in GrootCV

## 0.3.7

 - Add defensive check if no categorical found in the subsampling of the dataset
 - Re-run the notebooks with the new version
## 0.3.6

 - Fix clustering when plotting only strongly correlated predictors
 - Remove palettable dependencies for plotting 
 - Add default colormap but implement the user defined option
## 0.3.5

- Enable clustering before plotting the correlation/association matrix, optional
- Decrease fontsize for the lables of the correlation matrix

## 0.3.4

 - Update requirements 

## 0.3.3

 - Upgrade documentation

## 0.3.2

 - Fix typo for distributing the dataset and pinned the dependencies
## 0.3.1 

 - Update the syntax for computing associations using the latest version of dython

## 0.3.0

 - Fix the Boruta_py feature counts, now adds up to n_features
 - Fix the boxplot colours, when only rejected and accepted (no tentative) the background color was the tentative color
 - Numpy docstring style
 - Implement the new lightGBM callbacks. The new lgbm version (>3.3.0) implements the early stopping using a callback rather than an argument
 - Fix a bug for computing the shap importance when the estimator is lightGBM and the task is classification
 - Add ranking and absolute ranking attributes for all the classes
 - Fix future pandas TypeError when computing numerical values on a dataframe containing non-numerical columns
 - Add housing data to the distribution
 - Add "extreme" sampling methods
 - Re-run the NBs
 - reindex to keep the original columns order

## 0.2.3

 - Update syntax to stick to the new argument names in Dython

## 0.2.2

 - Check if no feature selected, warn rather than throw error

## 0.2.1

 - Fix a bug when removing collinear columns

## 0.2.0

 - Prefilters now support the filtering of continuous and nominal (categorical) collinear variables

## 0.1.6

 - improve the plot_y_vs_X function
 - remove gc.collect()

## 0.1.5

 - fix readme (typos)
 - move utilities in utils sub-package
 - make unit tests lighter

## 0.1.4

 - fix bug when using catboost, clone estimator (avoid error and be sure to use a non-fitted estimator)

## 0.1.3

 - change the defaut for categorical encoding in pre-filters (pd.cat to integers as default)
 - fix the unit tests with new defaults and names

## 0.1.2

 - change arguments name in pre-filters

## 0.1.1

 - remove old attribute names in unit-tests

## 0.1.0

 - Fix lightGBM warnings
 - Typo in repr
 - Provide load_data utility
 - Enhance jupyter NB examples
 - highlighting synthetic random predictors
 - Benchmark using sklearn permutation importance
 - Harmonization of the attributes and parameters
 - Fix categoricals handling

## 0.0.4

 - setting optimal number of features (according to "Elements of statistical learning") when using lightGBM random forest boosting.
 - Providing random forest, lightgbm implementation, estimators

## 0.0.3

 - Adding examples and expanding documentation

## 0.0.2

 - fix bug: relative import removed