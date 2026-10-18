# Changes

# Unreleased

- Adapt binary log loss/error metrics when `VariableImportance` promotes to multiclass, including metric lists and aliases, without mutating caller parameters; reject AUC/average precision before encoding or training because they have no direct multiclass equivalent.
- Assign rank 1 to selected features and rank 2 to rejected features in `BoostAGroota` and `GrootCV`, matching `support_` and `feature_names_in_`.
- Respect `VariableImportance` task defaults (`rmse`, `binary`, `multiclass`) when `lgb_kwargs` is `None` or omits `objective`, promote `binary` with more than two target classes to `multiclass` with a `UserWarning`, validate classification labels and `num_class` before encoding or training, and derive validation-split stratification from the resolved objective without mutating caller dictionaries.
- Keep zero-total SHAP and native iterations/folds as finite zeros in `BoostAGroota` and `GrootCV`, require positive summary importance so uninformative fits select nothing, and raise `ValueError` on nonfinite importances.
- Add opt-in MRMR row sampling through `sample_size` and `random_state`.
  Relevance and redundancy share one draw; full-data fitting remains the default.

Breaking migration (requires refitting before release):

 - `TreeDiscretizer` now bins numeric columns only. `all`, `numerical` and `None` select eligible numeric columns; explicitly selected categories raise an error. Replace mixed calls with `TreeGrouper` for categorical columns, followed by `TreeDiscretizer` for numeric columns. Forward `sample_weight` explicitly to each pipeline step.
 - Add `TreeGrouper(group_features="all", n_groups=10, boost_params=None, raw=False, task="regression")`. Nominal levels use native categorical splits; ordered categories retain their declared relative order with contiguous groups. Cast numeric category codes to categorical dtype explicitly.
 - Both fit one native LightGBM tree on all positive-weight training rows instead of the GBM wrapper's 80% training subset. Task, one round, leaf limit and all-row fitting override conflicting aliases and row-sampling settings without mutating supplied parameters. Validate finite numeric regression targets and numeric/Boolean binary `{0,1}` targets before dropping zero-weight rows. Constant targets remain valid.
 - Refit existing discretizers. `tree_dic` now contains native Boosters; remove `.model` access. Numeric `bin_upper_bound_dic` and `leaf_interval_dic_` remain; categorical `cat_bin_dict` moves to `TreeGrouper`, and the discretizer's `ordinal_encoder_dic` and `cat_features_` are removed. Grouper `category_dtypes_` and `group_dtypes_` record frozen input/output vocabularies. Missing and unseen levels follow native missing routes.
 - Numeric output options and response-scale `raw=True` remain, including binary probabilities. Use categorical bins and factor encoding for separate effects. Grouping/binning does not implement group/fused Lasso, guarantee better predictions or justify ordinary post-selection p-values. The tutorials now use training-only comparisons, explicit factor encoding and local data; preprocessing includes a known-truth factorial experiment and category-renaming counterexample. Saved outputs are cleared for the requested manual rerun, and the workflow guide records that execution/figure-inspection gap.

# 4.0.1 — 2026-10-16

 - Correct weighted Spearman ranks so zero-weight rows cannot change positive-weight ties; nullable numeric inputs now use explicit float arrays. Recheck feature selections after this rank correction.
 - Reuse centered NumPy products and per-column ranks for Pearson/Spearman; series calls use matrix–vector products. NumPy/BLAS controls numeric threading independently of validated `n_jobs`. Prepare category codes/marginals once for Theil’s U and correlation ratio, with optional joblib batches and unchanged custom callback dispatch.

# 4.0.0 — 2026-10-16

This major release includes the R01–R33 repairs. Refit models and recheck feature
selections and downstream metrics, since corrected behavior can change results.

Migration notes:

 - Remove `VariableImportance(n_iterations=...)`; importance now uses one fit.
 - Replace `cat_var` with fitted `OrdinalEncoderPandas` encoding. It returns a DataFrame, not the old four-item tuple.
 - Replace `validate_pandas_input` with pandas `.to_numpy()` when an array is needed.
 - Refit and save GBM models again; old Booster-only files are rejected. New artifacts require trusted input and a compatible environment.
 - Replace `TreeDiscretizer.tree_imputer` with `leaf_interval_dic_`; categorical maps now use native leaf IDs.
 - Preserve fitted column order for named encoders and prediction schemas; array inputs must have the fitted width.

Changes:

 - Retire the temporary Ruff baseline and require a clean direct check. Remove inherited lint debt while preserving Python 3.9 runtime typing, backend import checks and error wrappers; use immutable encoder defaults and raise the missing FastTreeSHAP error before training. R33 records the completed local release-evidence gates and their support limits.

 - Correct GrootCV's imbalance heuristic to use the largest class frequency, regardless of its integer label; retain the strict 80% unweighted-row cutoff.

 - Repair documentation examples, optional-backend setup, sampling limits and reduced-feature time-series calls. Build HTML from the local docs directory using the project docs extra and a system Pandoc executable; render saved notebook outputs without executing optional or large workloads. Remove unused Pandoc Python/tabs dependencies, fix API markup and qualify method, weighting and validation claims.

 - Use local seeded random draws for synthetic datasets and Titanic/cancer augmentation, including weights, without altering NumPy's global stream. Generate exact odd/even regression row counts and mark Titanic passengers alone only when they have no relatives.

 - Make plots follow the displayed data and fitted selection: preserve target-plot feature labels and single-panel layouts, color Leshy boxes by membership, and honor VariableImportance plot limits and terminal thresholds. Keep cumulative feature positions linear and shares relative to all features; render singleton association matrices without clustering.

 - Preserve mixed integer/string and tuple/MultiIndex labels in parallel association work. Use joblib's negative worker-count semantics, cap workers to nonempty chunks, propagate callback errors and return typed empty results. Sort results with stable ties and a consistent matrix index. Serial Pearson now returns both directions of distinct pairs without diagonal entries, matching the parallel path while retaining its native matrix calculation.

 - Make Gower nearest neighbors use the explicit one-row query, exclude its identifier and return the nearest finite-distance row per remaining key. Use linear distance storage, stable input-order ties and validated nonnegative counts; raise for too few eligible neighbors. Support Series queries and preserve candidate dtype inference. smallest_indices now returns only finite values and empty arrays for zero, without nonfinite-value replacement.

 - Bound isolation sampling to 20 attempts per distinct 5%–95% proper-subset size, starting at 5%. Return empty/single inputs unchanged and full original data on exhaustion. Describe the unweighted KS p-value screen as a heuristic, without representativeness or outlier-removal guarantees.

 - Repair sampling outputs: impute numeric columns separately, preserve feature names (including label), and return original isolation-selected rows and dtypes. Reject unsupported Gower row weights and entirely missing columns; forward validated observation weights to the isolation forest fit while keeping subset draws and KS screening unweighted. Learn isolation category codes from positive-weight rows only; unseen score-only categories use -1.

 - Correct Gower numeric scaling, cross-matrix indexing and pairwise missing-value weights. Return NaN for undefined pairs, validate feature weights and category masks, support nullable dtypes and reference Series, and retain pooled numeric ranges with float64 output.

 - Fit benchmark encoders on training rows only, support ndarray inputs and integer display names, and compare fresh Leshy/BoostAGroota configurations without mutating caller objects. Reject GrootCV in cross-estimator comparisons and refresh Leshy backend flags at fit time.

 - Save versioned GBM artifacts with the offset guard, category names and original prediction schema. Validate metadata before replacing loaded state and clear stale validation data/curves. Reject legacy Booster-only files because their baseline contract is unknown; refit and save again. Stage model and optional PNG writes before replacement; the PNG remains a separate file, and its replacement can fail after the model commits. Joblib artifacts require trusted input and a compatible environment.

 - Reject conflicting GBM group/stratified splitting and mismatched prediction schemas. Handle multiclass objective aliases and finite positional initial-score matrices; clear stale state after successful refits. Restrict automatic categories to pandas category/bool, honor explicit categorical disable, and explain unsupported dtype conversion. Initial-score predictions still require a new external baseline on the raw scale.

 - Preserve TreeDiscretizer, IntervalToMidpoint and PatsyTransformer constructor options across refits. Resolve fitted columns/formulas in bin_features_, cat_features_, cols_ and formula_; normalize array inputs consistently and preserve Patsy dictionary names. Named tree transforms require the full fitted schema; arrays require its width. Honor scalar midpoint names and Patsy's NA_action/add_intercept options. Explicit Patsy dropping removes predictor rows only, so targets and weights must be aligned by the caller.
 - Make TreeDiscretizer bins follow exact fitted thresholds and leaf IDs, preserve predictor columns, and use copied current one-tree parameters. Apply categorical encoding in raw mode; binary raw values are probabilities. Missing/unknown values follow native routes, including all-missing single-leaf bins. Categorical maps now use leaf IDs; leaf_interval_dic_ replaces tree_imputer.
 - Remove the public legacy helpers cat_var and validate_pandas_input. Use OrdinalEncoderPandas for fitted categorical encoding/inverse decoding; use pandas .to_numpy() for array extraction and keep existing NumPy arrays directly. The encoder returns a DataFrame, not cat_var's four-item tuple; no compatibility shim is provided.
 - Keep OrdinalEncoderPandas full input/output names separate from categorical validation, preserve inverse inputs and honor output_dtype on refit. Nullable strings and booleans are encoded with native missing/unknown handling; transforms require the fitted column order.
 - Validate response and observation-weight inputs, preserve category identity, and correct weighted association arithmetic.
 - Correct MRMR ranking, categorical shadow controls, Leshy support/history, weighted permutation scoring, and GrootCV output/threshold aggregation.
 - Return negative response-scale GLM deviance for sklearn scoring, preserve BIC minimization, and repair weighted CV splits, NumPy inputs and fold isolation.
 - Reuse fitted GLM designs for single-row inference, align named schemas, preserve supplied constants and resolve current family/link parameters at fit. Input/support metadata now excludes generated intercepts.
 - Omit zero-weight GLM rows before fitting/scoring/tuning, correct AIC/BIC dimensions and reject nonfinite search criteria. Document unpenalized post-selection refits and reject unsupported pure ridge fits.
 - Reject duplicate threshold-selector names, align labelled statistics, validate current parameters and expose fitted feature counts. Preserve named projection and strict threshold equality; handle empty collinearity stages and break ties in input order.
 - Make VariableImportance use one fit and remove n_iterations. Keep the cumulative crossing feature, exclude zero importance, reject nonfinite contributions and fix array multiclass/optional-backend errors. Training remains weighted; global importance averages validation rows equally.
 - Test installed wheels on Python 3.9–3.12 with locked dependencies, exact direct runtime minima on Python 3.9, and a newer Python 3.12 snapshot. Add a fixed held-out selection screen and document support and calibration limits.

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
