Methods overview
================

Boruta
------

The Boruta algorithm tries to capture all the important features you might have in your dataset with respect to an outcome variable. The procedure is as follows:

* Create duplicate copies of all independent variables. When the number of independent variables in the original data is less than 5, create at least 5 copies using existing variables.
* Shuffle the values of added duplicate copies to remove their correlations with the target variable. It is called shadow features or permuted copies.
* Combine the original ones with shuffled copies.
* Run a random forest classifier on the combined dataset and perform a variable importance measure (the default is Mean Decrease Accuracy) to evaluate the importance of each variable where higher means more important.
* Then Z score is computed. It means the mean of accuracy loss divided by the standard deviation of accuracy loss.
* Find the maximum Z score among shadow attributes (MZSA).
* Tag the variables as 'unimportant' when they have importance significantly lower than MZSA. Then we permanently remove them from the process.
* Tag the variables as 'important' when they have importance significantly higher than MZSA.
* Repeat the above steps for a predefined number of iterations (random forest runs), or until all attributes are either tagged 'unimportant' or 'important', whichever comes first.

At every iteration, the algorithm compares the Z-scores of the shuffled copies of the features and the original features to see if the latter performed better than the former. If it does, the algorithm will mark the feature as important. In essence, the algorithm is trying to validate the importance of the feature by comparing with randomly shuffled copies, which increases the robustness. This is done by simply comparing the number of times a feature did better with the shadow features using a binomial distribution. Since the whole process is done on the same train-test split, the variance of the variable importance comes only from the different re-fit of the model over the different iterations.


BoostARoota
-----------

BoostARoota follows closely the Boruta method but modifies a few things:

* One-Hot-Encode the feature set.
* Double width of the data set, making a copy of all features in the original dataset.
* Randomly shuffle the new features created in (2). These duplicated and shuffled features are referred to as "shadow features."
* Run XGBoost classifier on the entire data set ten times. Running it ten times allows for random noise to be smoothed, resulting in more robust estimates of importance. The number of repeats is a parameter that can be changed.
* Obtain importance values for each feature. This is a simple importance metric that sums up how many times the particular feature was split in the XGBoost algorithm.
* Compute "cutoff": the average feature importance value for all shadow features and divide by four. Shadow importance values are divided by four (parameters can be changed) to make it more difficult for the variables to be removed. With values lower than this, features are removed at too high of a rate.
* Remove features with average importance across the ten iterations that are less than the cutoff specified in (6).
* Go back to (2) until the number of features removed is less than ten percent of the total.
* The method returns the features remaining once completed.

Modifications to Boruta
-----------------------

Boruta --> Leshy:

For chronological development, see https://github.com/scikit-learn-contrib/boruta_py/pull/77 and https://github.com/scikit-learn-contrib/boruta_py/pull/100

Leshy vs. BorutaPy:
    To summarize, this PR solves/enhances:
    * Detect and encode categorical features. Codes need the estimator's categorical handling; encoding alone does not establish superiority over one-hot encoding.
    * Accept compatible sklearn-style tree estimators. CatBoost/XGBoost execution is outside the tested runtime matrix.
    * Forward observation weights to compatible estimators and validation scoring; their meaning depends on the estimator and loss.
    * 3 different feature importances: native, SHAP, and permutation. Native being the least consistent (because of the imp. biased towards numerical and large cardinality categorical) but the fastest of the 3.
    * Support LightGBM; runtime depends on the data, estimator and parameters.
    * Visualization like in the R package.

BorutaPy vs. Boruta R:
    The improvements of this implementation include:
    * Faster run times: Thanks to scikit-learn's fast implementation of the ensemble methods.
    * Scikit-learn like interface: Use BorutaPy just like any other scikit-learn: fit, fit_transform, and transform are all implemented in a similar fashion.
    * Modularity: Any ensemble method could be used: random forest, extra trees classifier, even gradient boosted trees.
    * Two-step correction: The original Boruta code corrects for multiple testing in an overly conservative way. In this implementation, the Benjamini Hochberg FDR is used to correct in each iteration across active features. This means only those features are included in the correction which are still in the selection process. Following this, each that passed goes through a regular Bonferroni correction to check for the repeated testing over the iterations.
    * Percentile: Instead of using the max values of the shadow features, the user can specify which percentile to use. This gives a finer control over this crucial parameter. For more info, please read about the perc parameter.
    * Automatic tree number: Setting the n_estimator to 'auto' will calculate the number of trees in each iteration based on the number of features under investigation. This way more trees are used when the training data has many features and fewer when most of the features have been rejected.
    * Ranking of features: After fitting BorutaPy, it provides the user with ranking of features. Confirmed ones are 1, Tentatives are 2, and the rejected are ranked starting from 3, based on their feature importance history through the iterations.
    * Using either the native variable importance, scikit permutation importance, SHAP importance.

    We highly recommend using pruned trees with a depth between 3-7. For more, see the docs of these functions, and the examples below. Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
    
BoostAGroota
------------

In each reduction round, BoostAGroota normalizes the real and shadow importances
separately for each model iteration, then takes each feature's mean across
iterations. The threshold is the maximum of the shadow means divided by
``cutoff``. Real features at or above the threshold are kept. For example,
one iteration with real scores ``[.3, .1]`` and shadow scores ``[.5, .1]``
uses a threshold of ``.5`` when ``cutoff=1``.

Permutation scores retain their signs and are normalized by their absolute
total; an all-zero vector stays zero. Only strictly positive mean permutation
scores can be selected. SHAP and native importances retain their existing
per-iteration total normalization. These empirical thresholds do not establish
false-discovery control.

GrootCV, a new method
---------------------

New: GrootCV:

- Repeated-fold feature importance based on LightGBM only; runtime and predictive accuracy depend on the data and parameters.
- SHAP importance is the mean absolute contribution over each fold's training
  rows, summed over outputs for multi-output models, then normalized across
  real and shadow features. Validation rows drive early stopping.
- Take each feature's median across all folds and repeats, then the maximum
  shadow median divided by ``cutoff``. Keep real medians at or above this
  threshold. Summary columns do not enter either calculation.
- For real scores ``[.57, .57, .16]`` and shadows ``[.35, .05, .70]`` and
  ``[.05, .35, .05]`` across three folds, the real median is ``.57`` and the
  threshold is ``.35`` at ``cutoff=1``, so the real feature is kept.
- Recognized multiclass classification objectives resolve to softmax, including
  binary classification with more than two target classes. SHAP aggregation
  follows the returned output axes rather than the objective's spelling.
- This empirical comparison does not establish calibrated false-discovery control.
- Not based on a given percentage of cols needed to be deleted.
- Plot method for var. imp.

MRmr
----

Re-implementing the Uber MRmr scheme using associations for handling continuous and categorical predictors.
Mixed-type relevance scores are ranking heuristics, not calibrated significance
tests. MRmr's selected set need not be a subset of an all-relevant selector's set.
- Theil's U statistics for the categorical-categorical association (correlation).
- Variance ratio for continuous-categorical association.
- Pearson or Spearman correlation for continuous-continuous association.

Lasso
-----

Performing a simple grid search with enforced lasso regularization.
The best model minimizes training BIC (``score="bic"``) or mean total validation
deviance across five folds (``score="deviance"``), and all non-zero coefficients are selected.
``EnetGLM.score`` returns negative response-scale deviance so sklearn searches
can maximize it. Weights enter as variance weights; the loss uses unit scale
and is not divided by fold size or weight mass.
Each fold fits an independent clone. Classifiers and binomial GLMs with discrete
targets use stratified folds; regressions, including two-valued numeric responses,
and fractional binomial responses use ordinary folds. Splits do not shuffle rows.
Fit records one design for prediction and transformation. Nonzero training
constants are preserved without adding another intercept; ``fit_intercept=False``
disables automatic addition and preserves supplied columns. A generated column
is named ``Intercept`` (with a numbered suffix to avoid input-name collisions).
It is still penalized and, if selected, included in transformed output.
``feature_names_in_``, ``n_features_in_`` and ``support_`` describe original
inputs; ``design_feature_names_`` and ``design_support_`` describe the selector's
coefficient design. Named inference inputs align to the fitted names; missing,
extra and duplicate columns are rejected. Arrays require fitted input width
and order. Family/link parameters are resolved at each fit.
Zero-weight rows are omitted after response/weight validation, before fitting,
scoring and forming GLM validation folds. Positive weights are not normalized.
``alpha`` is penalty strength; ``L1_wt`` is the L1 fraction of the selection
penalty. Surviving columns are refit without regularization, so returned
coefficients do not retain ridge shrinkage. Pure ridge (``L1_wt=0``) is unsupported.
Summary p-values do not account for using the same data to select variables.
The wrapper's AIC/BIC use the returned log-likelihood, retained row count and
selected design rank, counting mean parameters without dispersion. Backend
summary information criteria are unchanged and can differ from these corrected
attributes. A nonfinite candidate score aborts the search with its penalty value;
invalid candidates are not silently discarded. Variance-weighted fixed-scale
likelihoods are quasi-likelihoods; finite BIC does not establish calibrated
selection or post-selection significance.
The loss function can belong to the exponential family, as seen in the statsmodels GLM documentation.
Using the bic metric is faster since it is evaluated on the training data, making it unsuitable for the test data, whereas the deviance is cross-validated.

Combine ``TreeGrouper`` (categorical predictors) followed by ``TreeDiscretizer``
(numeric predictors) to introduce supervised univariate partitions. Both fit one
native LightGBM tree per selected feature using all positive-weight training rows;
the general GBM wrapper's validation split is not used. Validate outcomes and
weights before omitting zero-weight rows. Nominal levels use native categorical
splits; ordered categoricals use their declared order with contiguous groups.
Missing and unknown levels follow the fitted missing route, without an Other
level. Output category vocabularies are frozen at fit.

Fit these steps inside every training fold, forwarding observation weights to
each. Encode groups and numeric bins as factors (``num_bins_as_category=True``)
when they should have separate model effects. This does not implement group or
fused Lasso: selecting encoded columns does not jointly select the original
categorical variable. Supervised partitioning and selection invalidate ordinary
post-selection interpretations of reported p-values. Neither the partitions nor
a passing example establish predictive improvement.

References
----------

**Theory**
- [Consistent feature selection for pattern recognition in polynomial time](http://compmed.se/files/6914/2107/3475/pub_2007_5.pdf)
- [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://www.uber.com/blog/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform)

**Applications**
- [The Boruta paper](https://www.jstatsoft.org/article/view/v036i11/v36i11.pdf)
- [The python implementation](https://github.com/scikit-learn-contrib/boruta_py)
- [BoostARoota](https://github.com/chasedehan/BoostARoota)
