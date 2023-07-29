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
    * The categorical features (they are detected, encoded. The tree-based models are working better with integer encoding rather than with OHE, which leads to deep and unstable trees). If Catboost is used, then the cat.pred (if any) are set up.
    * Work with Catboost sklearn API.
    * Allow using sample_weight, for applications like Poisson regression or any requiring weights.
    * 3 different feature importances: native, SHAP, and permutation. Native being the least consistent (because of the imp. biased towards numerical and large cardinality categorical) but the fastest of the 3.
    * Using LightGBM as default speed up by an order of magnitude the running time.
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

GrootCV, a new method
---------------------

New: GrootCV:
- Cross-validated feature importance to smooth out the noise, based on lightGBM only (which is, most of the time, the fastest and more accurate Boosting).
- The feature importance is derived using SHAP importance.
- Taking the max of median of the shadow var. imp over folds otherwise not enough conservative and it improves the convergence (needs less evaluation to find a threshold).
- Not based on a given percentage of cols needed to be deleted.
- Plot method for var. imp.

MRmr
----

Re-implementing the Uber MRmr scheme using associations for handling continuous and categorical predictors.
- Theil's U statistics for the categorical-categorical association (correlation).
- Variance ratio for continuous-categorical association.
- Pearson or Spearman correlation for continuous-continuous association.

Lasso
-----

Performing a simple grid search

 with enforced lasso regularization.
The best model is chosen based on the minimum BIC or deviance score, and all non-zero coefficients are selected.
The loss function can belong to the exponential family, as seen in the statsmodels GLM documentation.
Using the bic metric is faster since it is evaluated on the training data, making it unsuitable for the test data, whereas the deviance is cross-validated.

This approach can be combined with the TreeDiscretizer transformer to introduce univariate non-linearities (tree-GAM) before feature selection.
This serves as a workaround to compensate for the absence of fused and grouped lasso regularization.

References
----------

**Theory**
- [Consistent feature selection for pattern recognition in polynomial time](http://compmed.se/files/6914/2107/3475/pub_2007_5.pdf)
- [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://www.uber.com/blog/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform)

**Applications**
- [The Boruta paper](https://www.jstatsoft.org/article/view/v036i11/v36i11.pdf)
- [The python implementation](https://github.com/scikit-learn-contrib/boruta_py)
- [BoostARoota](https://github.com/chasedehan/BoostARoota)