<img src="logo.png" alt="drawing" width="200"/>


# All relevant feature selection

All relevant feature selection means trying to find all features carrying information usable for prediction, rather than finding a possibly compact subset of features on which some particular model has a minimal error. This might include redundant predictors. All relevant feature selection is model agnostic in the sense that it doesn't optimize a scoring function for a *specific* model but rather tries to select all the predictors which are related to the response. 

This package implements 3 different methods (Leshy is an evolution of Boruta, BoostAGroota is an evolution of BoostARoota and GrootCV is a new one). They are sklearn compatible. See hereunder for details about those methods. You can use any sklearn compatible estimator with Leshy and BoostAGroota but I recommend lightGBM. It's fast, accurate and has SHAP values builtin.

## Installation

`pip install arfs`

## Example

For working examples, see the example NB and the unit tests.


```python
import arfs.allrelevant as arfsgroot

def LightForestRegressor(n_feat):
    return lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=100, bagging_fraction=0.632,
                                   feature_fraction=n_feat / (3*n_feat), boosting_type="rf", bagging_freq=1)

# The arfs methods work with regressors or classifiers
models = [RandomForestRegressor(n_jobs= 4, oob_score= True), 
          CatBoostRegressor(random_state=42, verbose=0), 
          XGBRegressor(random_state=42, verbose=0), 
          LGBMRegressor(random_state=42, verbose=-1),
          LightForestRegressor(n_feat=X.shape[1])]

model = models[3]

# Leshy
feat_selector = arfsgroot.Leshy(model, n_estimators = 100, verbose= 1, max_iter= 10, random_state=42, importance=varimp)
feat_selector.fit(X, y, sample_weight)
print(feat_selector.support_names_)
feat_selector.plot_importance(n_feat_per_inch=3)

# BoostAGroota
feat_selector = arfsgroot.BoostAGroota(est=model, cutoff=1, iters=10, max_rounds=10, delta=0.1, silent=False, imp='shap')
feat_selector.fit(X, y)
print(feat_selector.keep_vars_ )
feat_selector.plot_importance()

# GrootCV
feat_selector = arfsgroot.GrootCV(objective='rmse', cutoff = 1, n_folds=5, n_iter=5, silent=False)
feat_selector.fit(X, y, sample_weight)
print(feat_selector.keep_vars_ )
feat_selector.plot_importance(n_feat_per_inch=2)
```

## Boruta

The Boruta algorithm tries to capture all the important features you might have in your dataset with respect to an outcome variable. The procedure is the following:

 * Create duplicate copies of all independent variables. When the number of independent variables in the original data is less than 5, create at least 5 copies using existing variables.
 * Shuffle the values of added duplicate copies to remove their correlations with the target variable. It is called shadow features or permuted copies.
 * Combine the original ones with shuffled copies
 * Run a random forest classifier on the combined dataset and performs a variable importance measure (the default is Mean Decrease Accuracy) to evaluate the importance of each variable where higher means more important.
 * Then Z score is computed. It means mean of accuracy loss divided by the standard deviation of accuracy loss.
 * Find the maximum Z score among shadow attributes (MZSA)
 * Tag the variables as 'unimportant' when they have importance significantly lower than MZSA. Then we permanently remove them from the process.
 * Tag the variables as 'important' when they have importance significantly higher than MZSA.
 * Repeat the above steps for a predefined number of iterations (random forest runs), or until all attributes are either tagged 'unimportant' or 'important', whichever comes first.

At every iteration, the algorithm compares the Z-scores of the shuffled copies of the features and the original features to see if the latter performed better than the former. If it does, the algorithm will mark the feature as important. In essence, the algorithm is trying to validate the importance of the feature by comparing with randomly shuffled copies, which increases the robustness. This is done by simply comparing the number of times a feature did better with the shadow features using a binomial distribution. Since the whole process is done on the same train-test split, the variance of the variable importance comes only from the different re-fit of the model over the different iterations.


## BoostARoota

BoostARoota follows closely the Boruta method but modifies a few things:

 * One-Hot-Encode the feature set
 * Double width of the data set, making a copy of all features in the original dataset
 * Randomly shuffle the new features created in (2). These duplicated and shuffled features are referred to as "shadow features"
 * Run XGBoost classifier on the entire data set ten times. Running it ten times allows for random noise to be smoothed, resulting in more robust estimates of importance. The number of repeats is a parameter than can be changed.
 * Obtain importance values for each feature. This is a simple importance metric that sums up how many times the particular feature was split on in the XGBoost algorithm.
 * Compute "cutoff": the average feature importance value for all shadow features and divide by four. Shadow importance values are divided by four (parameter can be changed) to make it more difficult for the variables to be removed. With values lower than this, features are removed at too high of a rate.
 * Remove features with average importance across the ten iterations that is less than the cutoff specified in (6)
 * Go back to (2) until the number of features removed is less than ten per cent of the total.
 * Method returns the features remaining once completed.

In the spirit, the same heuristic than Boruta but using Boosting (originally Boruta was supporting only random forest). The validation of the importance is done by comparing to the maximum of the median var. imp of the shadow predictors (in Boruta, a statistical test is performed using the Z-score). Since the whole process is done on the same train-test split, the variance of the variable importance comes only from the different re-fit of the model over the different iterations.

## Modifications to Boruta and BoostARoota

 I forked both Boruta and BoostARoota and made the following changes (under PR):
 
**Boruta --> Leshy**:
 
  - The categorical features (they are detected, encoded. The tree-based models are working better with integer encoding rather than with OHE, which leads to deep and unstable trees). If Catboost is used, then the cat.pred (if any) are set up
  - Using lightGBM as the default speeds up by an order of magnitude the running time
  - Work with Catboost, sklearn API
  - Allow using sample_weight, for applications like Poisson regression or any requiring weights
  - Supports 3 different feature importances: native, SHAP and permutation. Native being the least consistent(because of the imp. biased towards numerical and large cardinality categorical) but the fastest of the 3. Indeed, the impurity var.imp. are biased en sensitive to large cardinality (see [scikit demo](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py))
  
**BoostARoota --> BoostAGroota**:

  - Replace XGBoost with LightGBM, you can still use tree-based scikitlearn models
  - Replace native var.imp by SHAP var.imp. Indeed, the impurity var.imp. are biased en sensitive to large cardinality (see [scikit demo](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py)). Moreover, the native var. imp are computed on the train set, here the data are split (internally) in train and test, var. imp computed on the test set.
  - Handling categorical predictors. Cat. predictors should NOT be one-hot encoded, it leads to deep unstable trees. Instead, it's better to use the native method of lightGBM or CatBoost. A preprocessing step is needed to encode (ligthGBM and CatBoost use integer encoding and reference to categorical columns. The splitting strategies are different then, see official doc).
  - Work with sample_weight, for Poisson or any application requiring a weighting.

## GrootCV, a new method

**New: GrootCV**:

  - Cross-validated feature importance to smooth out the noise, based on lightGBM only (which is, most of the time, the fastest and more accurate Boosting).
  - the feature importance is derived using SHAP importance
  - Taking the max of the median of the shadow var. imp over folds otherwise not enough conservative and it improves the convergence (needs less evaluation to find a threshold)
  - Not based on a given percentage of cols needed to be deleted
  - Plot method for var. imp
 
 
## References

**Theory**
 - [Consistent feature selection for pattern recognition in polynomial time](http://compmed.se/files/6914/2107/3475/pub_2007_5.pdf)

**Applications**
 - [The Boruta paper]([https://www.jstatsoft.org/article/view/v036i11/v36i11.pdf)
 - [The python implementation](https://github.com/scikit-learn-contrib/boruta_py)
 - [BoostARoota](https://github.com/chasedehan/BoostARoota)
