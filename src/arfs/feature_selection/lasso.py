"""LassoFeatureSelection Submodule

This module provides LASSO-based feature selection, specifically designed for use with Generalized Linear Models (GLM). 
The Lasso Regularized GLM introduces an L1 regularization penalty (Lasso regularization), 
encouraging some coefficients to become exactly zero during the model fitting process. 
This regularization effectively removes irrelevant features from the model, making it a 
powerful tool for feature selection, particularly in datasets with numerous variables.

Module Structure:
-----------------
- `EnetGLM`: class serves as a scikit-learn wrapper for the regularized statsmodels GLM, providing seamless integration with scikit-learn's ecosystem.
- `weighted_cross_val_score`: function allows users to pass weights to the model and define a custom scoring metric.
- `grid_search_cv`: function performs a weighted LASSO grid search to find the best Lasso parameter for the model.
- `LassoFeatureSelection`: class is the core feature selection class, estimating the Lasso parameter through 
    the grid search process, enabling efficient and effective feature selection.

With this submodule, users can easily leverage Lasso Regularized GLMs and conduct feature selection, 
improving model performance and interpretability in various datasets.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.model_selection import StratifiedKFold, KFold
from joblib import Parallel, delayed
from typing import Union, Optional


def _map_family_link(family: str = "gaussian", link: Optional[str] = None):
    family_mapping = {
        "gaussian": sm.families.Gaussian,
        "binomial": sm.families.Binomial,
        "poisson": sm.families.Poisson,
        "gamma": sm.families.Gamma,
        "negativebinomial": sm.families.NegativeBinomial,
        "tweedie": sm.families.Tweedie,
    }
    link_mapping = {
        "identity": sm.genmod.families.links.Identity(),
        "log": sm.genmod.families.links.Log(),
        "logit": sm.genmod.families.links.Logit(),
        "probit": sm.genmod.families.links.Probit(),
        "cloglog": sm.genmod.families.links.CLogLog(),
        "inverse_squared": sm.genmod.families.links.InverseSquared(),
    }
    if link is not None:
        objective = family_mapping[family](link_mapping[link])
    else:
        objective = family_mapping[family]()
    return objective


class EnetGLM(BaseEstimator, RegressorMixin):
    """
    Elastic Net Generalized Linear Model.

    Parameters
    ----------
    family : str, (default="gaussian")
        The distributional assumption of the model. It can be any of the statsmodels distribution:
        "gaussian", "binomial", "poisson", "gamma", "negativebinomial", "tweedie"
    link : str, optional
        the GLM link function. It can be any of the: "identity", "log", "logit", "probit", "cloglog", "inverse_squared"
    alpha : float, optional (default=0.0)
        The elastic net mixing parameter. 0 <= alpha <= 1.
        alpha = 0 is equivalent to ridge regression, alpha = 1 is equivalent to lasso regression.
    L1_wt : float, optional (default=0.0)
        The weight of the L1 penalty term. 0 <= L1_wt <= 1.
        The `L1_wt` parameter represents the weight of the L1 penalty term in the model and
        should be within the range 0 to 1. A value of 0 corresponds to ridge regression,
        while a value of 1 corresponds to lasso regression. However, for obtaining statistics,
        `L1_wt` should be set to a value greater than 0. If it is set to 0.0, statsmodels returns
        a ridge regularized wrapper without refitting the model, making the statistics unavailable
        and breaking the class. Nevertheless, you can set `L1_wt` to a very small value, such as 1e-9,
        to obtain close-to-ridge behavior while still obtaining the necessary statistics.
    fit_intercept : bool, optional (default=True)
        Whether to fit an intercept term in the model.
    """

    def __init__(
        self,
        family: str = "gaussian",
        link: Optional[str] = None,
        alpha: float = 0.0,
        L1_wt: float = 1e-6,
        fit_intercept: bool = True,
    ):
        """
        Initialize self.

        Parameters
        ----------
        family :
            The distributional assumption of the model.
        link:
            the GLM link function
        alpha :
            The penalty weight. If a scalar, the same penalty weight applies to all variables in the model.
            If a vector, it must have the same length as params, and contains a penalty weight for each coefficient.
        L1_wt :
            The `L1_wt` parameter represents the weight of the L1 penalty term in the model and
            should be within the range 0 to 1. A value of 0 corresponds to ridge regression,
            while a value of 1 corresponds to lasso regression. However, for obtaining statistics,
            `L1_wt` should be set to a value greater than 0. If it is set to 0.0, statsmodels returns
            a ridge regularized wrapper without refitting the model, making the statistics unavailable
            and breaking the class. Nevertheless, you can set `L1_wt` to a very small value, such as 1e-9,
            to obtain close-to-ridge behavior while still obtaining the necessary statistics.

        fit_intercept :
            Whether to fit an intercept term in the model.
        """
        self.family = family
        self.link = link
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.model = None
        self.result = None
        self.fit_intercept = fit_intercept
        self.objective = _map_family_link(family=family, link=link)

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        """
        Fit the model to the data.

        Notes
        -----
        In statsmodels and GLMs in general, you can use either an offset or a weight to account for
        differences in exposure between observations. However, if you choose to use an offset,
        you need to pass the number of cases (ncl) instead of the frequency and set the offset to
        the logarithm of the exposure due to the log link function. It is recommended to use the frequency
        and the weights instead of the offset because this ensures that all models have the same inputs.
        To use the frequency and the weights, you can fit the model using the following code:

        ```python
        self.model = sm.GLM(endog=y, exog=X, var_weights=sample_weight, family=self.family)
        ```

        This is equivalent to using the exposure and the log of the exposure internally, which can be done using the following code:

        ```python
        self.model = sm.GLM(endog=y, exog=sm.add_constant(X), exposure=sample_weight, family=sm.families.Poisson())
        self.result = self.model.fit()
        ```

        Parameters
        ----------
        X :
            array-like, shape (n_samples, n_features)
            The input data.
        y :
            array-like, shape (n_samples,)
            The target values.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights.

        Returns
        -------
        self : object
            Returns self.
        """

        # see the if kwargs.get("L1_wt", 1) == 0 condition in
        # https://www.statsmodels.org/dev/_modules/statsmodels/genmod/generalized_linear_model.html#GLM.fit_regularized
        # workaround to get the statistics
        if self.alpha == 0.0:
            self.alpha = 1e-9

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"pred_{i}" for i in range(X.shape[1])]

        if self.fit_intercept:
            X = sm.add_constant(X)
            X = X.rename(columns={"const": "Intercept"})
        else:
            X = drop_existing_sm_constant_from_df(X)

        self.n_features_in_ = X.shape[1]

        self.model = sm.GLM(
            endog=y,
            exog=X,
            var_weights=sample_weight,
            family=self.objective,
        )

        self.result = self.model.fit_regularized(
            method="elastic_net", alpha=self.alpha, L1_wt=self.L1_wt, refit=True
        )
        self.coef_ = self.result.params
        self.bse_ = self.result.bse
        self.deviance_ = self.result.deviance
        self.pseudo_rsquared_ = self.result.pseudo_rsquared()
        self.aic_ = self.result.aic
        self.bic_ = self.result.bic_llf
        self.pvalues_ = self.result.pvalues
        self.tvalues_ = self.result.tvalues
        self.pearson_chi2_ = self.result.pearson_chi2

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X :
            array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples,)
            The predicted target values.

        Raises
        ------
        ValueError
            If the model has not been fit.
        """
        if self.model is None:
            raise ValueError("Fit the model first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"pred_{i}" for i in range(X.shape[1])]

        if self.fit_intercept:
            X = sm.add_constant(X)
            X = X.rename(columns={"const": "Intercept"})

        return self.result.predict(X)

    def get_coef(self):
        """
        Get the estimated coefficients of the fitted model.

        Returns
        -------
        coef_ : array-like, shape (n_features,)
            The estimated coefficients of the fitted model.
        """
        return self.coef_

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        """
        Return the deviance of the fitted model.

        Parameters
        ----------
        X :
            array-like, shape (n_samples, n_features)
            The input data.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights.

        Returns
        -------
        deviance : float
            The deviance of the fitted model.
        """
        mu = self.objective.link.inverse(self.predict(X))

        var_weights = sample_weight if sample_weight is not None else 1.0
        return self.objective.deviance(endog=y, mu=mu, var_weights=var_weights)

    def summary(self):
        """
        Print a summary of the fitted model.

        Returns
        -------
        summary : str
            The summary of the fitted model.
        """
        return self.result.summary()


def weighted_cross_val_score(estimator, X, y, sample_weight=None, cv=5, n_jobs=-1):
    """
    Perform cross-validation for a scikit-learn estimator with a score function that requires sample_weight.

    Parameters
    ----------
    estimator : estimator
        The scikit-learn estimator object.
    X : array-like of shape (n_samples, n_features)
        The input features.
    y : array-like of shape (n_samples,)
        The target variable.
    sample_weight : array-like of shape (n_samples,), optional
        The sample weights for each data point.
    cv : int, default=5
        The number of cross-validation folds.
    n_jobs:
        the number of processes

    Returns
    -------
    scores : array of shape (cv,)
        The list of scores for each fold.
    average_score : float
        The average score across all folds.

    """

    # logging.info("Starting cross-validation...")

    splitter = (
        KFold(n_splits=cv) if len(np.unique(y)) > 2 else StratifiedKFold(n_splits=cv)
    )

    if not hasattr(estimator, "score") or not callable(getattr(estimator, "score")):
        raise ValueError(
            "The estimator does not have a score method that takes a sample_weight argument."
        )

    with Parallel(n_jobs=n_jobs) as parallel:
        scores = parallel(
            delayed(_fit_and_score)(
                estimator, X, y, train_index, test_index, sample_weight
            )
            for train_index, test_index in splitter.split(X)
        )

    # logging.info("Finished cross-validation.")
    return scores


def _fit_and_score(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    train_index: np.ndarray,
    test_index: np.ndarray,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
) -> float:
    """
    Fit and score an estimator on a specified train-test split.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object implementing the scikit-learn estimator interface.
    X : Union[pd.DataFrame, np.ndarray]
        The input features, can be either a pandas DataFrame or a numpy array.
    y : Union[pd.Series, np.ndarray]
        The target values, can be either a pandas Series or a numpy array.
    train_index : np.ndarray
        Array of indices representing the training data.
    test_index : np.ndarray
        Array of indices representing the test data.
    sample_weight : Optional[Union[pd.Series, np.ndarray]], default=None
        Sample weights to be used during training. Can be either a pandas Series or a numpy array.

    Returns
    -------
    float
        The score of the estimator on the test data.

    Raises
    ------
    ValueError
        If the input data is not of the correct format.
    """
    # X = X.values if isinstance(X, pd.DataFrame) else X
    y = y.values if isinstance(y, pd.Series) else y
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    if sample_weight is not None:
        sample_weight = (
            sample_weight.values
            if isinstance(sample_weight, pd.Series)
            else sample_weight
        )
        sample_weight_train = sample_weight[train_index]
        sample_weight_test = sample_weight[test_index]
        estimator.fit(X_train, y_train, sample_weight=sample_weight_train)
        score = estimator.score(X_test, y_test, sample_weight=sample_weight_test)
    else:
        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)

    return score


def grid_search_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    n_iterations: int = 10,
    family: str = "gaussian",
    link: Optional[str] = None,
    score: str = "bic",
    fit_intercept: bool = True,
    n_jobs: int = -1
) -> EnetGLM:
    """
    Perform grid search cross-validation for an Elastic Net Generalized Linear Model (EnetGLM).

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        The input features, can be either a pandas DataFrame or a numpy array.
    y : Union[pd.Series, np.ndarray]
        The target values, can be either a pandas Series or a numpy array.
    sample_weight : Optional[Union[pd.Series, np.ndarray]], default=None
        Sample weights to be used during training. Can be either a pandas Series or a numpy array.
    n_iterations : int, default=10
        Number of iterations for the grid search.
    family : str, default="gaussian"
        The family of the GLM. Options: "gaussian", "poisson", "gamma", "negativebinomial", "binomial", "tweedie".
    link : str, optional
        the GLM link function. It can be any of the: "identity", "log", "logit", "probit", "cloglog", "inverse_squared"
    score : str, default="bic"
        The score to use for model selection. Options: "bic" (Bayesian Information Criterion) or "mean_cv" (mean cross-validation score).
    n_jobs:
        the number of processes

    Returns
    -------
    EnetGLM
        The best estimator found after grid search cross-validation.

    Raises
    ------
    ValueError
        If the input data is not of the correct format or if an invalid family or score value is provided.
    """
    estimator = EnetGLM(family=family, link=link, L1_wt=1.0, fit_intercept=fit_intercept)

    # Check if X and y are pandas DataFrames/Series and convert them to numpy arrays if necessary
    # X = check_array(X, accept_sparse=True, force_all_finite=False)
    y = check_array(y, ensure_2d=False, force_all_finite=False)

    if score not in ["bic", "deviance"]:
        raise ValueError("Invalid score value. Options are: 'bic' or 'deviance'.")

    grid = np.logspace(-3, 3, n_iterations)
    param_score = []

    for param in grid:
        estimator = clone(estimator)
        estimator.set_params(
            **{
                "alpha": param,
                "L1_wt": 1.0,
                "fit_intercept": fit_intercept,
                "family": family,
            }
        )

        if score == "bic":
            estimator.fit(X=X, y=y, sample_weight=sample_weight)
            param_score.append(estimator.bic_)
        else:
            scores = weighted_cross_val_score(
                estimator, X, y, sample_weight=sample_weight, cv=5, n_jobs=n_jobs
            )
            param_score.append(np.mean(scores))
    # min deviance or min BIC
    best_alpha_value = grid[np.argmin(param_score)]
    best_estimator = clone(estimator)
    best_estimator.set_params(
        **{
            "alpha": best_alpha_value,
            "L1_wt": 1.0,
            "fit_intercept": fit_intercept,
            "family": family,
        }
    )
    best_estimator.fit(X, y, sample_weight=sample_weight)

    return best_estimator


class LassoFeatureSelection(BaseEstimator, TransformerMixin):
    """
    LassoFeatureSelection performs feature selection using GLM Lasso regularization.

    Parameters
    ----------
    family : str, (default="gaussian")
        The distributional assumption of the model. It can be any of the statsmodels distribution:
        "gaussian", "binomial", "poisson", "gamma", "negativebinomial", "tweedie"
    link : str, optional
        the GLM link function. It can be any of the: "identity", "log", "logit", "probit", "cloglog", "inverse_squared"
    n_iterations : int, default=10
        Number of iterations for the grid search.
    score : str, default="bic"
        The score to use for model selection. Options: "bic" (Bayesian Information Criterion) or "mean_cv" (mean cross-validation score).
    n_jobs: int, default=-1
        the number of processes. -1 means all the processes

    Attributes
    ----------
    family : str
        The family of the GLM.
    n_iterations : int
        Number of iterations for the grid search.
    best_estimator_ : EnetGLM
        The best estimator found after grid search cross-validation.
    selected_features_ : ndarray
        The selected feature names.
    support_ : ndarray
        The support of selected features (True for selected, False otherwise).
    feature_names_in_ : ndarray
        The input feature names.
    score : str
        The score used for model selection.
    n_jobs: int
        the number of processes. -1 means all the processes

    Methods
    -------
    fit(X, y=None, sample_weight=None)
        Fit the LassoFeatureSelection model and select the best features.
    transform(X)
        Transform the input data to keep only the selected features.
    get_feature_names_out()
        Get the names of the selected features.

    """

    def __init__(
        self,
        family: str = "gaussian",
        link: Optional[str] = None,
        n_iterations: int = 10,
        score: str = "bic",
        fit_intercept: bool = True,
        n_jobs: int = -1
    ):
        self.family = family
        self.link = link
        self.n_iterations = n_iterations
        self.best_estimator_ = None
        self.selected_features_ = None
        self.support_ = None
        self.feature_names_in_ = None
        self.score = score
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        """
        Fit the LassoFeatureSelection model and select the best features.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input features, can be either a pandas DataFrame or a numpy array.
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            The target values, can be either a pandas Series or a numpy array.
        sample_weight : Optional[Union[pd.Series, np.ndarray]], default=None
            Sample weights to be used during training. Can be either a pandas Series or a numpy array.

        Returns
        -------
        LassoFeatureSelection
            The fitted LassoFeatureSelection model.

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"pred_{i}" for i in range(X.shape[1])]

        if not self.fit_intercept:
            X = drop_existing_sm_constant_from_df(X)

        self.feature_names_in_ = (
            X.columns.insert(0, "Intercept")
            if self.fit_intercept and "Intercept" not in X.columns
            else X.columns
        )

        self.best_estimator_ = grid_search_cv(
            family=self.family,
            link=self.link,
            X=X,
            y=y,
            sample_weight=sample_weight,
            n_iterations=self.n_iterations,
            score=self.score,
            fit_intercept=self.fit_intercept,
            n_jobs=self.n_jobs,
        )
        self.support_ = self.best_estimator_.coef_ != 0
        self.selected_features_ = self.feature_names_in_[self.support_]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform the input data to keep only the selected features.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input features, can be either a pandas DataFrame or a numpy array.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            The transformed data with only the selected features.

        """

        if self.fit_intercept:
            X = sm.add_constant(X)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            # if not a DF, assuming the col orders is
            # the same, as required anyway
            X.columns = self.feature_names_in_

        X = X.rename(columns={"const": "Intercept"}) if "const" in X.columns else X
        return X[self.selected_features_]

    def get_feature_names_out(self) -> np.ndarray:
        """
        Get the names of the selected features.

        Returns
        -------
        np.ndarray
            The names of the selected features.

        """
        return self.feature_names_in_[self.support_]


def drop_existing_sm_constant_from_df(X):
    X = X.drop(columns=["Intercept"]) if "Intercept" in X.columns else X
    X = X.drop(columns=["const"]) if "const" in X.columns else X
    X = X.drop(columns=["intercept"]) if "intercept" in X.columns else X
    return X
