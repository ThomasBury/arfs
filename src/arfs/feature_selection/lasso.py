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

from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    clone,
    is_classifier,
)
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_is_fitted

from ..utils import validate_sample_weight, validate_target


def _glm_frame(X):
    if not isinstance(X, pd.DataFrame):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        X = pd.DataFrame(X, columns=[f"pred_{i}" for i in range(X.shape[1])])
    if not X.columns.is_unique:
        raise ValueError("X must not contain duplicate columns.")
    return X


def _positive_weight_rows(X, y, sample_weight):
    if sample_weight is not None and np.any(sample_weight == 0):
        keep = sample_weight > 0
        X = X.iloc[keep] if isinstance(X, pd.DataFrame) else np.asarray(X)[keep]
        return X, np.asarray(y)[keep], sample_weight[keep]
    return X, y, sample_weight


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
    Elastic-net variable selection followed by an unpenalized GLM refit.

    The penalty chooses nonzero coefficients; returned coefficients and
    predictions use an unpenalized refit on those columns. Summary statistics
    use the selection data and do not provide valid post-selection inference.
    Zero-weight rows are omitted before fitting and scoring, after response
    and weight validation. AIC/BIC count the rank of selected mean parameters,
    excluding dispersion, and use the retained observation count. Variance-
    weighted fixed-scale families have a quasi-likelihood interpretation;
    finite information criteria do not establish calibrated model selection.

    Parameters
    ----------
    family : str, (default="gaussian")
        The distributional assumption of the model. It can be any of the statsmodels distribution:
        "gaussian", "binomial", "poisson", "gamma", "negativebinomial", "tweedie"
    link : str, optional
        the GLM link function. It can be any of the: "identity", "log", "logit", "probit", "cloglog", "inverse_squared"
    alpha : float, optional (default=0.0)
        Penalty strength, not the mixing fraction. Zero uses a tiny positive
        fitting penalty (1e-9) to retain the backend refit/statistics path.
    L1_wt : float, optional (default=1e-6)
        L1 fraction of the selection penalty, with 0 < L1_wt <= 1.
        One selects using lasso; smaller values mix L1 and L2 penalties before
        refitting. Returned coefficients do not retain ridge shrinkage. Pure
        ridge (zero) is unsupported because its backend result lacks refit
        statistics required by this wrapper.
    fit_intercept : bool, optional (default=True)
        Add an intercept only when training inputs have no nonzero constant.
        Supplied columns, including constants, are always preserved. The fitted
        design governs prediction; named inputs are aligned to training names,
        while arrays must have the fitted input width and order.

    Attributes
    ----------
    feature_names_in_ : ndarray
        Original input names; arrays receive names pred_0, pred_1, etc.
    n_features_in_ : int
        Number of original input columns.
    design_feature_names_ : ndarray
        Actual coefficient names, including an automatically added intercept.
    added_intercept_ : str or None
        Name of the generated column, or None when none was added. The name
        is Intercept, with a numbered suffix if an input already uses it.
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
            L1 fraction of the selection penalty, with 0 < L1_wt <= 1.
            The surviving columns are then refit without regularization.

        fit_intercept :
            Add an intercept only if no nonzero constant was supplied.
        """
        self.family = family
        self.link = link
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.model = None
        self.result = None
        self.fit_intercept = fit_intercept

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        """
        Fit using the current family/link parameters and record one design.

        Notes
        -----
        For Poisson rates with a log link, positive exposure as a variance
        weight gives corresponding coefficient estimates and prediction loss
        to counts with log-exposure as an offset. This equivalence does not
        extend to arbitrary family/weight combinations. Fixed-scale variance-
        weighted likelihoods have a quasi-likelihood interpretation; ordinary
        likelihood-based significance claims are not justified by finite BIC.
        Zero-weight rows do not determine the design, penalty scaling or
        likelihood observation count. Positive weights are not normalized.

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
        alpha = 1e-9 if self.alpha == 0.0 else self.alpha
        if not 0 < self.L1_wt <= 1:
            raise ValueError("L1_wt must satisfy 0 < L1_wt <= 1 for an unpenalized refit.")

        y = validate_target(y, len(X)).to_numpy()
        sample_weight = validate_sample_weight(sample_weight, len(X))
        X = _glm_frame(X)
        X, y, sample_weight = _positive_weight_rows(X, y, sample_weight)
        input_names = X.columns.to_numpy(copy=True)
        added_intercept = None
        if self.fit_intercept:
            design = sm.add_constant(X, has_constant="skip")
            if design.shape[1] > X.shape[1]:
                added_intercept = "Intercept"
                suffix = 1
                while added_intercept in X.columns:
                    added_intercept = f"Intercept_{suffix}"
                    suffix += 1
                design.columns = [added_intercept, *X.columns]
            X = design

        objective = _map_family_link(family=self.family, link=self.link)
        model = sm.GLM(
            endog=y,
            exog=X,
            var_weights=sample_weight,
            family=objective,
        )

        result = model.fit_regularized(
            method="elastic_net", alpha=alpha, L1_wt=self.L1_wt, refit=True
        )
        self.model, self.result, self.objective = model, result, objective
        self.coef_ = self.result.params
        self.bse_ = self.result.bse
        self.deviance_ = self.result.deviance
        self.pseudo_rsquared_ = self.result.pseudo_rsquared()
        selected = np.asarray(self.coef_) != 0
        k_params = np.linalg.matrix_rank(model.exog[:, selected]) if selected.any() else 0
        # The backend refit's IC bookkeeping counts an extra parameter and row.
        self.aic_ = -2 * self.result.llf + 2 * k_params
        self.bic_ = -2 * self.result.llf + np.log(model.nobs) * k_params
        self.pvalues_ = self.result.pvalues
        self.tvalues_ = self.result.tvalues
        self.pearson_chi2_ = self.result.pearson_chi2
        self.feature_names_in_ = input_names
        self.n_features_in_ = len(input_names)
        self.design_feature_names_ = X.columns.to_numpy(copy=True)
        self.added_intercept_ = added_intercept
        return self

    def _prepare_design(self, X):
        check_is_fitted(self, "design_feature_names_")
        named = isinstance(X, pd.DataFrame)
        X = _glm_frame(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the fitted number of input features.")
        if named:
            if not X.columns.isin(self.feature_names_in_).all():
                raise ValueError("X columns must match the fitted input columns.")
            X = X.loc[:, self.feature_names_in_]
        else:
            X.columns = self.feature_names_in_
        if self.added_intercept_ is not None:
            X = X.copy()
            X.insert(0, self.added_intercept_, 1.0)
        return X

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
        NotFittedError
            If the model has not been fit.
        ValueError
            If input columns are duplicated or differ from the fitted schema.
        """
        X = self._prepare_design(X)
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
        Return negative response-scale deviance (higher is better).

        Parameters
        ----------
        X :
            array-like, shape (n_samples, n_features)
            The input data.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Negative total deviance, using variance weights and unit scale.
        """
        y = validate_target(y, len(X)).to_numpy()
        sample_weight = validate_sample_weight(sample_weight, len(X))
        X, y, sample_weight = _positive_weight_rows(X, y, sample_weight)
        mu = self.predict(X)

        var_weights = sample_weight if sample_weight is not None else 1.0
        return -self.objective.deviance(endog=y, mu=mu, var_weights=var_weights)

    def summary(self):
        """
        Print a summary of the fitted model.

        Returns
        -------
        summary : str
            The summary of the fitted model.

        Notes
        -----
        These statistics use the same data as variable selection and do not
        account for that selection; ordinary p-values are not selection-adjusted.
        The backend summary is returned unchanged. Its information criteria
        can differ from the wrapper's rank/count-corrected ``aic_`` and ``bic_``.
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
        The sample weights for each data point. For an EnetGLM argument,
        zero-weight rows are omitted before splitting. Other estimators retain
        their existing zero-weight and positive-fold-mass contracts.
    cv : int, default=5
        The number of cross-validation folds. Classifiers and binomial GLMs
        with discrete targets use stratified folds; other targets use KFold.
    n_jobs:
        the number of processes

    Returns
    -------
    scores : list of length cv
        Each independent clone's validation score, with validation weights
        when supplied. EnetGLM returns negative total deviance per fold.

    """

    validate_target(y, len(X))
    sample_weight = validate_sample_weight(sample_weight, len(X))
    if isinstance(estimator, EnetGLM):
        X, y, sample_weight = _positive_weight_rows(X, y, sample_weight)
    splitter = check_cv(
        cv,
        y,
        classifier=is_classifier(estimator)
        or getattr(estimator, "family", None) == "binomial",
    )

    if not hasattr(estimator, "score") or not callable(estimator.score):
        raise ValueError(
            "The estimator does not have a score method that takes a sample_weight argument."
        )

    with Parallel(n_jobs=n_jobs) as parallel:
        scores = parallel(
            delayed(_fit_and_score)(
                estimator, X, y, train_index, test_index, sample_weight
            )
            for train_index, test_index in splitter.split(X, y)
        )

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
        Sample weights for the chosen estimator, as a pandas Series or NumPy array.

    Returns
    -------
    float
        The score of the estimator on the test data.

    Raises
    ------
    ValueError
        If the input data is not of the correct format.
    """
    y = validate_target(y, len(X)).to_numpy()
    sample_weight = validate_sample_weight(sample_weight, len(X))
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    else:
        X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    estimator = clone(estimator)

    if sample_weight is not None:
        sample_weight_train = validate_sample_weight(
            sample_weight[train_index], len(train_index)
        )
        sample_weight_test = validate_sample_weight(
            sample_weight[test_index], len(test_index)
        )
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
    n_jobs: int = -1,
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
        Variance weights, as a pandas Series or NumPy array. Zero-weight rows
        are omitted before fitting or forming validation folds.
    n_iterations : int, default=10
        Number of iterations for the grid search.
    family : str, default="gaussian"
        The family of the GLM. Options: "gaussian", "poisson", "gamma", "negativebinomial", "binomial", "tweedie".
    link : str, optional
        the GLM link function. It can be any of the: "identity", "log", "logit", "probit", "cloglog", "inverse_squared"
    score : str, default="bic"
        Options: "bic" minimizes training BIC; "deviance" minimizes the mean
        total validation deviance across five folds (maximizes negative scores).
    fit_intercept : bool, default=True
        Add an intercept when no nonzero training constant exists. Supplied
        columns are preserved.
    n_jobs:
        the number of processes

    Returns
    -------
    EnetGLM
        The best estimator found after grid search cross-validation.

    Raises
    ------
    ValueError
        If the inputs, family or score are invalid, or any candidate criterion
        is nonfinite. The search fails without discarding invalid candidates.
    """
    estimator = EnetGLM(
        family=family, link=link, L1_wt=1.0, fit_intercept=fit_intercept
    )

    # Check if X and y are pandas DataFrames/Series and convert them to numpy arrays if necessary
    # X = check_array(X, accept_sparse=True, ensure_all_finite=False)
    y = validate_target(y, len(X)).to_numpy()
    sample_weight = validate_sample_weight(sample_weight, len(X))

    if score not in ["bic", "deviance"]:
        raise ValueError("Invalid score value. Options are: 'bic' or 'deviance'.")

    X, y, sample_weight = _positive_weight_rows(X, y, sample_weight)
    grid = np.logspace(-3, 3, n_iterations)
    param_score = []

    for param in grid:
        estimator = clone(estimator)
        estimator.set_params(
            alpha=param, L1_wt=1.0, fit_intercept=fit_intercept, family=family
        )

        if score == "bic":
            estimator.fit(X=X, y=y, sample_weight=sample_weight)
            param_score.append(estimator.bic_)
        else:
            scores = weighted_cross_val_score(
                estimator, X, y, sample_weight=sample_weight, cv=5, n_jobs=n_jobs
            )
            param_score.append(np.mean(scores))
        if not np.isfinite(param_score[-1]):
            raise ValueError(
                f"{score} for alpha={param:g} is not finite; cannot compare penalty candidates."
            )
    best_index = np.argmin(param_score) if score == "bic" else np.argmax(param_score)
    best_alpha_value = grid[best_index]
    best_estimator = clone(estimator)
    best_estimator.set_params(
        alpha=best_alpha_value, L1_wt=1.0, fit_intercept=fit_intercept, family=family
    )
    best_estimator.fit(X, y, sample_weight=sample_weight)

    return best_estimator


class LassoFeatureSelection(BaseEstimator, TransformerMixin):
    """
    LassoFeatureSelection selects variables with a penalty and refits surviving
    columns in an unpenalized GLM. Returned coefficients do not retain shrinkage.
    Zero-weight rows are omitted before fitting and tuning. Any nonfinite search
    criterion raises an error rather than returning an arbitrary penalty.

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
        Options: "bic" minimizes training BIC; "deviance" minimizes the mean
        total validation deviance across five folds (maximizes negative scores).
    n_jobs: int, default=-1
        the number of processes. -1 means all the processes
    fit_intercept : bool, default=True
        Add an intercept when no nonzero training constant exists. Supplied
        columns are preserved. A selected generated intercept is included in
        transform output, but not in input metadata or ``support_``.

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
        Support for original input columns, excluding a generated intercept.
    design_feature_names_ : ndarray
        Names of the fitted design, including a generated intercept if needed.
    design_support_ : ndarray
        Nonzero-coefficient support for the fitted design. A selected generated
        intercept remains in transform output and ``selected_features_``.
    feature_names_in_ : ndarray
        The input feature names.
    n_features_in_ : int
        Number of original input columns, excluding a generated intercept.
    score : str
        The score used for model selection.
    n_jobs: int
        the number of processes. -1 means all the processes


    """

    def __init__(
        self,
        family: str = "gaussian",
        link: Optional[str] = None,
        n_iterations: int = 10,
        score: str = "bic",
        fit_intercept: bool = True,
        n_jobs: int = -1,
    ):
        self.family = family
        self.link = link
        self.n_iterations = n_iterations
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
        X = _glm_frame(X)
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
        self.feature_names_in_ = self.best_estimator_.feature_names_in_.copy()
        self.n_features_in_ = self.best_estimator_.n_features_in_
        self.design_feature_names_ = self.best_estimator_.design_feature_names_.copy()
        self.design_support_ = np.asarray(self.best_estimator_.coef_) != 0
        self.support_ = self.design_support_[
            pd.Index(self.design_feature_names_).get_indexer(self.feature_names_in_)
        ]
        self.selected_features_ = self.design_feature_names_[self.design_support_]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Apply the fitted design and retain its selected columns.

        Named inputs are aligned to the complete fitted input schema; missing,
        extra or duplicate columns are rejected. Arrays use fitted input order.
        A generated intercept is recreated even for a single row.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input features, can be either a pandas DataFrame or a numpy array.

        Returns
        -------
        pd.DataFrame
            The transformed data with only the selected features.

        """

        check_is_fitted(self, "selected_features_")
        return self.best_estimator_._prepare_design(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Get the names of the selected features.

        Parameters
        ----------
        input_features : array-like, optional
            If supplied, must match the original input names and order.

        Returns
        -------
        np.ndarray
            The names of the selected features.

        """
        check_is_fitted(self, "selected_features_")
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_.")
        return self.selected_features_.copy()


def drop_existing_sm_constant_from_df(X):
    X = X.drop(columns=["Intercept"]) if "Intercept" in X.columns else X
    X = X.drop(columns=["const"]) if "const" in X.columns else X
    X = X.drop(columns=["intercept"]) if "intercept" in X.columns else X
    return X
