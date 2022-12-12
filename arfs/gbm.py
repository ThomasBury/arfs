"""
This module provides a class to train a base lightgbm and catboost models using early stopping as default.
The target can be finite discrete (classification) or continuous (regression).
It can boost from an initial score (aka baseline for catboost), take sample weight as input.

**The module structure is the following:**

- The ``GradientBoosting`` main class to train a lightGBM  or catboost with early stopping

"""
from typing import Union, Tuple, List, Optional, Dict, Any

from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
)
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation
from catboost import CatBoost, Pool
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
from datetime import date
import warnings
import gc
import os
from pathlib import Path

qual_colors_list = [
    (0.188235, 0.635294, 0.854902),
    (0.898039, 0.682353, 0.219608),
    (0.988235, 0.309804, 0.188235),
    (0.427451, 0.564706, 0.309804)
    ]

def set_my_plt_style(height=3, width=5, linewidth: Any = 2):
    """This set the style of matplotlib to fivethirtyeight with some modifications (colours, axes)

    Parameters
    ----------
    height : float, default = 3
        fig height in inches (yeah they're still struggling with the metric system) (default ``3``)
    width : float, default = 5
        fig width in inches (yeah they're still struggling with the metric system) (default ``5``)
    linewidth :
         (default ``2``)

    """
    plt.style.use("fivethirtyeight")
    my_colors_list = qual_colors_list  # Bold_10.hex_colors
    # myorder = [2, 3, 4, 1, 0, 6, 5, 8, 9, 7]
    # my_colors_list = [my_colors_list[i] for i in myorder]
    bckgnd_color = "#f5f5f5"
    params = {
        "figure.figsize": (width, height),
        "axes.prop_cycle": plt.cycler(color=my_colors_list),
        "axes.facecolor": bckgnd_color,
        "patch.edgecolor": bckgnd_color,
        "figure.facecolor": bckgnd_color,
        "axes.edgecolor": bckgnd_color,
        "savefig.edgecolor": bckgnd_color,
        "savefig.facecolor": bckgnd_color,
        "grid.color": "#d2d2d2",
        "lines.linewidth": linewidth,
        "grid.alpha": 0.5,
    }  # plt.cycler(color=my_colors_list)
    mpl.rcParams.update(params)

class GradientBoosting:
    """Performs the training of a base lightGBM/CatBoost using early stopping. It works for any of the
    supported loss function (lightGBM/CatBoost), so for regression and classification you can use an instance of
    this class. For the early stopping process, 20% of the data set is used and a fix seed is used for
    reproducibility.

    The resulting model can be saved at the desired location.
    Last, you can pass relevant lightGBM/Catboost parameters and/or sample weights (exposure, etc.) if needed.

    Init score of Booster to start from, if required (like for GLM residuals modelling using GBM).

    For the catboost (method='cat') you can set 'allow_writing_files': False in the params dictionary.
    It will disable the writing of analytical and snapshot files during training (text and json files).
    However, the snapshot and data visualization tools are unavailable.

    Parameters
    ----------
    method : str, default "lgb"
        the boosting method, either "lgb" for lightGBM or "cat" for catboost
    cat_feat : List[str], 'auto' or None,
        The list of column names of the categorical predictors. For catboost, much more efficient if those columns
        are of dtype pd.Categorical. For lightGBM, most of the time better to integer encode and NOT consider
        them as categorical (set this parameter as None).
    params : dict, default=None
        you can pass the parameters that you want to lightGBM/Catboost, as long as they are valid.
        If None, default parameters are passed.
    stratified : bool, default=False
        stratified shuffle split for the early stopping process. For classification problem, it guarantees
        the same proportion
    show_learning_curve : bool, default=True
        if show or not the learning curve
    verbose_eval : int, default=50
        period for printing the train and validation results. If < 1, no output

    Attributes
    ----------
    method : str
        the boosting method
    cat_feat : Union[str, List[str], None]
        The list of categorical predictors after pre-processing
    model_params : Dict
        the dictionary of model parameters
    learning_curve : plt.figure
        the learning curve
    is_init_score :  bool
        boosted from an initial score or not
    stratified : bool
        either if stratified shuffle split was used or not for the early stopping process

    Example
    -------
    >>> # set up the trainer
    >>> save_path = "C:/Users/mtpl_bi_pp/base/"
    >>> gbm_model = GradientBoosting(objective='tweedie',
    >>>                              method='lgb',
    >>>                              cat_feat='auto',
    >>>                              stratified=False,
    >>>                              params={
    >>>                                 'objective': 'tweedie',
    >>>                                 'tweedie_variance_power': 1.1
    >>>                             })
    >>>
    >>> # train the model
    >>> gbm_model.fit(X=X_tr,y=y_tr,sample_weight=exp_tr)
    >>>
    >>> # predict new values (test set)
    >>> y_bt = gbm_model.predict(X_tt)
    >>>
    >>> # save the model
    >>> gbm_model.save(save_path='C:/models/', name="my_fancy_model")

    """

    def __init__(
        self, method="lgb", cat_feat="auto", params=None, stratified=False, show_learning_curve=True, verbose_eval=50
    ):
        self.method = method
        self.model = None
        self.cat_feat = cat_feat
        self.model_params = None
        self.params = params
        self.learning_curve = None
        self.is_init_score = False
        self.stratified = stratified
        self.show_learning_curve = show_learning_curve
        self.verbose_eval = verbose_eval

    def __repr__(self):
        s = (
            "GradientBoosting(method={method}, \n"
            "                 cat_feat={cat_feat},\n"
            "                 params={params})".format(method=self.method, cat_feat=self.cat_feat, params=self.params)
        )
        return s

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        groups=None,
    ):
        """Fit the lightGBM/Catboost either using the python API and early stopping

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            the predictors' matrix
        y : pd.Series or np.ndarray
            the target series/array
        sample_weight : pd.Series or np.ndarray, optional
            the sample_weight series/array, if relevant. If not None, it should be of the same length as the
            target (default ``None``)
        init_score : pd.Series or np.ndarray, optional
            the initial score to boost from (series/array), if relevant. If not None,
            it should be of the same length as the target (default ``None``)
        groups : pd.Series or np.ndarray, optional
            the groups (e.g. polID) for robust cross validation.
            The same group will not appear in two different folds.

        """
        if (self.params is not None) and (not isinstance(self.params, dict)):
            raise TypeError("params should be either None or a dictionary of lightgbm params")
        elif (isinstance(self.params, dict)) and ("objective" not in self.params.keys()):
            raise KeyError("Provide the objective in the params dict")

        if self.cat_feat == "auto":
            self.cat_feat = list(set(list(X.columns)) - set(list(X.select_dtypes(include=[np.number]))))

        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if sample_weight is not None:
            if not isinstance(sample_weight, pd.Series):
                sample_weight = pd.Series(sample_weight)

        if init_score is not None:
            self.is_init_score = True
            if not isinstance(init_score, pd.Series):
                init_score = pd.Series(init_score)

        if self.method == "cat":
            output = _fit_early_stopped_catboost(
                X=X,
                y=y,
                sample_weight=sample_weight,
                params=self.params,
                init_score=init_score,
                cat_feat=self.cat_feat,
                stratified=self.stratified,
                groups=groups,
                learning_curve=self.show_learning_curve,
                verbose_eval=self.verbose_eval,
            )
        elif self.method == "lgb":
            output = _fit_early_stopped_lgb(
                X=X,
                y=y,
                sample_weight=sample_weight,
                params=self.params,
                init_score=init_score,
                cat_feat=self.cat_feat,
                stratified=self.stratified,
                groups=groups,
                learning_curve=self.show_learning_curve,
                verbose_eval=self.verbose_eval,
            )
        else:
            raise Exception("method not found")

        if self.show_learning_curve:
            self.model, self.learning_curve = output[0], output[1]
        else:
            self.model = output

        self.model_params = self.model.params if self.method == "lgb" else self.model.get_all_params()

    def predict(self, X, predict_proba=False):
        """Predict the new values using the fitted model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            the predictors' matrix
        predict_proba : bool, default=False
            returns probabilities (only for classification) (default ``False``)

        Raises
        ------
        Exception
            "method not found" if the method specified in the init differs from "lgb" or "cat"

        """
        if self.is_init_score:
            raise AttributeError(
                "The model is fitted from an initial score, use the `predict_raw` method instead\n"
                "Please also check what is returned by the predicted method, for the raw version\n"
                "you might have to apply `exp` (CatBoost returns rates if the loss is \n"
                " Poisson or Tweedie)"
            )

        obj_fn = self.model_params["objective"] if self.method == "lgb" else self.model_params["loss_function"]
        # self.params['objective']
        # LightGBM
        if self.method == "lgb":
            if not predict_proba and ("binary" in obj_fn):
                # rounding the values and convert to integer
                return self.model.predict(X).round(0).astype(int)
            elif not predict_proba and ("multi" in obj_fn):
                y_pred = self.model.predict(X)
                # find the class using the argmax function
                # one proba per class and pick the largest prob
                return np.array([np.argmax(line) for line in y_pred])
            else:
                return self.model.predict(X)

        # CatBoost
        elif self.method == "cat":
            test_pool = Pool(X, cat_features=self.cat_feat)
            if predict_proba:
                return self.model.predict(test_pool, prediction_type="Probability")[:, 1]
            elif ("Tweedie" in obj_fn) or ("Poisson" in obj_fn):
                return np.exp(self.model.predict(test_pool))
            else:
                return self.model.predict(test_pool)

        else:
            raise Exception("method not found")

    def predict_raw(self, X: Union[pd.DataFrame, np.ndarray], **kwargs):
        """The native (either lightGBM or CatBoost) predict method, if you need raw_score, etc.


        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            the predictors' matrix

        **kwargs : dict, optional
            optional dictionary of other parameters for the prediction.
            See the ``lightgbm`` and ``catboost`` documentation for details.

        Raises
        ------
        Exception
            "method not found" if the method specified in the init differs from "lgb" or "cat"

        """
        if self.method == "lgb":
            return self.model.predict(X, **kwargs)
        elif self.method == "cat":
            test_pool = Pool(X, cat_features=self.cat_feat)
            return self.model.predict(test_pool, **kwargs)
        else:
            raise Exception("method not found")

    def save(self, save_path=None, name=None):
        """Save method, saves the model as pkl file in the specified folder as name.pkl
        If the path is None, then the model is saved in the current working directory.
        If the name is not specified, the model is saved as 'gbm_base_model_[TIMESTAMP].pkl

        Parameters
        ----------
        save_path : str, optional
            folder where to save the model, as a pickle/joblib file
        name : str, optional
            name of the model name

        Returns
        -------
        str
            where the pkl file is saved

        """
        if name:
            file_name = f"{str(name)}.joblib"
            fig_name = f"{str(name)}_learning_curve.png"
        else:
            file_name = f"gbm_base_model_{str(self.params['objective'])}.joblib"
            fig_name = f"gbm_base_model_learning_curve_{str(date.today())}.png"

        if save_path:
            file_path = os.path.join(save_path, file_name)
            fig_path = os.path.join(save_path, fig_name)
        else:
            file_path = file_name
            fig_path = fig_name
        print(f"Saving model as: {file_path}")
        joblib.dump(self.model, file_path)

        self.learning_curve.savefig(fig_path, bbox_inches="tight")  # save the figure to file
        return file_path

    def load(self, model_path):
        if Path(model_path).is_file():
            # load model and update method
            self.model = joblib.load(model_path)
            self.method = gbm_flavour(self.model)
            self.model_params = self.model.params if self.method == "lgb" else self.model.get_all_params()
            self.cat_feat = self.model.params if self.method == "lgb" else self.model.get_cat_feature_indices()
        else:
            raise ValueError("The model file does not exist, please check the path")


def _fit_early_stopped_catboost(
    X,
    y,
    sample_weight=None,
    groups=None,
    init_score=None,
    params=None,
    cat_feat=None,
    stratified=False,
    learning_curve=True,
    verbose_eval=0,
) -> Tuple[object, plt.figure]:

    """CONVENIENCE FUNCTION, IT SHOULD NOT BE ACCESSED/IMPORTED BY THE USER

    Early stopping for catboost, using Pool and setting categorical feature, sample weights
    and baseline (init_score), if any. User defined params can be passed.
    It works for classification and regression.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        the predictors' matrix
    y : pd.Series or np.ndarray
        the target series/array
    sample_weight : pd.Series or np.ndarray, optional
        the sample_weight series/array, if relevant. If not None, it should be of the same length as the
        target (default ``None``)
    groups : pd.Series or np.ndarray, optional
        the groups (e.g. polID) for robust cross validation.
        The same group will not appear in two different folds.
    params : dict, optional
        you can pass the parameters that you want to lightGBM/Catboost, as long as they are valid.
        If None, default parameters are passed.
    init_score : pd.Series or np.ndarray, optional
        the initial score to boost from (series/array), if relevant. If not None,
        it should be of the same length as the target (default ``None``)
    cat_feat : str or list of strings, optional
        The list of column names of the categorical predictors. For catboost, much more efficient if those columns
        are of dtype pd.Categorical. For lightGBM, most of the time better to integer encode and NOT consider
        them as categorical (set this parameter as None). (default ``None``)
    stratified : bool, default=False
        stratified shuffle split for the early stopping process. For classification problem, it guarantees
        the same proportion
    learning_curve : bool, default=True
        if show or not the learning curve
    verbose_eval : int, default=0
        period for printing the train and validation results. If < 1, no output

    Raises
    ------
    TypeError
        if params are not a dictionary of CatBoost params or None

    Returns
    -------
    model : object
        model object
    fig : plt.figure
        the learning curves

    """

    (
        x_train,
        y_train,
        x_val,
        y_val,
        sample_weight_val,
        sample_weight_train,
        init_score_val,
        init_score_train,
    ) = _make_split(
        X=X,
        y=y,
        sample_weight=sample_weight,
        init_score=init_score,
        groups=groups,
        stratified=stratified,
        test_size=0.2,
    )
    d_train = Pool(x_train, label=y_train, cat_features=cat_feat)
    d_valid = Pool(x_val, label=y_val, cat_features=cat_feat)

    if sample_weight is not None:
        d_train = d_train.set_weight(sample_weight_train)
        d_valid = d_valid.set_weight(sample_weight_val)

    if init_score is not None:
        d_train = d_train.set_baseline(init_score_train)
        d_valid = d_valid.set_baseline(init_score_val)

    if params is None:
        warnings.warn("No params dictionary provided, using RMSE as default")
        params = {
            "loss_function": "RMSE",
            "n_estimators": 10000,
            "silent": False,
            "od_wait": 10,
        }
    elif not isinstance(params, dict):
        raise TypeError("params should be either None or a dictionary of catboost params")

    catmodel = CatBoost(params)
    model = catmodel.fit(d_train, eval_set=d_valid, early_stopping_rounds=10, verbose_eval=verbose_eval)

    # get the score values on training and validation set
    scoring_dic = model.get_evals_result()
    loss = list(scoring_dic["learn"].keys())[0]
    if learning_curve:
        set_my_plt_style()
        fig, ax = plt.subplots()
        ax.set_title(f"{loss} during training")
        ax.set_ylabel(loss)
        ax.set_xlabel("iterations")
        ax.plot(scoring_dic["learn"][loss], label="train")
        ax.plot(scoring_dic["validation"][loss], label="valid")
        ax.axvline(x=model.get_best_iteration(), color="grey", linestyle="--", label="best_iter")
        ax.legend(loc="best")
        plt.tight_layout()

        del d_train
        del d_valid
        gc.enable()
        gc.collect()

        return model, fig
    else:
        del d_train
        del d_valid
        gc.enable()
        gc.collect()
        return model


def _fit_early_stopped_lgb(
    X,
    y,
    sample_weight=None,
    groups=None,
    init_score=None,
    params=None,
    cat_feat=None,
    stratified=False,
    learning_curve=True,
    verbose_eval=0,
):
    """convenience function, early stopping for lightGBM, using dataset and setting categorical feature, sample weights
    and baseline (init_score), if any. User defined params can be passed.
    It works for classification and regression.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        the predictors' matrix
    y : pd.Series or np.ndarray
        the target series/array
    sample_weight : pd.Series or np.ndarray, optional
        the sample_weight series/array, if relevant. If not None, it should be of the same length as the
        target (default ``None``)
    groups : pd.Series or np.ndarray, optional
        the groups (e.g. polID) for robust cross validation.
        The same group will not appear in two different folds.
    params : dict, optional
        you can pass the parameters that you want to lightGBM/Catboost, as long as they are valid.
        If None, default parameters are passed.
    init_score : pd.Series or np.ndarray, optional
        the initial score to boost from (series/array), if relevant. If not None,
        it should be of the same length as the target (default ``None``)
    cat_feat : str or list of strings, optional
        Categorical features. If list of int, interpreted as indices. If list of strings, interpreted as feature names
        (need to specify ``feature_name`` as well). If 'auto' and data is pandas DataFrame, pandas unordered categorical
        columns are used. All values in categorical features should be less than int32 max value (2147483647). Large
        values could be memory consuming. Consider using consecutive integers starting from zero. All negative values
        in categorical features will be treated as missing values. The output cannot be monotonically constrained with
        respect to a categorical feature (default ``None``)
    stratified : bool, default = False
        stratified shuffle split for the early stopping process. For classification problem, it guarantees
        the same proportion
    learning_curve : bool, default = False
        if show or not the learning curve
    verbose_eval : int, default = 0
        period for printing the train and validation results. If < 1, no output

    Returns
    -------
    model : object
        model object
    fig : plt.figure
        the learning curves, matplotlib figure object

    """
    (
        x_train,
        y_train,
        x_val,
        y_val,
        sample_weight_val,
        sample_weight_train,
        init_score_val,
        init_score_train,
    ) = _make_split(
        X=X,
        y=y,
        sample_weight=sample_weight,
        init_score=init_score,
        groups=groups,
        stratified=stratified,
        test_size=0.2,
    )

    col_list = list(X.columns)
    # cat_idx = [col_list.index(c) for c in cat_feat if c in x]
    d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=cat_feat, free_raw_data=False)
    d_valid = lgb.Dataset(
        x_val,
        label=y_val,
        categorical_feature=cat_feat,
        reference=d_train,
        free_raw_data=False,
    )
    # set weight if any
    if sample_weight is not None:
        d_train = d_train.set_weight(sample_weight_train)
        d_valid = d_valid.set_weight(sample_weight_val)

    # set initial score if any
    if init_score is not None:
        d_train = d_train.set_init_score(init_score_train)
        d_valid = d_valid.set_init_score(init_score_val)

    # check that if the params argument is not None, it is a dictionary
    if params is None:
        warnings.warn("No params dictionary provided, using RMSE as default")
        params = {"objective": "rmse", "metric": "rmse"}
    elif not isinstance(params, dict):
        raise TypeError("params should be either None or a dictionary of lightgbm params")

    # Check if the objective is passed as an argument, dictionary key or both
    if "objective" not in params:
        raise KeyError("No objective provided in the params dictionary")
    # if no metric provided --> set to same as objective (early stopping)
    # requires a metric
    if "metric" not in params and not callable(params["objective"]):
        params["metric"] = params["objective"]
    elif "metric" not in params and callable(params["objective"]):
        raise KeyError(
            "No metric provided for early stopping and could not set objective as metric (scoring)\n"
            "because the objective is user defined"
        )

    if "metric" in params:
        if isinstance(params["metric"], str):
            feval_call = None
        else:
            feval_call = params["metric"]
            params["metric"] = "custom"
    else:
        feval_call = None

    if callable(params["objective"]):
        fobj_call = params["objective"]
        params.pop("objective")
    else:
        fobj_call = None

    # if 'categorical_feature' in params.keys():
    #     params.pop('categorical_feature')

    watchlist = [d_train, d_valid]
    evals_result = {}
    params["verbosity"] = -1

    model = lgb.train(
        params,
        train_set=d_train,
        valid_sets=watchlist,
        num_boost_round=10000,
        feval=feval_call,
        fobj=fobj_call,
        callbacks=[
            early_stopping(10, verbose=False),
            log_evaluation(verbose_eval),
            record_evaluation(eval_result=evals_result),
        ],
    )

    if learning_curve:
        set_my_plt_style()
        fig, ax = plt.subplots()
        ax = lgb.plot_metric(evals_result, ax=ax)
        up_lim = model.best_iteration + 50
        ax.axvline(x=model.best_iteration, color="grey", linestyle="--", label="best_iter")
        ax.set_xlim([0, up_lim])

        del d_train
        del d_valid
        gc.enable()
        gc.collect()

        return model, fig
    else:
        return model


def _make_split(
    X,
    y,
    sample_weight=None,
    init_score=None,
    groups=None,
    stratified=False,
    test_size=0.2,
):
    """_summary_

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        the predictors' matrix
    y : pd.Series or np.ndarray
        the target series/array
    sample_weight : pd.Series or np.ndarray, optional
        the sample_weight series/array, if relevant. If not None, it should be of the same length as the
        target (default ``None``)
    groups : pd.Series or np.ndarray, optional
        the groups (e.g. polID) for robust cross validation.
        The same group will not appear in two different folds.
    stratified : bool, default False
        stratified shuffle split for the early stopping process. For classification problem, it guarantees
        the same proportion
    test_size : float, default 0.2
        test set size, percentage of the total number of rows, by default .2

    Returns
    -------
    Tuple[Union[pd.DataFrame, pd.Series]]
        the split data, target, weights and initial scores (if any)
    """

    if stratified:
        rs = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        splitter = rs.split(X, y)
    elif (not stratified) and (groups is not None):
        rs = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        splitter = rs.split(X, y, groups)
    else:
        rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        splitter = rs.split(X, y)

    for train_index, test_index in splitter:
        x_val, y_val = X.iloc[test_index], y.iloc[test_index]
        x_train, y_train = X.iloc[train_index], y.iloc[train_index]
        if sample_weight is not None:
            sample_weight_val, sample_weight_train = (
                sample_weight.iloc[test_index],
                sample_weight.iloc[train_index],
            )
        else:
            sample_weight_val, sample_weight_train = None, None

        if init_score is not None:
            init_score_val, init_score_train = (
                init_score.iloc[test_index],
                init_score.iloc[train_index],
            )
        else:
            init_score_val, init_score_train = None, None

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        sample_weight_val,
        sample_weight_train,
        init_score_val,
        init_score_train,
    )


def gbm_flavour(estimator):
    model_str = str(type(estimator))
    if "lightgbm" in model_str:
        method = "lgb"
    elif "catboost" in model_str:
        method = "cat"
    else:
        method = "unknown"
    return method