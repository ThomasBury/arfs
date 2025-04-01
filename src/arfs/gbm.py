"""GBM Wrapper

This module offers a class to train base LightGBM models, with early stopping
as the default behavior. The target variable can be finite discrete (classification)
or continuous (regression). Additionally, the model allows boosting from an
initial score and accepts sample weights as input.

This module is part of the 'arfs' package and relies on 'arfs.utils'.

Module Structure:
-----------------
- ``GradientBoosting``: main class to train a lightGBM with early stopping

Dependencies:
-------------
- Requires 'arfs.utils' for 'create_dtype_dict'.
"""

# Standard library imports
import gc
import warnings
from datetime import date
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any

# Third-party imports
import joblib
import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
# Local imports
from arfs.utils import create_dtype_dict


# --- Matplotlib Configuration ---
QUAL_COLORS = [
    (0.188235, 0.635294, 0.854902),
    (0.898039, 0.682353, 0.219608),
    (0.988235, 0.309804, 0.188235),
    (0.427451, 0.564706, 0.309804),
]
BCKGRND_COLOR = "#f5f5f5"

MPL_PARAMS = {
    "figure.figsize": (5, 3),
    "axes.prop_cycle": plt.cycler(color=QUAL_COLORS),
    "axes.facecolor": BCKGRND_COLOR,
    "patch.edgecolor": BCKGRND_COLOR,
    "figure.facecolor": BCKGRND_COLOR,
    "axes.edgecolor": BCKGRND_COLOR,
    "savefig.edgecolor": BCKGRND_COLOR,
    "savefig.facecolor": BCKGRND_COLOR,
    "grid.color": "#d2d2d2",
    "lines.linewidth": 2,
    "grid.alpha": 0.5,
}


# --- Main GradientBoosting Class ---
class GradientBoosting:
    """Performs the training of a base LightGBM using early stopping.

    Works for regression and classification objectives supported by LightGBM.
    Uses a fixed 20% validation split for early stopping (stratified if specified).
    Allows boosting from an initial score and using sample weights.

    Parameters
    ----------
    cat_feat : List[str], 'auto', or None, default='auto'
        List of categorical feature names.
        If 'auto', uses `arfs.utils.create_dtype_dict` to identify columns
        with dtypes 'object', 'category', 'bool', 'datetime', 'timedelta',
        'datetimetz', and any unrecognized types as categorical for LightGBM.
        If None, no features are treated as categorical by LightGBM.
        Note: For LightGBM, integer-encoded features often perform well even
              when not explicitly marked as categorical.
    params : dict, optional
        LightGBM parameters. Must include 'objective'. If None, uses default
        RMSE objective with 10,000 boosting rounds (subject to early stopping).
    stratified : bool, default=False
        Whether to use StratifiedShuffleSplit for the validation set. Ensures
        class proportions are maintained in classification tasks.
    show_learning_curve : bool, default=True
        If True, generates and stores the learning curve plot.
    verbose_eval : int, default=50
        Period (in boosting rounds) for printing training/validation metrics.
        Set to 0 or False to disable logging during training.
    return_valid_features : bool, default=False
        If True, stores the validation features (X_val) used for early stopping.

    Attributes
    ----------
    model : lgb.Booster or None
        The trained LightGBM Booster object.
    cat_feat : Union[List[str], None]
        Categorical features used (after potential 'auto' detection).
    model_params : Dict[str, Any] or None
        Parameters of the trained LightGBM model.
    params : Dict[str, Any] or None
        Original parameters passed during initialization.
    learning_curve : plt.Figure or None
        Matplotlib figure object of the learning curve, if generated.
    is_init_score : bool
        True if the model was trained with an initial score.
    stratified : bool
        Whether stratified splitting was used.
    show_learning_curve : bool
        Whether the learning curve was requested.
    verbose_eval : int
        Verbosity level used during training.
    return_valid_features : bool
        Whether validation features were stored.
    valid_features : pd.DataFrame or None
        Validation features (X_val), if `return_valid_features` was True.

    Example
    -------
    >>> # Example Usage (assuming X_tr, y_tr, X_tt exist)
    >>> gbm_trainer = GradientBoosting(
    ...     cat_feat='auto', # Automatically detect categorical/object/bool/time cols
    ...     stratified=False,
    ...     params={'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 500}
    ... )
    >>> # Train the model (assuming sample_weight 'exp_tr' exists if needed)
    >>> # gbm_trainer.fit(X=X_tr, y=y_tr, sample_weight=exp_tr)
    >>> gbm_trainer.fit(X=X_tr, y=y_tr) # Without sample weight
    >>>
    >>> # Predict on test data
    >>> y_pred = gbm_trainer.predict(X_tt)
    >>>
    >>> # Save the model
    >>> # gbm_trainer.save(save_path='./models/', name="my_regression_model")
    """

    def __init__(
        self,
        cat_feat: Union[List[str], str, None] = "auto",
        params: Optional[Dict[str, Any]] = None,
        stratified: bool = False,
        show_learning_curve: bool = True,
        verbose_eval: int = 50,
        return_valid_features: bool = False,
    ):
        self.model: Optional[lgb.Booster] = None
        self.cat_feat_input = cat_feat # Store original input
        self.cat_feat: Optional[List[str]] = None # Processed list
        self.model_params: Optional[Dict[str, Any]] = None
        self.params: Optional[Dict[str, Any]] = params
        self.learning_curve: Optional[plt.Figure] = None
        self.is_init_score: bool = False
        self.stratified: bool = stratified
        self.show_learning_curve: bool = show_learning_curve
        # Ensure verbose_eval is usable by log_evaluation (expects int or bool)
        self.verbose_eval: Union[int, bool] = verbose_eval if verbose_eval > 0 else False
        self.return_valid_features: bool = return_valid_features
        self.valid_features: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        """Provides a string representation of the GradientBoosting object."""
        return (
            f"{self.__class__.__name__}("
            f"cat_feat={self.cat_feat_input!r}, "
            f"params={self.params!r}, "
            f"stratified={self.stratified!r}, "
            f"show_learning_curve={self.show_learning_curve!r}, "
            f"verbose_eval={self.verbose_eval!r}, "
            f"return_valid_features={self.return_valid_features!r})"
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        init_score: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """Fits the LightGBM model using early stopping.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Predictor matrix (features).
        y : pd.Series or np.ndarray
            Target variable.
        sample_weight : pd.Series or np.ndarray, optional
            Sample weights. Must have the same length as y.
        init_score : pd.Series or np.ndarray, optional
            Initial scores to boost from. Must have the same length as y.
        groups : pd.Series or np.ndarray, optional
            Group labels for GroupShuffleSplit. Ensures samples from the same
            group are not in both train and validation sets.
        """
        # --- Input Validation and Preparation ---
        if self.params is not None and not isinstance(self.params, dict):
            raise TypeError("params must be None or a dictionary.")
        if isinstance(self.params, dict) and "objective" not in self.params:
            raise KeyError("params dictionary must include an 'objective'.")

        # Ensure X is a DataFrame for potential 'auto' cat_feat detection
        if not isinstance(X, pd.DataFrame):
            # Warning: Column names will be lost if originally numpy
            warnings.warn("Input X is not a pandas DataFrame. Converting, column names might be lost.")
            X = pd.DataFrame(X) # Potential high memory usage for large arrays

        # Handle categorical features
        if self.cat_feat_input == "auto":
            try:
                # Use create_dtype_dict to find column names by type groups
                # It identifies:
                # 'cat': object, category, bool
                # 'time': datetime, timedelta, datetimetz
                # 'unk': Any other non-numeric, non-interval types
                dtypes_dic = create_dtype_dict(df=X, dic_keys="dtypes")
                # Combine columns identified as cat, time, or unknown
                # These will be treated as categorical by LightGBM
                category_cols = (
                    dtypes_dic.get("cat", [])
                    + dtypes_dic.get("time", [])
                    + dtypes_dic.get("unk", [])
                )
                self.cat_feat = category_cols if category_cols else None
                if self.cat_feat:
                     print(f"Auto-detected categorical features (cat/time/unk dtypes): {self.cat_feat}")
            except Exception as e:
                 warnings.warn(f"Error during auto-detection of categorical features: {e}. Proceeding with cat_feat=None.")
                 self.cat_feat = None
        elif isinstance(self.cat_feat_input, list):
             # Validate that provided categorical features exist in X
             missing_cols = [col for col in self.cat_feat_input if col not in X.columns]
             if missing_cols:
                 raise ValueError(f"Categorical features not found in X: {missing_cols}")
             self.cat_feat = self.cat_feat_input
        elif self.cat_feat_input is None:
             self.cat_feat = None
        else:
             raise TypeError("cat_feat must be 'auto', a list of column names, or None.")

        # Ensure y is a Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="target") # Assign a default name

        # Ensure sample_weight and init_score are Series if provided
        if sample_weight is not None:
            if not isinstance(sample_weight, pd.Series):
                sample_weight = pd.Series(sample_weight, name="sample_weight")
            if len(sample_weight) != len(y):
                 raise ValueError("Length of sample_weight must match length of y.")

        if init_score is not None:
            self.is_init_score = True
            if not isinstance(init_score, pd.Series):
                init_score = pd.Series(init_score, name="init_score")
            if len(init_score) != len(y):
                 raise ValueError("Length of init_score must match length of y.")

        # Ensure groups is a Series if provided
        if groups is not None:
            if not isinstance(groups, pd.Series):
                groups = pd.Series(groups, name="groups")
            if len(groups) != len(y):
                 raise ValueError("Length of groups must match length of y.")


        # --- Model Training ---
        output = _fit_early_stopped_lgb(
            X=X,
            y=y,
            sample_weight=sample_weight,
            params=self.params.copy() if self.params else None, # Pass a copy
            init_score=init_score,
            cat_feat=self.cat_feat, # Use processed list
            stratified=self.stratified,
            groups=groups,
            show_learning_curve=self.show_learning_curve,
            verbose_eval=self.verbose_eval,
            return_valid_features=self.return_valid_features,
        )

        # --- Process Output ---
        if self.return_valid_features and self.show_learning_curve:
            self.model, self.valid_features, self.learning_curve = output
        elif self.return_valid_features and not self.show_learning_curve:
            self.model, self.valid_features = output
        elif not self.return_valid_features and self.show_learning_curve:
            self.model, self.learning_curve = output
        else: # Not returning valid features, not showing learning curve
            self.model = output

        # Store final model parameters
        if self.model:
            self.model_params = self.model.params


    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], predict_proba: bool = False
    ) -> np.ndarray:
        """Predicts target values or probabilities for new data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Predictor matrix for which to make predictions.
        predict_proba : bool, default=False
            If True and the objective is classification, returns class
            probabilities. Otherwise, returns predicted values (regression)
            or class labels (classification).

        Returns
        -------
        np.ndarray
            Predicted values or probabilities.

        Raises
        ------
        AttributeError
            If the model was trained with `init_score` (use `predict_raw`).
        ValueError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if self.is_init_score:
            raise AttributeError(
                "Model was fitted with init_score. Use predict_raw() instead "
                "for raw outputs. Apply transformations (e.g., exp) manually "
                "if needed."
            )

        # Ensure X is DataFrame if model was trained on DataFrame (for feature names)
        if isinstance(self.model.feature_name(), list) and not isinstance(X, pd.DataFrame):
             warnings.warn("Model was trained with feature names, but input X for prediction is not a DataFrame. Converting.")
             # Assuming columns match the order during training if no names provided
             try:
                 X = pd.DataFrame(X, columns=self.model.feature_name())
             except ValueError:
                  raise ValueError(f"Input X has {X.shape[1]} columns, but model expects {len(self.model.feature_name())}.")


        obj_fn = self.model_params.get("objective", "") if self.model_params else ""

        # Standard prediction
        y_pred_raw = self.model.predict(X)

        # Post-processing based on objective and predict_proba
        if "binary" in obj_fn:
            if predict_proba:
                # Return probabilities for the positive class
                # LightGBM binary predict often returns probabilities directly
                # Ensure it's 1D array for consistency if needed
                return y_pred_raw if y_pred_raw.ndim == 1 else y_pred_raw[:, 1]
            else:
                # Return class labels (0 or 1)
                return (y_pred_raw > 0.5).astype(int)
        elif "multiclass" in obj_fn:
            if predict_proba:
                # Return probabilities for all classes
                return y_pred_raw
            else:
                # Return the class index with the highest probability
                return np.argmax(y_pred_raw, axis=1)
        else: # Regression or other objectives
            if predict_proba:
                 warnings.warn("predict_proba=True is ignored for non-classification objectives.")
            return y_pred_raw


    def predict_raw(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Provides direct access to the underlying LightGBM predict method.

        Useful for obtaining raw scores, leaf indices, etc., especially when
        `init_score` was used during training.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Predictor matrix.
        **kwargs : dict, optional
            Additional keyword arguments passed directly to `lgb.Booster.predict()`.
            Examples: `raw_score=True`, `pred_leaf=True`. See LightGBM docs.

        Returns
        -------
        np.ndarray
            The raw prediction output from the LightGBM model.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Ensure X is DataFrame if model was trained on DataFrame (for feature names)
        if isinstance(self.model.feature_name(), list) and not isinstance(X, pd.DataFrame):
             warnings.warn("Model was trained with feature names, but input X for prediction is not a DataFrame. Converting.")
             try:
                 X = pd.DataFrame(X, columns=self.model.feature_name())
             except ValueError:
                  raise ValueError(f"Input X has {X.shape[1]} columns, but model expects {len(self.model.feature_name())}.")

        return self.model.predict(X, **kwargs)


    def save(self, save_path: Optional[str] = None, name: Optional[str] = None) -> str:
        """Saves the trained model and learning curve (if generated).

        Model is saved using joblib. Learning curve is saved as a PNG image.

        Parameters
        ----------
        save_path : str, optional
            Directory path to save the files. If None, saves in the current
            working directory. The directory will be created if it doesn't exist.
        name : str, optional
            Base name for the saved files (without extension). If None, defaults
            to 'gbm_base_model_{objective}_{date}'.

        Returns
        -------
        str
            The full path to the saved model file (.joblib).

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        TypeError
            If the learning curve exists but is not a matplotlib Figure.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Cannot save.")

        # Determine file names
        if name:
            base_name = str(name)
        else:
            obj = self.model_params.get('objective', 'unknown_obj')
            base_name = f"gbm_base_model_{obj}_{date.today()}"

        model_file_name = f"{base_name}.joblib"
        fig_file_name = f"{base_name}_learning_curve.png"

        # Determine save directory
        if save_path:
            save_dir = Path(save_path)
            # Create directory if it doesn't exist
            save_dir.mkdir(parents=True, exist_ok=True)
            model_file_path = save_dir / model_file_name
            fig_file_path = save_dir / fig_file_name
        else:
            model_file_path = Path(model_file_name)
            fig_file_path = Path(fig_file_name)

        # Save model
        print(f"Saving model to: {model_file_path}")
        joblib.dump(self.model, model_file_path)

        # Save learning curve if it exists
        if self.learning_curve:
            if isinstance(self.learning_curve, plt.Figure):
                print(f"Saving learning curve to: {fig_file_path}")
                self.learning_curve.savefig(fig_file_path, bbox_inches="tight")
                plt.close(self.learning_curve) # Close figure to free memory
            else:
                 warnings.warn("Learning curve attribute exists but is not a matplotlib Figure. Cannot save.")

        return str(model_file_path)


    def load(self, model_path: str) -> None:
        """Loads a previously saved LightGBM model.

        Overwrites the current `model` and `model_params` attributes.

        Parameters
        ----------
        model_path : str
            Path to the saved model file (.joblib).

        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the specified path.
        """
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_file)
        if self.model:
            self.model_params = self.model.params
            # Attempt to infer cat_feat from loaded model if possible (might not be stored directly)
            # This part is heuristic; cat_feat isn't directly stored in older booster files this way
            if hasattr(self.model, 'pandas_categorical') and self.model.pandas_categorical:
                 self.cat_feat = self.model.pandas_categorical
                 print(f"Inferred categorical features from loaded model: {self.cat_feat}")
            else:
                 # Cannot reliably get cat_feat from older models, keep original setting or None
                 print("Could not reliably infer categorical features from the loaded model.")
                 self.cat_feat = self.cat_feat_input if isinstance(self.cat_feat_input, list) else None
        else:
             raise ValueError("Failed to load model from file.")


# --- Helper Functions ---
def _fit_early_stopped_lgb(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    groups: Optional[pd.Series],
    init_score: Optional[pd.Series],
    params: Optional[Dict[str, Any]],
    cat_feat: Optional[List[str]],
    stratified: bool,
    show_learning_curve: bool,
    verbose_eval: Union[int, bool],
    return_valid_features: bool,
) -> Union[
        lgb.Booster,
        Tuple[lgb.Booster, pd.DataFrame],
        Tuple[lgb.Booster, plt.Figure],
        Tuple[lgb.Booster, pd.DataFrame, plt.Figure],
    ]:
    """Internal function to train LightGBM with early stopping."""

    # --- Data Splitting ---
    (
        X_train, y_train, X_val, y_val,
        sw_val, sw_train, # sample weights
        is_val, is_train, # init scores
    ) = _make_split(
        X=X, y=y, sample_weight=sample_weight, init_score=init_score,
        groups=groups, stratified=stratified, test_size=0.2,
    )

    # --- Prepare LightGBM Datasets ---
    # Note: LightGBM recommends using pd.Categorical dtype for categorical features
    #       for optimal performance, but handles string/object types too.
    #       Consider converting specified cat_feat columns to pd.Categorical
    #       before creating lgb.Dataset if performance is critical.
    # Example:
    # if cat_feat:
    #     for col in cat_feat:
    #         X_train[col] = X_train[col].astype('category')
    #         X_val[col] = X_val[col].astype('category')

    # If cat_feat list is provided or detected, pass it. Otherwise 'auto'.
    categorical_feature_param = cat_feat if cat_feat else 'auto'

    d_train = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=categorical_feature_param,
        free_raw_data=False # Keep data for potential future use/inspection
    )
    d_valid = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=categorical_feature_param,
        reference=d_train, # Important for consistency
        free_raw_data=False
    )

    # Set weights and initial scores if provided
    if sw_train is not None:
        d_train.set_weight(sw_train)
    if sw_val is not None:
        d_valid.set_weight(sw_val)
    if is_train is not None:
        d_train.set_init_score(is_train)
    if is_val is not None:
        d_valid.set_init_score(is_val)

    # --- Parameter Handling ---
    train_params = params.copy() if params else {}

    # Default parameters if none provided
    if not train_params:
        warnings.warn("No params dictionary provided. Using default RMSE objective.")
        train_params = {"objective": "rmse", "metric": "rmse"}

    # Ensure objective is present
    if "objective" not in train_params:
        # This case should be caught in the main class, but double-check
        raise KeyError("No 'objective' provided in the params dictionary.")

    # Set default metric if missing and objective is standard string
    if "metric" not in train_params and isinstance(train_params["objective"], str):
        # Avoid setting metric if objective is not suitable (e.g., 'custom')
        if train_params["objective"] not in ['custom', 'None', None]:
             train_params["metric"] = train_params["objective"]
             print(f"No 'metric' provided, using objective '{train_params['objective']}' as metric.")
        else:
             raise KeyError(
                 f"Objective '{train_params['objective']}' requires an explicit 'metric' for early stopping."
             )
    elif "metric" not in train_params and callable(train_params["objective"]):
        raise KeyError(
            "A 'metric' must be provided in params for early stopping when "
            "using a custom objective function."
        )

    # Handle n_estimators / num_boost_round
    # Use 'n_estimators' as the preferred key, fallback to 'num_boost_round'
    n_trees = train_params.pop('n_estimators', train_params.pop('num_boost_round', 10000))
    if n_trees <= 0:
         warnings.warn(f"n_estimators/num_boost_round ({n_trees}) is <= 0. Setting to default 10000.")
         n_trees = 10000
    print(f"Training up to {n_trees} boosting rounds.")


    # Handle custom evaluation metric (feval)
    feval_callback = None
    if "metric" in train_params and callable(train_params["metric"]):
        feval_callback = train_params["metric"]
        # LightGBM needs a metric name even for custom feval
        # Use 'custom' or the name of the function if available
        train_params["metric"] = "custom"
        print("Using custom evaluation metric.")


    # Set verbosity for LightGBM internal messages (-1 = Fatal, 0 = Error/Warning, 1 = Info)
    # Keep user-controlled printouts via log_evaluation callback
    train_params["verbosity"] = -1 # Suppress internal LightGBM logs

    # --- Callbacks ---
    evals_result = {}
    callbacks = [
        # Stop if validation metric doesn't improve for 10 rounds.
        # `verbose=False` here prevents early_stopping's own messages.
        lgb.early_stopping(stopping_rounds=10, verbose=False),
        # Log metrics every `verbose_eval` rounds using print().
        lgb.log_evaluation(period=verbose_eval if isinstance(verbose_eval, int) and verbose_eval > 0 else 0),
        # Store metric history.
        lgb.record_evaluation(eval_result=evals_result),
    ]

    # --- Training ---
    model = lgb.train(
        params=train_params,
        train_set=d_train,
        num_boost_round=n_trees,
        valid_sets=[d_train, d_valid], # Use both for monitoring
        valid_names=['train', 'valid'], # Assign names
        feval=feval_callback, # Custom metric function, if any
        callbacks=callbacks,
    )

    # --- Post-Training ---
    fig = None
    if show_learning_curve:
        try:
            with mpl.rc_context(MPL_PARAMS):
                fig, ax = plt.subplots()
                lgb.plot_metric(evals_result, ax=ax, xlabel='Boosting Round', ylabel='Metric Value')
                # Add vertical line for best iteration
                if model.best_iteration:
                    ax.axvline(
                        x=model.best_iteration, color="grey", linestyle="--",
                        label=f"Best Iteration ({model.best_iteration})"
                    )
                    ax.legend() # Show legend including the best iteration label
                    # Adjust x-limit for better visualization
                    up_lim = max(50, model.best_iteration + 50) # Show at least 50 rounds
                    ax.set_xlim([0, up_lim])
                ax.set_title("LightGBM Learning Curve")
                fig.tight_layout()
        except Exception as e:
             warnings.warn(f"Could not generate learning curve plot: {e}")
             if fig:
                 plt.close(fig) # Close figure if created but failed during plotting
             fig = None # Ensure fig is None if plotting fails


    # --- Cleanup and Return ---
    # Explicitly delete datasets to potentially free memory sooner
    del d_train, d_valid
    gc.collect() # Suggest garbage collection

    # Return based on user flags
    if return_valid_features and show_learning_curve:
        return model, X_val, fig
    elif return_valid_features and not show_learning_curve:
        return model, X_val
    elif not return_valid_features and show_learning_curve:
        return model, fig
    else: # Not returning valid features, not showing learning curve
        return model


def _make_split(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    init_score: Optional[pd.Series],
    groups: Optional[pd.Series],
    stratified: bool,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, # X_train, y_train, X_val, y_val
        Optional[pd.Series], Optional[pd.Series],       # sample_weight_val, sample_weight_train
        Optional[pd.Series], Optional[pd.Series],       # init_score_val, init_score_train
    ]:
    """Splits data into training and validation sets."""

    # Choose the appropriate splitter
    if stratified:
        # StratifiedShuffleSplit requires y for stratification
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        split_generator = splitter.split(X, y)
    elif groups is not None:
        # GroupShuffleSplit requires groups
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        split_generator = splitter.split(X, y, groups=groups)
    else:
        # Default ShuffleSplit
        splitter = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        split_generator = splitter.split(X, y)

    # Get the indices
    try:
        train_index, val_index = next(split_generator)
    except StopIteration:
         # Should not happen with n_splits=1, but handle defensively
         raise RuntimeError("Failed to generate train/validation split.")


    # Perform the split using iloc for robustness
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Split optional arrays if they exist
    sw_train, sw_val = (None, None)
    if sample_weight is not None:
        sw_train, sw_val = sample_weight.iloc[train_index], sample_weight.iloc[val_index]

    is_train, is_val = (None, None)
    if init_score is not None:
        is_train, is_val = init_score.iloc[train_index], init_score.iloc[val_index]

    print(f"Data split: Train={len(X_train)} samples, Validation={len(X_val)} samples.")

    return X_train, y_train, X_val, y_val, sw_val, sw_train, is_val, is_train


# --- Optional Utility Function ---
def gbm_flavour(estimator: object) -> str:
    """Identifies the type of GBM estimator (basic check)."""
    model_str = str(type(estimator)).lower()
    if "lightgbm" in model_str:
        return "lgb"
    elif "catboost" in model_str:
        return "cat"
    elif "xgboost" in model_str:
        return "xgb"
    else:
        # Could add checks for sklearn GBMs etc.
        return "unknown"

