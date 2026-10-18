"""
This module provides preprocessing classes

Module Structure:
-----------------
- ``OrdinalEncoderPandas``: main class for ordinal encoding, takes in a DF and returns a DF of the same shape
- ``dtype_column_selector``: for standardizing selection of columns based on their dtypes
- ``TreeDiscretizer``: supervised numeric binning
- ``TreeGrouper``: supervised categorical grouping
- ``IntervalToMidpoint``: class for converting pandas numerical intervals into their float midpoint
- ``PatsyTransformer``: class for encoding data for (generalized) linear models, leveraging Patsy
"""

from collections.abc import Mapping
from numbers import Integral
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from patsy import INTERCEPT, EvalEnvironment, ModelDesc, dmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from .utils import create_dtype_dict, validate_sample_weight, validate_target


class OrdinalEncoderPandas(OrdinalEncoder):
    """Encode categorical features as an integer array and returns a pandas DF.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.
    Read more in the scikit-learn OrdinalEncoder documentation

    Parameters
    ----------
    pattern : str, default=None
        Name of columns containing this regex pattern will be included. If
        None, column selection will not be selected based on pattern.
    dtype_include : column dtype or sequence of column dtypes, default=("category", "object", "string", "bool")
        A selection of dtypes to include. For more details, see
        `pandas.DataFrame.select_dtypes`.
    dtype_exclude : column dtype or sequence of column dtypes, default=(np.number,)
        A selection of dtypes to exclude. For more details, see
        `pandas.DataFrame.select_dtypes`.
    exclude_cols : list of str, optional
        columns to not encode
    output_dtype : number type, default np.float64
        Desired dtype of encoded columns, resolved at fit. Passthrough columns
        keep their dtype.
    handle_unknown : {'error', 'use_encoded_value'}, default='use_encoded_value'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to
        'use_encoded_value', the encoded value of unknown categories will be
        set to the value given for the parameter `unknown_value`. In
        `inverse_transform`, an unknown category will be denoted as None.
    unknown_value : int or np.nan, default=np.nan
        When the parameter handle_unknown is set to 'use_encoded_value', this
        parameter is required and will set the encoded value of unknown
        categories. It has to be distinct from the values used to encode any of
        the categories in `fit`. If set to np.nan, the `dtype` parameter must
        be a float dtype.
    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories. If set to `np.nan`, then the `dtype`
        parameter must be a float dtype.
    return_pandas_categorical : bool, default=False
        return encoded columns as pandas category dtype or as float

    Attributes
    ----------
    categories_ : list of arrays
        The categories of encoded features determined during ``fit``, in
        ``categorical_features_`` order. Passthrough columns have no entry.
        This does not include categories that weren't seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Full input column names, including passthrough columns. Transform and
        inverse_transform require these same columns in the fitted order.
    n_features_in_ : int
        Full input width, including passthrough columns.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.
    >>> ord_enc = OrdinalEncoderPandas(exclude_cols=["PARENT1", "SEX"])
    >>> X_enc = ord_enc.fit_transform(X)
    >>> X_original = ord_enc.inverse_transform(X_enc)
    """

    def __init__(
        self,
        dtype_include=("category", "object", "string", "bool"),
        dtype_exclude=(np.number,),
        pattern=None,
        exclude_cols=None,
        output_dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
        return_pandas_categorical=False,
    ):
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude
        self.pattern = pattern
        self.exclude_cols = exclude_cols
        self.output_dtype = output_dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.return_pandas_categorical = return_pandas_categorical

        super().__init__(
            categories="auto",
            dtype=self.output_dtype,
            handle_unknown=self.handle_unknown,
            unknown_value=self.unknown_value,
            encoded_missing_value=self.encoded_missing_value,
        )

    def fit(self, X, y=None):
        """
        Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : pd.DataFrame, of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self :
            Fitted encoder.
        """

        cat_features_selector = dtype_column_selector(
            dtype_include=self.dtype_include,
            dtype_exclude=self.dtype_exclude,
            pattern=self.pattern,
            exclude_cols=self.exclude_cols,
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not X.columns.is_unique:
            raise ValueError("X columns must be unique.")
        self.categorical_features_ = cat_features_selector(X)
        # The categorical encoder owns its subset schema; ours describes all output.
        self.encoder_ = OrdinalEncoder(
            dtype=self.output_dtype,
            handle_unknown=self.handle_unknown,
            unknown_value=self.unknown_value,
            encoded_missing_value=self.encoded_missing_value,
        )
        self.encoder_.fit(
            X[self.categorical_features_]
            .astype(object)
            .where(X[self.categorical_features_].notna(), np.nan)
        )
        self.categories_ = self.encoder_.categories_
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = X.shape[1]
        self.dtype = self.output_dtype
        return self

    def _check_schema(self, X):
        check_is_fitted(self, "n_features_in_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not X.columns.equals(pd.Index(self.feature_names_in_)):
            raise ValueError("X columns must match the fitted columns in order.")

    def transform(self, X, y=None, sample_weight=None):
        """
        Transform X to ordinal codes.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : pd.DataFrame (n_samples, n_features)
            Transformed input.
        """
        self._check_schema(X)
        X_trans = X.copy()
        X_trans[self.categorical_features_] = self.encoder_.transform(
            X[self.categorical_features_]
            .astype(object)
            .where(X[self.categorical_features_].notna(), np.nan)
        )

        if self.return_pandas_categorical:
            X_trans[self.categorical_features_] = X_trans[
                self.categorical_features_
            ].astype("category")
        return X_trans

    def fit_transform(self, X, y=None, sample_weight=None, **fit_params):
        """
        Fit to data, then transform it.
        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        X_new : pd.DataFrame of shape (n_samples, n_features)
            Encoded columns and unchanged passthrough columns.
        """
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.
        Distinct unknown codes decode to None; fitted missing codes decode to
        np.nan. When both codes are np.nan (the default), they decode as missing.
        Returns a copy and leaves passthrough columns unchanged.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_encoded_features)
            The transformed data.
        Returns
        -------
        X_tr : pd.Dataframe of shape (n_samples, n_features)
            Inverse transformed array.
        """

        self._check_schema(X)
        X_trans = X.copy()
        if self.categorical_features_:
            # Native inverse decoding writes into codes; pandas may expose read-only arrays.
            X_trans[self.categorical_features_] = self.encoder_.inverse_transform(
                X[self.categorical_features_].to_numpy(copy=True)
            )
        return X_trans


class dtype_column_selector:
    """Create a callable to select columns to be used with
    :class:`ColumnTransformer`.
    :func:`dtype_column_selector` can select columns based on datatype or the
    columns name with a regex. When using multiple selection criteria, **all**
    criteria must match for a column to be selected.

    Parameters
    ----------
    pattern : str, default=None
        Name of columns containing this regex pattern will be included. If
        None, column selection will not be selected based on pattern.
    dtype_include : column dtype or list of column dtypes, default=None
        A selection of dtypes to include. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.
    dtype_exclude : column dtype or list of column dtypes, default=None
        A selection of dtypes to exclude. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.
    exclude_cols : list of column names, default=None
        A selection of columns to exclude

    Returns
    -------
    selector : callable
        Callable for column selection to be used by a
        :class:`ColumnTransformer`.

    See Also
    --------
    ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> from arfs.preprocessing import dtype_column_selector
    >>> import numpy as np
    >>> import pandas as pd  # doctest: +SKIP
    >>> X = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
    ...                   'rating': [5, 3, 4, 5]})  # doctest: +SKIP
    >>> ct = make_column_transformer(
    ...       (StandardScaler(),
    ...        dtype_column_selector(dtype_include=np.number)),  # rating
    ...       (OneHotEncoder(),
    ...        dtype_column_selector(dtype_include=object)))  # city
    >>> ct.fit_transform(X)
    array([[ 0.90453403,  1.        ,  0.        ,  0.        ],
           [-1.50755672,  1.        ,  0.        ,  0.        ],
           [-0.30151134,  0.        ,  1.        ,  0.        ],
           [ 0.90453403,  0.        ,  0.        ,  1.        ]])
    """

    def __init__(
        self, pattern=None, *, dtype_include=None, dtype_exclude=None, exclude_cols=None
    ):
        self.pattern = pattern
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude
        self.exclude_cols = exclude_cols

    def __call__(self, df):
        """Callable for column selection to be used by a
        :class:`ColumnTransformer`.
        Parameters
        ----------
        df : pd.DataFrame of shape (n_features, n_samples)
            DataFrame to select columns from.
        """
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        df_row = df.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(
                include=self.dtype_include, exclude=self.dtype_exclude
            )
        cols = df_row.columns
        if self.pattern is not None:
            cols = cols[cols.str.contains(self.pattern, regex=True)]

        if self.exclude_cols is not None:
            cols = cols[~cols.isin(self.exclude_cols)]

        return cols.tolist()


def _tree_frame(X, names=None):
    """Own the output frame and enforce one fitted positional schema."""
    if not isinstance(X, pd.DataFrame):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if names is not None and array.shape[1] != len(names):
            raise ValueError("Array width must match the fitted input.")
        X = pd.DataFrame(
            array,
            columns=names
            if names is not None
            else [f"pred_{i}" for i in range(array.shape[1])],
        )
    if names is not None and not X.columns.equals(pd.Index(names)):
        raise ValueError("Feature names and order must match the fitted input.")
    if not X.columns.is_unique:
        raise ValueError("Feature names must be unique.")
    return X.copy()


def _tree_features(X, selection, categorical):
    eligible = list(
        X.select_dtypes(
            include=["category", "object", "string", "bool"]
            if categorical
            else ["number"],
            exclude=None if categorical else ["complex", "timedelta"],
        ).columns
    )
    automatic = ("all", "categorical") if categorical else ("all", "numerical")
    if selection is None or (isinstance(selection, str) and selection in automatic):
        return eligible
    if not categorical and isinstance(selection, str) and selection == "categorical":
        raise ValueError("Use TreeGrouper for categorical features.")
    selected = [selection] if isinstance(selection, str) else list(selection)
    if not pd.Index(selected).is_unique:
        raise ValueError("Selected feature names must be unique.")
    for col in selected:
        if col not in X.columns:
            raise ValueError(f"Unknown selected feature: {col!r}.")
        if col not in eligible:
            message = (
                "TreeGrouper requires categorical features; cast numeric codes to category dtype."
                if categorical
                else "TreeDiscretizer requires numeric features; use TreeGrouper for categories."
            )
            raise ValueError(message)
    return selected


def _tree_fit_inputs(X, y, sample_weight, task):
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or binary 'classification'.")
    target = validate_target(y, len(X))
    if not pd.api.types.is_numeric_dtype(target.dtype) or pd.api.types.is_complex_dtype(
        target.dtype
    ):
        raise ValueError(
            "y must contain finite numeric responses (binary {0, 1} for classification)."
        )
    values = target.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("y must contain finite numeric responses.")
    if task == "classification" and not np.isin(values, [0, 1]).all():
        raise ValueError(
            "Classification y must contain only numeric/Boolean {0, 1} responses."
        )
    weights = validate_sample_weight(sample_weight, len(X))
    keep = np.ones(len(X), dtype=bool) if weights is None else weights > 0
    return X.iloc[keep], values[keep], None if weights is None else weights[keep]


def _tree_params(boost_params, task, count, count_name):
    if boost_params is not None and not isinstance(boost_params, dict):
        raise TypeError("boost_params must be a dictionary or None.")
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, Integral)
        or count < 2
    ):
        raise ValueError(f"{count_name} must be an integer at least 2.")
    params = {} if boost_params is None else boost_params.copy()
    # The supervised partition always uses one tree and all retained rows.
    for name in (
        "num_iterations",
        "num_iteration",
        "n_iter",
        "num_tree",
        "num_trees",
        "num_round",
        "num_rounds",
        "nrounds",
        "num_boost_round",
        "n_estimators",
        "max_iter",
        "num_leaves",
        "num_leaf",
        "max_leaves",
        "max_leaf",
        "max_leaf_nodes",
        "objective",
        "app",
        "application",
        "loss",
        "objective_type",
        "boosting",
        "boosting_type",
        "boost",
        "data_sample_strategy",
        "bagging_fraction",
        "sub_row",
        "subsample",
        "bagging",
        "bagging_freq",
        "subsample_freq",
        "pos_bagging_fraction",
        "neg_bagging_fraction",
    ):
        params.pop(name, None)
    params.setdefault("verbosity", -1)
    params.setdefault("num_threads", 1)
    params.update(
        objective="rmse" if task == "regression" else "binary",
        num_iterations=1,
        num_leaves=int(count),
        boosting="gbdt",
        data_sample_strategy="bagging",
        bagging_fraction=1.0,
        bagging_freq=0,
        pos_bagging_fraction=1.0,
        neg_bagging_fraction=1.0,
    )
    return params


def _fit_one_tree(values, y, weights, params, categorical=False):
    tree = lgb.train(
        params.copy(),
        lgb.Dataset(
            values,
            label=y,
            weight=weights,
            feature_name=["feature"],
            categorical_feature=[0] if categorical else [],
        ),
        num_boost_round=1,
    )
    if tree.num_trees() != 1:
        raise ValueError("Tree preprocessing requires exactly one tree per feature.")
    return tree


def _tree_feature_names(estimator, input_features):
    check_is_fitted(estimator, "n_features_in_")
    if input_features is not None and not np.array_equal(
        input_features, estimator.feature_names_in_
    ):
        raise ValueError("input_features must match feature_names_in_.")
    return estimator.feature_names_in_.copy()


class TreeDiscretizer(BaseEstimator, TransformerMixin):
    """Bin numeric predictors with one supervised LightGBM tree per feature.

    Fit on all supplied training rows with positive observation weight. Fit only
    inside training folds. Other columns pass through unchanged; categorical
    grouping belongs to :class:`TreeGrouper`. Existing mixed discretizers must
    be refitted. Output is a copied DataFrame with the fitted column order.

    Parameters
    ----------
    bin_features : list, str or None, default="all"
        Explicit numeric columns, or "all", "numerical" or None for all numeric
        columns. Explicit categorical selections raise an error.
    n_bins : int, default=10
        Maximum leaves per feature, at least 2. Constants may have one leaf.
    n_bins_max : int or None, default=None
        Only bin selected columns with more than this many distinct values.
        Zero disables this filter.
    num_bins_as_category : bool, default=False
        Return ordered categorical intervals instead of interval dtype. Use
        factors and factor encoding when bins represent separate model effects.
    boost_params : dict or None, default=None
        Copied LightGBM parameters. Task, one round, leaf limit and all-row
        fitting take precedence over conflicting aliases and row sampling.
    raw : bool, default=False
        Return native responses (binary probabilities) instead of intervals.
    task : {"regression", "classification"}, default="regression"
        Finite numeric response or numeric/Boolean binary {0, 1} response.
        Constant responses are valid. Weights and targets align positionally.

    Attributes
    ----------
    tree_dic : dict
        Selected column names mapped to native LightGBM Boosters. The former
        GradientBoosting wrapper and its ``.model`` access are removed.
    bin_upper_bound_dic : dict
        Exact split thresholds bounded by negative and positive infinity.
    leaf_interval_dic_ : dict
        Native leaf IDs mapped to right-closed numeric intervals. Missing values
        follow native routes. Constants/all-missing columns have (-inf, inf].
    bin_features_ : list
        Resolved and filtered numeric columns.
    n_unique_table_ : pandas.Series
        Distinct counts on positive-weight rows before filtering.
    boost_params_ : dict
        Resolved parameters; constructor parameters remain unchanged.
    feature_names_in_ : numpy.ndarray
        Full fitted schema. Arrays fitted without names use pred_0, pred_1, etc.;
        subsequent arrays use fitted names and must match fitted width.
    n_features_in_ : int
        Full fitted input width, including passthrough columns.

    Notes
    -----
    Binning does not implement group/fused Lasso, guarantee predictive gains or
    justify ordinary post-selection inference. Interval endpoints describe
    comparisons, including native missing/zero routing, rather than imputation.
    """

    def __init__(
        self,
        bin_features="all",
        n_bins=10,
        n_bins_max=None,
        num_bins_as_category=False,
        boost_params=None,
        raw=False,
        task="regression",
    ):
        self.bin_features = bin_features
        self.n_bins = n_bins
        self.n_bins_max = n_bins_max
        self.num_bins_as_category = num_bins_as_category
        self.boost_params = boost_params
        self.raw = raw
        self.task = task

    def fit(self, X, y, sample_weight=None):
        """Learn numeric partitions from training data and positional weights."""
        params = _tree_params(self.boost_params, self.task, self.n_bins, "n_bins")
        X = _tree_frame(X)
        features = _tree_features(X, self.bin_features, categorical=False)
        if self.n_bins_max is not None and (
            isinstance(self.n_bins_max, (bool, np.bool_))
            or not isinstance(self.n_bins_max, Integral)
            or self.n_bins_max < 0
        ):
            raise ValueError("n_bins_max must be a nonnegative integer or None.")
        names = X.columns.to_numpy(copy=True)
        X, y, weights = _tree_fit_inputs(X, y, sample_weight, self.task)
        self.n_unique_table_ = X[features].nunique()
        self.bin_features_ = [
            col
            for col in features
            if not self.n_bins_max or self.n_unique_table_[col] > self.n_bins_max
        ]
        self.tree_dic = {}
        self.bin_upper_bound_dic = {}
        self.leaf_interval_dic_ = {}
        self.boost_params_ = params
        for col in self.bin_features_:
            values = X[[col]].to_numpy(dtype=float, na_value=np.nan)
            tree = _fit_one_tree(values, y, weights, params)
            self.tree_dic[col] = tree
            intervals = {}

            def collect(node, lower, upper, intervals=intervals):
                if "split_index" not in node:
                    intervals[node.get("leaf_index", 0)] = pd.Interval(lower, upper)
                    return
                if node["decision_type"] != "<=":
                    raise ValueError(
                        "TreeDiscretizer requires numerical univariate splits."
                    )
                threshold = float(node["threshold"])
                collect(node["left_child"], lower, threshold)
                collect(node["right_child"], threshold, upper)

            collect(
                tree.dump_model()["tree_info"][0]["tree_structure"], -np.inf, np.inf
            )
            self.leaf_interval_dic_[col] = intervals
            self.bin_upper_bound_dic[col] = np.unique(
                [
                    bound
                    for interval in intervals.values()
                    for bound in (interval.left, interval.right)
                ]
            )
        self.feature_names_in_ = names
        self.n_features_in_ = len(names)
        return self

    def transform(self, X):
        """Return fitted intervals or responses, preserving passthrough columns."""
        check_is_fitted(self, "n_features_in_")
        X = _tree_frame(X, self.feature_names_in_)
        for col in self.bin_features_:
            values = X[[col]].to_numpy(dtype=float, na_value=np.nan)
            if self.raw:
                X[col] = self.tree_dic[col].predict(values)
            else:
                leaves = pd.Series(
                    self.tree_dic[col].predict(values, pred_leaf=True).ravel(),
                    index=X.index,
                )
                intervals = self.leaf_interval_dic_[col]
                X[col] = leaves.map(intervals)
                if self.num_bins_as_category:
                    X[col] = pd.Categorical(
                        X[col], categories=sorted(intervals.values()), ordered=True
                    )
        return X

    def get_feature_names_out(self, input_features=None):
        """Return the full fitted output names, including passthrough columns."""
        return _tree_feature_names(self, input_features)


class TreeGrouper(BaseEstimator, TransformerMixin):
    """Group categorical predictors with one supervised LightGBM tree each.

    Nominal levels use native categorical splits; ordered categories use numeric
    splits of their declared order, so their groups are contiguous. Missing and
    unseen levels follow the native missing route. Vocabularies and trees learn
    only from positive-weight training rows. Fit only inside training folds.

    Parameters
    ----------
    group_features : list, str or None, default="all"
        Explicit category/object/string/Boolean columns, or "all", "categorical"
        or None for all eligible columns. Numeric codes require category dtype.
        Unselected columns pass through unchanged.
    n_groups : int, default=10
        Maximum leaves per feature, at least 2. Constants can have one group.
    boost_params : dict or None, default=None
        Copied LightGBM parameters; task, one round, leaf limit and all-row
        fitting override conflicting aliases and row sampling.
    raw : bool, default=False
        Return native responses, including binary probabilities, rather than
        categorical leaf labels.
    task : {"regression", "classification"}, default="regression"
        Finite numeric response or numeric/Boolean binary {0, 1} response.
        Constant responses are valid. Weights and targets align positionally.

    Attributes
    ----------
    tree_dic : dict
        Grouped columns mapped to native LightGBM Boosters.
    cat_bin_dict : dict
        Native leaf IDs mapped to display labels containing distinct leaf IDs.
    category_dtypes_ : dict
        Fitted input vocabularies, excluding unused and zero-weight-only levels.
        Ordered categories retain their declared relative order.
    group_dtypes_ : dict
        Frozen output categories; unordered for nominal input, ordered for
        ordered input. Batch composition cannot change the output dtype.
    group_features_ : list
        Resolved categorical columns.
    boost_params_ : dict
        Resolved parameters without mutating constructor parameters.
    feature_names_in_ : numpy.ndarray
        Full fitted column names/order. Unnamed arrays use pred_0, pred_1, etc.;
        later arrays use fitted names and must match fitted width.
    n_features_in_ : int
        Full input width. Output is always a copied pandas DataFrame preserving
        index, column order and passthrough values.

    Notes
    -----
    Grouping does not implement group/fused Lasso or imply predictive improvement.
    No Other level or outcome imputation is introduced. Group labels are for
    display; native leaf identities determine groups, even for equal responses.
    """

    def __init__(
        self,
        group_features="all",
        n_groups=10,
        boost_params=None,
        raw=False,
        task="regression",
    ):
        self.group_features = group_features
        self.n_groups = n_groups
        self.boost_params = boost_params
        self.raw = raw
        self.task = task

    def _feature_values(self, values, col):
        dtype = self.category_dtypes_[col]
        codes = pd.Categorical(values, dtype=dtype).codes.astype(float)
        missing = codes < 0
        # Start ordered codes at one so zero_as_missing cannot merge distant levels.
        if dtype.ordered:
            codes += 1
        codes[missing] = np.nan
        return codes.reshape(-1, 1)

    def fit(self, X, y, sample_weight=None):
        """Learn category vocabularies and groups from positive-weight rows."""
        params = _tree_params(self.boost_params, self.task, self.n_groups, "n_groups")
        X = _tree_frame(X)
        features = _tree_features(X, self.group_features, categorical=True)
        names = X.columns.to_numpy(copy=True)
        X, y, weights = _tree_fit_inputs(X, y, sample_weight, self.task)
        self.group_features_ = features
        self.boost_params_ = params
        self.tree_dic = {}
        self.cat_bin_dict = {}
        self.category_dtypes_ = {}
        self.group_dtypes_ = {}
        for col in features:
            observed = pd.Categorical(X[col]).remove_unused_categories()
            dtype = observed.dtype
            self.category_dtypes_[col] = dtype
            values = self._feature_values(X[col], col)
            tree = _fit_one_tree(
                values, y, weights, params, categorical=not dtype.ordered
            )
            self.tree_dic[col] = tree
            # Include the native missing route even if no training row was missing.
            levels = list(dtype.categories) + [np.nan]
            leaves = tree.predict(
                self._feature_values(levels, col), pred_leaf=True
            ).ravel()
            labels = {}
            for leaf in dict.fromkeys(leaves):
                members = [
                    str(level) for level, group in zip(levels, leaves) if group == leaf
                ]
                text = " / ".join(members)
                if len(text) > 25:
                    text = text[:7] + "/.../" + text[-7:]
                labels[leaf] = f"{leaf}: {text}"
            self.cat_bin_dict[col] = labels
            self.group_dtypes_[col] = pd.CategoricalDtype(
                list(labels.values()), ordered=dtype.ordered
            )
        self.feature_names_in_ = names
        self.n_features_in_ = len(names)
        return self

    def transform(self, X):
        """Apply frozen vocabularies and native missing/unseen category routes."""
        check_is_fitted(self, "n_features_in_")
        X = _tree_frame(X, self.feature_names_in_)
        for col in self.group_features_:
            values = self._feature_values(X[col], col)
            if self.raw:
                X[col] = self.tree_dic[col].predict(values)
            else:
                leaves = pd.Series(
                    self.tree_dic[col].predict(values, pred_leaf=True).ravel(),
                    index=X.index,
                )
                X[col] = pd.Categorical(
                    leaves.map(self.cat_bin_dict[col]), dtype=self.group_dtypes_[col]
                )
        return X

    def get_feature_names_out(self, input_features=None):
        """Return the full fitted output names, including passthrough columns."""
        return _tree_feature_names(self, input_features)


def highlight_discarded(s):
    """
    highlight X in red and V in green.

    Parameters
    ----------
    s : np.arrays

    Returns
    -------
    list

    """
    is_X = s == 0
    return [
        "background-color: #d65f5f" if v else "background-color: #33a654" for v in is_X
    ]


class IntervalToMidpoint(BaseEstimator, TransformerMixin):
    """
    IntervalToMidpoint is a transformer that converts numerical intervals in a pandas DataFrame to their midpoints.

    Parameters
    ----------
    cols : list of str or str, default "all"
        The column(s) to transform. If "all", all columns with numerical intervals will be transformed.

    Attributes
    ----------
    cols : list of str or str
        The column(s) to transform.
    float_interval_cols_ : list of str
        The columns with numerical interval data types in the input DataFrame.
    columns_to_transform_ : list of str
        The columns to be transformed based on the specified `cols` attribute.

    """

    def __init__(self, cols: Union[list[str], str] = "all"):
        self.cols = cols

    def fit(self, X: pd.DataFrame = None, y: pd.Series = None):
        """
        Fit the transformer on the input data.

        Parameters
        ----------
        X :
            The input data to fit the transformer on.
        y :
            Ignored parameter.

        Returns
        -------
        self : IntervalToMidpoint
            The fitted transformer object.
        """
        if isinstance(self.cols, str):
            self.cols_ = list(X.columns) if self.cols == "all" else [self.cols]
        else:
            self.cols_ = list(self.cols)

        self.float_interval_cols_ = create_dtype_dict(X, dic_keys="dtypes")[
            "num_interval"
        ]
        self.columns_to_transform_ = [
            col for col in self.float_interval_cols_ if col in self.cols_
        ]
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data by converting numerical intervals to midpoints.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        X : pd.DataFrame
            The transformed data with numerical intervals replaced by their midpoints.
        """
        check_is_fitted(self, "columns_to_transform_")
        X = X.copy()
        for c in self.columns_to_transform_:
            X[c] = find_interval_midpoint(X[c])
        return X

    def inverse_transform(self, X: pd.DataFrame):
        """
        Inverse transform is not implemented for this transformer.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to perform inverse transform on.

        Raises
        ------
        NotImplementedError
            Raised since inverse transform is not implemented for this transformer.
        """
        raise NotImplementedError(
            "inverse_transform is not implemented for this transformer."
        )


def transform_interval_to_midpoint(
    X: pd.DataFrame, cols: Union[list[str], str] = "all"
) -> pd.DataFrame:
    """
    Transforms interval columns in a pandas DataFrame to their midpoint values.

    Notes
    -----
    Equivalent function to ``IntervalToMidpoint`` without the estimator API

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing the data to be transformed.
    cols : list of str or str
        The columns to be transformed. Defaults to "all" which transforms all columns.

    Returns
    -------
    pd.DataFrame :
        The transformed DataFrame with interval columns replaced by their midpoint values.

    Raises
    ------
    TypeError :
        If the input data is not a pandas DataFrame.
    """
    if isinstance(cols, str):
        cols = list(X.columns) if cols == "all" else [cols]

    X = X.copy()
    float_interval_cols_ = create_dtype_dict(X, dic_keys="dtypes")["num_interval"]
    columns_to_transform_ = [col for col in float_interval_cols_ if col in cols]
    for c in columns_to_transform_:
        X[c] = find_interval_midpoint(X[c])
    return X


def find_interval_midpoint(interval_series: pd.Series) -> np.ndarray:
    """Find the midpoint (or left/right bound if the interval contains Inf).

    Parameters
    ----------
    interval_series : pd.Series
        series of pandas intervals.

    Returns
    -------
    np.ndarray
        Array of midpoints or bounds of the intervals.
    """
    left = interval_series.array.left
    right = interval_series.array.right
    mid = interval_series.array.mid
    left_inf = np.isinf(left)
    right_inf = np.isinf(right)

    return np.where(
        left_inf & right_inf,
        np.inf,
        np.where(left_inf, right, np.where(right_inf, left, mid)),
    )


class PatsyTransformer(BaseEstimator, TransformerMixin):
    """Transformer using patsy-formulas.

    PatsyTransformer transforms a pandas DataFrame, named dictionary or array
    according to the formula. Arrays receive names pred_0, pred_1, etc. at fit;
    later arrays use the fitted names and must have the same width.

    Parameters
    ----------
    formula : string or ModelDesc, default=None
        Patsy formula used to transform the data. None infers a new formula
        at each fit, excluding the named target if it appears in the input.

    add_intercept : boolean, default=True
        Allow the formula's intercept. False removes it even if specified;
        True respects explicit formula exclusions such as "0 + x".

    eval_env : environment or int, default=0
        Environment in which to evaluate the formula.
        Pass a Patsy EvalEnvironment for explicit callable bindings.

    NA_action : string or NAAction, default="drop"
        What to do with rows that contain missing values. You can ``"drop"``
        them, ``"raise"`` an error, or for customization, pass an `NAAction`
        object.  See ``patsy.NAAction`` for details on what values count as
        'missing' (and how to alter this).
        Drop removes predictor rows only; this transformer cannot also align
        y or sample weights. Use "raise" in pipelines that require every row.

    return_type : string, default="dataframe"
        Output type: pandas DataFrame or NumPy array.

    Attributes
    ----------
    feature_names_in_ : ndarray
        Column names / keys of training data.
    formula_ : string or ModelDesc
        Resolved fitted formula; the constructor's formula stays unchanged.
    design_ : DesignInfo
        Fitted design, including stateful transforms and categorical levels.

    Note
    ----
    As scikit-learn transformers can not output y, the formula
    should not contain a left hand side.
    """

    def __init__(
        self,
        formula=None,
        add_intercept=True,
        eval_env=0,
        NA_action="drop",
        return_type="dataframe",
    ):
        self.formula = formula
        self.eval_env = eval_env
        self.add_intercept = add_intercept
        self.NA_action = NA_action
        self.return_type = return_type

    def fit(self, data, y=None):
        """Fit the scikit-learn model using the formula.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        self._fit_transform(data, y)
        return self

    def fit_transform(self, data, y=None):
        """Fit the scikit-learn model using the formula and transform it.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.

        Returns
        -------
        X_transform : ndarray
            Transformed data
        """
        return self._fit_transform(data, y)

    def _fit_transform(self, data, y=None):
        data = self._prepare_input_dataframe(data)
        self.feature_names_in_ = data.columns.to_numpy()
        self.n_features_in_ = data.shape[1]
        target_name = y.name if isinstance(y, pd.Series) else None
        formula = self.formula
        if formula is None:
            formula = " + ".join(data.columns.difference([target_name])) or "1"
        self.formula_ = _drop_intercept(formula, self.add_intercept)
        eval_env = EvalEnvironment.capture(self.eval_env, reference=2)

        design = dmatrix(
            self.formula_,
            data,
            NA_action=self.NA_action,
            return_type="dataframe",
            eval_env=eval_env,
        )
        self.design_ = design.design_info

        if self.return_type == "dataframe":
            return design
        else:
            return np.array(design)

    def _prepare_input_dataframe(self, data, fitting=True):
        if isinstance(data, (pd.DataFrame, Mapping)):
            return pd.DataFrame(data).copy()
        data = pd.DataFrame(data).copy()
        if fitting:
            data.columns = [f"pred_{i}" for i in range(data.shape[1])]
        else:
            if data.shape[1] != self.n_features_in_:
                raise ValueError("Array width must match the fitted input.")
            data.columns = self.feature_names_in_
        return data

    def transform(self, data):
        """Transform with estimator using formula.

        Transform the data using formula, then transform it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        check_is_fitted(self, "design_")
        data = self._prepare_input_dataframe(data, fitting=False)
        design = dmatrix(
            self.design_, data, NA_action=self.NA_action, return_type="dataframe"
        )
        return design if self.return_type == "dataframe" else np.array(design)


def _drop_intercept(formula, add_intercept):
    """Drop the intercept from formula if not add_intercept"""
    if not add_intercept:
        if not isinstance(formula, ModelDesc):
            formula = ModelDesc.from_formula(formula)
        return ModelDesc(
            formula.lhs_termlist,
            [term for term in formula.rhs_termlist if term != INTERCEPT],
        )
    return formula
