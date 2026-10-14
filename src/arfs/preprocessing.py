"""
This module provides preprocessing classes

Module Structure:
-----------------
- ``OrdinalEncoderPandas``: main class for ordinal encoding, takes in a DF and returns a DF of the same shape
- ``dtype_column_selector``: for standardizing selection of columns based on their dtypes
- ``TreeDiscretizer``: class for discretizing continuous columns and auto-group levels of categorical columns
- ``IntervalToMidpoint``: class for converting pandas numerical intervals into their float midpoint
- ``PatsyTransformer``: class for encoding data for (generalized) linear models, leveraging Patsy
"""

# Settings and libraries
from __future__ import print_function
from numbers import Integral
from collections.abc import Mapping

# pandas
import pandas as pd

# numpy
import numpy as np

# regular expression
import re

# sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# patsy
from patsy import dmatrix, EvalEnvironment, ModelDesc, INTERCEPT

# typing
from typing import Any, Callable, Union, List, Tuple, Optional, Dict

# ARFS
from .gbm import GradientBoosting
from .utils import create_dtype_dict, concat_or_group


# fix random seed for reproducibility
np.random.seed(7)


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
    dtype_include : column dtype or list of column dtypes, default=None
        A selection of dtypes to include. For more details, see
        `pandas.DataFrame.select_dtypes`.
    dtype_exclude : column dtype or list of column dtypes, default=None
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
        dtype_include=["category", "object", "string", "bool"],
        dtype_exclude=[np.number],
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
            X[self.categorical_features_].astype(object).where(
                X[self.categorical_features_].notna(), np.nan
            )
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
            X[self.categorical_features_].astype(object).where(
                X[self.categorical_features_].notna(), np.nan
            )
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


class TreeDiscretizer(BaseEstimator, TransformerMixin):
    """
    Discretize continuous and/or categorical data using univariate regularized trees, returning a pandas DataFrame.
    The TreeDiscretizer is designed to support regression and binary classification tasks.
    Discretization, also known as quantization or binning, allows for the partitioning of continuous features into discrete values.
    In certain datasets with continuous attributes, discretization can be beneficial as it transforms the dataset into one with only nominal attributes.
    Additionally, for categorical predictors, grouping levels can help reduce overfitting and create meaningful clusters.

    By encoding discretized features, a model can become more expressive while maintaining interpretability.
    For example, preprocessing with a discretizer can introduce nonlinearity to linear models.
    For more advanced possibilities, particularly smooth ones, you can refer to the section on generating polynomial features.
    The TreeDiscretizer function utilizes univariate regularized trees, with one tree per column to be binned.
    It returns the fitted tree's numerical intervals for continuous columns and
    pd.Categorical leaf groups for categorical columns. Split thresholds and
    leaf IDs determine the partition, even when separate leaves predict the same value.
    This approach groups similar levels together, reducing dimensionality and regularizing the model.

    Missing numerical values and missing/unknown categories follow the fitted
    tree's native missing-value route. An all-missing numerical feature has one
    bin, (-inf, inf]; no observed-value imputation is invented.

    Notes
    -----
    Retains GradientBoosting's fixed 80/20 training/validation split. Fit this
    supervised transformer only on training folds. Binning does not implement
    GroupLasso or FusedLasso penalties or guarantee better held-out performance.
    Named transforms require the full fitted column names and order. Arrays
    receive pred_0, pred_1, etc. at fit; later arrays use the fitted names and
    must have the fitted width. Output is always a copied DataFrame.

    Parameters
    ----------
    bin_features : List of string or None
        The list of names of the variable that has to be binned, or "all", "numerical" or "categorical"
        for splitting and grouping all, only numerical or only categorical columns.
    n_bins : int, default=10
        Maximum number of leaves/bins per feature, at least 2. A constant or
        all-missing feature can have fewer bins, including a single bin.
    n_bins_max : int, optional
        The maximum number of levels that a categorical column can have to avoid being binned.
    num_bins_as_category: bool, default=False
        Save the numeric bins as pandas category or as pandas interval.
    boost_params : dict, optional
        Additional LightGBM parameters, copied at fit. Current task, one boosting
        round and n_bins take precedence over objective, iteration and leaf-count
        aliases. The supplied dictionary is not modified.
    raw : bool, default=False
        Return native tree response values instead of intervals/group labels.
        Binary classification returns probabilities, not class labels or logits.
    task : str
        Either regression or classification (binary).

    Attributes
    ----------
    tree_dic : dict
        The dictionary keys are binned column names and items are the univariate trees.
    bin_upper_bound_dic : dict
        Exact numerical split thresholds, bounded by negative/positive infinity.
    cat_bin_dict : dict
        Categorical leaf-ID-to-label mappings. Labels include the leaf ID so
        equal predictions or truncated display text cannot merge groups.
    boost_params_ : dict
        Resolved fit-time parameters; boost_params remains the constructor input.
    leaf_interval_dic_ : dict
        Numerical leaf-ID-to-interval mappings. Missing values use their fitted
        leaf's interval; interval endpoints describe numerical comparisons.
    ordinal_encoder_dic : dict
        Dictionary with the fitted encoder, if any.
    bin_features_ : list
        Resolved columns to bin; bin_features remains the constructor option.
    cat_features_ : list
        Names of the found categorical columns.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the transformer object on data.
    transform(X)
        Apply the fitted transformer object on new data.
    fit_transform(X)
        Fit and apply the transformer object on data.

    Example
    -------
    >>> lgb_params = {'min_split_gain': 5}
    >>> disc = TreeDiscretizer(bin_features='all', n_bins=10)
    >>> disc.fit(X=df[predictors], y=df['Frequency'], sample_weight=df['Exposure'])
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
        """
        Fit the TreeDiscretizer on the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The predictor dataframe.
        y : array-like of shape (n_samples,)
            The target vector.
        sample_weight : array-like of shape (n_samples,), optional
            The weight vector, by default None.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.boost_params is not None and not isinstance(self.boost_params, dict):
            raise TypeError("boost_params must be a dictionary or None.")
        if (
            isinstance(self.n_bins, (bool, np.bool_))
            or not isinstance(self.n_bins, Integral)
            or self.n_bins < 2
        ):
            raise ValueError("n_bins must be an integer at least 2.")
        if self.task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or binary 'classification'.")
        self.boost_params_ = (
            self.boost_params.copy() if self.boost_params is not None else {}
        )
        # Remove conflicting aliases before both the wrapper and LightGBM resolve them.
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
        ):
            self.boost_params_.pop(name, None)
        self.boost_params_.update(
            objective="rmse" if self.task == "regression" else "binary",
            num_iterations=1,
            num_boost_round=1,
            num_leaves=int(self.n_bins),
        )
        self.tree_dic = {}
        self.bin_upper_bound_dic = {}
        self.cat_bin_dict = {}
        self.ordinal_encoder_dic = {}
        self.leaf_interval_dic_ = {}
        X, self.feature_names_in_ = self._prepare_input_dataframe(X)
        if not X.columns.is_unique:
            raise ValueError("Feature names must be unique.")
        self.n_features_in_ = X.shape[1]
        self.bin_features_, self.cat_features_ = self._determine_bin_and_cat_features(
            X, self.bin_features
        )
        self.n_unique_table_ = X[self.bin_features_].nunique()
        self.bin_features_ = self._filter_bin_features(
            self.bin_features_, self.n_unique_table_, self.n_bins_max
        )
        X, self.ordinal_encoder_dic = self._encode_categorical_features(
            X, self.bin_features_, self.cat_features_
        )

        for col in self.bin_features_:
            is_categorical = col in self.cat_features_
            self._fit_tree_and_create_bins(X, col, y, sample_weight, is_categorical)

        return self

    def _prepare_input_dataframe(self, X, fitting=True):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            if fitting:
                X.columns = [f"pred_{i}" for i in range(X.shape[1])]
            else:
                if X.shape[1] != self.n_features_in_:
                    raise ValueError("Array width must match the fitted input.")
                X.columns = self.feature_names_in_

        return X.copy(), X.columns.to_numpy()

    def _determine_bin_and_cat_features(self, X, bin_features):
        if bin_features is None or (
            isinstance(bin_features, str) and (bin_features == "numerical")
        ):
            bin_features = list(X.select_dtypes("number").columns)
        elif isinstance(bin_features, str) and (bin_features == "all"):
            bin_features = list(X.columns)
        elif isinstance(bin_features, str) and (bin_features == "categorical"):
            bin_features = create_dtype_dict(X, dic_keys="dtypes")["cat"]

        cat_features = [
            col
            for col in bin_features
            if col not in X[bin_features].select_dtypes("number").columns
        ]
        return bin_features, cat_features

    def _filter_bin_features(self, bin_features, n_unique_table_, n_bins_max):
        return (
            n_unique_table_[n_unique_table_ > n_bins_max].index.to_list()
            if n_bins_max
            else bin_features
        )

    def _encode_categorical_features(self, X, bin_features, cat_features):
        ordinal_encoder_dic = {}
        for col in bin_features:
            if col in cat_features:
                encoder = OrdinalEncoderPandas(dtype_include=None, dtype_exclude=None)
                ordinal_encoder_dic[col] = encoder.fit(X[[col]])
                X[col] = encoder.transform(X[[col]])[col]

        return X, ordinal_encoder_dic

    def _fit_tree_and_create_bins(self, X, col, y, sample_weight, is_categorical):
        gbm_param = self.boost_params_.copy()
        tree = GradientBoosting(
            cat_feat=None, params=gbm_param, show_learning_curve=False, verbose_eval=0
        )
        tree.fit(X[[col]], y, sample_weight=sample_weight)
        self.tree_dic[col] = tree

        if tree.model.num_trees() != 1:
            raise ValueError("TreeDiscretizer requires exactly one tree per feature.")
        if is_categorical:
            original = self.ordinal_encoder_dic[col].inverse_transform(X[[col]])[col]
            leaves = tree.predict_raw(X[[col]], pred_leaf=True).ravel()
            self.cat_bin_dict[col] = {
                leaf: f"{leaf}: {concat_or_group(col, group.to_frame(name=col), max_length=25)}"
                for leaf, group in original.groupby(leaves, sort=True)
            }
        else:
            intervals = {}

            def collect(node, lower, upper):
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
                tree.model.dump_model()["tree_info"][0]["tree_structure"],
                -np.inf,
                np.inf,
            )
            self.leaf_interval_dic_[col] = intervals
            self.bin_upper_bound_dic[col] = np.unique(
                [
                    bound
                    for interval in intervals.values()
                    for bound in (interval.left, interval.right)
                ]
            )

    def transform(self, X):
        """
        Apply the discretizer on `X`. Only the columns with more than n_bins_max unique values will be transformed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data with shape (n_samples, n_features), where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        X : pd.DataFrame
            DataFrame with the binned and grouped columns.
        """
        check_is_fitted(self, "feature_names_in_")
        X, names = self._prepare_input_dataframe(X, fitting=False)
        if not np.array_equal(names, self.feature_names_in_):
            raise ValueError("Feature names and order must match the fitted input.")

        for col in self.bin_features_:
            frame = X[[col]]
            if col in self.ordinal_encoder_dic:
                frame = self.ordinal_encoder_dic[col].transform(frame)
            if self.raw:
                X[col] = self.tree_dic[col].predict_raw(frame)
            else:
                leaves = pd.Series(
                    self.tree_dic[col].predict_raw(frame, pred_leaf=True).ravel(),
                    index=X.index,
                )
                if col in self.ordinal_encoder_dic:
                    X[col] = pd.Categorical(leaves.map(self.cat_bin_dict[col]))
                else:
                    intervals = self.leaf_interval_dic_[col]
                    X[col] = leaves.map(intervals)
                    if self.num_bins_as_category:
                        X[col] = pd.Categorical(
                            X[col], categories=sorted(intervals.values()), ordered=True
                        )
        return X


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

    Methods
    -------
    fit(X, y=None)
        Fit the transformer on the input data.
    transform(X)
        Transform the input data by converting numerical intervals to midpoints.
    inverse_transform(X)
        Inverse transform is not implemented for this transformer.
    """

    def __init__(self, cols: Union[List[str], str] = "all"):
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
    X: pd.DataFrame, cols: Union[List[str], str] = "all"
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
