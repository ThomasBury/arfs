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
from tqdm.auto import tqdm

# pandas
import pandas as pd
from pandas.api.types import IntervalDtype

# numpy
import numpy as np

# regular expression
import re

# sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

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
    # class OrdinalEncoderPandas(BaseEstimator, TransformerMixin):
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
        Desired dtype of output.
    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to
        'use_encoded_value', the encoded value of unknown categories will be
        set to the value given for the parameter `unknown_value`. In
        `inverse_transform`, an unknown category will be denoted as None.
    unknown_value : int or np.nan, default=None
        When the parameter handle_unknown is set to 'use_encoded_value', this
        parameter is required and will set the encoded value of unknown
        categories. It has to be distinct from the values used to encode any of
        the categories in `fit`. If set to np.nan, the `dtype` parameter must
        be a float dtype.
    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories. If set to `np.nan`, then the `dtype`
        parameter must be a float dtype.
    return_pandas_categorical : bool, defult=False
        return encoded columns as pandas category dtype or as float

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during ``fit`` (in order of
        the features in X and corresponding with the output of ``transform``).
        This does not include categories that weren't seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

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
        dtype_include=["category", "object", "bool"],
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

        self.feature_names_in_ = X.columns.to_numpy()
        self.categorical_features_ = cat_features_selector(X)

        super(OrdinalEncoderPandas, self).fit(X[self.categorical_features_])
        # self.feature_names_in_ = X.columns.to_numpy()
        return self

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
        X_trans = X.copy()
        X_trans[self.categorical_features_] = super(
            OrdinalEncoderPandas, self
        ).transform(X_trans[self.categorical_features_])

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
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        self = self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.
        When unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category. If the
        feature with the unknown category has a dropped category, the dropped
        category will be its inverse.
        For a given input feature, if there is an infrequent category,
        'infrequent_sklearn' will be used to represent the infrequent category.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_encoded_features)
            The transformed data.
        Returns
        -------
        X_tr : pd.Dataframe of shape (n_samples, n_features)
            Inverse transformed array.
        """

        X[self.categorical_features_] = super(
            OrdinalEncoderPandas, self
        ).inverse_transform(X[self.categorical_features_])
        return X


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


def cat_var(data, col_excl=None, return_cat=True):
    """Ad hoc categorical encoding (as integer). Automatically detect the non-numerical columns,
    save the index and name of those columns, encode them as integer,
    save the direct and inverse mappers as
    dictionaries.
    Return the data-set with the encoded columns with a data type either int or pandas categorical.

    Parameters
    ----------
    data: pd.DataFrame
        the dataset
    col_excl: list of str, default=None
        the list of columns names not being encoded (e.g. the ID column)
    return_cat: bool, default=True
        return encoded object columns as pandas categoricals or not.

    Returns
    -------
    df: pd.DataFrame
        the dataframe with encoded columns
    cat_var_df: pd.DataFrame
        the dataframe with the indices and names of the categorical columns
    inv_mapper: dict
        the dictionary to map integer --> category
    mapper: dict
        the dictionary to map category --> integer
    """
    df = data.copy()
    if col_excl is None:
        non_num_cols = list(
            set(list(df.columns)) - set(list(df.select_dtypes(include=[np.number])))
        )
    else:
        non_num_cols = list(
            set(list(df.columns))
            - set(list(df.select_dtypes(include=[np.number])))
            - set(col_excl)
        )
    cat_var_index = [df.columns.get_loc(c) for c in non_num_cols if c in df]
    cat_var_df = pd.DataFrame({"cat_ind": cat_var_index, "cat_name": non_num_cols})
    # avoid having datetime objects as keys in the mapping dic
    date_cols = [s for s in list(df) if "date" in s]
    df.loc[:, date_cols] = df.loc[:, date_cols].astype(str)
    cols_need_mapped = cat_var_df.cat_name.to_list()
    inv_mapper = {
        col: dict(enumerate(df[col].astype("category").cat.categories))
        for col in df[cols_need_mapped]
    }
    mapper = {
        col: {v: k for k, v in inv_mapper[col].items()} for col in df[cols_need_mapped]
    }
    progress_bar = tqdm(cols_need_mapped)
    for c in progress_bar:
        progress_bar.set_description("Processing {0:<30}".format(c))
        df.loc[:, c] = df.loc[:, c].map(mapper[c]).fillna(0).astype(int)
        # I could have use df[c].update(df[c].map(mapper[c])) while slower,
        # prevents values not included in an incomplete map from being changed to nans.
        # But then I could have outputs
        # with mixed types in the case of different dtypes mapping (like str -> int).
        # This would eventually break any flow.
        # Map is faster than replace
    if return_cat:
        df.loc[:, non_num_cols] = df.loc[:, non_num_cols].astype("category")
    return df, cat_var_df, inv_mapper, mapper







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
    It finds the optimal partition and returns numerical intervals for numerical continuous columns and pd.Categorical for categorical columns.
    This approach groups similar levels together, reducing dimensionality and regularizing the model.

    TreeDiscretizer handles missing values for both numerical and categorical predictors,
    eliminating the need for encoding categorical predictors separately.

    Notes
    -----
    This is a substitution to proper regularization schemes such as:
    - GroupLasso: Categorical predictors, which are usually encoded as multiple dummy variables,
                  are considered together rather than separately.
    - FusedLasso: Takes into account the ordering of the features.

    Parameters
    ----------
    bin_features : List of string or None
        The list of names of the variable that has to be binned, or "all", "numerical" or "categorical"
        for splitting and grouping all, only numerical or only categorical columns.
    n_bins : int
        The number of bins that has to be created while binning the variables in the "bin_features" list.
    n_bins_max : int, optional
        The maximum number of levels that a categorical column can have to avoid being binned.
    num_bins_as_category: bool, default=False
        Save the numeric bins as pandas category or as pandas interval.
    boost_params : dict
        The boosting parameters dictionary.
    raw : bool
        Returns raw levels (non-human-interpretable) or levels matching the original ones.
    task : str
        Either regression or classification (binary).

    Attributes
    ----------
    tree_dic : dict
        The dictionary keys are binned column names and items are the univariate trees.
    bin_upper_bound_dic : dict
        The upper bound of the numerical intervals.
    cat_bin_dict : dict
        The mapping dictionary for the categorical columns.
    tree_imputer : dict
        The missing values are split by the tree and lead to similar splits and are mapped to this value.
    ordinal_encoder_dic : dict
        Dictionary with the fitted encoder, if any.
    cat_features : list
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
        if (boost_params is not None) & (not isinstance(boost_params, dict)):
            raise TypeError("boost_kwargs should be a dictionary")

        self.bin_features = bin_features
        self.n_bins = n_bins
        self.n_bins_max = n_bins_max
        self.num_bins_as_category = num_bins_as_category
        self.boost_params = {}
        self.raw = raw
        self.task = task
        if boost_params is not None:
            self.boost_params = boost_params

        # force some params
        if self.task == "regression":
            self.boost_params["objective"] = "rmse"
        elif self.task == "classification":
            self.boost_params["objective"] = "binary"

        self.boost_params["num_boost_round"] = 1
        self.boost_params["max_leaf"] = self.n_bins
        self.tree_dic = {}
        self.bin_upper_bound_dic = {}
        self.cat_bin_dict = {}
        self.tree_imputer = {}
        self.ordinal_encoder_dic = {}
        self.cat_features = None

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
        X, self.feature_names_in_ = self._prepare_input_dataframe(X)
        self.bin_features, self.cat_features = self._determine_bin_and_cat_features(X, self.bin_features, self.cat_features)
        self.n_unique_table_ = X[self.bin_features].nunique()
        self.bin_features = self._filter_bin_features(self.bin_features, self.n_unique_table_, self.n_bins_max)
        X, self.ordinal_encoder_dic = self._encode_categorical_features(X, self.bin_features, self.cat_features)
        
        for col in self.bin_features:
            is_categorical = (self.cat_features is not None) and (col in self.cat_features)
            self._fit_tree_and_create_bins(X, col, y, sample_weight, is_categorical)
        
        return self
    
    def _prepare_input_dataframe(self, X):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"pred_{i}" for i in range(X.shape[1])]

        return X, X.columns.to_numpy()
    
    def _determine_bin_and_cat_features(self, X, bin_features, cat_features):
        
        if bin_features is None or (isinstance(bin_features, str) and (bin_features == "numerical")):
            bin_features = list(X.select_dtypes("number").columns)
        elif isinstance(bin_features, str) and (bin_features == "all"):
            bin_features = list(X.columns)
        elif isinstance(bin_features, str) and (bin_features == "categorical"):
            bin_features = list(X.select_dtypes(["category", "object", "bool"]).columns)

        # Calculate cat_features by subtracting bin_features from all numeric columns
        cat_features = list(set(bin_features) - set(list(X[bin_features].select_dtypes("number").columns)))
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
                # encode and create a category for missing
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                X[col] = (
                    X[col]
                    .astype("category")
                    .cat.add_categories("missing_added")
                    .fillna("missing_added")
                )
                ordinal_encoder_dic[col] = encoder.fit(X[[col]])
                dum = encoder.transform(X[[col]])
                if isinstance(dum, pd.DataFrame):
                    X[col] = dum.values.ravel()
                else:
                    X[col] = dum.ravel()

        return X, ordinal_encoder_dic
    
    def _fit_tree_and_create_bins(self, X, col, y, sample_weight, is_categorical):
        gbm_param = self.boost_params.copy()
        tree = GradientBoosting(
            cat_feat=None, params=gbm_param, show_learning_curve=False
        )
        tree.fit(X[[col]], y, sample_weight=sample_weight)
        self.tree_dic[col] = tree

        # Create bins and handle categorical features
        X[f"{col}_g"] = tree.predict(X[[col]])

        if is_categorical:
            dum = self.ordinal_encoder_dic[col].inverse_transform(X[[col]])
            if isinstance(dum, pd.DataFrame):
                X[col] = dum.values.ravel()
            else:
                X[col] = dum.ravel()

            self.cat_bin_dict[col] = (
                X[[f"{col}_g", col]]
                .groupby(f"{col}_g")
                .apply(lambda x: concat_or_group(col, x, max_length=25)) #" / ".join(map(str, x[col].unique())))
                .to_dict()
            )
        else:
            bin_array = (
                X[[f"{col}_g", col]]
                .groupby(f"{col}_g")
                .aggregate(max)
                .sort_values(col)
                .values.ravel()
            )
            bin_array = np.delete(bin_array, [np.argmax(bin_array)])
            bin_array = np.unique(np.append(bin_array, [-np.Inf, np.Inf]))
            self.bin_upper_bound_dic[col] = bin_array

            nan_pred_val = tree.predict(np.expand_dims([np.nan], axis=1))[0]
            non_nan_values = X[col].dropna().unique()
            pred_values = tree.predict(np.expand_dims(non_nan_values, axis=1))
            self.tree_imputer[col] = non_nan_values.flat[
                np.abs(pred_values - nan_pred_val).argmin()
            ]

        del tree


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
        X = X.copy()

        for col in self.bin_features:
            if self.raw:
                # predict each univariate tree
                X[col] = self.tree_dic[col].predict(X[[col]])
            else:
                if (self.cat_features is not None) and (col in self.cat_features):
                    # apply the systematic imputation (missing might be grouped
                    # with other categories depending on the results of the tree
                    # splitting)
                    X[col] = (
                        X[col]
                        .astype("category")
                        .cat.add_categories("missing_added")
                        .fillna("missing_added")
                    )
                    dum = self.ordinal_encoder_dic[col].transform(X[[col]])
                    X[col] = self.tree_dic[col].predict(dum)
                    X[col] = X[col].map(self.cat_bin_dict[col])
                else:
                    # retrieve the association the tree learnt for missing values
                    X[col].fillna(self.tree_imputer[col], inplace=True)
                    # apply the binning
                    X[col] = pd.cut(
                        X[col],
                        bins=self.bin_upper_bound_dic[col],
                        include_lowest=True,
                        precision=2,
                    )

                    if not self.num_bins_as_category:
                        X[col] = X[col].astype(IntervalDtype())
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
        data = X.copy()

        if self.cols == "all":
            self.cols = data.columns

        self.float_interval_cols_ = create_dtype_dict(X, dic_keys="dtypes")[
            "num_interval"
        ]
        self.columns_to_transform_ = list(
            set(self.cols).intersection(set(self.float_interval_cols_))
        )
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
        X = X.copy()
        for c in self.columns_to_transform_:
            X.loc[:, c] = find_interval_midpoint(X[c])
            X.loc[:, c] = X[c].astype(float)
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
    if cols == "all":
        cols = X.columns

    X = X.copy()
    float_interval_cols_ = create_dtype_dict(X, dic_keys="dtypes")["num_interval"]
    columns_to_transform_ = list(set(cols).intersection(set(float_interval_cols_)))
    for c in columns_to_transform_:
        X.loc[:, c] = find_interval_midpoint(X[c])
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

    PatsyTransformer transforms a pandas DataFrame (or dict-like)
    according to the formula and produces a numpy array.

    Parameters
    ----------
    formula : string or formula-like
        Pasty formula used to transform the data.

    add_intercept : boolean, default=False
        Wether to add an intersept. By default scikit-learn has built-in
        intercepts for all models, so we don't add an intercept to the data,
        even if one is specified in the formula.

    eval_env : environment or int, default=0
        Envirionment in which to evalute the formula.
        Defaults to the scope in which PatsyModel was instantiated.

    NA_action : string or NAAction, default="drop"
        What to do with rows that contain missing values. You can ``"drop"``
        them, ``"raise"`` an error, or for customization, pass an `NAAction`
        object.  See ``patsy.NAAction`` for details on what values count as
        'missing' (and how to alter this).

    Attributes
    ----------
    feature_names_ : list of string
        Column names / keys of training data.

    return_type : string, default="dataframe"
        data type that transform method will return. Default is ``"dataframe"``
        for numpy array, but if you would like to get Pandas dataframe (for
        example for using it in scikit transformers with dataframe as input
        use ``"dataframe"`` and if numpy array use ``"ndarray"``

    Note
    ----
    PastyTransformer does by default not add an intercept, even if you
    specified it in the formula. You need to set add_intercept=True.

    As scikit-learn transformers can not ouput y, the formula
    should not contain a left hand side.  If you need to transform both
    features and targets, use PatsyModel.
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
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
            data.columns = [f"pred_{i}" for i in range(data.shape[1])]

        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            y.name = "target"

        target_name = y.name if y is not None else "y"
        self.formula = self.formula or " + ".join(
            data.columns.difference([target_name])
        )
        eval_env = EvalEnvironment.capture(self.eval_env, reference=2)
        # self.formula = _drop_intercept(self.formula, self.add_intercept)

        design = dmatrix(
            self.formula,
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

    def transform(self, data):
        """Transform with estimator using formula.

        Transform the data using formula, then transform it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        if self.return_type == "dataframe":
            return dmatrix(self.design_, data, return_type="dataframe")
        else:
            return np.array(dmatrix(self.design_, data))


def _drop_intercept(formula, add_intercept):
    """Drop the intercept from formula if not add_intercept"""
    if not add_intercept:
        if not isinstance(formula, ModelDesc):
            formula = ModelDesc.from_formula(formula)
        if INTERCEPT in formula.rhs_termlist:
            formula.rhs_termlist.remove(INTERCEPT)
        return formula
    return formula
