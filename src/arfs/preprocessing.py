"""
This module provides preprocessing classes

**The module structure is the following:**

- The ``OrdinalEncoderPandas`` main class for ordinal encoding, takes in a DF and returns a DF of the same shape
- The ``dtype_column_selector`` for standardizing selection of columns based on their dtypes
- The ``TreeDiscretizer`` for discretizing continuous columns and auto-group levels of categorical columns

"""

# Settings and libraries
from __future__ import print_function
from tqdm.auto import tqdm

# pandas
import pandas as pd
from pandas.api.types import IntervalDtype

# numpy
import numpy as np

# sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# ARFS
from .gbm import GradientBoosting


# fix random seed for reproducibility
np.random.seed(7)

__all__ = [
    "OrdinalEncoderPandas",
    "dtype_column_selector",
    "TreeDiscretizer",
]


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
    """The purpose of the function is to discretize continuous and/or categorical data, returning a pandas DataFrame. 
    It is designed to support regression and binary classification tasks. Discretization, also known as quantization or binning, 
    allows for the partitioning of continuous features into discrete values. In certain datasets with continuous attributes, 
    discretization can be beneficial as it transforms the dataset into one with only nominal attributes. 
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
     This is a substitution to proper regularization scheme such as
     - GroupLasso (categorical predictors, which are usually encoded as multiple dummy variables: one for each category.
                   It makes sense in many analyses to consider these dummy variables (representing one categorical predictor)
                   together rather than separately)
     - FusedLasso (One drawback of the Lasso is it ignores ordering of the features, FusedLasso takes into account the ordering)

    Parameters
    ----------
    bin_features : List of string, str or None
        the list of names of the variable that has to be binned, or "all", "numerical" or "categorical"
        for splitting and grouping all, only numerical or only categorical columns.
    n_bins : int
        the number of bins that has to be created while binning the variables in "bin_features" list
    n_bins_max : int, optional
        the maximum number of levels that a categorical column can have in order to avoid being binned
    num_bins_as_category: bool, default=False
        save the numeric bins as pandas category or as pandas interval
    boost_params : dic
        the boosting parameters dictionary
    raw : bool
        returns raw levels (non human-interpretable) or levels matching the orginal ones
    task : str
        either regression or classification (binary)

    Attributes
    ----------
    tree_dic : dic
        the dictionary keys are binned column names and items are the univariate trees
    bin_upper_bound_dic : dic
        the upper bound of the numerical intervals
    cat_bin_dict : dic
        the mapping dictionary for the categorical columns
    tree_imputer : dic
        the missing values are split by the tree and lead to similar splits and are mapped to this value
    ordinal_encoder_dic : dic
        dictionary with the fitted encoder, if any
    cat_features : list
        names of the found categorical columns

    Methods
    -------
    fit(X, y, sample_weight=None)
        fit the transformer object on data
    transform(x)
        apply the fitted transformer object on new data
    fit_transform(x)
        fit and apply the transformer object on data

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
            self.boost_params["objective"] = "RMSE"
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
        """Fit the discretizer on `X`.
        Parameters
        ----------
        X :
            Input data shape (n_samples, n_features), where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y :
            target for internally fitting the tree(s)
        sample_weight :
            sample weight (e.g. exposure) if any

        Returns
        -------
        X :
            pd.DataFrame with the binned and grouped columns
        """
        X = X.copy()
        
        self.n_unique_table_ = X[self.bin_features].nunique()
        # transform only the columns with more than n_bins_max
        self.bin_features = (self.n_unique_table_ > self.n_bins_max).index.to_list() if self.n_bins_max else self.bin_features
        
        if self.bin_features is None:
            self.bin_features = list(X.select_dtypes("number").columns)
            self.cat_features = []
        elif isinstance(self.bin_features, list):
            self.cat_features = list(
                set(self.bin_features)
                - set(list(X[self.bin_features].select_dtypes("number").columns))
            )
        elif isinstance(self.bin_features, str) and (self.bin_features == "all"):
            self.bin_features = list(X.columns)
            self.cat_features = list(
                set(self.bin_features) - set(list(X.select_dtypes("number").columns))
            )
        elif isinstance(self.bin_features, str) and (self.bin_features == "numerical"):
            self.bin_features = list(X.select_dtypes("number").columns)
        elif isinstance(self.bin_features, str) and (
            self.bin_features == "categorical"
        ):
            self.bin_features = list(X.select_dtypes(["category", "object"]).columns)
            self.cat_features = self.bin_features

        for col in self.bin_features:
            is_categorical = (self.cat_features is not None) and (
                col in self.cat_features
            )
            if is_categorical:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                )
                # create a category for missing
                X[col] = (
                    X[col]
                    .astype("category")
                    .cat.add_categories("missing_added")
                    .fillna("missing_added")
                )
                # encode
                self.ordinal_encoder_dic[col] = encoder.fit(X[[col]])
                X[col] = encoder.transform(X[[col]]).ravel()
            else:
                encoder = None

            gbm_param = self.boost_params.copy()
            tree = GradientBoosting(
                cat_feat=None, params=gbm_param, show_learning_curve=False
            )
            tree.fit(X[[col]], y, sample_weight=sample_weight)

            # store each fitted tree in a dictionary
            self.tree_dic[col] = tree

            # create the bins series
            # predict, group by original values
            # create monotonicly increasing bin for pd.cut
            X[f"{col}_g"] = tree.predict(X[[col]])

            if is_categorical:
                # retrieve original values
                X[col] = encoder.inverse_transform(X[[col]]).ravel()
                self.cat_bin_dict[col] = (
                    X[[f"{col}_g", col]]
                    .groupby(f"{col}_g")
                    .apply(lambda x: " / ".join(map(str, x[col].unique())))
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
                # overwrite max for handling unseen larger values than max val on the train set
                bin_array = np.delete(bin_array, [np.argmax(bin_array)])
                # append -Inf and Inf for covering the whole interval, as in KBinsDiscretizer
                bin_array = np.unique(np.append(bin_array, [-np.Inf, np.Inf]))
                self.bin_upper_bound_dic[col] = bin_array
                # the tree imputer: if nan is passed, the predicted value will be the
                # same than another non-missing value and nan should fall into this bin
                # the predicted value of a NaN
                nan_pred_val = tree.predict(np.expand_dims([np.nan], axis=1))[0]
                # the closest predicted value for non NaN inputs
                non_nan_values = X[col].dropna().unique()
                pred_values = tree.predict(np.expand_dims(non_nan_values, axis=1))
                # store the value for knowing the bin into which the NaN should fall
                self.tree_imputer[col] = non_nan_values.flat[
                    np.abs(pred_values - nan_pred_val).argmin()
                ]

            del tree

        return self

    def transform(self, X):
        """Apply the discretizer on `X`. Only the columns with more than n_bins_max unique values will be transformed.
        
        Parameters
        ----------
        X :
            Input data shape (n_samples, n_features), where `n_samples` is the number of samples and
            `n_features` is the number of features.


        Returns
        -------
        X :
            pd.DataFrame with the binned and grouped columns
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
                        precision=1,
                    )
                    
                    if not self.num_bins_as_category:
                        X[col] = X[col].astype(IntervalDtype())
        return X


def highlight_discarded(s):
    """highlight X in red and V in green.

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


def make_fs_summary(selector_pipe):
    """make_fs_summary makes a summary dataframe highlighting at which step a
    given predictor has been rejected (if any).

    Parameters
    ----------
    selector_pipe : sklearn.pipeline.Pipeline
        the feature selector pipeline.

    Examples
    --------
    >>> groot_pipeline = Pipeline([
    ... ('missing', MissingValueThreshold()),
    ... ('unique', UniqueValuesThreshold()),
    ... ('cardinality', CardinalityThreshold()),
    ... ('collinearity', CollinearityThreshold(threshold=0.5)),
    ... ('lowimp', VariableImportance(eval_metric='poisson', objective='poisson', verbose=2)),
    ... ('grootcv', GrootCV(objective='poisson', cutoff=1, n_folds=3, n_iter=5))])
    >>> groot_pipeline.fit_transform(
        X=df[predictors],
        y=df[target],
        lowimp__sample_weight=df[weight],
        grootcv__sample_weight=df[weight])
    >>> fs_summary_df = make_fs_summary(groot_pipeline)
    """

    tag_df = pd.DataFrame({"predictor": selector_pipe[0].feature_names_in_})
    for selector_name in selector_pipe.named_steps.keys():
        if hasattr(selector_pipe.named_steps[selector_name], "feature_names_in_"):
            feature_in = selector_pipe.named_steps[selector_name].feature_names_in_
            to_drop = list(
                set(selector_pipe.named_steps[selector_name].feature_names_in_)
                - set(selector_pipe.named_steps[selector_name].get_feature_names_out())
            )
            tag_df[selector_name] = np.where(
                tag_df["predictor"].isin(to_drop), 0, 1
            ) * np.where(tag_df["predictor"].isin(feature_in), 1, np.nan)

    col_to_apply_style = tag_df.columns[1:]
    tag_df = (
        tag_df.style.apply(highlight_discarded, subset=col_to_apply_style)
        .applymap(lambda x: "" if x == x else "background-color: #ffa500")
        .format(precision=0)
    )
    return tag_df
