"""Utility and validation functions
"""

from __future__ import print_function, division

import lightgbm as lgb
import matplotlib as mpl
import numpy as np
import pandas as pd

from pkg_resources import resource_filename
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch
import joblib

qualitative_colors = [
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    "#008695",
    "#CF1C90",
    "#F97B72",
]

#####################
#                   #
#     Utilities     #
#                   #
#####################


def concat_or_group(col, x, max_length=25):
    """
    Concatenate unique values from a column or return a group value.

    Parameters
    ----------
    col : str
        The name of the column to process.
    x : pd.DataFrame
        The DataFrame containing the data.
    max_length : int, optional
        The maximum length for concatenated strings, beyond which grouping is performed,
        by default 40.

    Returns
    -------
    str
        A concatenated string of unique values if the length is less than `max_length`,
        otherwise, a unique group value from the specified column.

    Notes
    -----
    If the concatenated string length is greater than or equal to `max_length`, this
    function returns the unique group value from the column with a "_g" suffix.

    Examples
    --------
    >>> data = {
    >>> 'Category_g': [1, 1, 2, 2, 3],
    >>> 'Category': ['AAAAAAAAAAAAAAA', 'Bovoh', 'Ccccccccccccccc', 'D', 'E']}
    >>> cat_bin_dict = {}
    >>> col = 'Category'
    >>> cat_bin_dict[col] = (
    >>>     X[[f"{col}_g", col]]
    >>>     .groupby(f"{col}_g")
    >>>     .apply(lambda x: concat_or_group(col, x))
    >>>     .to_dict()
    >>> )
    >>> print(cat_bin_dict)
    >>> {'Category': {1: 'gr_1', 2: 'gr_2', 3: 'E'}}
    """
    unique_values = x[col].unique()
    concat_str = " / ".join(map(str, unique_values))
    return (
        concat_str
        if len(concat_str) < max_length
        else concat_str[:7] + "/.../" + concat_str[-7:]
    )


def reset_plot():
    """Reset plot style"""
    # plt.rcParams = plt.rcParamsDefault
    mpl.rcParams.update(plt.rcParamsDefault)


def set_my_plt_style(height=3, width=5, linewidth=2):
    """This set the style of matplotlib to fivethirtyeight with some modifications (colours, axes)

    Parameters
    ----------
    linewidth: int, default=2
        line width
    height: int, default=3
        fig height in inches (yeah they're still struggling with the metric system)
    width: int, default=5
        fig width in inches (yeah they're still struggling with the metric system)

    """
    plt.style.use("fivethirtyeight")
    my_colors_list = qualitative_colors
    myorder = [2, 3, 4, 1, 0, 6, 5, 8, 9, 7]
    my_colors_list = [my_colors_list[i] for i in myorder]
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
        "grid.color": "#9e9e9e",
        "lines.linewidth": linewidth,
    }  # plt.cycler(color=my_colors_list)
    mpl.rcParams.update(params)


def create_dtype_dict(df: pd.DataFrame, dic_keys: str = "col_names") -> dict:
    """Create a custom dictionary of data type for adding suffixes
    to column names in the plotting utility for association matrix.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe used for computing the association matrix.
    dic_keys : str
        Either "col_names" or "dtypes" for returning either a dictionary
        with column names or dtypes as keys.

    Returns
    -------
    dict
        A dictionary with either column names or dtypes as keys.

    Raises
    ------
    ValueError
        If `dic_keys` is not either "col_names" or "dtypes".
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be a pandas DataFrame")

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    time_cols = df.select_dtypes(
        include=["datetime", "timedelta", "datetimetz"]
    ).columns
    numerical_interval_cols = df.select_dtypes(
        ["Interval[float]", "Interval[int]"]
    ).columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    remaining_cols = (
        df.columns.difference(categorical_cols)
        .difference(numerical_cols)
        .difference(time_cols)
        .difference(numerical_interval_cols)
    )

    if dic_keys == "col_names":
        cat_dict = dict.fromkeys(categorical_cols, "cat")
        num_dict = dict.fromkeys(numerical_cols, "num")
        num_interval_dict = dict.fromkeys(numerical_interval_cols, "num_interval")
        time_dict = dict.fromkeys(time_cols, "time")
        remaining_dict = dict.fromkeys(remaining_cols, "unk")
        return {
            **cat_dict,
            **num_dict,
            **num_interval_dict,
            **time_dict,
            **remaining_dict,
        }

    if dic_keys == "dtypes":
        return {
            "cat": categorical_cols.tolist(),
            "num": numerical_cols.tolist(),
            "num_interval": numerical_interval_cols.tolist(),
            "time": time_cols.tolist(),
            "unk": remaining_cols.tolist(),
        }

    raise ValueError("dic_keys should be either 'col_names' or 'dtypes'")


def get_pandas_cat_codes(X):
    """
    Converts categorical and time features in a pandas DataFrame into numerical codes.

    Parameters
    ----------
    X : pandas DataFrame
        The input DataFrame containing categorical and/or time features.

    Returns
    -------
    X : pandas DataFrame
        The modified input DataFrame with categorical and time features replaced by numerical codes.
    obj_feat : list or None
        List of column names that were converted to numerical codes. Returns None if no categorical or time features found.
    cat_idx : list or None
        List of column indices for the columns in obj_feat. Returns None if no categorical or time features found.
    """
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")
    obj_feat = dtypes_dic["cat"] + dtypes_dic["time"] + dtypes_dic["unk"]

    if obj_feat:
        for obj_column in obj_feat:
            column = X[obj_column].astype("str").astype("category")
            # performs label encoding
            _, inverse = np.unique(column, return_inverse=True)
            X[obj_column] = inverse
        cat_idx = [X.columns.get_loc(col) for col in obj_feat]
    else:
        obj_feat = None
        cat_idx = None

    return X, obj_feat, cat_idx


def validate_sample_weight(sample_weight):
    """Ensures sample_weight parameter is a numpy array."""
    if isinstance(sample_weight, pd.Series):
        return sample_weight.values
    elif isinstance(sample_weight, np.ndarray):
        return sample_weight
    elif sample_weight is None:
        return None
    else:
        raise ValueError("sample_weight must be an array-like object or None.")


def validate_sample_weight(sample_weight):
    """
    Validate the sample_weight parameter.

    Parameters
    ----------
    sample_weight : array-like or None
        Input sample weights.

    Returns
    -------
    np.ndarray or None
        If sample_weight is a Pandas Series, its values are returned as a
        numpy array. If sample_weight is already a numpy array, it is
        returned unmodified. If sample_weight is None, None is returned.

    Raises
    ------
    ValueError
        If sample_weight is not an array-like object or None.
    """
    if isinstance(sample_weight, pd.Series):
        return sample_weight.values
    elif isinstance(sample_weight, np.ndarray):
        return sample_weight
    elif sample_weight is None:
        return None
    else:
        raise ValueError("sample_weight must be an array-like object or None.")


def validate_pandas_input(arg):
    """Validate if pandas or numpy arrays are provided
    Parameters
    ----------
    arg : pd.DataFrame or np.array
        the object to validate
    Raises
    ------
    TypeError
        error if pandas or numpy arrays are not provided
    """
    try:
        return arg.values
    except AttributeError:
        raise TypeError("input needs to be a numpy array or pandas data frame.")


def check_if_tree_based(model):
    """check if estimator is tree based

    Parameters
    ----------
    model : object
        the estimator to check

    Returns
    -------
    condition : boolean
        if tree based or not
    """
    tree_based_models = [
        "lightgbm",
        "lgbm",
        "xgboost",
        "xgb",
        "catboost",
        "forest",
        "boosting",
        "tree",
    ]
    return any(m in model.__class__.__name__.lower() for m in tree_based_models)


def is_lightgbm(estimator):
    """check if estimator is lightgbm

    Parameters
    ----------
    model : object
        the estimator to check

    Returns
    -------
    condition : boolean
        if lgbm based or not
    """
    is_lgb = "lgbm" in estimator.__class__.__name__.lower()
    return is_lgb


def is_catboost(estimator):
    """check if estimator is catboost

    Parameters
    ----------
    model : object
        the estimator to check

    Returns
    -------
    condition : boolean
        if catboost based or not
    """
    is_cat = "catboost" in estimator.__class__.__name__.lower()
    return is_cat


def is_xgboost(estimator):
    """check if estimator is xgboost

    Parameters
    ----------
    model : object
        the estimator to check

    Returns
    -------
    condition : boolean
        if xgboost based or not
    """
    is_xgb = "xgb" in estimator.__class__.__name__.lower()
    return is_xgb


def LightForestRegressor(n_feat, n_estimators=10):
    """lightGBM implementation of the Random Forest regressor with the
    ideal number of features, according to Elements of statistical learning

    Parameters
    ----------
    n_feat: int
        the number of predictors (nbr of columns of the X matrix)
    n_estimators : int, optional
        the number of trees/estimators, by default 10

    Returns
    -------
    lightgbm regressor
        sklearn random forest estimator based on lightgbm
    """

    feat_frac = n_feat / (3 * n_feat)
    return lgb.LGBMRegressor(
        verbose=-1,
        force_col_wise=True,
        n_estimators=n_estimators,
        subsample=0.632,
        colsample_bytree=feat_frac,
        boosting_type="rf",
        subsample_freq=1,
    )


def LightForestClassifier(n_feat, n_estimators=10):
    """lightGBM implementation of the Random Forest classifier with the
    ideal number of features, according to Elements of statistical learning

    Parameters
    ----------
    n_feat: int
        the number of predictors (nbr of columns of the X matrix)
    n_estimators : int, optional
        the number of trees/estimators, by default 10

    Returns
    -------
    lightgbm classifier
        sklearn random forest estimator based on lightgbm
    """
    feat_frac = np.sqrt(n_feat) / n_feat
    return lgb.LGBMClassifier(
        verbose=-1,
        force_col_wise=True,
        n_estimators=n_estimators,
        subsample=0.632,
        colsample_bytree=feat_frac,
        boosting_type="rf",
        subsample_freq=1,
    )


def is_list_of_str(str_list):
    """Check if ``str_list`` is a list of strings.

    Parameters
    ----------
    str_list : list or None
        The list to check.

    Returns
    -------
    bool
        True if the list is a list of strings, False otherwise.
    """
    if (
        str_list is not None
        and isinstance(str_list, list)
        and all(isinstance(s, str) for s in str_list)
    ):
        return True
    else:
        return False


def is_list_of_bool(bool_list):
    """Check if ``bool_list`` is not a list of Booleans

    Parameters
    ----------
    bool_list : list of bool
        the list we want to check for

    Returns
    -------
    bool
        True if list of Booleans, else False
    """
    if (
        bool_list is not None
        and isinstance(bool_list, list)
        and all(isinstance(s, bool) for s in bool_list)
    ):
        return True
    else:
        return False


def is_list_of_int(int_list):
    """Check if ``int_list`` is not a list of integers

    Parameters
    ----------
    int_list : list of int
        the list we want to check for

    Returns
    -------
    bool
        True if list of integers, else False
    """
    if (
        int_list is not None
        and isinstance(int_list, list)
        and all(isinstance(s, int) for s in int_list)
    ):
        return True
    else:
        return False


def _get_titanic_data():
    """Load Titanic data and add dummies (random predictors, numeric and categorical) and
    a genuine one, for benchmarking purpose. Classification (binary)

    Returns
    -------
    object
        Bunch sklearn, extension of dictionary
    """
    # Fetch Titanic data and add random cat and numbers
    # Example taken from https://scikit-learn.org/stable/auto_examples/inspection/
    # plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
    X, y = fetch_openml(
        "titanic", version=1, as_frame=True, return_X_y=True, parser="auto"
    )
    rng = np.random.RandomState(seed=42)
    nice_guys = ["Rick", "Bender", "Cartman", "Morty", "Fry", "Vador", "Thanos"]
    X["random_cat"] = np.random.choice(nice_guys, X.shape[0])
    X["random_num"] = rng.randn(X.shape[0])
    X["family_size"] = X["parch"] + X["sibsp"]
    X.drop(["parch", "sibsp"], axis=1, inplace=True)
    X["is_alone"] = np.where(X["family_size"] > 1, 0, 1)
    X["title"] = (
        X["name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    )
    X.loc[X["title"] == "Miss", "title"] = "Mrs"
    title_counts = X["title"].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    X.loc[X["title"].isin(rare_titles), "title"] = "rare"
    categorical_columns = [
        "pclass",
        "sex",
        "embarked",
        "random_cat",
        "is_alone",
        "title",
    ]
    numerical_columns = ["age", "family_size", "fare", "random_num"]
    X = X[categorical_columns + numerical_columns]

    # Preprocessing
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing")
    )
    numerical_pipe = make_pipeline(SimpleImputer(strategy="mean"))
    preprocessor = make_column_transformer(
        (categorical_pipe, categorical_columns),
        (numerical_pipe, numerical_columns),
    )
    X = preprocessor.fit_transform(X)

    # Encode categorical variables
    X = pd.DataFrame(X, columns=categorical_columns + numerical_columns)
    X[categorical_columns] = X[categorical_columns].astype(str)
    X[numerical_columns] = X[numerical_columns].astype(float)

    # Create sample weights
    sample_weight = np.random.uniform(0, 1, len(y))

    return Bunch(
        data=X,
        target=y,
        sample_weight=sample_weight,
        categorical=categorical_columns,
    )


def _get_cancer_data():
    """Load breast cancer data and add dummies (random predictors) and a genuine one, for benchmarking purpose
    Classification (binary)

    Returns
    -------
    object
        Bunch sklearn, extension of dictionary
    """

    rng = np.random.RandomState(seed=42)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X = pd.DataFrame(X)
    X.columns = data.feature_names
    X["random_num1"] = rng.randn(X.shape[0])
    X["random_num2"] = np.random.poisson(1, X.shape[0])
    z = y.astype(int)
    X["genuine_num"] = z * np.abs(
        np.random.normal(0, 0.1, X.shape[0])
    ) + np.random.normal(0, 0.1, X.shape[0])
    y = pd.Series(y)
    return Bunch(data=X, target=y, sample_weight=None, categorical=None)


def _load_boston_data():
    """Load Boston data and add dummies (random predictors, numeric and categorical) and
    a genuine one, for benchmarking purpose. Regression (positive domain).

    Returns
    -------
    object
        Bunch sklearn, extension of dictionary
    """

    data_file_name = resource_filename(__name__, "dataset/data/boston_bunch.joblib")
    return joblib.load(data_file_name)


def _load_housing(as_frame: bool = False):
    """Load the California housing data. See here
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
    for the downloadable version.

    Parameters
    ----------
    as_frame :
        return a pandas dataframe? if not then a "Bunch" (enhanced dictionary) is returned (default ``True``)

    Returns
    -------
    pd.DataFrame or Bunch
        the dataset

    """
    fdescr_name = resource_filename(__name__, "dataset/descr/housing.rst")
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = resource_filename(__name__, "dataset/data/housing.zip")
    data = pd.read_csv(data_file_name)
    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    if as_frame:
        return data
    else:
        return Bunch(
            data=data[feature_names].values,
            target=data["target"].values,
            feature_names=feature_names,
            DESCR=descr_text,
            filename=data_file_name,
        )


def plot_y_vs_X(X, y, ncols=2, figsize=(10, 10)):
    """Plot target vs relevant and non-relevant predictors

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame of the predictors.
    y : np.array
        The target.
    ncols : int, optional
        The number of columns in the facet plot. Default is 2.
    figsize : tuple, optional
        The figure size. Default is (10, 10).

    Returns
    -------
    plt.figure
        The univariate plots y vs pred_i.
    """
    n_cols_to_plot = X.shape[1]
    n_rows = int(np.ceil(n_cols_to_plot / ncols))

    # Create figure and axes
    f, axs = plt.subplots(nrows=n_rows, ncols=ncols, figsize=figsize)

    for i, col in enumerate(X.columns):
        row = i // ncols
        col = i % ncols
        axs[row, col].scatter(X[col], y, alpha=0.1)
        axs[row, col].set_title(col)

    # Hide unused subplots
    for i in range(n_cols_to_plot, n_rows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].set_axis_off()

    # Adjust spacing between subplots
    plt.tight_layout()

    return f


def load_data(name="Titanic"):
    """Load some toy data set to test the All Relevant Feature Selection methods.
    Dummies (random) predictors are added and ARFS should be able to filter them out.
    The Titanic predictors are encoded (needed for scikit estimators).

    Titanic and cancer are for binary classification, they contain synthetic random (dummies) predictors and a
    noisy but genuine synthetic predictor. Hopefully, a good All Relevant FS should be able to detect all the
    predictors genuinely related to the target.

    Boston is for regression, this data set contains

    Parameters
    ----------
    name : str, optional
        the name of the data set. Titanic is for classification with sample_weight,
        Boston for regression and cancer for classification (without sample weight), by default 'Titanic'

    Returns
    -------
    Bunch
        extension of dictionary, accessible by key

    Raises
    ------
    ValueError
        if the dataset name is invalid
    """

    if name == "Titanic":
        return _get_titanic_data()
    elif name == "Boston":
        return _load_boston_data()
    elif name == "cancer":
        return _get_cancer_data()
    elif name == "housing":
        return _load_housing(as_frame=False)
    else:
        raise ValueError(
            "`name should be in ['Titanic', 'Boston', 'cancer', 'housing']`"
        )


def _make_corr_dataset_regression(size=1000):
    """Generate an artificial dataset for regression tasks with columns that
    are correlated, have no variance, large cardinality, numerical and categorical.

    Parameters
    ----------
    size : int, optional
        number of rows to generate, by default 1000

    Returns
    -------
    pd.DataFrame, pd.Series, pd.Series
        the predictors matrix, the target and the weights
    """
    # generate weights
    w = np.random.beta(a=1, b=0.5, size=size)

    # set seed for reproducibility
    np.random.seed(42)

    # generate target variable
    sigma = 0.2
    y = np.random.normal(1, sigma, size)

    # generate correlated features
    z = y - np.random.normal(1, sigma / 5, size) + np.random.normal(1, sigma / 5, size)
    X = pd.DataFrame(
        {
            "var0": z,
            "var1": y * np.abs(np.random.normal(0, sigma * 2, size))
            + np.random.normal(0, sigma / 10, size),
            "var2": -y + np.random.normal(0, sigma, size),
            "var3": y**2 + np.random.normal(0, sigma, size),
            "var4": np.sqrt(y) + np.random.gamma(1, 0.2, size),
            "var5": np.random.normal(0, 1, size),
            "var6": np.random.poisson(1, size),
            "var7": np.random.binomial(1, 0.3, size),
            "var8": np.random.normal(0, 1, size),
            "var9": np.random.poisson(1, size),
            "var10": np.ones(size),
            "var11": np.concatenate(
                [
                    np.arange(start=0, stop=int(size / 2), step=1),
                    np.arange(start=0, stop=int(size / 2), step=1),
                ]
            ),
            "var12": y**3 + np.abs(np.random.normal(0, 1, size)),
        }
    )

    # introduce missing values
    idx_nan = np.random.choice(size, int(round(size / 2)), replace=False)
    X.loc[idx_nan, "var12"] = np.nan

    # set column names and types
    X.columns = ["var" + str(i) for i in range(13)]
    X["var11"] = X["var11"].astype("category")
    X["nice_guys"] = np.random.choice(
        [
            "Rick",
            "Bender",
            "Cartman",
            "Morty",
            "Fry",
            "Vador",
            "Thanos",
            "Bejita",
            "Cell",
            "Tinkywinky",
            "Lecter",
            "Alien",
            "Terminator",
            "Drago",
            "Dracula",
            "Krueger",
            "Geoffrey",
            "Goldfinder",
            "Blackbeard",
            "Excel",
            "SAS",
            "Bias",
            "Variance",
            "Scrum",
            "Human",
            "Garry",
            "Coldplay",
            "Imaginedragons",
            "Platist",
            "Creationist",
            "Gruber",
            "KeyserSoze",
            "Luthor",
            "Klaue",
            "Bane",
            "MarkZ",
        ],
        size,
    )

    return X, y, w


def _make_corr_dataset_classification(size=1000):
    """
    Generate an artificial dataset for classification tasks. Some columns are correlated,
    have no variance, large cardinality, numerical and categorical.

    Parameters:
        size (int): The number of rows to generate. Default is 1000.

    Returns:
        tuple: A tuple containing the predictors matrix, the target, and the weights.
    """
    # Generate weights
    w = np.random.beta(a=1, b=0.5, size=size)

    # Fix the seed and generate the target
    np.random.seed(42)
    y = np.random.binomial(1, 0.5, size)

    # Generate the predictors matrix
    X = np.zeros((size, 13))

    z = y - np.random.binomial(1, 0.1, size) + np.random.binomial(1, 0.1, size)
    z[z == -1] = 0
    z[z == 2] = 1

    # Generate 5 relevant features, with positive and negative correlation to the target
    X[:, 0] = z
    X[:, 1] = y * np.abs(np.random.normal(0, 1, size)) + np.random.normal(0, 0.1, size)
    X[:, 2] = -y + np.random.normal(0, 1, size)
    X[:, 3] = y**2 + np.random.normal(0, 1, size)
    X[:, 4] = np.sqrt(y) + np.random.binomial(2, 0.1, size)

    # Generate 5 irrelevant features
    X[:, 5:10] = np.random.normal(0, 1, size=(size, 5))

    # Generate a column with zero variance
    X[:, 10] = np.ones(size)

    # Generate a column with high cardinality
    X[:, 11] = np.arange(start=0, stop=size, step=1)

    # Generate a column with a lot of missing values
    idx_nan = np.random.choice(size, int(round(size / 2)), replace=False)
    X[:, 12] = y**3 + np.abs(np.random.normal(0, 1, size))
    X[idx_nan, 12] = np.nan

    # Make the predictors matrix a pandas DataFrame
    column_names = ["var" + str(i) for i in range(13)]
    column_names[11] = "dummy"
    X = pd.DataFrame(X, columns=column_names)
    X["dummy"] = X["dummy"].astype("category")

    # Add a column of random values from a list
    nice_guys = [
        "Rick",
        "Bender",
        "Cartman",
        "Morty",
        "Fry",
        "Vador",
        "Thanos",
        "Bejita",
        "Cell",
        "Tinkywinky",
        "Lecter",
        "Alien",
        "Terminator",
        "Drago",
        "Dracula",
        "Krueger",
        "Geoffrey",
        "Goldfinder",
        "Blackbeard",
        "Excel",
        "SAS",
        "Bias",
        "Variance",
        "Scrum",
        "Human",
        "Garry",
        "Coldplay",
        "Imaginedragons",
        "Platist",
        "Creationist",
        "Gruber",
        "KeyserSoze",
        "Luthor",
        "Klaue",
        "Bane",
        "MarkZ",
    ]

    return X, y, w
