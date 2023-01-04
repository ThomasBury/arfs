"""
Utility and validation functions
  
"""

from __future__ import print_function, division

import lightgbm as lgb
import matplotlib as mpl
import numpy as np
import pandas as pd

from pkg_resources import resource_filename
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.utils import Bunch

qualitative_colors = ['#7F3C8D', 
                      '#11A579',
                      '#3969AC',
                      '#F2B701',
                      '#E73F74',
                      '#80BA5A',
                      '#E68310',
                      '#008695',
                      '#CF1C90',
                      '#F97B72']

#####################
#                   #
#     Utilities     #
#                   #
#####################


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



def create_dtype_dict(df: pd.DataFrame, dic_keys: str = 'col_names'):
    """create a custom dictionary of data type for adding suffixes
    to column names in the plotting utility for association matrix

    Parameters
    ----------
    df :
        the dataframe used for computing the association matrix
    dic_keys :
        Either "col_names" or "dtypes" for returning either a dictionary
        with column names or dtypes as keys.
    """
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'bool']))
    time_cols = list(df.select_dtypes(include=['datetime', 'timedelta', 'datetimetz']))
    num_cols = list(df.select_dtypes(include=[np.number]))
    remaining_cols = list(set(df.columns) - set(cat_cols).union(set(num_cols)).union(time_cols))

    if dic_keys == 'col_names':
        cat_dic = {c: "cat" for c in cat_cols}
        num_dic = {c: "num" for c in num_cols}
        time_dic = {c: "time" for c in time_cols}
        remainder_dic = {c: "unk" for c in remaining_cols}
        return {**cat_dic, **num_dic, **time_dic,**remainder_dic}
    elif dic_keys == 'dtypes':
        cat_dic = {"cat": cat_cols}
        num_dic = {"num": num_cols}
        time_dic = {"time": time_cols}
        remainder_dic = {"unk": remaining_cols}
        return {**cat_dic, **num_dic, **time_dic,**remainder_dic}
    else:
        raise ValueError("'dic_keys' should be either 'col_names' or 'dtypes'")
        

def get_pandas_cat_codes(X):
    dtypes_dic = create_dtype_dict(X, dic_keys="dtypes")
    obj_feat = dtypes_dic['cat'] + dtypes_dic['time'] + dtypes_dic['unk']  

    if obj_feat:                
        cat = X[obj_feat].stack().astype("str").astype("category").cat.codes.unstack()
        X = pd.concat([X[X.columns.difference(obj_feat)], cat], axis=1)
        cat_idx = [X.columns.get_loc(col) for col in obj_feat]
    else:
        obj_feat = None
        cat_idx = None
        
    return X, obj_feat, cat_idx


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
    tree_based_models = ["lightgbm", "xgboost", "catboost", "_forest", "boosting"]
    condition = any(i in str(type(model)).lower() for i in tree_based_models)
    return condition


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
    is_lgb = "lightgbm" in str(type(estimator))
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
    is_cat = "catboost" in str(type(estimator))
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
    is_xgb = "xgboost" in str(type(estimator))
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
    """Check if ``str_list`` is not a list of strings

    Parameters
    ----------
    str_list : list of str
        the list we want to check for

    Returns
    -------
    bool
        True if list of strings, else False
    """
    if str_list is not None:
        if not (
            isinstance(str_list, list) and all(isinstance(s, str) for s in str_list)
        ):
            return False
        else:
            return True


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
    if bool_list is not None:
        if not (
            isinstance(bool_list, list) and all(isinstance(s, bool) for s in bool_list)
        ):
            return False
        else:
            return True


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
    if int_list is not None:
        if not (
            isinstance(int_list, list) and all(isinstance(s, int) for s in int_list)
        ):
            return False
        else:
            return True


def set_my_plt_style(height=3, width=5, linewidth=2):
    """This set the style of matplotlib to fivethirtyeight with some modifications (colours, axes)

    Parameters
    ----------
    height : int, optional
        global line width, by default 3
    width : int, optional
        fig height in inches, by default 5
    linewidth : int, optional
        fig width in inches, by default 2
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
        "grid.color": "#d2d2d2",
        "lines.linewidth": linewidth,
    }  # plt.cycler(color=my_colors_list)
    mpl.rcParams.update(params)





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
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    rng = np.random.RandomState(seed=42)
    nice_guys = ["Rick", "Bender", "Cartman", "Morty", "Fry", "Vador", "Thanos"]
    X["random_cat"] = np.random.choice(nice_guys, X.shape[0])
    X["random_num"] = rng.randn(X.shape[0])
    X["family_size"] = X["parch"] + X["sibsp"]
    X.drop(["parch", "sibsp"], axis=1, inplace=True)
    X["is_alone"] = 1
    X.loc[X["family_size"] > 1, "is_alone"] = 0
    X["title"] = (
        X["name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    )
    X.loc[X.title == "Miss", "title"] = "Mrs"
    title_counts = X["title"].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    X.loc[X["title"].isin(rare_titles), "title"] = "rare"
    # X['title'] = X.title.apply(lambda x: 'rare' if rare_titles[x] else x)

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
    # Impute
    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="constant", fill_value="missing"))]
    )
    numerical_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    preprocessing = ColumnTransformer(
        [
            ("cat", categorical_pipe, categorical_columns),
            ("num", numerical_pipe, numerical_columns),
        ]
    )
    X_trans = preprocessing.fit_transform(X)
    X = pd.DataFrame(X_trans, columns=X.columns)
    # encode
    # X, cat_var_df, inv_mapper = cat_var(X)
    X[categorical_columns] = X[categorical_columns].astype(str)
    X[numerical_columns] = X[numerical_columns].astype(float)
    # sample weight is just a dummy random vector for testing purpose
    sample_weight = np.random.uniform(0, 1, len(y))
    return Bunch(
        data=X, target=y, sample_weight=sample_weight, categorical=categorical_columns
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


def _get_boston_data():
    """Load Boston data and add dummies (random predictors, numeric and categorical) and
    a genuine one, for benchmarking purpose. Regression (positive domain).

    Returns
    -------
    object
        Bunch sklearn, extension of dictionary
    """

    boston = load_boston()
    rng = np.random.RandomState(seed=42)
    X = pd.DataFrame(boston.data)
    X.columns = boston.feature_names
    X["random_num1"] = rng.randn(X.shape[0])
    X["random_num2"] = np.random.poisson(1, X.shape[0])
    # high cardinality
    X["random_cat"] = rng.randint(10 * X.shape[0], size=X.shape[0])
    X["random_cat"] = "cat_" + X["random_cat"].astype("str")
    # low cardinality
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
    X["random_cat_2"] = np.random.choice(nice_guys, X.shape[0])
    y = pd.Series(boston.target)
    # non linear noisy but genuine predictor to test the ability to detect even genuine noisy non-linearities
    X["genuine_num"] = np.sqrt(y) + np.random.gamma(2, 0.5, X.shape[0])
    cat_f = ["CHAS", "RAD", "random_cat", "random_cat_2"]
    X[cat_f] = X[cat_f].astype(str).astype("category")
    return Bunch(data=X, target=y, sample_weight=None, categorical=cat_f)


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
        the pd DF of the predictors
    y : np.array
        the target
    ncols : int, optional
        the number of columns in the facet plot, by default 2
    figsize : tuple, optional
        the figure size, by default (10, 10)

    Returns
    -------
    plt.figure
        the univariate plots y vs pred_i
    """


    X = pd.DataFrame(X)
    ncols_to_plot = X.shape[1]
    n_rows = int(np.ceil(ncols_to_plot / ncols))

    # Create figure and axes (this time it's 9, arranged 3 by 3)
    f, axs = plt.subplots(nrows=n_rows, ncols=ncols, figsize=figsize)

    # delete non-used axes
    n_charts = ncols_to_plot
    n_subplots = n_rows * ncols
    cols_to_enum = X.columns

    # Make the axes accessible with single indexing
    if n_charts > 1:
        axs = axs.flatten()

    for i, col in enumerate(cols_to_enum):
        # select the axis where the map will go
        if n_charts > 1:
            ax = axs[i]
        else:
            ax = axs

        ax.scatter(X[col], y, alpha=0.1)
        ax.set_title(col)

    if n_subplots > n_charts > 1:
        for i in range(n_charts, n_subplots):
            ax = axs[i]
            ax.set_axis_off()

    # Display the figure
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
        return _get_boston_data()
    elif name == "cancer":
        return _get_cancer_data()
    elif name == "housing":
        return _load_housing(as_frame=False)
    else:
        raise ValueError(
            "`name should be in ['Titanic', 'Boston', 'cancer', 'housing']`"
        )


def _generated_corr_dataset_regr(size=1000):
    """Generate artificial dataset for regression tasks. Some columns are
    correlated, have no variance, large cardinality, numerical and categorical.

    Parameters
    ----------
    size : int, optional
        number of rows to generate, by default 1000

    Returns
    -------
    pd.DataFrame, pd.Series, pd.Series
        the predictors matrix, the target and the weights
    """
    # weights
    w = np.random.beta(a=1, b=0.5, size=size)
    # fixing the seed and the target
    np.random.seed(42)
    sigma = 0.2
    y = np.random.normal(1, sigma, size)
    z = y - np.random.normal(1, sigma / 5, size) + np.random.normal(1, sigma / 5, size)
    X = np.zeros((size, 13))

    # 5 relevant features, with positive and negative correlation to the target
    X[:, 0] = z
    X[:, 1] = y * np.abs(np.random.normal(0, sigma * 2, size)) + np.random.normal(
        0, sigma / 10, size
    )
    X[:, 2] = -y + np.random.normal(0, sigma, size)
    X[:, 3] = y**2 + np.random.normal(0, sigma, size)
    X[:, 4] = np.sqrt(y) + np.random.gamma(1, 0.2, size)

    # 5 irrelevant features
    X[:, 5] = np.random.normal(0, 1, size)
    X[:, 6] = np.random.poisson(1, size)
    X[:, 7] = np.random.binomial(1, 0.3, size)
    X[:, 8] = np.random.normal(0, 1, size)
    X[:, 9] = np.random.poisson(1, size)
    # zero variance
    X[:, 10] = np.ones(size)
    # high cardinality
    half_size = int(size/2)
    X[:, 11] = np.concatenate([np.arange(start=0, stop=half_size, step=1), np.arange(start=0, stop=size-half_size, step=1)])
    # a lot of missing values
    idx_nan = np.random.choice(size, int(round(size / 2)), replace=False)
    X[:, 12] = y**3 + np.abs(np.random.normal(0, 1, size))
    X[idx_nan, 12] = np.nan

    # make  it a pandas DF
    column_names = ["var" + str(i) for i in range(13)]
    column_names[11] = "dummy_cat"
    X = pd.DataFrame(X)
    X.columns = column_names
    X["dummy_cat"] = X["dummy_cat"].astype("category")
    # low cardinality
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
    X["nice_guys"] = np.random.choice(nice_guys, X.shape[0])

    return X, y, w


def _generated_corr_dataset_classification(size=1000):
    """Generate artificial dataset for classification tasks. Some columns are
    correlated, have no variance, large cardinality, numerical and categorical.

    Parameters
    ----------
    size : int, optional
        number of rows to generate, by default 1000

    Returns
    -------
    pd.DataFrame, pd.Series, pd.Series
        the predictors matrix, the target and the weights
    """
    # weights
    w = np.random.beta(a=1, b=0.5, size=size)
    # fixing the seed and the target
    np.random.seed(42)
    y = np.random.binomial(1, 0.5, size)
    X = np.zeros((size, 13))

    z = y - np.random.binomial(1, 0.1, size) + np.random.binomial(1, 0.1, size)
    z[z == -1] = 0
    z[z == 2] = 1

    # 5 relevant features, with positive and negative correlation to the target
    X[:, 0] = z
    X[:, 1] = y * np.abs(np.random.normal(0, 1, size)) + np.random.normal(0, 0.1, size)
    X[:, 2] = -y + np.random.normal(0, 1, size)
    X[:, 3] = y**2 + np.random.normal(0, 1, size)
    X[:, 4] = np.sqrt(y) + np.random.binomial(2, 0.1, size)

    # 5 irrelevant features
    X[:, 5] = np.random.normal(0, 1, size)
    X[:, 6] = np.random.poisson(1, size)
    X[:, 7] = np.random.binomial(1, 0.3, size)
    X[:, 8] = np.random.normal(0, 1, size)
    X[:, 9] = np.random.poisson(1, size)
    # zero variance
    X[:, 10] = np.ones(size)
    # high cardinality
    X[:, 11] = np.arange(start=0, stop=size, step=1)
    # a lot of missing values
    idx_nan = np.random.choice(size, int(round(size / 2)), replace=False)
    X[:, 12] = y**3 + np.abs(np.random.normal(0, 1, size))
    X[idx_nan, 12] = np.nan

    # make  it a pandas DF
    column_names = ["var" + str(i) for i in range(13)]
    column_names[11] = "dummy"
    X = pd.DataFrame(X)
    X.columns = column_names
    X["dummy"] = X["dummy"].astype("category")

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
    X["nice_guys"] = np.random.choice(nice_guys, X.shape[0])

    return X, y, w
