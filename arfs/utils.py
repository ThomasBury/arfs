"""
Utility and validation functions
  
"""

from __future__ import print_function, division

import lightgbm as lgb
import matplotlib as mpl
import itertools
import numpy as np
import pandas as pd

from pkg_resources import resource_filename
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.base import clone
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


def highlight_tick(str_match, figure, color="red", axis="y"):
    """Highlight the x/y tick-labels if they contains a given string

    Parameters
    ----------
    str_match : str
        the substring to match
    figure : object
        the matplotlib figure
    color : str, optional
        the matplotlib color for highlighting tick-labels, by default 'red'
    axis : str, optional
        axis to use for highlighting, by default 'y'

    Returns
    -------
    plt.figure
        the modified matplotlib figure

    Raises
    ------
    ValueError
        if axis is not 'x' or 'y'
    """

    if axis == "y":
        labels = [item.get_text() for item in figure.gca().get_yticklabels()]
        indices = [i for i, s in enumerate(labels) if str_match in s]
        [figure.gca().get_yticklabels()[idx].set_color(color) for idx in indices]
    elif axis == "x":
        labels = [item.get_text() for item in figure.gca().get_xticklabels()]
        indices = [i for i, s in enumerate(labels) if str_match in s]
        [figure.gca().get_xticklabels()[idx].set_color(color) for idx in indices]
    else:
        raise ValueError("`axis` should be a string, either 'y' or 'x'")

    return figure


def sklearn_pimp_bench(model, X, y, task="regression", sample_weight=None):
    """Benchmark using sklearn permutation importance, works for regression and classification.

    Parameters
    ----------
    model: object
        An estimator that has not been fitted, sklearn compatible.
    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.
    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or None for unsupervised.
    task : str, optional
        kind of task, either 'regression' or 'classification", by default 'regression'
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights, by default None

    Returns
    -------
    plt.figure
        the figure corresponding to the feature selection

    Raises
    ------
    ValueError
        if task is not 'regression' or 'classification'
    """

    # for lightGBM cat feat as contiguous int
    # https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
    # same for Random Forest and XGBoost (OHE leads to deep and sparse trees).
    # For illustrations, see
    # https://towardsdatascience.com/one-hot-encoding-is-making-
    # your-tree-based-ensembles-worse-heres-why-d64b282b5769
    X, cat_var_df, inv_mapper = cat_var(X)

    if task == "regression":
        stratify = None
    elif task == "classification":
        stratify = y
    else:
        raise ValueError("`task` should be either 'regression' or 'classification' ")

    if sample_weight is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weight, stratify=stratify, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, random_state=42
        )
        w_train, w_test = None, None

    # lightgbm faster and better than RF

    model.fit(X_train, y_train, sample_weight=w_train)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=2,
        sample_weight=w_test,
    )

    sorted_idx = result.importances_mean.argsort()
    # Plot (5 predictors per inch)
    fig, ax = plt.subplots(figsize=(16, X.shape[1] / 5))
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx]
    )
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    indices = [i for i, s in enumerate(X_test.columns[sorted_idx]) if "random" in s]
    [fig.gca().get_yticklabels()[idx].set_color("red") for idx in indices]
    indices = [i for i, s in enumerate(X_test.columns[sorted_idx]) if "genuine" in s]
    [fig.gca().get_yticklabels()[idx].set_color("green") for idx in indices]
    plt.show()
    return fig


def compare_varimp(feat_selector, models, X, y, sample_weight=None):
    """Utility function to compare the results for the three possible king of feature importance

    Parameters
    ----------
    feat_selector : object
        an instance of either Leshy, BoostaGRoota or GrootCV
    models : list of objects
        list of tree based scikit-learn estimators
    X : pd.DataFrame, shape (n_samples, n_features)
        the predictors frame
    y : pd.Series
        the target (same length as X)
    sample_weight : None or pd.Series, optional
        sample weights if any, by default None
    """

    varimp_list = ["shap", "pimp", "native"]
    for model, varimp in itertools.product(models, varimp_list):
        print(
            "=" * 20
            + " "
            + str(feat_selector.__class__.__name__)
            + " - testing: {mod:>25} for var.imp: {vimp:<15} ".format(
                mod=str(model.__class__.__name__), vimp=varimp
            )
            + "=" * 20
        )
        # change the varimp
        feat_selector.importance = varimp
        # change model
        mod_clone = clone(model, safe=True)
        feat_selector.estimator = mod_clone
        # fit the feature selector
        feat_selector.fit(X=X, y=y, sample_weight=sample_weight)
        # print the results
        print(feat_selector.support_names_)
        fig = feat_selector.plot_importance(n_feat_per_inch=5)

        if fig is not None:
            # highlight synthetic random variable
            fig = highlight_tick(figure=fig, str_match="random")
            fig = highlight_tick(figure=fig, str_match="genuine", color="green")
            plt.show()


def cat_var(df, col_excl=None, return_cat=True):
    """Encode categorical variables using integers

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to be encoded
    col_excl : None or list of str, optional
        list of column names to exclude from the encoding, by default None
    return_cat : bool, optional
        return or not as pandas categorical columns, by default True

    Returns
    -------
    pd.DataFrame, pd.DataFrame, dict
        The encoded pandas dataframe, the dataframe with the details and the mapping dictionary
    """

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

    cols_need_mapped = cat_var_df.cat_name.to_list()
    inv_mapper = {
        col: dict(enumerate(df[col].astype("category").cat.categories))
        for col in df[cols_need_mapped]
    }
    mapper = {
        col: {v: k for k, v in inv_mapper[col].items()} for col in df[cols_need_mapped]
    }

    for c in cols_need_mapped:
        df.loc[:, c] = df.loc[:, c].map(mapper[c]).fillna(0).astype(int)

    if return_cat:
        df[non_num_cols] = df[non_num_cols].astype("category")
    return df, cat_var_df, inv_mapper


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
    X[:, 11] = np.arange(start=0, stop=size, step=1)
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
