"""
This module provides a class for basic feature selection based on the following rationals:
        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find collinear variables with a correlation greater than a specified correlation coefficient
        4. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
        5. Find low importance features that do not contribute to a specified cumulative
           feature importance from the gbm SHAP feature importance.

It is the first step of the feature selection, ideally followed by a
"all relevant feature selection" step.

"""

# from https://github.com/WillKoehrsen
# modified by Thomas Bury

# memory management
import gc
# utilities
from itertools import chain
import time
import random
# numpy and pandas for data manipulation
import pandas as pd
import numpy as np
from dython.nominal import compute_associations
# model used for feature importance, Shapley values are builtin
import lightgbm as lgb
# visualizations
import matplotlib as mpl
import matplotlib.pyplot as plt
from palettable.cmocean.diverging import Curl_5_r
from palettable.cartocolors.qualitative import Bold_10
import holoviews as hv
import panel as pn
# progress bar
from tqdm.autonotebook import trange, tqdm
# ML
import scipy.cluster.hierarchy as sch

# set style
hv.extension('bokeh', logo=False)
hv.renderer('bokeh').theme = 'light_minimal'


#####################
#                   #
#     Utilities     #
#                   #
#####################

def reset_plot():
    """
    reset matplotlib default
    :return: None
    """
    plt.rcParams = plt.rcParamsDefault


def set_my_plt_style(height=3, width=5, linewidth=2):
    """
    This set the style of matplotlib to fivethirtyeight with some modifications (colours, axes)

    :param linewidth: int, default=2
        line width
    :param height: int, default=3
        fig height in inches (yeah they're still struggling with the metric system)
    :param width: int, default=5
        fig width in inches (yeah they're still struggling with the metric system)
    :return: Nothing
    """
    plt.style.use('fivethirtyeight')
    my_colors_list = Bold_10.hex_colors
    myorder = [2, 3, 4, 1, 0, 6, 5, 8, 9, 7]
    my_colors_list = [my_colors_list[i] for i in myorder]
    bckgnd_color = "#f5f5f5"
    params = {'figure.figsize': (width, height), "axes.prop_cycle": plt.cycler(color=my_colors_list),
              "axes.facecolor": bckgnd_color, "patch.edgecolor": bckgnd_color,
              "figure.facecolor": bckgnd_color,
              "axes.edgecolor": bckgnd_color, "savefig.edgecolor": bckgnd_color,
              "savefig.facecolor": bckgnd_color, "grid.color": "#d2d2d2",
              'lines.linewidth': linewidth}  # plt.cycler(color=my_colors_list)
    mpl.rcParams.update(params)


def cat_var(df, col_excl=None, return_cat=True):
    """
    Categorical encoding (as integer). Automatically detect the non-numerical columns,
    save the index and name of those columns, encode them as integer,
    save the direct and inverse mappers as
    dictionaries.
    Return the data-set with the encoded columns with a data type either int or pandas categorical.

    :param df: pd.DataFrame
        the dataset
    :param col_excl: list of str, default=None
        the list of columns names not being encoded (e.g. the ID column)
    :param return_cat: bool, default=True
        return encoded object columns as pandas categoricals or not.
    :return:
     df: pd.DataFrame
        the dataframe with encoded columns
     cat_var_df: pd.DataFrame
        the dataframe with the indices and names of the categorical columns
     inv_mapper: dict
        the dictionary to map integer --> category
     mapper: dict
        the dictionary to map category --> integer
    """

    if col_excl is None:
        non_num_cols = list(set(list(df.columns)) - set(list(df.select_dtypes(include=[np.number]))))
    else:
        non_num_cols = list(
            set(list(df.columns)) - set(list(df.select_dtypes(include=[np.number]))) - set(col_excl))

    cat_var_index = [df.columns.get_loc(c) for c in non_num_cols if c in df]

    cat_var_df = pd.DataFrame({'cat_ind': cat_var_index,
                               'cat_name': non_num_cols})

    # avoid having datetime objects as keys in the mapping dic
    date_cols = [s for s in list(df) if "date" in s]
    df[date_cols] = df[date_cols].astype(str)

    cols_need_mapped = cat_var_df.cat_name.to_list()
    inv_mapper = {col: dict(enumerate(df[col].astype('category').cat.categories))
                  for col in df[cols_need_mapped]}
    mapper = {col: {v: k for k, v in inv_mapper[col].items()} for col in df[cols_need_mapped]}

    progress_bar = tqdm(cols_need_mapped)
    for c in progress_bar:
        progress_bar.set_description('Processing {0:<30}'.format(c))
        df[c] = df[c].map(mapper[c]).fillna(0).astype(int)
        # I could have use df[c].update(df[c].map(mapper[c])) while slower,
        # prevents values not included in an incomplete map from being changed to nans.
        # But then I could have outputs
        # with mixed types in the case of different dtypes mapping (like str -> int).
        # This would eventually break any flow.
        # Map is faster than replace

    if return_cat:
        df[non_num_cols] = df[non_num_cols].astype('category')
    return df, cat_var_df, inv_mapper, mapper


def plot_corr(df, size=10):
    """
    Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """
    set_my_plt_style()
    corr = df.corr()
    # Re-order the rows and columns using clustering
    d = sch.distance.pdist(corr)
    L = sch.linkage(d, method='ward')
    ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    corr = corr.reindex(columns, axis=1)
    corr = corr.reindex(columns, axis=0)

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap=Curl_5_r.mpl_colormap, vmin=-1, vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, -0.5, 0, 0.5, 1], aspect=10, shrink=.8)
    # plt.show()
    return fig


def plot_associations(df, features=None, size=1200, theil_u=False):

    if features is None:
        features = df.columns

    # continuous features
    con_features = set(features).intersection(set(list(df.select_dtypes(include=[np.number]))))

    # nominal features
    nom_features = set(features) - set(con_features)

    assoc_df = compute_associations(df,
                                    nominal_columns=nom_features,
                                    mark_columns=True,
                                    theil_u=theil_u,
                                    clustering=True,
                                    bias_correction=True,
                                    nan_strategy='drop_samples')

    heatmap = hv.HeatMap((assoc_df.columns, assoc_df.index, assoc_df)).redim.range(z=(-1, 1))

    heatmap.opts(tools=['tap', 'hover'], height=size, width=size + 50, toolbar='left', colorbar=True,
                cmap=Curl_5_r.mpl_colormap, fontsize={'title': 12, 'ticks': 12, 'minor_ticks': 12}, xrotation=90,
                invert_xaxis=False, invert_yaxis=True,  # title=title_str,
                xlabel='', ylabel=''
                     )
    title_str = "**Continuous (con) and Categorical (nom) Associations **"
    sub_title_str = "*Categorical(nom): uncertainty coefficient & correlation ratio from 0 to 1. The uncertainty " \
                        "coefficient is assymmetrical, (approximating how much the elements on the " \
                        "left PROVIDE INFORMATION on elements in the row). Continuous(con): symmetrical numerical " \
                        "correlations (Pearson's) from -1 to 1*"
    panel_layout = pn.Column(
            pn.pane.Markdown(title_str, align="start"),  # bold
            pn.pane.Markdown(sub_title_str, align="start"),  # italic
            heatmap, background='#ebebeb'
    )

    gc.enable()
    del assoc_df
    gc.collect()
    return panel_layout


######################
#                    #
#     main class     #
#                    #
######################


class FeatureSelector:
    """
    Class for performing feature selection for machine learning or data preprocessing.
    Heavily inspired from https://github.com/WillKoehrsen

    Implements five different methods to identify features for removal

        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find collinear variables with a correlation greater than a specified correlation coefficient
        4. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
        5. Find low importance features that do not contribute to a specified
        cumulative feature importance from
           the gbm SHAP feature importance.

    Parameters
    ----------
        X : pd.DataFrame
            A dataset with observations in the rows and features in the columns

        y : array or series, default = None
            Array of labels for training the machine learning model to find feature importances.
            These can be either binary labels (if task is 'classification') or continuous targets
            (if task is 'regression').
            If no labels are provided, then the feature importance based methods are not available.

        sample_weight : pd.Series or np.array.
            Sample weights, if any

    Attributes
    ----------

    cat_features : list of str
        list of column names for the categorical columns. Note that lightGBM is
        working better if you integer encode the categorical predictors and leave
        this argument to None, even if there is order relationship.
        See lightGBM doc and blog pages.

    record_missing : pd.DataFrame
        The fraction of missing values for features with missing fraction above threshold

    record_single_unique : dataframe
        Records the features that have a single unique value

    record_collinear : dataframe
        Records the pairs of collinear variables with a correlation
        coefficient above the threshold

    record_zero_importance : dataframe
        Records the zero importance features in the data according to the gbm,
        using SHAP feature importance

    record_low_importance : dataframe
        Records the lowest importance features not needed to reach the
        threshold of cumulative importance
        according to the gbm, using SHAP feature importance

    missing_stats : dataframe
        The fraction of missing values for all features

    unique_stats : dataframe
        Number of unique values for all features

    cardinality_stats : dataframe
        Cardinality for all features (number of unique values for categorical columns

    corr_matrix : dataframe
        All correlations between all features in the data

    correlation_threshold: float between 0 and 1
        the pariwise correlation threshold, above this threshold one of the two
        predictors is tagged 'to be dropped'

    feature_importances : dataframe
        All feature importances (SHAP) from the gradient boosting machine

    ops : dict
        Dictionary of operations run and features identified for removal

    cat_var_df : pd.DataFrame
        dataframe with the indices and names of the categorical columns

    removed_features : list of str
        the list of features to drop

    missing_threshold : float between 0 and 1
        the fraction of missing values to tag the column 'to drop"

    cumulative_importance : float between 0 and 1,
        the fraction of the feature importance. The features are sorted by importance.
        The selector drop all the feature contributing for the part > cumulative_importance.
        Example: 0.9, then all the predictors contributing upto 90%
        of the total feature importance are kept, the others being discarded.

    all_identified : list of str
        the name of all the predictors to drop

    n_identified : int
        the number of predictors to drop

    encoded : bool
        True if the categorical and object columns are integer encoded or not during the process

    mapper : dict
        The dictionary of the encoding mapper, if any

    tag_df : pd.DataFrame
        the dataframe with the tag "to drop" or "to keep" (== not to drop)


    Methods:
    --------
    identify_patterns(patterns=None):
        Drop the columns by identifying patterns in their name

    identify_missing(missing_threshold=0.1):
        Find the features with a fraction of missing values above `missing_threshold`

    identify_single_unique():
        find the zero variance columns

    identify_high_cardinality(max_card=1000):
        Finds the categorical columns with more than max_card unique values
        (might be relevant for GLM or if the categorical column has as many
        levels than rows in the data set)

    encode_cat_var(df=None, col_excl=None):
        Categorical encoding (as integer). Automatically detect the non-numerical columns,
        save the index and name of those columns, encode them as integer,
        save the direct and inverse mappers as dictionaries.
        Return the data-set with the encoded columns with a data type either int or pandas categorical.

    identify_collinear(correlation_threshold, encode=False, method='spearman'):
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.

    identify_zero_importance(task, eval_metric=None, objective=None,
                             n_iterations=10, early_stopping=True, missing=0):
        Identify the features with zero importance according to a gradient boosting machine.
        The gbm can be trained with early stopping using a utils set to prevent overfitting.
        The feature importances are averaged over `n_iterations` to reduce variance.

    identify_low_importance(cumulative_importance):
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance from the gradient boosting machine. As an example, if cumulative
        importance is set to 0.95, this will retain only the most important features needed to
        reach 95% of the total feature importance. The identified features are those not needed.

    identify_all(selection_params):
        Use all methods to identify features to remove.

    check_removal():
        Check the identified features before removal. Returns a list of the unique features identified

    remove(methods):
        Remove the features from the data according to the specified methods.

    plot_missing():
        Histogram of missing fraction in each feature

    plot_unique():
        Histogram of the number of unique values per column

    plot_collinear(plot_all=False, size=1000):
        plot the sorted correlation matrix as an interactive heatmap

    plot_feature_importances(plot_n=15, threshold=None):
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach
        `threshold` cumulative importance.


    Example:
    --------

    # Initiate an instance of the feature selector
    fs = noglmfs.FeatureSelector(data = df[predictors], labels = df.re_cl_RCC, weight = df.exp_yr)

    # A dictionary to store the output of each step (to combine with further selection,
    not provided by this FS, or to split the process in two parts to speed up the
    computation of the correlation matrix.)
    fsDic = {}
    # Missing values
    fs.identify_missing(missing_threshold=0.2)
    fsDic['starting_list'] = predictors
    fsDic['missing'] = fs.ops['missing']
    fs.plot_missing()

    # Unique value
    fs.identify_single_unique()
    fs.plot_unique()
    fsDic['single_unique'] = fs.ops['single_unique']

    # Large cardinality
    fs.identify_high_cardinality(max_card=2500)
    fs.plot_cardinality()
    fsDic['high_cardinality'] = fs.ops['high_cardinality']

    # Remove the tagged predictors so far to speed up the computation of the sorted correlation matrix
    cols_to_drop = fs.check_removal()
    filtered_features = list( set(predictors) - set(cols_to_drop) )
    survivors_cols = targets + exposure  + filtered_features
    df_red = df[survivors_cols].copy()
    fs_df = fs.tag_df

    # New instance of the feature selector
    gc.enable()
    del(fs)
    gc.collect()
    fs = noglmfs.FeatureSelector(data = df_red[filtered_features],
                                 labels = df_red.re_cl_RCC,
                                 weight = df_red.exp_yr)

    # Correlation
    fs.identify_collinear(correlation_threshold=0.85, encode=False)
    fs_df = fs_df.merge(fs.tag_df, how='left') # tag the discarded predictors and store the results
    fsDic['collinear'] = sorted(fs.ops['collinear']
    heatmap = fs.plot_collinear(plot_all=True, size=1500)
    hv.save(heatmap, outpath + "heatmap_corr_TPLMD_freq.html")

    # Drop the predictors tagged by the correlation step and remove them in order to speed up the next steps
    prefiltered_feat_to_remove = fs.check_removal()
    features_all_corrfilt = list(set(filtered_features) - set(prefiltered_feat_to_remove))

    # New instance of the feature selector
    gc.enable()
    del(fs)
    gc.collect()
    X = df[features_all_corrfilt].copy()
    fs = noglmfs.FeatureSelector(data = X, labels = df.re_cl_RCC, weight = df.exp_yr)
    fs.encoded = True

    # Identify the zero and low importance predictors
    fs.identify_zero_importance(task = 'regression', eval_metric = 'poisson',
                                n_iterations = 10, early_stopping = True)
    cum_imp_threshold = 0.95
    fs.identify_low_importance(cumulative_importance = cum_imp_threshold)
    fsDic['zero_importance'] = fs.ops['zero_importance']
    fsDic['low_importance'] = fs.ops['low_importance']
    fs_df = fs_df.merge(fs.tag_df, how='left')
    feat_imp = fs.plot_feature_importances(threshold = 0.9, plot_n = 50)
    hv.save(feat_imp, outpath+"feat_imp_TPLMD_freq.html")
    """

    def __init__(self, X, y=None, sample_weight=None):

        # Dataset and optional training labels
        self.data = X
        self.labels = y
        self.weight = sample_weight

        if y is None:
            print('No labels provided. Feature importance based methods are not available.')

        self.base_features = list(self.data.columns)
        self.cat_features = None

        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None

        self.missing_stats = None
        self.unique_stats = None
        self.cardinality_stats = None
        self.corr_matrix = None
        self.correlation_threshold = 0.5
        self.feature_importances = None
        self.cat_var_df = None
        self.removed_features = None
        self.missing_threshold = None
        self.cumulative_importance = None
        self.all_identified = None
        self.n_identified = None
        self.encoded = False
        self.mapper = None
        self.collinear_method = None
        self.tag_df = pd.DataFrame({'predictor': self.base_features})

        # Dictionary to hold removal operations
        self.ops = {}

        self.encoded_correlated = False

    def __repr__(self):
        s = "FeatureSelector(data, labels, weight)"
        return s

    def identify_patterns(self, patterns=None):
        """
        Drop the columns by identifying patterns in their name
        :param patterns: str
            the pattern to look for e.g. emb_ or bel_
        :return:
         self : object
            Nothing but attributes
        """
        if patterns is None:
            patterns = ['_Prop']
        to_drop = self.data.columns[self.data.columns.str.contains('|'.join(patterns))].tolist()
        self.ops['pattern'] = to_drop

        self.tag_df['pattern'] = 1
        self.tag_df['pattern'] = np.where(self.tag_df['predictor'].isin(to_drop), 0, 1)

        print('%d features with the pattern(s). \n' % len(self.ops['pattern']))

    def identify_missing(self, missing_threshold=0.1):
        """
        Find the features with a fraction of missing values above `missing_threshold`
        :param missing_threshold: float, between 0 and 1. Default=0.1
        :return:
         self : object
            Nothing but attributes
        """

        self.missing_threshold = missing_threshold

        # Calculate the fraction of missing in each column
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(
            columns={'index': 'feature', 0: 'missing_fraction'}
        )

        # Sort with highest number of missing values on top
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending=False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index(). \
            rename(columns={
            'index': 'feature',
            0: 'missing_fraction'})
        to_drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.ops['missing'] = to_drop
        # tagging
        self.tag_df['missing'] = 1
        self.tag_df['missing'] = np.where(self.tag_df['predictor'].isin(to_drop), 0, 1)

        print('%d features with greater than %0.2f missing values.\n' % (
            len(self.ops['missing']), self.missing_threshold))

    def identify_single_unique(self):
        """
        Finds features with only a single unique value. NaNs do not count as a unique value.
        :return:
         self : object
            Nothing but attributes
        """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={'index': 'feature',
                     0: 'nunique'})

        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop
        # tagging
        self.tag_df['single_unique'] = 1
        self.tag_df['single_unique'] = np.where(self.tag_df['predictor'].isin(to_drop), 0, 1)

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))

    def identify_high_cardinality(self, max_card=1000):
        """
        Finds the categorical columns with more than max_card unique values
        (might be relevant for GLM or if the categorical column has as many
        levels than rows in the data set)

        :param max_card: int
            the maximum number of the unique values
        :return:
         self : object
            Nothing but attributes
        """

        col_names = list(self.data)

        char_cols = list(set(list(self.data.columns)) - set(list(self.data.select_dtypes(include=[np.number]))))

        # cat_var_index = [i for i, x in enumerate(self.data.dtypes.tolist()) if
        #                  isinstance(x, pd.CategoricalDtype) or x == 'object']
        # char_cols = [x for i, x in enumerate(col_names) if i in cat_var_index]

        unique_counts = self.data[char_cols].nunique()
        self.cardinality_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        self.cardinality_stats = self.cardinality_stats.sort_values('nunique', ascending=False)

        record_high_cardinality = pd.DataFrame(unique_counts[unique_counts > max_card]).reset_index().rename(
            columns={'index': 'feature', 0: 'nunique'})
        to_drop = list(record_high_cardinality['feature'])
        self.ops['high_cardinality'] = to_drop
        # tagging
        self.tag_df['high_cardinality'] = 1
        self.tag_df['high_cardinality'] = np.where(self.tag_df['predictor'].isin(to_drop), 0, 1)

        print('{0:d} features with a cardinality larger than {1:d}'.format(
            len(self.ops['high_cardinality']), max_card)
        )

    def encode_cat_var(self, df=None, col_excl=None, return_cat=False):
        """
        Categorical encoding (as integer). Automatically detect the non-numerical columns,
        save the index and name of those columns, encode them as integer,
        save the direct and inverse mappers as dictionaries.
        Return the data-set with the encoded columns with a data type either int or pandas categorical.


        :param df: pd.DataFrame
            the dataset
        :param col_excl: list of str, default=None
            the list of columns names not being encoded (e.g. the ID column)
        :param return_cat: Boolean, default=False
            wether or not return pandas categorical (if false --> integer)

        :return:
         df: pd.DataFrame
            the dataframe with encoded columns
        """
        if df is None:
            is_external_data = False
            df = self.data
        else:
            is_external_data = True

        df, cat_var_df, inv_mapper, mapper = cat_var(df, col_excl=col_excl, return_cat=return_cat)

        self.cat_var_df = cat_var_df
        self.mapper = inv_mapper

        if not is_external_data:
            self.encoded = True
            self.data = df

        return df

    def identify_collinear(self, correlation_threshold, encode=False, method='association'):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.

        Using code adapted from:
        https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

        Parameters
        --------

        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying correlation features

        encode : boolean, default = False
            Whether to encode the features before calculating the correlation coefficients

        method: str
            association, spearman or pearson correlation coefficient. If you have categorical variables, use "association"

        """

        self.correlation_threshold = correlation_threshold
        self.encoded_correlated = encode

        # Calculate the correlations between every column
        tic = time.time()
        # Calculate the correlations between every column
        if encode:
            self.data = self.encode_cat_var()
            print('Encoding done, {0:4.0f} min'.format(round((tic - time.time()) / 60)))

        tic_corr = time.time()

        self.collinear_method = method
        if method == 'association':
            features = self.data.columns

            # continuous features
            con_features = set(features).intersection(set(list(self.data.select_dtypes(include=[np.number]))))

            # nominal features
            nom_features = set(features) - set(con_features)

            self.corr_matrix = compute_associations(self.data,
                                                    nominal_columns=nom_features,
                                                    mark_columns=True,
                                                    theil_u=True,
                                                    clustering=True,
                                                    bias_correction=True,
                                                    nan_strategy='drop_samples')

            upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if
                       any(upper[column].abs() > correlation_threshold)]


        elif method == 'spearman':
            self.corr_matrix = self.data.corr(method="spearman").fillna(0)
            # Extract the upper triangle of the correlation matrix
            upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if
                       any(upper[column].abs() > correlation_threshold)]
        elif method == 'pearson':
            # Using both Spearman and Pearson because we might have ordinal and non ordinal respectively.
            corr_matrix = self.data.corr(method='pearson').fillna(0)
            # Extract the upper triangle of the correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if
                       any(upper[column].abs() > correlation_threshold)]
        else:
            raise ValueError('method should be spearman or pearson')

        time_corr = round((time.time() - tic_corr) / 60)
        time_start = round((time.time() - tic) / 60)
        print('Corr matrix done, {0:4.0f} min and {1:4.0f} min from start'.format(time_corr, time_start))

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index=True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop
        # tagging
        self.tag_df['collinear'] = 1
        self.tag_df['collinear'] = np.where(self.tag_df['predictor'].isin(to_drop), 0, 1)

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
            len(self.ops['collinear']), self.correlation_threshold))

    def identify_zero_importance(self, task, eval_metric=None, objective=None,
                                 n_iterations=10, early_stopping=True, zero_as_missing=False):
        """

        Identify the features with zero importance according to a gradient boosting machine.
        The gbm can be trained with early stopping using a utils set to prevent overfitting.
        The feature importances are averaged over `n_iterations` to reduce variance.

        Uses the LightGBM implementation (http://lightgbm.readthedocs.io/en/latest/index.html)

        Parameters
        --------

        eval_metric : string
            Evaluation metric to use for the gradient boosting machine for early stopping. Must be
            provided if `early_stopping` is True

        objective : string
            define the LGBM specific objective if any

        task : string
            The machine learning task, either 'classification' or 'regression'

        n_iterations : int, default = 10
            Number of iterations to train the gradient boosting machine

        early_stopping : boolean, default = True
            Whether or not to use early stopping with a utils set when training

        zero_as_missing : Bool, default = False
           consider or not zero as missing


        Notes
        --------

        - Features are one-hot encoded to handle the categorical variables before training.
        - The gbm is not optimized for any particular task and might need some hyperparameter tuning
        - Feature importances, including zero importance features,
          can change across runs, using Shapley values.

        """

        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. 
            Examples include "auc" for classification or "l2" for regression.""")

        if self.labels is None:
            raise ValueError("No training labels provided.")

        if self.encoded:
            features = self.data
        else:
            features = self.encode_cat_var()

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1,))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        print('Training Gradient Boosting Model\n')
        progress_bar = trange(n_iterations)
        # Iterate through each fold
        for _ in progress_bar:
            progress_bar.set_description('Iteration nb: {0:<3}'.format(_))

            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1,
                                           zero_as_missing=zero_as_missing)

            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1,
                                          zero_as_missing=zero_as_missing)  # , objective='gamma')

            else:
                raise ValueError('Task must be either "classification" or "regression"')

            if objective is not None:
                model.set_params(**{'objective': objective})

            # If training using early stopping need a utils set
            if early_stopping:
                train_idx = random.sample(range(features.shape[0]), k=int(features.shape[0] * .85))
                mask = np.zeros(features.shape[0], dtype=bool)
                mask[train_idx] = True
                weight = self.weight
                train_features = features[mask]
                valid_features = features[~mask]  # features.drop(features.index[train_idx])
                train_labels = labels[mask]
                valid_labels = labels[~mask]
                train_weight = weight[mask]
                valid_weight = weight[~mask]

                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric=eval_metric,
                          eval_set=[(valid_features, valid_labels)],
                          early_stopping_rounds=100, verbose=-1, sample_weight=train_weight,
                          eval_sample_weight=[valid_weight])
                # pimp cool but too slow
                # perm_imp =  permutation_importance(
                # model, valid_features, valid_labels, n_repeats=10, random_state=42, n_jobs=-1
                # )
                # perm_imp = perm_imp.importances_mean

                shap_matrix = model.predict(valid_features, pred_contrib=True)
                shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)

                # Clean up memory
                del train_features, train_labels, valid_features, valid_labels

            else:
                model.fit(features, labels, sample_weight=self.weight)
                # perm_imp =  permutation_importance(
                # model, features, labels, n_repeats=10, random_state=42, n_jobs=-1
                # )
                # perm_imp = perm_imp.importances_mean

                shap_matrix = model.predict(features, pred_contrib=True)
                shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)

            # Record the feature importances
            feature_importance_values += shap_imp / n_iterations  # model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names,
                                            'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values(
            'importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances[
            'importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop
        # tagging
        self.tag_df['zero_importance'] = 1
        self.tag_df['zero_importance'] = np.where(self.tag_df['predictor'].isin(to_drop), 0, 1)

        print('\n%d features with zero importance after encoding.\n' % len(self.ops['zero_importance']))

    def identify_low_importance(self, cumulative_importance):
        """
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance from the gradient boosting machine. As an example, if cumulative
        importance is set to 0.95, this will retain only the most important features needed to
        reach 95% of the total feature importance. The identified features are those not needed.

        Parameters
        --------
        cumulative_importance : float between 0 and 1
            The fraction of cumulative importance to account for

        """

        self.cumulative_importance = cumulative_importance

        # The feature importances need to be calculated before running
        if self.feature_importances is None:
            raise NotImplementedError("""Feature importances have not yet been determined. 
                                         Call the `identify_zero_importance` method first.""")

        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[
            self.feature_importances['cumulative_importance'] > cumulative_importance]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop

        # tagging
        self.tag_df['low_importance'] = 1
        self.tag_df['low_importance'] = np.where(self.tag_df['predictor'].isin(to_drop), 0, 1)

        print('%d features required for cumulative importance of %0.2f after encoding.' % (
            len(self.feature_importances) -
            len(self.record_low_importance), self.cumulative_importance))
        print('%d features do not contribute to cumulative '
              'importance of %0.2f.\n' % (len(self.ops['low_importance']), self.cumulative_importance))

    def identify_all(self, selection_params):
        """
        Use all five of the methods to identify features to remove.

        Parameters
        --------

        selection_params : dict Parameters to use in the five feature selection methhods.
        Params must contain the keys
        ['patterns', 'missing_threshold', 'max_card','correlation_threshold', 'eval_metric', 'task',
        'cumulative_importance']
        identify_all = {'patterns': 'emb_',
                        'missing_threshold': 0.1,
                        'max_card':100,
                        'correlation_threshold': 0.5,
                        'eval_metric':'l2',
                        'task': 'regression',
                        'cumulative_importance': 0.95}
        """

        # Check for all required parameters
        list_of_params = ['patterns', 'missing_threshold', 'max_card', 'correlation_threshold',
                          'eval_metric', 'task', 'cumulative_importance']

        progress_bar = tqdm(list_of_params)
        for param in progress_bar:
            progress_bar.set_description('Processing method: {0:<23}'.format(param))
            if param not in selection_params.keys():
                raise ValueError('%s is a required parameter for this method.' % param)

        # Implement each of the five methods
        self.identify_patterns(selection_params['patterns'])
        self.identify_missing(selection_params['missing_threshold'])
        self.identify_single_unique()
        self.identify_high_cardinality(selection_params['max_card'])
        self.identify_collinear(selection_params['correlation_threshold'])
        self.identify_zero_importance(task=selection_params['task'], eval_metric=selection_params['eval_metric'])
        self.identify_low_importance(selection_params['cumulative_importance'])

        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)

        print('%d total features out of %d identified for removal.\n' % (self.n_identified, self.data.shape[1]))

    def check_removal(self):
        """Check the identified features before removal. Returns a list of the unique features identified."""

        self.all_identified = set(list(chain(*list(self.ops.values()))))
        print('Total of %d features identified for removal' % len(self.all_identified))

        return list(self.all_identified)

    def remove(self, methods):
        """
        Remove the features from the data according to the specified methods.

        Parameters
        --------
            methods : 'all' or list of methods
                If methods == 'all', any methods that have identified features will be used
                Otherwise, only the specified methods will be used.
                Can be one of ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
        Return
        --------
            data : dataframe
                Dataframe with identified features removed


        Notes
        --------
            - If feature importances are used, the one-hot encoded columns will be
              added to the data (and then may be removed)
            - Check the features that will be removed before transforming data!

        """

        features_to_drop = []

        if methods == 'all':

            # Need to use one-hot encoded data as well
            data = self.data

            print('{} methods have been run\n'.format(list(self.ops.keys())))

            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))

        else:
            data = self.data
            # Iterate through the specified methods
            for method in methods:

                # Check to make sure the method has been run
                if method not in self.ops.keys():
                    raise NotImplementedError('%s method has not been run' % method)

                # Append the features identified for removal
                else:
                    features_to_drop.append(self.ops[method])

            # Find the unique features to drop
            features_to_drop = set(list(chain(*features_to_drop)))

        features_to_drop = list(features_to_drop)

        # Remove the features and return the data
        data = data.drop(columns=features_to_drop)
        self.removed_features = features_to_drop

        return data

    def plot_missing(self):
        """Histogram of missing fraction in each feature"""
        if self.record_missing is None:
            raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")

        # self.reset_plot()
        set_my_plt_style()

        # Histogram of missing values
        # plt.style.use('seaborn-white')
        plt.figure(figsize=(7, 5))
        plt.hist(self.missing_stats['missing_fraction'], bins=np.linspace(0, 1, 21))
        # , edgecolor='k', color='red', linewidth=1.5)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel('Missing Fraction', size=14)
        plt.ylabel('Count of Features', size=14)
        plt.title("Fraction of Missing Values Histogram", size=16)
        plt.yscale("log", nonpositive='clip')
        return plt.show()

    def plot_unique(self):
        """Histogram of number of unique values in each feature"""
        if self.record_single_unique is None:
            raise NotImplementedError('Unique values have not been calculated. '
                                      'Run `identify_single_unique`')

        # self.reset_plot()
        set_my_plt_style()

        # Histogram of number of unique values
        lower_bound = max([self.unique_stats.min()[0], 1])
        logbins = np.logspace(np.log10(lower_bound), np.log10(self.unique_stats.max()[0]), 21)
        self.unique_stats.plot.hist(figsize=(7, 5), bins=logbins)

        plt.ylabel('Frequency', size=14)
        plt.xlabel('Unique Values', size=14)
        plt.title('Number of Unique Values Histogram', size=16)
        plt.xscale("log", nonpositive='clip')
        plt.yscale("log", nonpositive='clip')

    def plot_cardinality(self):
        """Histogram of number of unique values in each feature"""
        if self.cardinality_stats is None:
            raise NotImplementedError('Cardinality values have not been '
                                      'calculated. Run `identify_cardinality`')

        # self.reset_plot()
        set_my_plt_style()

        # Histogram of number of unique values
        logbins = np.logspace(np.log10(self.cardinality_stats.min()[0]),
                              np.log10(self.cardinality_stats.max()[0]), 21)
        self.cardinality_stats.plot.hist(figsize=(7, 5), bins=logbins)

        plt.ylabel('Frequency', size=14)
        plt.xlabel('Cardinality', size=14)
        plt.title('Cardinality of categorical predictors', size=16)
        plt.xscale("log", nonpositive='clip')
        # plt.yscale("log", nonposy='clip')

    def plot_collinear(self, plot_all=False, size=1000):
        """
        Heatmap of the correlation values. If plot_all = True plots all the correlations otherwise
        plots only those features that have a correlation above the threshold

        Notes
        --------
            - Not all of the plotted correlations are above the threshold because this plots
            all the variables that have been idenfitied as having even one correlation above the threshold
            - The features on the x-axis are those that will be removed. The features on the y-axis
            are the correlated features with those on the x-axis

        Code adapted from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
        """

        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been identified. '
                                      'Run `identify_collinear`.')

        if plot_all:
            corr_matrix_plot = self.corr_matrix
            subtitle_str = 'All Correlations'

        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])),
                                                    list(set(self.record_collinear['drop_feature']))]
            subtitle_str = "Correlations Above Threshold"

            d = sch.distance.pdist(corr_matrix_plot)
            L = sch.linkage(d, method='ward')
            ind = sch.fcluster(L, 0.5 * np.nanmax(d), 'distance')
            columns = [corr_matrix_plot.columns.tolist()[i] for i in list((np.argsort(ind)))]
            corr_matrix_plot = corr_matrix_plot.reindex(columns, axis=1)
            corr_matrix_plot = corr_matrix_plot.reindex(columns, axis=0)
        # interactive plot of the correlation matrix
        heatmap = hv.HeatMap((corr_matrix_plot.columns, corr_matrix_plot.index, corr_matrix_plot)).redim.range(
                                 z=(-1, 1))

        heatmap.opts(tools=['tap', 'hover'], height=size, width=size + 50, toolbar='left', colorbar=True,
                cmap=Curl_5_r.mpl_colormap, fontsize={'title': 12, 'ticks': 12, 'minor_ticks': 12}, xrotation=90,
                invert_xaxis=False, invert_yaxis=True,  # title=title_str,
                xlabel='', ylabel=''
                     )
        if self.collinear_method == 'association':
            title_str = "**Continuous (con) and Categorical (nom) Associations **"
            sub_title_str = "*Categorical(nom): uncertainty coefficient & correlation ratio from 0 to 1. The uncertainty " \
                        "coefficient is assymmetrical, (approximating how much the elements on the " \
                        "left PROVIDE INFORMATION on elements in the row). Continuous(con): symmetrical numerical " \
                        "correlations (Pearson's) from -1 to 1*"
        else:
            title_str = "**Correlations (continuous variables only) **"
            subtitle_str = "Use `method='association'` if there are categorical variables. "  + subtitle_str 

            
        panel_layout = pn.Column(
                pn.pane.Markdown(title_str, align="start"),  # bold
                pn.pane.Markdown(sub_title_str, align="start"),  # italic
                heatmap, background='#ebebeb'
        )

        return panel_layout


    def plot_feature_importances(self, plot_n=15, threshold=None, figsize=None):
        """
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold`
        cumulative importance.

        Parameters
        --------

        plot_n : int, default = 15
            Number of most important features to plot. Defaults to 15 or the maximum
            number of features whichever is smaller

        threshold : float, between 0 and 1 default = None
            Threshold for printing information about cumulative importances

        """

        if self.record_zero_importance is None:
            raise NotImplementedError('Feature importances have not been determined. '
                                      'Run `idenfity_zero_importance`')

        # Need to adjust number of features if greater than the features in the data
        if plot_n > self.feature_importances.shape[0]:
            plot_n = self.feature_importances.shape[0] - 1

        reset_plot()

        if figsize is not None:
            plot_width = figsize[0]
            plot_height = figsize[1]
        else:
            plot_height = 500
            plot_width = 500
            if plot_n <= 20:
                plot_height = 500
                plot_width = 500
            elif (plot_n > 20) and (plot_n < 30):
                plot_height = 800
                plot_width = 800
            elif (plot_n >= 30) and (plot_n < 50):
                plot_height = 1000
                plot_width = 1000
            elif (plot_n >= 50) and (plot_n < 100):
                plot_height = 1500
                plot_width = 1500

        bars = hv.Bars(self.feature_importances.iloc[:plot_n], kdims='feature',
                       vdims='normalized_importance').opts(color='#2961ba', invert_axes=True,
                                                           invert_yaxis=True, width=400,
                                                           height=plot_height, tools=['hover'],
                                                           fontsize={'title': 20, 'ticks': 10}, xlabel='')

        curve = hv.Scatter(self.feature_importances.iloc[:plot_n], kdims='feature',
                           vdims='cumulative_importance').opts(color='#2961ba', width=plot_width, size=8,
                                                               height=plot_height, tools=['hover'],
                                                               fontsize={'title': 20, 'ticks': 10}, xrotation=45,
                                                               xlabel='')
        plot = bars + curve

        if threshold:
            importance_index = np.min(np.where(self.feature_importances['cumulative_importance'] > threshold))
            curve = curve * hv.VLine(importance_index + 1).opts(color='darkred', line_dash='dashed')
            plot = bars + curve
            plot = plot.opts(title="Importance from the GBM dummy  model")
            print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
        return plot.opts(shared_axes=False, axiswise=True)
