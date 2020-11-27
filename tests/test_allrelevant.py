import pytest
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from arfs.allrelevant import Leshy, BoostAGroota, GrootCV


def _generated_corr_dataset_regr():
    # weights
    size = 1000
    w = np.random.beta(a=1, b=0.5, size=size)
    # fixing the seed and the target
    np.random.seed(42)
    sigma = 0.2
    y = np.random.gamma(1, 4, size)
    z = y + np.random.gamma(2, 2, size) - 2 * np.random.gamma(1, 1, size)
    X = np.zeros((size, 11))

    # 5 relevant features, with positive and negative correlation to the target
    # and non-linearity
    X[:, 0] = z
    X[:, 1] = y * np.random.gamma(5, .5, size) + np.random.normal(2, sigma, size)
    X[:, 2] = -y * z + np.random.normal(0, sigma, size)
    X[:, 3] = y ** (2 + np.random.normal(0, sigma / 2, size))
    X[:, 4] = np.sqrt(y) + np.random.gamma(1, .2, size)
    X[:, 5] = X[:, 3] * X[:, 0] / X[:, 1]

    # 5 irrelevant features (with one having high cardinality)
    X[:, 6] = np.random.gamma(1, .2, size)
    X[:, 7] = np.random.binomial(1, 0.3, size)
    X[:, 8] = np.random.normal(0, 1, size)
    X[:, 9] = np.random.gamma(1, 2, size)
    X[:, 10] = np.arange(start=0, stop=size, step=1)

    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(11)]
    X = pd.DataFrame(X)
    X.columns = column_names

    return X, y, w


def _generated_corr_dataset_classification():
    # weights
    size = 1000
    w = np.random.beta(a=1, b=0.5, size=size)
    # fixing the seed and the target
    np.random.seed(42)
    y = np.random.binomial(1, 0.5, size)
    X = np.zeros((size, 11))

    z = y - np.random.binomial(1, 0.1, size) + np.random.binomial(1, 0.1, size)
    z[z == -1] = 0
    z[z == 2] = 1

    # 5 relevant features, with positive and negative correlation to the target
    # and non-linearity
    X[:, 0] = z
    X[:, 1] = y * np.abs(np.random.normal(0, .1, size)) + np.random.normal(0, 0.1, size)
    X[:, 2] = -y + np.random.normal(0, 1, size)
    X[:, 3] = y ** 2 + np.random.normal(0, 1, size)
    X[:, 4] = X[:, 3] * X[:, 2]  # np.sqrt(y) + np.random.binomial(2, 0.1, size)

    # 6 irrelevant features (with one having high cardinality)
    X[:, 5] = np.random.normal(0, 1, size)
    X[:, 6] = np.random.poisson(1, size)
    X[:, 7] = np.random.binomial(1, 0.3, size)
    X[:, 8] = np.random.normal(0, 1, size)
    X[:, 9] = np.random.poisson(1, size)
    X[:, 10] = np.arange(start=0, stop=size, step=1)

    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(11)]
    X = pd.DataFrame(X)
    X.columns = column_names

    return X, y, w


def _plot_y_vs_X(X, y):
    """
    Plot target vs relevant and non-relevant predictors

    :param X: pd.DataFrame
        the pd DF of the predictors
    :param y: np.array
        the target
    :return: g1 and g2, matplotlib objects
        the univariate plots y vs pred_i
    """
    data = X.copy()
    data['target'] = y
    x_vars = ["var0", "var1", "var2", "var3", "var4", "var5"]
    y_vars = ["target"]
    g1 = sns.PairGrid(data, x_vars=x_vars, y_vars=y_vars)
    g1.map(plt.scatter, alpha=0.1)

    x_vars = ["var6", "var7", "var8", "var9", "var10"]
    y_vars = ["target"]
    g2 = sns.PairGrid(data, x_vars=x_vars, y_vars=y_vars)
    g2.map(plt.scatter, alpha=0.1)
    return g1, g2


class TestLeshy:
    """
    Test suite for all-relevant FS boruta-like method: Leshy
    """

    def test_boruta_rfc_vs_boruta_lightgbm_implementation_of_rndforest_clf(self):
        # sklearn random forest implementation
        X, y, w = _generated_corr_dataset_classification()
        rfc = RandomForestClassifier(max_features='sqrt', max_samples=0.632, n_estimators=100)
        bt = BorutaPy(rfc)
        bt.fit(X.values, y)
        sklearn_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        n_feat = X.shape[1]
        rfc = lgb.LGBMClassifier(verbose=-1, force_col_wise=True,
                                 n_estimators=100, bagging_fraction=0.632,
                                 feature_fraction=np.sqrt(n_feat) / n_feat,
                                 boosting_type="rf", bagging_freq=1)
        bt = BorutaPy(rfc)
        bt.fit(X.values, y)
        lightgbm_rfc_list = sorted(list(X.columns[bt.support_]))

        assert bool(set(sklearn_rfc_list) & set(lightgbm_rfc_list)), "expect non-empty intersection"

    def test_boruta_rfr_vs_boruta_lightgbm_implementation_of_rndforest_rgr(self):
        # sklearn random forest implementation
        X, y, w = _generated_corr_dataset_regr()
        rfr = RandomForestRegressor(max_features=0.3, max_samples=0.632, n_estimators=100)
        bt = BorutaPy(rfr)
        bt.fit(X.values, y)
        sklearn_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        n_feat = X.shape[1]
        rfc = lgb.LGBMRegressor(verbose=-1, force_col_wise=True,
                                n_estimators=100, bagging_fraction=0.632,
                                feature_fraction=n_feat / (n_feat * 3),
                                boosting_type="rf", bagging_freq=1)
        bt = BorutaPy(rfc)
        bt.fit(X.values, y)
        lightgbm_rfc_list = sorted(list(X.columns[bt.support_]))

        assert bool(set(sklearn_rfc_list) & set(lightgbm_rfc_list)), "expect non-empty intersection"

    def test_borutaPy_vs_leshy_with_rfc_and_native_feature_importance(self):
        # sklearn random forest implementation
        X, y, w = _generated_corr_dataset_classification()
        rfc = RandomForestClassifier(max_features='sqrt', max_samples=0.632, n_estimators=100)
        bt = BorutaPy(rfc)
        bt.fit(X.values, y)
        borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        arfs = Leshy(rfc, verbose=0, max_iter=10, random_state=42, importance='native')
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.support_names_)

        assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"

    def test_borutaPy_vs_leshy_with_rfr_and_native_feature_importance(self):
        # sklearn random forest implementation
        X, y, w = _generated_corr_dataset_regr()
        rfr = RandomForestRegressor(max_features=0.3, max_samples=0.632, n_estimators=100)
        bt = BorutaPy(rfr)
        bt.fit(X.values, y)
        borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        arfs = Leshy(rfr, verbose=0, max_iter=10, random_state=42, importance='native')
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.support_names_)

        assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"

    def test_borutaPy_vs_leshy_with_rfc_and_shap_feature_importance(self):
        # sklearn random forest implementation
        X, y, w = _generated_corr_dataset_classification()
        rfc = RandomForestClassifier(max_features='sqrt', max_samples=0.632, n_estimators=100)
        bt = BorutaPy(rfc)
        bt.fit(X.values, y)
        borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        n_feat = X.shape[1]
        model = lgb.LGBMClassifier(verbose=-1, force_col_wise=True, n_estimators=100, bagging_fraction=0.632,
                                   feature_fraction=np.sqrt(n_feat) / n_feat, boosting_type="rf", bagging_freq=1)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance='shap')
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.support_names_)

        assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"

    def test_borutaPy_vs_leshy_with_rfr_and_shap_feature_importance(self):
        # sklearn random forest implementation
        X, y, w = _generated_corr_dataset_regr()
        rfr = RandomForestRegressor(max_features=0.3, max_samples=0.632, n_estimators=100)
        bt = BorutaPy(rfr)
        bt.fit(X.values, y)
        borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        n_feat = X.shape[1]
        model = lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=100, bagging_fraction=0.632,
                                  feature_fraction=n_feat / (3*n_feat), boosting_type="rf", bagging_freq=1)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance='shap')
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.support_names_)

        assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"

    def test_leshy_clf_with_lgb_and_shap_feature_importance_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4']

        X, y, w = _generated_corr_dataset_classification()
        model = lgb.LGBMClassifier(verbose=-1, force_col_wise=True, n_estimators=100)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance='shap')
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.support_names_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"

    def test_leshy_regr_with_lgb_and_shap_feature_importance_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5']

        X, y, w = _generated_corr_dataset_classification()
        model = lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=100)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance='shap')
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.support_names_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"


class TestBoostAGroota:
    """
    Test suite for all-relevant FS boruta-like method: Leshy
    """

    def test_boostagroota_clf_with_lgb_and_shap_feature_importance_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4']

        X, y, w = _generated_corr_dataset_classification()
        model = lgb.LGBMClassifier(verbose=-1, force_col_wise=True, n_estimators=100)
        arfs = BoostAGroota(est=model, cutoff=1, iters=10, max_rounds=10, delta=0.1, silent=False, imp='shap')
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.keep_vars_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"

    def test_boostagroota_clf_with_lgb_and_pimp_feature_importance_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4']

        X, y, w = _generated_corr_dataset_classification()
        model = lgb.LGBMClassifier(verbose=-1, force_col_wise=True, n_estimators=100)
        arfs = BoostAGroota(est=model, cutoff=1, iters=10, max_rounds=10, delta=0.1, silent=False, imp='pimp')
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.keep_vars_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"

    def test_boostagroota_rgr_with_lgb_and_shap_feature_importance_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5']

        X, y, w = _generated_corr_dataset_regr()
        model = lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=100)
        arfs = BoostAGroota(est=model, cutoff=1, iters=10, max_rounds=10, delta=0.1, silent=False, imp='shap')
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.keep_vars_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"

    def test_boostagroota_regr_with_lgb_and_pimp_feature_importance_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5']

        X, y, w = _generated_corr_dataset_regr()
        model = lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=100)
        arfs = BoostAGroota(est=model, cutoff=1, iters=10, max_rounds=10, delta=0.1, silent=False, imp='pimp')
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.keep_vars_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"


class TestGrootCV:
    """
    Test suite for all-relevant FS boruta-like method: Leshy
    """

    def test_grootcv_classification_with_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4']

        X, y, w = _generated_corr_dataset_classification()
        arfs = GrootCV(objective='binary', cutoff=1, n_folds=5, n_iter=5, silent=False)
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.keep_vars_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"

    def test_grootcv_regression_with_and_sample_weight(self):
        baseline_list = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5']

        X, y, w = _generated_corr_dataset_regr()
        arfs = GrootCV(objective='l2', cutoff=1, n_folds=5, n_iter=5, silent=False)
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.keep_vars_)

        assert bool(set(baseline_list) & set(leshy_list)), "expect non-empty intersection"
