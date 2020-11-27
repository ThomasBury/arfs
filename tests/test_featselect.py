import pytest
import numpy as np
import pandas as pd
from arfs.featselect import FeatureSelector


def _generated_corr_dataset_regr():
    # weights
    w = np.random.beta(a=1, b=0.5, size=1000)
    # fixing the seed and the target
    np.random.seed(42)
    sigma = 0.2
    y = np.random.normal(1, sigma, 1000)
    z = y - np.random.normal(1, sigma / 5, 1000) + np.random.normal(1, sigma / 5, 1000)
    X = np.zeros((1000, 13))

    # 5 relevant features, with positive and negative correlation to the target
    X[:, 0] = z
    X[:, 1] = y * np.abs(np.random.normal(0, sigma * 2, 1000)) + np.random.normal(0, sigma / 10, 1000)
    X[:, 2] = -y + np.random.normal(0, sigma, 1000)
    X[:, 3] = y ** 2 + np.random.normal(0, sigma, 1000)
    X[:, 4] = np.sqrt(y) + np.random.gamma(1, .2, 1000)

    # 5 irrelevant features
    X[:, 5] = np.random.normal(0, 1, 1000)
    X[:, 6] = np.random.poisson(1, 1000)
    X[:, 7] = np.random.binomial(1, 0.3, 1000)
    X[:, 8] = np.random.normal(0, 1, 1000)
    X[:, 9] = np.random.poisson(1, 1000)
    # zero variance
    X[:, 10] = np.ones(1000)
    # high cardinality
    X[:, 11] = np.arange(start=0, stop=1000, step=1)
    # a lot of missing values
    idx_nan = np.random.choice(1000, 500, replace=False)
    X[:, 12] = y ** 3 + np.abs(np.random.normal(0, 1, 1000))
    X[idx_nan, 12] = np.nan

    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(13)]
    column_names[11] = 'emb_dummy'
    X = pd.DataFrame(X)
    X.columns = column_names
    X['emb_dummy'] = X['emb_dummy'].astype('category')

    return X, y, w


def _generated_corr_dataset_classification():
    # weights
    w = np.random.beta(a=1, b=0.5, size=1000)
    # fixing the seed and the target
    np.random.seed(42)
    y = np.random.binomial(1, 0.5, 1000)
    X = np.zeros((1000, 13))

    z = y - np.random.binomial(1, 0.1, 1000) + np.random.binomial(1, 0.1, 1000)
    z[z == -1] = 0
    z[z == 2] = 1

    # 5 relevant features, with positive and negative correlation to the target
    X[:, 0] = z
    X[:, 1] = y * np.abs(np.random.normal(0, 1, 1000)) + np.random.normal(0, 0.1, 1000)
    X[:, 2] = -y + np.random.normal(0, 1, 1000)
    X[:, 3] = y ** 2 + np.random.normal(0, 1, 1000)
    X[:, 4] = np.sqrt(y) + np.random.binomial(2, 0.1, 1000)

    # 5 irrelevant features
    X[:, 5] = np.random.normal(0, 1, 1000)
    X[:, 6] = np.random.poisson(1, 1000)
    X[:, 7] = np.random.binomial(1, 0.3, 1000)
    X[:, 8] = np.random.normal(0, 1, 1000)
    X[:, 9] = np.random.poisson(1, 1000)
    # zero variance
    X[:, 10] = np.ones(1000)
    # high cardinality
    X[:, 11] = np.arange(start=0, stop=1000, step=1)
    # a lot of missing values
    idx_nan = np.random.choice(1000, 500, replace=False)
    X[:, 12] = y ** 3 + np.abs(np.random.normal(0, 1, 1000))
    X[idx_nan, 12] = np.nan

    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(13)]
    column_names[11] = 'emb_dummy'
    X = pd.DataFrame(X)
    X.columns = column_names
    X['emb_dummy'] = X['emb_dummy'].astype('category')

    return X, y, w


class TestFeatSelectPattern:
    """
    Test suite for FeatureSelector, pattern identification
    """

    def test_identify_patterns_for_classification(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_patterns(patterns='emb_')
        message = "Expected: {0}, Actual: {1}".format('emb_dummy', fs.ops['pattern'])
        assert fs.ops['pattern'] == ['emb_dummy'], message

    def test_identify_patterns_for_regression(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_patterns(patterns='emb_')
        message = "Expected: {0}, Actual: {1}".format('emb_dummy', fs.ops['pattern'])
        assert fs.ops['pattern'] == ['emb_dummy'], message


class TestFeatSelectMissing:
    """
    Test suite for FeatureSelector, missing values
    """

    def test_identify_missing_for_classification(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_missing()
        message = "Expected: {0}, Actual: {1}".format('var12', fs.ops['missing'])
        assert fs.ops['missing'] == ['var12'], message

    def test_identify_missing_for_regression(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_missing()
        message = "Expected: {0}, Actual: {1}".format('var12', fs.ops['missing'])
        assert fs.ops['missing'] == ['var12'], message


class TestFeatSelectZeroVariance:
    """
    Test suite for FeatureSelector, missing values
    """

    def test_identify_single_unique_classification(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_single_unique()
        message = "Expected: {0}, Actual: {1}".format('var10', fs.ops['single_unique'])
        assert fs.ops['single_unique'] == ['var10'], message

    def test_identify_single_unique_regression(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_single_unique()
        message = "Expected: {0}, Actual: {1}".format('var10', fs.ops['single_unique'])
        assert fs.ops['single_unique'] == ['var10'], message


class TestFeatSelectHighCardinality:
    """
    Test suite for FeatureSelector, high cardinality
    """

    def test_identify_high_cardinality_classification(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_high_cardinality(max_card=100)
        message = "Expected: {0}, Actual: {1}".format('emb_dummy', fs.ops['high_cardinality'])
        assert fs.ops['high_cardinality'] == ['emb_dummy'], message

    def test_identify_high_cardinality_regression(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_high_cardinality(max_card=100)
        message = "Expected: {0}, Actual: {1}".format('emb_dummy', fs.ops['high_cardinality'])
        assert fs.ops['high_cardinality'] == ['emb_dummy'], message


class TestFeatSelectCollinearity:
    """
    test suite for FeatureSelector, high cardinality
    """

    def test_identify_collinear_spearman_no_encoding(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=False, method='spearman')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var4', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var4', 'var12'], message

    def test_identify_collinear_pearson_no_encoding(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=False, method='pearson')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var12'], message

    def test_identify_collinear_spearman_with_encoding(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=True, method='spearman')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var4', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var4', 'var12'], message

    def test_identify_collinear_pearson_with_encoding(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=True, method='pearson')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var12'], message


class TestFeatSelectZeroImportance:
    """
    test suite for FeatureSelector, high cardinality
    """

    def test_identify_zero_importance_for_regression_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_regression_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective='l2', n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_regression_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective=None, n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_regression_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric='l2', objective='l2', n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    def test_identify_zero_importance_for_regression_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=5,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    def test_identify_zero_importance_for_regression_without_early_stopping_no_objective(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='regression', n_iterations=5,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    def test_identify_zero_importance_for_classification_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_classification_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_classification_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, objective=None, n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_classification_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric='auc', objective='cross-entropy', n_iterations=5,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    def test_identify_zero_importance_for_classification_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='classification', objective='binary', n_iterations=5,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message

    def test_identify_zero_importance_for_classification_without_early_stopping_no_objective(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='classification', n_iterations=5,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert fs.ops['zero_importance'] == ['var10'], message


class TestFeatSelectLowImportance:
    """
    test suite for FeatureSelector, high cardinality
    """

    def test_identify_low_importance_for_regression_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=5,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_regression_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective='l2', n_iterations=5,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_regression_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective=None, n_iterations=5,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_regression_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric='l2', objective='l2', n_iterations=5,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_regression_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='regression', objective='l2', n_iterations=5, early_stopping=False)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_regression_without_early_stopping_no_objective(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='regression', n_iterations=5, early_stopping=False)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_classification_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=5, early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_classification_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, n_iterations=5,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_classification_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, objective=None, n_iterations=5,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_classification_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric='auc', objective='cross-entropy', n_iterations=5,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_classification_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification()
        fs = FeatureSelector(data=X, labels=y, weight=w)
        fs.identify_zero_importance(task='classification', n_iterations=5, early_stopping=False)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message


class TestFeatSelectAllMethods:
    def test_identify_all(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)

        selection_params_dic = {'patterns': 'emb_',
                                'missing_threshold': 0.1,
                                'max_card': 100,
                                'correlation_threshold': 0.5,
                                'eval_metric': 'l2',
                                'task': 'regression',
                                'cumulative_importance': 0.95}

        fs.identify_all(selection_params=selection_params_dic)

        assert len(fs.all_identified) > 4, "there should be at least 4 useless predictors"

    def test_remove_all_identified(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)

        selection_params_dic = {'patterns': 'emb_',
                                'missing_threshold': 0.1,
                                'max_card': 100,
                                'correlation_threshold': 0.5,
                                'eval_metric': 'l2',
                                'task': 'regression',
                                'cumulative_importance': 0.99}

        fs.identify_all(selection_params=selection_params_dic)
        X_new = fs.remove(methods='all')

        assert sorted(list(X_new.columns)) == sorted(set(X.columns) - set(fs.all_identified))

    def test_remove_identified_after_some_methods(self):
        X, y, w = _generated_corr_dataset_regr()
        fs = FeatureSelector(data=X, labels=y, weight=w)

        selection_params_dic = {'patterns': 'emb_',
                                'missing_threshold': 0.1,
                                'max_card': 100,
                                'correlation_threshold': 0.5,
                                'eval_metric': 'l2',
                                'task': 'regression',
                                'cumulative_importance': 0.99}

        fs.identify_all(selection_params=selection_params_dic)
        X_new = fs.remove(methods=['pattern', 'missing', 'single_unique', 'high_cardinality'])

        assert len(list(X_new.columns)) == 10, "Only 4 columns should be ruled out"
