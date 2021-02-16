import pytest
import numpy as np
import pandas as pd
from arfs.featselect import FeatureSelector
from arfs.utils import _generated_corr_dataset_regr, _generated_corr_dataset_classification


class TestFeatSelectPattern:
    """
    Test suite for FeatureSelector, pattern identification
    """

    def test_identify_patterns_for_classification(self):
        # not task dependent (same for clf and regr)
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_patterns(patterns='emb_')
        message = "Expected: {0}, Actual: {1}".format('emb_dummy', fs.ops['pattern'])
        assert fs.ops['pattern'] == ['emb_dummy'], message


class TestFeatSelectMissing:
    """
    Test suite for FeatureSelector, missing values
    """

    def test_identify_missing_for_classification(self):
        # not task dependent (same for clf and regr)
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_missing(missing_threshold=0.01)
        message = "Expected: {0}, Actual: {1}".format('var12', fs.ops['missing'])
        assert fs.ops['missing'] == ['var12'], message


class TestFeatSelectZeroVariance:
    """
    Test suite for FeatureSelector, missing values
    """

    def test_identify_single_unique_classification(self):
        # not task dependent (same for clf and regr)
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_single_unique()
        message = "Expected: {0}, Actual: {1}".format('var10', fs.ops['single_unique'])
        assert fs.ops['single_unique'] == ['var10'], message


class TestFeatSelectHighCardinality:
    """
    Test suite for FeatureSelector, high cardinality
    """

    def test_identify_high_cardinality_classification(self):
        # not task dependent (same for clf and regr)
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_high_cardinality(max_card=5)
        message = "Expected: {0}, Actual: {1}".format('emb_dummy', fs.ops['high_cardinality'])
        assert fs.ops['high_cardinality'] == ['emb_dummy'], message


class TestFeatSelectCollinearity:
    """
    test suite for FeatureSelector, high cardinality
    """

    def test_identify_collinear_spearman_no_encoding(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=False, method='spearman')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var4', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var4', 'var12'], message

    def test_identify_collinear_pearson_no_encoding(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=False, method='pearson')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var12'], message

    def test_identify_collinear_spearman_with_encoding(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=True, method='spearman')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var4', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var4', 'var12'], message

    def test_identify_collinear_pearson_with_encoding(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_collinear(correlation_threshold=0.5, encode=True, method='pearson')
        message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var12'], fs.ops['collinear'])
        assert fs.ops['collinear'] == ['var2', 'var3', 'var12'], message


class TestFeatSelectZeroImportance:
    """
    test suite for FeatureSelector, high cardinality
    """

    def test_identify_zero_importance_for_regression_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_regression_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective='l2', n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_regression_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective=None, n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_regression_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_regr(size=10)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric='l2', objective='l2', n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    def test_identify_zero_importance_for_regression_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=2,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    def test_identify_zero_importance_for_regression_without_early_stopping_no_objective(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='regression', n_iterations=2,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    def test_identify_zero_importance_for_classification_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_classification_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_classification_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, objective=None, n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    @pytest.mark.xfail
    def test_identify_zero_importance_for_classification_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric='auc', objective='cross-entropy', n_iterations=2,
                                    early_stopping=True)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    def test_identify_zero_importance_for_classification_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='classification', objective='binary', n_iterations=2,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message

    def test_identify_zero_importance_for_classification_without_early_stopping_no_objective(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='classification', n_iterations=2,
                                    early_stopping=False)
        message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
        assert 'var10' in fs.ops['zero_importance'], message


class TestFeatSelectLowImportance:
    """
    test suite for FeatureSelector, high cardinality
    """

    def test_identify_low_importance_for_regression_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=2,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_regression_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective='l2', n_iterations=2,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_regression_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric=None, objective=None, n_iterations=2,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_regression_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric='l2', objective='l2', n_iterations=2,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_regression_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='regression', objective='l2', n_iterations=2, early_stopping=False)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_regression_without_early_stopping_no_objective(self):
        X, y, w = _generated_corr_dataset_regr(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='regression', n_iterations=2, early_stopping=False)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_classification_with_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=2, early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_classification_with_early_stopping_no_eval_metric(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, n_iterations=2,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_classification_with_early_stopping_no_eval_metric_no_objective(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='classification', eval_metric=None, objective=None, n_iterations=2,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    @pytest.mark.xfail
    def test_identify_low_importance_for_classification_with_early_stopping_wrong_task(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        # Xfail: expected to fail because the eval metric is not provided
        fs.identify_zero_importance(task='regression', eval_metric='auc', objective='cross-entropy', n_iterations=2,
                                    early_stopping=True)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message

    def test_identify_low_importance_for_classification_without_early_stopping(self):
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = FeatureSelector(X=X, y=y, sample_weight=w)
        fs.identify_zero_importance(task='classification', n_iterations=2, early_stopping=False)
        cum_imp_threshold = 0.95
        fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
        expected = 1
        message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
        assert len(fs.ops['low_importance']) >= expected, message


# class TestFeatSelectAllMethods:
#     def test_identify_all(self):
#         X, y, w = _generated_corr_dataset_regr()
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#
#         selection_params_dic = {'patterns': 'emb_',
#                                 'missing_threshold': 0.1,
#                                 'max_card': 100,
#                                 'correlation_threshold': 0.5,
#                                 'eval_metric': 'l2',
#                                 'task': 'regression',
#                                 'cumulative_importance': 0.95}
#
#         fs.identify_all(selection_params=selection_params_dic)
#
#         assert len(fs.all_identified) > 4, "there should be at least 4 useless predictors"
#
#     def test_remove_all_identified(self):
#         X, y, w = _generated_corr_dataset_regr()
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#
#         selection_params_dic = {'patterns': 'emb_',
#                                 'missing_threshold': 0.1,
#                                 'max_card': 100,
#                                 'correlation_threshold': 0.5,
#                                 'eval_metric': 'l2',
#                                 'task': 'regression',
#                                 'cumulative_importance': 0.99}
#
#         fs.identify_all(selection_params=selection_params_dic)
#         X_new = fs.remove(methods='all')
#
#         assert sorted(list(X_new.columns)) == sorted(set(X.columns) - set(fs.all_identified))
#
#     def test_remove_identified_after_some_methods(self):
#         X, y, w = _generated_corr_dataset_regr()
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#
#         selection_params_dic = {'patterns': 'emb_',
#                                 'missing_threshold': 0.1,
#                                 'max_card': 100,
#                                 'correlation_threshold': 0.5,
#                                 'eval_metric': 'l2',
#                                 'task': 'regression',
#                                 'cumulative_importance': 0.99}
#
#         fs.identify_all(selection_params=selection_params_dic)
#         X_new = fs.remove(methods=['pattern', 'missing', 'single_unique', 'high_cardinality'])
#
#         assert len(list(X_new.columns)) == 10, "Only 4 columns should be ruled out"
