import pytest
import numpy as np
import pandas as pd
from arfs.feature_selection import (
    MissingValueThreshold,
    UniqueValuesThreshold,
    CardinalityThreshold,
    CollinearityThreshold,
)
from arfs.utils import (
    _generated_corr_dataset_regr,
    _generated_corr_dataset_classification,
)


class TestFeatSelectMissing:
    """
    Test suite for FeatureSelector, missing values
    """

    def test_identify_missing_for_classification(self):
        # not task dependent (same for clf and regr)
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = MissingValueThreshold(threshold=0.01)
        fs.fit(X)
        message = "Expected: {0}, Actual: {1}".format(
            "var12", fs.not_selected_features_
        )
        assert fs.not_selected_features_ == ["var12"], message


class TestFeatSelectZeroVariance:
    """
    Test suite for FeatureSelector, missing values
    """

    def test_identify_single_unique_classification(self):
        # not task dependent (same for clf and regr)
        X, y, w = _generated_corr_dataset_classification(size=10)
        fs = UniqueValuesThreshold(threshold=2)
        fs.fit(X)
        message = "Expected: {0}, Actual: {1}".format(
            "var10", fs.not_selected_features_
        )
        assert fs.not_selected_features_ == ["var10"], message


class TestFeatSelectHighCardinality:
    """
    Test suite for FeatureSelector, high cardinality
    """

    def test_identify_high_cardinality_classification(self):
        # not task dependent (same for clf and regr)
        X, y, w = _generated_corr_dataset_classification(size=100)
        fs = CardinalityThreshold(threshold=5)
        fs.fit(X)
        expected = sorted(["dummy", "nice_guys"])
        actual = sorted(list(fs.not_selected_features_))
        message = "Expected: {0}, Actual: {1}".format(expected, actual)
        assert actual == expected, message


# class TestFeatSelectCollinearity:
#     """
#     test suite for FeatureSelector, high cardinality
#     """

#     def test_identify_collinear_spearman_no_encoding(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_collinear(correlation_threshold=0.5, encode=False, method='spearman')
#         message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var4', 'var12'], fs.ops['collinear'])
#         assert fs.ops['collinear'] == ['var2', 'var3', 'var4', 'var12'], message

#     def test_identify_collinear_pearson_no_encoding(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_collinear(correlation_threshold=0.5, encode=False, method='pearson')
#         message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var12'], fs.ops['collinear'])
#         assert fs.ops['collinear'] == ['var2', 'var3', 'var12'], message

#     def test_identify_collinear_spearman_with_encoding(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_collinear(correlation_threshold=0.5, encode=True, method='spearman')
#         message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var4', 'var12'], fs.ops['collinear'])
#         assert fs.ops['collinear'] == ['var2', 'var3', 'var4', 'var12'], message

#     def test_identify_collinear_pearson_with_encoding(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_collinear(correlation_threshold=0.5, encode=True, method='pearson')
#         message = "Expected: {0}, Actual: {1}".format(['var2', 'var3', 'var12'], fs.ops['collinear'])
#         assert fs.ops['collinear'] == ['var2', 'var3', 'var12'], message


# class TestFeatSelectZeroImportance:
#     """
#     test suite for FeatureSelector, high cardinality
#     """

#     def test_identify_zero_importance_for_regression_with_early_stopping(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     @pytest.mark.xfail
#     def test_identify_zero_importance_for_regression_with_early_stopping_no_eval_metric(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='regression', eval_metric=None, objective='l2', n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     @pytest.mark.xfail
#     def test_identify_zero_importance_for_regression_with_early_stopping_no_eval_metric_no_objective(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='regression', eval_metric=None, objective=None, n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     @pytest.mark.xfail
#     def test_identify_zero_importance_for_regression_with_early_stopping_wrong_task(self):
#         X, y, w = _generated_corr_dataset_regr(size=10)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='classification', eval_metric='l2', objective='l2', n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     def test_identify_zero_importance_for_regression_without_early_stopping(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=2,
#                                     early_stopping=False)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     def test_identify_zero_importance_for_regression_without_early_stopping_no_objective(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='regression', n_iterations=2,
#                                     early_stopping=False)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     def test_identify_zero_importance_for_classification_with_early_stopping(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     @pytest.mark.xfail
#     def test_identify_zero_importance_for_classification_with_early_stopping_no_eval_metric(self):
#         X, y, w = _generated_corr_dataset_classification(size=10)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='classification', eval_metric=None, n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     @pytest.mark.xfail
#     def test_identify_zero_importance_for_classification_with_early_stopping_no_eval_metric_no_objective(self):
#         X, y, w = _generated_corr_dataset_classification(size=10)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='classification', eval_metric=None, objective=None, n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     @pytest.mark.xfail
#     def test_identify_zero_importance_for_classification_with_early_stopping_wrong_task(self):
#         X, y, w = _generated_corr_dataset_classification(size=10)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='regression', eval_metric='auc', objective='cross-entropy', n_iterations=2,
#                                     early_stopping=True)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     def test_identify_zero_importance_for_classification_without_early_stopping(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='classification', objective='binary', n_iterations=2,
#                                     early_stopping=False)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message

#     def test_identify_zero_importance_for_classification_without_early_stopping_no_objective(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='classification', n_iterations=2,
#                                     early_stopping=False)
#         message = "Expected: {0}, Actual: {1}".format(['var10'], fs.ops['zero_importance'])
#         assert 'var10' in fs.ops['zero_importance'], message


# class TestFeatSelectLowImportance:
#     """
#     test suite for FeatureSelector, high cardinality
#     """

#     def test_identify_low_importance_for_regression_with_early_stopping(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='regression', eval_metric='l2', objective='l2', n_iterations=2,
#                                     early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     @pytest.mark.xfail
#     def test_identify_low_importance_for_regression_with_early_stopping_no_eval_metric(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='regression', eval_metric=None, objective='l2', n_iterations=2,
#                                     early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     @pytest.mark.xfail
#     def test_identify_low_importance_for_regression_with_early_stopping_no_eval_metric_no_objective(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='regression', eval_metric=None, objective=None, n_iterations=2,
#                                     early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     @pytest.mark.xfail
#     def test_identify_low_importance_for_regression_with_early_stopping_wrong_task(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='classification', eval_metric='l2', objective='l2', n_iterations=2,
#                                     early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     def test_identify_low_importance_for_regression_without_early_stopping(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='regression', objective='l2', n_iterations=2, early_stopping=False)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     def test_identify_low_importance_for_regression_without_early_stopping_no_objective(self):
#         X, y, w = _generated_corr_dataset_regr(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='regression', n_iterations=2, early_stopping=False)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     def test_identify_low_importance_for_classification_with_early_stopping(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=2, early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     @pytest.mark.xfail
#     def test_identify_low_importance_for_classification_with_early_stopping_no_eval_metric(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='classification', eval_metric=None, n_iterations=2,
#                                     early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     @pytest.mark.xfail
#     def test_identify_low_importance_for_classification_with_early_stopping_no_eval_metric_no_objective(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='classification', eval_metric=None, objective=None, n_iterations=2,
#                                     early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     @pytest.mark.xfail
#     def test_identify_low_importance_for_classification_with_early_stopping_wrong_task(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         # Xfail: expected to fail because the eval metric is not provided
#         fs.identify_zero_importance(task='regression', eval_metric='auc', objective='cross-entropy', n_iterations=2,
#                                     early_stopping=True)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message

#     def test_identify_low_importance_for_classification_without_early_stopping(self):
#         X, y, w = _generated_corr_dataset_classification(size=100)
#         fs = FeatureSelector(X=X, y=y, sample_weight=w)
#         fs.identify_zero_importance(task='classification', n_iterations=2, early_stopping=False)
#         cum_imp_threshold = 0.95
#         fs.identify_low_importance(cumulative_importance=cum_imp_threshold)
#         expected = 1
#         message = "Expected at least one predictor ruled out, Actual: {0}".format(sorted(fs.ops['low_importance']))
#         assert len(fs.ops['low_importance']) >= expected, message
