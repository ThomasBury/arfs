import pytest
import numpy as np
import lightgbm as lgb
from arfs.feature_selection.allrelevant import Leshy, BoostAGroota, GrootCV
from arfs.utils import (
    _generated_corr_dataset_regr,
    _generated_corr_dataset_classification,
)
from arfs.utils import LightForestClassifier, LightForestRegressor


class TestLeshy:
    """
    Test suite for all-relevant FS boruta-like method: Leshy
    """

    def test_borutaPy_vs_leshy_with_rfc_and_native_feature_importance(self):
        # too slow for circleci to run them in a reasonable time
        # takes 2 min on laptop, 1h or more on circleci
        # sklearn random forest implementation
        # X, y, w = _generated_corr_dataset_classification()
        # rfc = RandomForestClassifier(max_features='sqrt', max_samples=0.632, n_estimators=100)
        # bt = BorutaPy(rfc)
        # bt.fit(X.values, y)
        # borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]
        X, y, w = _generated_corr_dataset_classification(size=100)
        n_feat = X.shape[1]
        rfc = LightForestClassifier(n_feat)
        # RandomForestClassifier(max_features='sqrt', max_samples=0.632, n_estimators=100) # --> too slow
        arfs = Leshy(rfc, verbose=0, max_iter=10, random_state=42, importance="native")
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.feature_names_in_[arfs.support_])

        # assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"
        assert bool(
            set(baseline_list) & set(leshy_rfc_list)
        ), "expect non-empty intersection"

    def test_borutaPy_vs_leshy_with_rfr_and_native_feature_importance(self):
        # too slow for circleci to run them in a reasonable time
        # takes 2 min on laptop, 1h or more on circleci
        # # sklearn random forest implementation
        # X, y, w = _generated_corr_dataset_regr()
        # rfr = RandomForestRegressor(max_features=0.3, max_samples=0.632, n_estimators=100)
        # bt = BorutaPy(rfr)
        # bt.fit(X.values, y)
        # borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]
        X, y, w = _generated_corr_dataset_regr(size=100)
        n_feat = X.shape[1]
        rfr = LightForestRegressor(n_feat)
        # rfr = RandomForestRegressor(max_features=0.3, max_samples=0.632, n_estimators=10)
        arfs = Leshy(rfr, verbose=0, max_iter=10, random_state=42, importance="native")
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.feature_names_in_[arfs.support_])

        # assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"
        assert bool(
            set(baseline_list) & set(leshy_rfc_list)
        ), "expect non-empty intersection"

    def test_borutaPy_vs_leshy_with_rfc_and_shap_feature_importance(self):
        # too slow for circleci to run them in a reasonable time
        # takes 2 min on laptop, 1h or more on circleci
        # # sklearn random forest implementation
        # X, y, w = _generated_corr_dataset_classification()
        # rfc = RandomForestClassifier(max_features='sqrt', max_samples=0.632, n_estimators=100)
        # bt = BorutaPy(rfc)
        # bt.fit(X.values, y)
        # borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]
        X, y, w = _generated_corr_dataset_classification(size=100)
        n_feat = X.shape[1]
        model = LightForestClassifier(n_feat)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance="shap")
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.feature_names_in_[arfs.support_])

        # assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"
        assert bool(
            set(baseline_list) & set(leshy_rfc_list)
        ), "expect non-empty intersection"

    def test_borutaPy_vs_leshy_with_rfr_and_shap_feature_importance(self):
        # too slow for circleci to run them in a reasonable time
        # takes 2 min on laptop, 1h or more on circleci
        # # sklearn random forest implementation
        # X, y, w = _generated_corr_dataset_regr()
        # rfr = RandomForestRegressor(max_features=0.3, max_samples=0.632, n_estimators=100)
        # bt = BorutaPy(rfr)
        # bt.fit(X.values, y)
        # borutapy_rfc_list = sorted(list(X.columns[bt.support_]))

        # lightGBM random forest implementation
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]
        X, y, w = _generated_corr_dataset_regr(size=500)
        n_feat = X.shape[1]
        model = LightForestRegressor(n_feat)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance="shap")
        arfs.fit(X, y)
        leshy_rfc_list = sorted(arfs.feature_names_in_[arfs.support_])

        # assert borutapy_rfc_list == leshy_rfc_list, "same selected features are expected"
        assert bool(
            set(baseline_list) & set(leshy_rfc_list)
        ), "expect non-empty intersection"

    def test_leshy_clf_with_lgb_and_shap_feature_importance_and_sample_weight(self):
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]

        X, y, w = _generated_corr_dataset_classification(size=500)
        model = lgb.LGBMClassifier(verbose=-1, force_col_wise=True, n_estimators=10)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance="shap")
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(leshy_list)
        ), "expect non-empty intersection"

    def test_leshy_regr_with_lgb_and_shap_feature_importance_and_sample_weight(self):
        baseline_list = ["var0", "var1", "var2", "var3", "var4", "var5"]

        X, y, w = _generated_corr_dataset_classification(size=500)
        model = lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=10)
        arfs = Leshy(model, verbose=0, max_iter=10, random_state=42, importance="shap")
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(leshy_list)
        ), "expect non-empty intersection"


class TestBoostAGroota:
    """
    Test suite for all-relevant FS boruta-like method: Leshy
    """

    def test_boostagroota_clf_with_lgb_and_shap_feature_importance_and_sample_weight(
        self,
    ):
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]

        X, y, w = _generated_corr_dataset_classification(size=500)
        model = lgb.LGBMClassifier(verbose=-1, force_col_wise=True, n_estimators=10)
        arfs = BoostAGroota(
            est=model,
            cutoff=1,
            iters=3,
            max_rounds=3,
            delta=0.1,
            silent=False,
            importance="shap",
        )
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(leshy_list)
        ), "expect non-empty intersection"

    def test_boostagroota_clf_with_lgb_and_pimp_feature_importance_and_sample_weight(
        self,
    ):
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]

        X, y, w = _generated_corr_dataset_classification(size=500)
        model = lgb.LGBMClassifier(verbose=-1, force_col_wise=True, n_estimators=10)
        arfs = BoostAGroota(
            est=model,
            cutoff=1,
            iters=3,
            max_rounds=3,
            delta=0.1,
            silent=False,
            importance="pimp",
        )
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(leshy_list)
        ), "expect non-empty intersection"

    def test_boostagroota_rgr_with_lgb_and_shap_feature_importance_and_sample_weight(
        self,
    ):
        baseline_list = ["var0", "var1", "var2", "var3", "var4", "var5"]

        X, y, w = _generated_corr_dataset_regr(size=500)
        model = lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=10)
        arfs = BoostAGroota(
            est=model,
            cutoff=1,
            iters=3,
            max_rounds=3,
            delta=0.1,
            silent=False,
            importance="shap",
        )
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(leshy_list)
        ), "expect non-empty intersection"

    def test_boostagroota_regr_with_lgb_and_pimp_feature_importance_and_sample_weight(
        self,
    ):
        baseline_list = ["var0", "var1", "var2", "var3", "var4", "var5"]

        X, y, w = _generated_corr_dataset_regr(size=500)
        model = lgb.LGBMRegressor(verbose=-1, force_col_wise=True, n_estimators=10)
        arfs = BoostAGroota(
            est=model,
            cutoff=1,
            iters=3,
            max_rounds=3,
            delta=0.1,
            silent=False,
            importance="pimp",
        )
        arfs.fit(X, y, w)
        leshy_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(leshy_list)
        ), "expect non-empty intersection"


class TestGrootCV:
    """
    Test suite for all-relevant FS boruta-like method: Leshy
    """

    def test_grootcv_classification_with_and_sample_weight(self):
        baseline_list = ["var0", "var1", "var2", "var3", "var4"]

        X, y, w = _generated_corr_dataset_classification(size=100)
        arfs = GrootCV(objective="binary", cutoff=1, n_folds=3, n_iter=3, silent=False)
        arfs.fit(X, y, w)
        grootcv_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(grootcv_list)
        ), "expect non-empty intersection"

    def test_grootcv_regression_with_and_sample_weight(self):
        baseline_list = ["var0", "var1", "var2", "var3", "var4", "var5"]

        X, y, w = _generated_corr_dataset_regr(size=100)
        arfs = GrootCV(objective="l2", cutoff=1, n_folds=3, n_iter=3, silent=False)
        arfs.fit(X, y, w)
        grootcv_list = sorted(arfs.feature_names_in_[arfs.support_])

        assert bool(
            set(baseline_list) & set(grootcv_list)
        ), "expect non-empty intersection"
