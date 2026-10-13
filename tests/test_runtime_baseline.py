"""Small signal/noise screens, not evidence of calibrated selection error rates."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from arfs.feature_selection import BoostAGroota, GrootCV, Leshy, UniqueValuesThreshold
from arfs.utils import _make_corr_dataset_classification


def test_classification_fixture_schema_and_signal():
    frame, target, weights = _make_corr_dataset_classification(size=1000)
    assert frame.columns.tolist() == [
        *(f"var{i}" for i in range(11)),
        "dummy",
        "var12",
        "nice_guys",
    ]
    assert frame.shape == (1000, 14)
    assert target.shape == weights.shape == (1000,)
    assert set(target) == {0, 1}
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert isinstance(frame["dummy"].dtype, pd.CategoricalDtype)
    assert frame["dummy"].nunique() == 1000
    assert frame["nice_guys"].map(type).eq(str).all()
    assert frame["nice_guys"].nunique() == 36
    assert frame["var10"].eq(1).all()
    assert frame["var12"].isna().sum() == 500
    correlations = frame.drop(columns=["dummy", "nice_guys", "var10"]).corrwith(
        pd.Series(target)
    )
    assert (correlations[[f"var{i}" for i in range(5)]].abs() > 0.3).all()
    assert (correlations[[f"var{i}" for i in range(5, 10)]].abs() < 0.1).all()


@pytest.mark.parametrize(
    "threshold, selected", [(1, ["binary", "ternary"]), (2, ["ternary"])]
)
def test_unique_threshold_excludes_equality(threshold, selected):
    frame = pd.DataFrame(
        {"constant": [1, 1, 1], "binary": [0, 1, 0], "ternary": [0, 1, 2]}
    )
    selector = UniqueValuesThreshold(threshold=threshold).fit(frame)
    assert selector.selected_features_.tolist() == selected
    assert selector.transform(frame).columns.tolist() == selected


@pytest.mark.parametrize("method", ["leshy", "boostagroota", "grootcv"])
def test_selector_recovers_signal_and_rejects_noise(method):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(600, 9)), columns=["signal", *(f"noise{i}" for i in range(8))]
    )
    target = 5 * frame["signal"].to_numpy() + rng.normal(scale=0.1, size=len(frame))
    model = LGBMRegressor(n_estimators=40, n_jobs=1, random_state=42, verbosity=-1)
    if method == "leshy":
        selector = Leshy(
            model,
            n_estimators=40,
            max_iter=15,
            random_state=42,
            importance="native",
            verbose=0,
        )
    elif method == "boostagroota":
        selector = BoostAGroota(
            model, cutoff=1, iters=3, max_rounds=3, importance="native"
        )
    else:
        selector = GrootCV(
            objective="l2", cutoff=1, n_folds=3, n_iter=1, n_jobs=1, silent=True
        )
    state = np.random.get_state()
    try:
        np.random.seed(42)
        selector.fit(frame, target)
    finally:
        np.random.set_state(state)
    selected = set(selector.selected_features_)
    assert "signal" in selected
    assert len(selected - {"signal"}) <= 2
