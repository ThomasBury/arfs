"""Remaining optional-entry boundaries, without installing plotting backends."""

import sys
import types
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from arfs.association import plot_association_matrix_int
from arfs.feature_selection import BoostAGroota, Leshy


def test_interactive_plot_missing_holoviews_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "holoviews", None)
    with pytest.raises(ImportError, match="Holoviews is not installed"):
        plot_association_matrix_int(pd.DataFrame([[1.0]]))


def test_interactive_plot_missing_panel_is_actionable(monkeypatch):
    # A placeholder gets past only the first import; no rendering is simulated.
    monkeypatch.setitem(sys.modules, "holoviews", types.ModuleType("holoviews"))
    monkeypatch.setitem(sys.modules, "panel", None)
    with pytest.raises(ImportError, match="Panel is not installed"):
        plot_association_matrix_int(pd.DataFrame([[1.0]]))


@pytest.mark.parametrize("selector", [Leshy, BoostAGroota])
def test_fastshap_fallback_is_explicit_before_input_validation(monkeypatch, selector):
    monkeypatch.setitem(sys.modules, "fasttreeshap", None)
    instance = selector(estimator=RandomForestRegressor(n_estimators=1), importance="fastshap")
    with pytest.warns(UserWarning, match="Fallback to shap"):
        with pytest.raises(TypeError, match="dataframe"):
            instance.fit([[1.0]], [1.0])
    assert instance.importance == "shap"


@pytest.mark.xfail(strict=True, reason="07 F1: time-series post-selection plot still trains and predicts with X")
def test_timeseries_postselection_plot_uses_reduced_features():
    repo = Path(__file__).resolve().parents[3]
    notebook = json.loads((repo / "docs/notebooks/arfs_timeseries.ipynb").read_text())
    cell = "".join(notebook["cells"][10]["source"])
    seen = []
    frame = pd.DataFrame({"signal": np.arange(60), "random_num": np.arange(60), "holiday": np.zeros(60)})
    model = types.SimpleNamespace(
        fit=lambda X, y: seen.append(list(X.columns)),
        predict=lambda X: (seen.append(list(X.columns)), np.zeros(len(X)))[1],
    )
    namespace = {"X": frame, "y": pd.Series(np.arange(60)), "gbrt": model,
                 "ts_cv": TimeSeriesSplit(n_splits=2), "plt": plt}
    try:
        exec(compile(cell, "arfs_timeseries.ipynb:cell10", "exec"), namespace)
        assert list(namespace["X_fs"].columns) == ["signal"]
        assert seen == [["signal"], ["signal"]]
    finally:
        plt.close("all")
