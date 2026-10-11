"""Run separately on unit 0's current-dependency wheel environment."""
import numpy as np
import pandas as pd
import pytest
from arfs.feature_selection.lasso import LassoFeatureSelection


@pytest.mark.xfail(strict=True, reason="F1: force_all_finite was removed from allowed sklearn versions")
def test_public_selector_fits_with_current_dependencies():
    x = np.arange(30.)
    model = LassoFeatureSelection(n_iterations=3).fit(pd.DataFrame({"x": x}), x + np.sin(x))
    assert len(model.selected_features_) > 0
