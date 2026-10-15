"""Check the R33 experiment's leakage and weighted-loss boundaries."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from tests.selection_screen import loss, prepare


def test_screen_encoder_learns_no_held_out_category():
    train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"])}, index=[0, 1, 2])
    test = pd.DataFrame({"cat": pd.Categorical(["held-out", "b"])}, index=[3, 4])
    encoded_train, encoded_test, encoder = prepare(train, test)
    assert encoder.categories_[0].tolist() == ["a", "b"]
    assert np.isnan(encoded_test.iloc[0, 0])
    assert encoded_test.iloc[1, 0] == encoded_train.iloc[1, 0]
    assert encoded_test.index.tolist() == [3, 4]


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_screen_weighted_losses_match_integer_replication(task):
    y = np.array([0, 1, 1])
    prediction = np.array([0.2, 0.8, 0.3])
    weights = np.array([1, 4, 2])
    rows = np.repeat(np.arange(len(y)), weights)
    assert_allclose(
        loss(task, y, prediction, weights),
        loss(task, y[rows], prediction[rows]),
    )
