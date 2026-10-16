"""R30 fixture construction and RNG isolation, not selection calibration."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from arfs import utils

GENERATORS = [
    utils._make_corr_dataset_regression,
    utils._make_corr_dataset_classification,
]


@pytest.fixture(autouse=True)
def restore_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def local_titanic(monkeypatch):
    frame = pd.DataFrame(
        {
            "pclass": [1, 2, 3, 1],
            "sex": ["female", "male", "female", "male"],
            "embarked": ["S", "C", "Q", "S"],
            "age": [20.0, 30.0, 40.0, 50.0],
            "parch": [0, 0, 1, 1],
            "sibsp": [0, 1, 0, 1],
            "fare": [10.0, 20.0, 30.0, 40.0],
            "name": ["A, Mrs. One", "B, Mr. Two", "C, Miss. Three", "D, Mr. Four"],
        }
    )
    target = pd.Series(pd.Categorical(["1", "0", "1", "0"]))

    def fetch(name, **kwargs):
        assert name == "titanic"
        assert kwargs == {
            "version": 1,
            "as_frame": True,
            "return_X_y": True,
            "parser": "auto",
        }
        return frame.copy(deep=True), target.copy(deep=True)

    monkeypatch.setattr(utils, "fetch_openml", fetch)
    return target


@pytest.mark.parametrize("generator", GENERATORS)
def test_generator_preserves_global_random_stream(generator):
    for seed in [19, 83]:
        np.random.seed(seed)
        generator(size=20)
        np.testing.assert_array_equal(
            np.random.random(5), np.random.RandomState(seed).random_sample(5)
        )


@pytest.mark.parametrize("generator", GENERATORS)
def test_generator_repeats_features_target_and_weights(generator):
    np.random.seed(11)
    first = generator(size=1000)
    np.random.seed(12)
    for other in [generator(size=1000), generator(size=1000)]:
        assert_frame_equal(first[0], other[0])
        np.testing.assert_array_equal(first[1], other[1])
        np.testing.assert_array_equal(first[2], other[2])


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize("size", [0, 1, 2, 3, 10, 11, 1001])
def test_generator_returns_exact_requested_rows_and_schema(generator, size):
    frame, target, weights = generator(size=size)
    assert frame.shape == (size, 14)
    assert target.shape == weights.shape == (size,)
    assert np.isfinite(weights).all() and (weights > 0).all()
    category = "var11" if generator is GENERATORS[0] else "dummy"
    assert isinstance(frame[category].dtype, pd.CategoricalDtype)
    assert frame[category].nunique() == (
        (size + 1) // 2 if generator is GENERATORS[0] else size
    )
    assert frame["var10"].eq(1).all()
    assert frame["var12"].isna().sum() == round(size / 2)
    if generator is GENERATORS[0]:
        np.testing.assert_array_equal(
            frame[category].to_numpy(),
            np.concatenate([np.arange((size + 1) // 2), np.arange(size // 2)]),
        )


@pytest.mark.parametrize("generator", GENERATORS)
def test_generator_keeps_intended_strong_and_noise_features(generator):
    frame, target, _ = generator(size=1000)
    assert np.corrcoef(frame["var0"], target)[0, 1] > 0.7
    assert np.corrcoef(frame["var2"], target)[0, 1] < -0.4
    assert abs(np.corrcoef(frame["var5"], target)[0, 1]) < 0.12
    assert frame["nice_guys"].nunique() == 36


@pytest.mark.parametrize("name", ["cancer", "Titanic"])
def test_augmented_loader_preserves_global_stream(name, local_titanic):
    for seed in [19, 83]:
        np.random.seed(seed)
        utils.load_data(name)
        np.testing.assert_array_equal(
            np.random.random(5), np.random.RandomState(seed).random_sample(5)
        )


@pytest.mark.parametrize("name", ["cancer", "Titanic"])
def test_augmented_loader_repeats_all_generated_values(name, local_titanic):
    np.random.seed(11)
    first = utils.load_data(name)
    np.random.seed(12)
    for other in [utils.load_data(name), utils.load_data(name)]:
        assert_frame_equal(first.data, other.data)
        assert_series_equal(first.target, other.target)
        assert first.categorical == other.categorical
        if name == "Titanic":
            np.testing.assert_array_equal(first.sample_weight, other.sample_weight)
        else:
            assert first.sample_weight is other.sample_weight is None


def test_titanic_alone_means_no_relatives(local_titanic):
    data = utils.load_data("Titanic")
    assert data.data.shape == (4, 10)
    assert data.data["family_size"].tolist() == [0, 1, 1, 2]
    assert data.data["is_alone"].astype(int).tolist() == [1, 0, 0, 0]
    assert_series_equal(data.target, local_titanic)
    assert data.sample_weight.shape == (4,)
    assert np.isfinite(data.sample_weight).all() and (data.sample_weight > 0).all()
