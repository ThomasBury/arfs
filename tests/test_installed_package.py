"""Installation contracts, also runnable against a wheel outside the checkout."""

import importlib
import re
from importlib import metadata, resources
from pathlib import Path
from zipfile import Path as ZipPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "arfs",
        "arfs.association",
        "arfs.benchmark",
        "arfs.feature_selection",
        "arfs.feature_selection.allrelevant",
        "arfs.feature_selection.base",
        "arfs.feature_selection.lasso",
        "arfs.feature_selection.mrmr",
        "arfs.feature_selection.summary",
        "arfs.feature_selection.unsupervised",
        "arfs.feature_selection.variable_importance",
        "arfs.gbm",
        "arfs.parallel",
        "arfs.preprocessing",
        "arfs.sampling",
        "arfs.utils",
    ],
)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_declared_exports_exist():
    import arfs.feature_selection as selection
    from arfs.feature_selection.base import BaseThresholdSelector

    for name in selection.__all__:
        assert getattr(selection, name) is not None
    assert selection.BaseThresholdSelector is BaseThresholdSelector


@pytest.mark.parametrize(
    ("name", "shape"),
    [("Boston", (506, 18)), ("cancer", (569, 33)), ("housing", (20640, 8))],
)
def test_local_datasets(name, shape):
    from arfs.utils import load_data

    dataset = load_data(name)
    assert dataset.data.shape == shape
    assert len(dataset.target) == shape[0]
    if name == "housing":
        assert "California" in dataset.DESCR
        assert Path(dataset.filename).is_file()
        assert pd.read_csv(dataset.filename).shape == (20640, 9)


def test_styling_dependency_is_declared_and_usable():
    requirements = metadata.requires("arfs")
    assert any(re.match(r"(?i)^jinja2\b", requirement) for requirement in requirements)
    assert "predictor" in pd.DataFrame({"predictor": ["a"]}).style.to_html()


def test_dataset_streams_from_zip(tmp_path, monkeypatch):
    from arfs import utils

    expected_boston = utils.load_data("Boston")
    expected_housing = utils.load_data("housing")
    root = resources.files("arfs")
    archive_path = tmp_path / "resources.zip"
    resource_names = (
        "dataset/data/boston_bunch.joblib",
        "dataset/data/housing.zip",
        "dataset/descr/housing.rst",
    )
    with ZipFile(archive_path, "w") as archive:
        for name in resource_names:
            archive.writestr(f"arfs/{name}", root.joinpath(name).read_bytes())

    with ZipFile(archive_path) as archive:
        # Python 3.9 paths share the ZipFile handle; retain one root until loading ends.
        zip_root = ZipPath(archive, "arfs/")
        monkeypatch.setattr(utils, "files", lambda package: zip_root)
        boston = utils.load_data("Boston")
        housing = utils.load_data("housing")
        frame = utils._load_housing(as_frame=True)

    # Loaded values remain usable after the resource archive and streams close.
    pd.testing.assert_frame_equal(boston.data, expected_boston.data)
    np.testing.assert_array_equal(boston.target, expected_boston.target)
    np.testing.assert_array_equal(housing.data, expected_housing.data)
    np.testing.assert_array_equal(housing.target, expected_housing.target)
    np.testing.assert_array_equal(frame.iloc[:, :8], housing.data)
    np.testing.assert_array_equal(frame["target"], housing.target)
    assert housing.DESCR == expected_housing.DESCR
    assert housing.filename == f"{archive_path}/arfs/dataset/data/housing.zip"
