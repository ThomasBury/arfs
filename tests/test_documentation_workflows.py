"""R32 source examples; CI copies docs alongside tests for installed-wheel checks."""

# ruff: noqa: S102
# These tests execute only repository-owned notebook cells.

import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def cell(notebook, index):
    data = json.loads((DOCS / "notebooks" / (notebook + ".ipynb")).read_text())
    source = "".join(data["cells"][index]["source"])
    return source.removeprefix("%%time\n")


@pytest.mark.parametrize("notebook", ["arfs_classification", "arfs_regression"])
def test_core_imports_need_no_optional_comparison_backends(notebook):
    with plt.rc_context():
        exec(compile(cell(notebook, 1), notebook, "exec"), {})


@pytest.mark.parametrize("index", [15, 21, 26])
def test_regression_fastshap_examples_do_not_run_by_default(index):
    exec(compile(cell("arfs_regression", index), "optional-fastshap", "exec"), {})


def test_classification_defines_encoding_before_first_fit():
    frame = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0, 4.0],
            "category": pd.Categorical(["a", "b", "a", "b"]),
        }
    )
    data = SimpleNamespace(
        data=frame,
        target=pd.Series(pd.Categorical(["1", "0", "1", "0"])),
        categorical=["category"],
    )
    namespace = {"load_data": lambda **kwargs: data}
    exec(
        compile(cell("arfs_classification", 7), "classification-setup", "exec"),
        namespace,
    )
    encoded = namespace["X_encoded"]
    assert encoded.columns.tolist() == frame.columns.tolist()
    assert encoded["category"].tolist() == [0, 1, 0, 1]
    assert len(encoded) == len(namespace["y"]) == 4


def test_sampling_keeps_presampling_limit_before_distance_allocation():
    data = pd.DataFrame({"value": np.arange(10_005)})
    seen = []

    def sample(df, n, sample_weight, method):
        seen.append((len(df), n, method))
        return df.iloc[:n].copy()

    namespace = {"data": data, "sample": sample}
    exec(
        compile(cell("arfs_large_data_sampling", 15), "presampling", "exec"), namespace
    )
    assert seen == [(10_000, 1000, "gower")]
    assert len(namespace["data_g_samp"]) == 1000
    assert len(data) == 10_005


def test_timeseries_fit_and_predict_use_reduced_columns():
    seen = []
    frame = pd.DataFrame(
        {"signal": np.arange(60), "random_num": np.arange(60), "holiday": np.zeros(60)}
    )
    model = SimpleNamespace(
        fit=lambda X, y: seen.append(list(X.columns)),
        predict=lambda X: (seen.append(list(X.columns)), np.zeros(len(X)))[1],
    )
    namespace = {
        "X": frame,
        "y": pd.Series(np.arange(60)),
        "gbrt": model,
        "ts_cv": TimeSeriesSplit(n_splits=2),
        "plt": plt,
    }
    try:
        exec(
            compile(cell("arfs_timeseries", 10), "reduced-features", "exec"), namespace
        )
        assert seen == [["signal"], ["signal"]]
    finally:
        plt.close("all")


@pytest.mark.parametrize("launcher", ["Makefile", "make.bat"])
def test_documentation_launcher_points_to_configuration(launcher):
    match = re.search(r"SOURCEDIR\s*=\s*(\S+)", (DOCS / launcher).read_text())
    assert (DOCS / match[1] / "conf.py").is_file()


def test_readme_and_documentation_local_links_exist():
    text = (ROOT / "README.md").read_text()
    targets = re.findall(r"\]\((\.[^ )]+)\)", text)
    targets += re.findall(r'src="([^"h][^"]*)"', text)
    for target in targets:
        assert (ROOT / unquote(target)).exists(), target
    for path in DOCS.glob("*.rst"):
        in_tree = False
        for line in path.read_text().splitlines():
            if line.startswith(".. toctree::"):
                in_tree = True
            elif line and not line.startswith(" "):
                in_tree = False
            elif in_tree and line.strip() and not line.strip().startswith(":"):
                target = line.strip()
                suffix = "" if target.endswith(".ipynb") else ".rst"
                assert (path.parent / (target + suffix)).is_file(), (path, target)
    for path in (DOCS / "notebooks").glob("*.ipynb"):
        for entry in json.loads(path.read_text())["cells"]:
            if entry["cell_type"] == "markdown":
                for target in re.findall(
                    r"\]\(([^ )]+\.ipynb)\)", "".join(entry["source"])
                ):
                    if "://" not in target:
                        assert (path.parent / unquote(target)).is_file(), (path, target)
