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


def run_tagged(notebook, tags, namespace=None):
    """Run the named computational cells without executing figures or a kernel."""
    namespace = {} if namespace is None else namespace
    data = json.loads((DOCS / "notebooks" / (notebook + ".ipynb")).read_text())
    for tag in tags:
        matches = [
            entry for entry in data["cells"] if tag in entry["metadata"].get("tags", [])
        ]
        assert len(matches) == 1, (notebook, tag)
        source = "".join(matches[0]["source"])
        exec(compile(source, notebook + ":" + tag, "exec"), namespace)
    return namespace


def test_crafted_partition_tutorial_recovers_stated_truth_and_deployment_contracts():
    with plt.rc_context():
        result = run_tagged(
            "preprocessing",
            [
                "setup",
                "crafted-data",
                "split",
                "partition-checks",
                "renaming-check",
                "composition-checks",
            ],
        )
    assert len(result["X"]) == 4320
    assert result["native_unchanged"] and result["naive_changed"]
    assert result["X_train"].stage.cat.ordered
    assert result["training_factors"].constant.nunique() == 1
    assert result["training_factors"].all_missing.nunique() == 1
    assert set(result["X_train"].index).isdisjoint(result["X_test"].index)


@pytest.mark.parametrize(
    "notebook,setup_tags",
    [
        ("preprocessing", ["setup", "crafted-data", "split"]),
        ("lasso_feature_selection", ["comparison-setup", "comparison-data"]),
        ("mrmr_feature_selection", ["setup", "classification-data"]),
    ],
)
def test_tutorial_comparisons_fit_only_training_rows_and_reuse_factor_design(
    notebook, setup_tags, monkeypatch
):
    from sklearn.compose import ColumnTransformer

    from arfs.feature_selection import MinRedundancyMaxRelevance
    from arfs.feature_selection.lasso import LassoFeatureSelection
    from arfs.preprocessing import TreeDiscretizer, TreeGrouper

    with plt.rc_context():
        namespace = run_tagged(notebook, setup_tags + ["design"])
    X_train, w_train = namespace["X_train"], namespace["w_train"]
    fitted = []

    def watch(cls, method, weighted):
        original = getattr(cls, method)

        def checked(self, X, y=None, *args, **kwargs):
            assert len(X) == len(X_train)
            if isinstance(X, pd.DataFrame):
                assert X.index.equals(X_train.index)
            if weighted:
                weights = kwargs.get("sample_weight", args[0] if args else None)
                np.testing.assert_array_equal(weights, w_train)
            fitted.append(cls.__name__)
            return original(self, X, y, *args, **kwargs)

        monkeypatch.setattr(cls, method, checked)

    for cls in [
        TreeGrouper,
        TreeDiscretizer,
        LassoFeatureSelection,
        MinRedundancyMaxRelevance,
    ]:
        watch(cls, "fit", True)
    watch(ColumnTransformer, "fit_transform", False)
    result = run_tagged(notebook, ["comparison"], namespace)
    assert {"TreeGrouper", "TreeDiscretizer", "ColumnTransformer"}.issubset(fitted)
    for name, pipeline in result["models"].items():
        if hasattr(pipeline, "named_steps") and "bin" in pipeline.named_steps:
            grouped = pipeline.named_steps["group"].transform(X_train)
            factors = pipeline.named_steps["bin"].transform(grouped)
            assert all(
                isinstance(dtype, pd.CategoricalDtype) for dtype in factors.dtypes
            )
        if hasattr(pipeline, "named_steps"):
            design_pipe = pipeline[:-1]
            design = design_pipe.transform(result["X_test"])
            assert np.isfinite(design.to_numpy()).all()
            assert len(design) == len(result["X_test"])
        pred = result["predictions"][name]
        assert pred.index.equals(result["X_test"].index)
        assert np.isfinite(pred).all()
        if notebook == "mrmr_feature_selection":
            from sklearn.metrics import log_loss

            expected = log_loss(
                result["y_test"], pred, sample_weight=result["w_test"], labels=[0, 1]
            )
            assert "dummy" not in result["X_train"]
        else:
            from sklearn.metrics import mean_squared_error

            expected = mean_squared_error(
                result["y_test"], pred, sample_weight=result["w_test"]
            )
        assert result["scores"][name] == pytest.approx(expected)


def test_poisson_illustration_generates_counts_from_exposure_times_rate():
    namespace = {
        "np": np,
        "pd": pd,
        "train_test_split": __import__(
            "sklearn.model_selection", fromlist=["train_test_split"]
        ).train_test_split,
        "plot_y_vs_X": lambda *args, **kwargs: None,
    }
    result = run_tagged("lasso_feature_selection", ["poisson-data"], namespace)
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(1000, 6))
    rate = np.exp(0.5 + features @ np.array([0.6, -0.4, 0.25, 0.0, 0.0, 0.0]))
    exposure = rng.uniform(30 / 365, 1, 1000)
    counts = rng.poisson(exposure * rate)
    np.testing.assert_array_equal(result["counts"], counts)
    np.testing.assert_allclose(result["y"] * result["w"], counts)
    assert len(result["X_train"]) == 750 and len(result["X_test"]) == 250


@pytest.mark.parametrize(
    "notebook", ["preprocessing", "lasso_feature_selection", "mrmr_feature_selection"]
)
def test_refreshed_tutorial_sources_compile_and_use_local_inputs(notebook):
    data = json.loads((DOCS / "notebooks" / (notebook + ".ipynb")).read_text())
    for entry in data["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]).removeprefix("%%time\n"), notebook, "exec")
    if notebook == "mrmr_feature_selection":
        source = "".join(
            "".join(entry["source"])
            for entry in data["cells"]
            if entry["cell_type"] == "code"
        )
        assert "Titanic" not in source and "load_data(" not in source
