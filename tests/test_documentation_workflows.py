"""Documentation examples and links, including installed-wheel checks."""

# ruff: noqa: S102
# These tests execute only repository-owned notebook cells.

import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


@pytest.mark.parametrize("notebook", ["arfs_classification", "arfs_regression"])
def test_core_imports_need_no_optional_comparison_backends(notebook):
    run_tagged(notebook, ["setup"])


def test_groot_notebook_calls_use_supported_constructor_arguments():
    for path in (DOCS / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        for entry in notebook["cells"]:
            source = "".join(entry["source"])
            if entry["cell_type"] != "code" or "GrootCV(" not in source:
                continue
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and (
                    getattr(node.func, "id", None) == "GrootCV"
                    or getattr(node.func, "attr", None) == "GrootCV"
                ):
                    assert "fastshap" not in {kw.arg for kw in node.keywords}, path


@pytest.mark.parametrize("size", [500, 605])
def test_sampling_keeps_presampling_limit_before_distance_allocation(size):
    data = pd.DataFrame({"value": np.arange(size)})
    seen = []

    def sample(df, n, method):
        seen.append((len(df), n, method))
        return df.iloc[:n].copy()

    result = run_tagged(
        "arfs_large_data_sampling",
        ["gower-presample"],
        {"data": data, "sample": sample},
    )
    assert seen == [(min(size, 600), 120, "gower")]
    assert len(result["gower"]) == 120
    assert len(data) == size


def test_timeseries_refits_inside_each_window_and_reuses_selected_columns(monkeypatch):
    from arfs.feature_selection import GrootCV

    result = run_tagged("arfs_timeseries", ["setup"])
    X = result["X"]
    fit = GrootCV.fit
    transform = GrootCV.transform
    fitted = []
    projected = []

    def record_fit(self, X_train, y_train):
        fitted.append((self, X_train.index))
        assert isinstance(self.folds, TimeSeriesSplit)
        for train, validation in self.folds.split(X_train):
            assert train[-1] + self.folds.gap < validation[0]
        return fit(self, X_train, y_train)

    def record_transform(self, frame):
        reduced = transform(self, frame)
        projected.append((self, frame.index, reduced.columns.tolist()))
        return reduced

    monkeypatch.setattr(GrootCV, "fit", record_fit)
    monkeypatch.setattr(GrootCV, "transform", record_transform)
    run_tagged("arfs_timeseries", ["outer-evaluation"], result)

    windows = list(result["outer"].split(X))
    assert len({id(selector) for selector, _ in fitted}) == len(fitted) == len(windows)
    assert len(projected) == 2 * len(windows)
    for i, ((selector, fitted_rows), (train, test)) in enumerate(zip(fitted, windows)):
        assert fitted_rows.equals(X.index[train])
        assert projected[2 * i][0] is projected[2 * i + 1][0] is selector
        assert projected[2 * i][1].equals(X.index[train])
        assert projected[2 * i + 1][1].equals(X.index[test])
        assert projected[2 * i][2] == projected[2 * i + 1][2]
        assert "weekly_signal" in projected[2 * i][2]


@pytest.mark.parametrize("launcher", ["Makefile", "make.bat"])
def test_documentation_launcher_points_to_configuration(launcher):
    match = re.search(r"SOURCEDIR\s*=\s*(\S+)", (DOCS / launcher).read_text())
    assert (DOCS / match[1] / "conf.py").is_file()


def test_readme_and_documentation_local_links_exist():
    text = (ROOT / "README.md").read_text()
    targets = re.findall(r"\]\((\.[^ )]+)\)", text)
    targets += re.findall(r'src="([^"h][^"]*)"', text)
    for target in targets:
        assert (ROOT / unquote(target.split("#", 1)[0])).exists(), target
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


@pytest.mark.parametrize("selector", ["Leshy", "BoostAGroota"])
def test_selector_notebooks_use_supported_importance(selector):
    for path in (DOCS / "notebooks").glob("*.ipynb"):
        for entry in json.loads(path.read_text())["cells"]:
            source = "".join(entry["source"])
            if entry["cell_type"] != "code" or f"{selector}(" not in source:
                continue
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and (
                    getattr(node.func, "id", None) == selector
                    or getattr(node.func, "attr", None) == selector
                ):
                    for keyword in node.keywords:
                        if keyword.arg == "importance" and isinstance(
                            keyword.value, ast.Constant
                        ):
                            assert keyword.value.value in ("shap", "pimp", "native"), (
                                path
                            )
