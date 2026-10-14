"""R22: trusted wrapper artifacts preserve prediction semantics and safe writes."""

import os
import subprocess
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from arfs import gbm
from arfs.gbm import GradientBoosting


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def fitted():
    rng = np.random.default_rng(91)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a b", "noise"])
    X.index = np.arange(len(X)) * 7 + 10
    y = pd.Series(2 * X["a b"].to_numpy() + rng.normal(size=len(X)))
    model = make_model()
    model.fit(X, y)
    return model, X, y


def make_model(objective="regression", **kwargs):
    params = {
        "objective": objective,
        "metric": "l2",
        "n_estimators": 15,
        "num_threads": 1,
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "force_col_wise": True,
        "seed": 91,
    }
    if objective == "binary":
        params["metric"] = "binary_logloss"
    elif objective == "softmax":
        params.update(num_class=3, metric="multi_logloss")
    return GradientBoosting(
        params=params, show_learning_curve=False, verbose_eval=0, **kwargs
    )


CHILD = """
import sys
from pathlib import Path
import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from arfs.gbm import GradientBoosting
assert Path(sys.prefix) in Path(__import__('arfs').__file__).resolve().parents
X, raw, prediction, names, categories, offset = joblib.load(sys.argv[2])
model = GradientBoosting(cat_feat=None, show_learning_curve=False)
model.load(sys.argv[1])
assert model.feature_names_in_ == names
assert model.cat_feat == categories
assert model.is_init_score is offset
assert model.valid_features is None and model.learning_curve is None
assert_allclose(model.predict_raw(X, raw_score=True), raw, rtol=0, atol=0)
assert_allclose(model.predict_raw(X, raw_score=True, validate_features=True), raw, rtol=0, atol=0)
if offset:
    try:
        model.predict(X)
    except AttributeError as error:
        assert 'init_score' in str(error)
    else:
        raise AssertionError('Offset contract was lost')
else:
    assert_array_equal(model.predict(X), prediction)
print('Fresh-process prediction contract passed')
"""


@pytest.mark.parametrize("task", ["regression", "binary", "softmax", "category"])
@pytest.mark.parametrize("offset", [False, True])
def test_fresh_process_round_trip_on_new_rows(fitted, tmp_path, task, offset):
    _, X, y = fitted
    if task == "category":
        X[7] = pd.Categorical(["red", "green"] * (len(X) // 2))
        X["flag"] = np.arange(len(X)) % 2 == 0
    elif task == "binary":
        y = (y > 0).astype(int)
    elif task == "softmax":
        y = pd.qcut(y, 3, labels=False)
    model = make_model("regression" if task == "category" else task)
    initial = (
        np.column_stack([X["a b"] / 4, X.noise / 5, np.zeros(len(X))])
        if task == "softmax"
        else np.linspace(-0.4, 0.7, len(X))
    )
    model.fit(X, y, init_score=initial if offset else None)
    # These rows and category levels were not passed to fit; no stored offset can supply their baseline.
    query = X.iloc[:4].copy()
    query["a b"] += 5
    if task == "category":
        query[7] = pd.Categorical(["new", "green", None, "red"])
    raw = model.predict_raw(query, raw_score=True)
    prediction = None if offset else model.predict(query)
    artifact_path = model.save(str(tmp_path / "nested"), "model")
    input_path = tmp_path / "trusted-input.joblib"
    joblib.dump(
        (query, raw, prediction, model.feature_names_in_, model.cat_feat, offset),
        input_path,
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, artifact_path, str(input_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("offset", [False, True])
def test_loaded_offsets_require_external_baseline(fitted, tmp_path, offset):
    _, X, y = fitted
    model = make_model()
    model.fit(X, y, init_score=np.zeros(len(X)) if offset else None)
    restored = make_model()
    restored.load(model.save(str(tmp_path), "model"))
    raw = restored.predict_raw(X.iloc[:2], raw_score=True)
    assert_allclose(raw, model.predict_raw(X.iloc[:2], raw_score=True), rtol=0, atol=0)
    if offset:
        with pytest.raises(AttributeError, match="init_score"):
            restored.predict(X.iloc[:2])
        baseline_a = np.array([2.0, 4.0])
        baseline_b = baseline_a + 10
        assert_allclose((raw + baseline_b) - (raw + baseline_a), 10)
    else:
        assert_allclose(restored.predict(X.iloc[:2]), model.predict(X.iloc[:2]))
    payload = joblib.load(tmp_path / "model.joblib")
    assert set(payload) == {
        "format",
        "version",
        "model",
        "is_init_score",
        "cat_feat",
        "feature_names_in_",
    }
    assert isinstance(payload["model"], lgb.Booster)


@pytest.mark.parametrize("load_offset", [False, True])
def test_load_replaces_fitted_state_and_retains_constructor_options(
    fitted, tmp_path, load_offset
):
    _, X, y = fitted
    X[7] = pd.Categorical(["red", "green"] * (len(X) // 2))
    saved = make_model()
    saved.fit(X, y, init_score=np.zeros(len(X)) if load_offset else None)
    path = saved.save(str(tmp_path), "saved")
    receiver = make_model(cat_feat=None, return_valid_features=True)
    receiver.show_learning_curve = True
    receiver.fit(
        X.rename(columns={"a b": "old"}),
        y,
        init_score=None if load_offset else np.zeros(len(X)),
    )
    old_curve = receiver.learning_curve
    old_params = receiver.params
    assert receiver.valid_features is not None and old_curve is not None
    receiver.load(path)
    assert receiver.model is not saved.model
    assert receiver.model_params == saved.model_params
    assert receiver.feature_names_in_ == saved.feature_names_in_
    assert receiver.cat_feat == [7]
    assert receiver.is_init_score is load_offset
    assert receiver.valid_features is None and receiver.learning_curve is None
    assert receiver.params is old_params and receiver.cat_feat_input is None
    assert receiver.show_learning_curve and receiver.return_valid_features
    assert plt.fignum_exists(old_curve.number)
    assert_allclose(
        receiver.predict_raw(X, raw_score=True), saved.predict_raw(X, raw_score=True)
    )
    receiver.show_learning_curve = False
    receiver.fit(X, y)
    assert receiver.is_init_score is False and receiver.cat_feat is None
    assert receiver.valid_features is not None and receiver.learning_curve is None
    assert_allclose(receiver.predict(X), receiver.model.predict(X))


@pytest.mark.parametrize(
    "bad",
    [
        "legacy",
        "foreign",
        "foreign_object",
        "corrupt",
        "missing",
        "format",
        "version",
        "bool_version",
        "missing_field",
        "model",
        "flag",
        "names",
        "duplicate_names",
        "wrong_names",
        "categories",
        "unknown_category",
    ],
)
def test_invalid_artifact_preserves_previous_state(fitted, tmp_path, bad):
    model, X, y = fitted
    model.return_valid_features = True
    model.show_learning_curve = True
    model.fit(X, y)
    good_path = model.save(str(tmp_path), "good")
    payload = joblib.load(good_path)
    if not isinstance(payload, dict):
        # Produce the intended malformed future-format fixture when checking the old wheel.
        payload = {
            "format": "arfs.GradientBoosting",
            "version": 1,
            "model": model.model,
            "is_init_score": False,
            "cat_feat": None,
            "feature_names_in_": list(map(str, X.columns)),
        }
    bad_path = tmp_path / "bad.joblib"
    if bad == "missing":
        expected_error = FileNotFoundError
    elif bad == "corrupt":
        bad_path.write_bytes(b"not a pickle")
        expected_error = Exception
    else:
        expected_error = ValueError
        if bad == "legacy":
            payload = model.model
            expected_error = TypeError
        elif bad == "foreign":
            payload = {"unrelated": True}
        elif bad == "foreign_object":
            payload = ["unrelated"]
            expected_error = TypeError
        elif bad == "format":
            payload["format"] = "not arfs"
        elif bad == "version":
            payload["version"] = 2
        elif bad == "bool_version":
            payload["version"] = True
        elif bad == "missing_field":
            del payload["is_init_score"]
        elif bad == "model":
            payload["model"] = "not a Booster"
        elif bad == "flag":
            payload["is_init_score"] = "False"
        elif bad == "names":
            payload["feature_names_in_"] = "a b"
        elif bad == "duplicate_names":
            payload["feature_names_in_"] = ["noise", "noise"]
        elif bad == "wrong_names":
            payload["feature_names_in_"] = ["noise", "a b"]
        elif bad == "categories":
            payload["cat_feat"] = "noise"
        elif bad == "unknown_category":
            payload["cat_feat"] = ["absent"]
        joblib.dump(payload, bad_path)
    attributes = [
        "model",
        "model_params",
        "feature_names_in_",
        "cat_feat",
        "is_init_score",
        "valid_features",
        "learning_curve",
    ]
    previous = [getattr(model, name) for name in attributes]
    prediction = model.predict(X)
    with pytest.raises(expected_error):
        model.load(str(bad_path))
    for name, old_value in zip(attributes, previous):
        assert getattr(model, name) is old_value
    assert_allclose(model.predict(X), prediction)


@pytest.mark.parametrize(
    "failure",
    ["dump", "fsync", "render", "model_replace", "curve_replace", "invalid_curve"],
)
def test_save_failure_does_not_truncate_existing_artifacts(
    fitted, tmp_path, monkeypatch, failure
):
    model, X, y = fitted
    model.show_learning_curve = True
    model.fit(X, y)
    path = Path(model.save(str(tmp_path), "saved"))
    curve_path = tmp_path / "saved_learning_curve.png"
    old_model, old_curve = path.read_bytes(), curve_path.read_bytes()
    model.fit(X, y + 10)
    figure = model.learning_curve
    if failure == "dump":

        def broken_dump(value, filename, **kwargs):
            if hasattr(filename, "write"):
                filename.write(b"partial")
            else:
                Path(filename).write_bytes(b"partial")
            raise OSError("forced serialization failure")

        monkeypatch.setattr(gbm.joblib, "dump", broken_dump)
    elif failure == "fsync":

        def broken_fsync(fd):
            raise OSError("forced sync failure")

        monkeypatch.setattr(gbm.os, "fsync", broken_fsync)
    elif failure == "render":

        def broken_savefig(path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("forced rendering failure")

        monkeypatch.setattr(figure, "savefig", broken_savefig)
    elif failure in ("model_replace", "curve_replace"):
        real_replace = Path.replace

        def broken_replace(source, target):
            if source.name == (
                "model.joblib" if failure == "model_replace" else "curve.png"
            ):
                raise OSError("forced replacement failure")
            return real_replace(source, target)

        monkeypatch.setattr(Path, "replace", broken_replace)
    else:
        model.learning_curve = "not a figure"
    with pytest.raises(TypeError if failure == "invalid_curve" else OSError):
        model.save(str(tmp_path), "saved")
    assert curve_path.read_bytes() == old_curve
    if failure == "curve_replace":
        # The documented commit point is the model replacement; PNG replacement follows it.
        assert path.read_bytes() != old_model
        restored = make_model()
        restored.load(str(path))
        assert_allclose(restored.predict(X), model.predict(X))
    else:
        assert path.read_bytes() == old_model
    assert not list(tmp_path.glob(".arfs-*"))
    assert plt.fignum_exists(figure.number)


def test_save_success_paths_overwrite_and_curve(fitted, tmp_path, monkeypatch):
    model, X, y = fitted
    model.show_learning_curve = True
    model.fit(X, y)
    figure = model.learning_curve
    path = Path(model.save(str(tmp_path / "nested"), "saved"))
    assert path.is_file() and path.with_name("saved_learning_curve.png").is_file()
    assert not plt.fignum_exists(figure.number)
    old = path.read_bytes()
    model.fit(X, y + 10)
    assert model.save(str(path.parent), "saved") == str(path)
    assert path.read_bytes() != old
    assert not list(path.parent.glob(".arfs-*"))
    monkeypatch.chdir(tmp_path)
    assert model.save(name="relative") == "relative.joblib"
    assert Path("relative.joblib").is_file()
    occupied = tmp_path / "occupied"
    occupied.write_text("sentinel")
    with pytest.raises(FileExistsError):
        model.save(str(occupied), "saved")
    assert occupied.read_text() == "sentinel"


def test_unfitted_save_rejected(tmp_path):
    with pytest.raises(ValueError, match="not been trained"):
        make_model().save(str(tmp_path), "model")
    assert not list(tmp_path.iterdir())


def test_explicit_duplicate_category_selection_round_trips(fitted, tmp_path):
    _, X, y = fitted
    X[7] = pd.Categorical(["red", "green"] * (len(X) // 2))
    model = make_model(cat_feat=[7, 7])
    model.fit(X, y)
    restored = make_model()
    restored.load(model.save(str(tmp_path), "duplicates"))
    assert restored.cat_feat == [7, 7]
    assert_allclose(restored.predict(X), model.predict(X))
