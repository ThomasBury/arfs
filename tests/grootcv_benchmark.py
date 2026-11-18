"""Explicit CPU benchmark; run each case in a fresh process.

Set PYTHONPATH to the source snapshot under comparison, then run:
python tests/grootcv_benchmark.py tall OUTPUT.json
Cases: tall, wide, multiclass, default. No boosting-round cap is supplied.
"""

import hashlib
import importlib.metadata
import inspect
import json
import platform
import resource
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from arfs.feature_selection import allrelevant as a


def measure(selector, X, y, weight):
    legacy = "fastshap" in inspect.signature(a.GrootCV).parameters
    timings = {"training": 0.0, "explanation": 0.0, "preparation": 0.0}
    calls = {"training": 0, "explanation": 0, "preparation": 0}

    def timed(function, stage):
        def wrapped(*args, **kwargs):
            start = perf_counter()
            result = function(*args, **kwargs)
            timings[stage] += perf_counter() - start
            calls[stage] += 1
            return result

        return wrapped

    train_fold = a._train_lgb_model
    predict = lgb.Booster.predict

    def explain_legacy(*args, **kwargs):
        training_before = timings["training"]
        start = perf_counter()
        result = train_fold(*args, **kwargs)
        timings["explanation"] += (
            perf_counter() - start - (timings["training"] - training_before)
        )
        calls["explanation"] += 1
        return result

    def predict_native(self, *args, **kwargs):
        if kwargs.get("pred_contrib"):
            return timed(predict, "explanation")(self, *args, **kwargs)
        return predict(self, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(lgb, "train", timed(lgb.train, "training")))
        for name in ["_split_data", "_create_shadow", "_set_lgb_parameters"]:
            stack.enter_context(
                patch.object(a, name, timed(getattr(a, name), "preparation"))
            )
        if legacy:
            stack.enter_context(patch.object(a, "_train_lgb_model", explain_legacy))
        else:
            stack.enter_context(patch.object(lgb.Booster, "predict", predict_native))
        start = perf_counter()
        selector.fit(X, y, sample_weight=weight)
        total = perf_counter() - start
    assert (
        calls["training"] == calls["explanation"] == selector.n_folds * selector.n_iter
    )
    return {
        **{f"{k}_seconds": v for k, v in timings.items()},
        "aggregation_and_bookkeeping_seconds": total - sum(timings.values()),
        "total_seconds": total,
    }


def run(case):
    rows, features, classes, n_folds, repeats = {
        "tall": (3000, 24, 1, 3, 2),
        "wide": (500, 300, 1, 3, 2),
        "multiclass": (1500, 24, 3, 3, 2),
        "default": (800, 12, 1, 5, 5),
    }[case]
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(rows, features)), columns=[f"x{i}" for i in range(features)]
    )
    X["category"] = pd.Categorical(rng.integers(0, 4, rows))
    signal = 3 * X.x0 + 2 * X.x1
    y = signal + rng.normal(size=rows)
    if classes > 1:
        y = pd.Series(np.digitize(signal, [-1, 1]))
    weight = np.where(X.x0 > 0, 4.0, 1.0)
    params = {"num_leaves": 7, "max_depth": 3, "min_data_in_leaf": 15, "seed": 42}

    # Warm native code before timing, without consuming the selector's RNG.
    lgb.train(
        {"objective": "l2", "num_threads": 1, "verbosity": -1},
        lgb.Dataset(X.iloc[:100], label=np.asarray(y)[:100]),
        num_boost_round=2,
    )
    selector = a.GrootCV(
        objective="multiclass" if classes > 1 else "l2",
        n_folds=n_folds,
        n_iter=repeats,
        n_jobs=1,
        random_state=42,
        lgbm_params=params,
    )
    measurements = []
    reference = None
    for _ in range(3):
        measurements.append(measure(selector, X, y, weight))
        assert np.isfinite(selector.cv_df.filter(like="fscore")).all().all()
        if reference is not None:
            assert_frame_equal(selector.cv_df, reference)
        reference = selector.cv_df.copy()
    return {
        "case": case,
        "shape": list(X.shape),
        "folds": n_folds,
        "repeats": repeats,
        "seed": 42,
        "threads": 1,
        "learner_params": params,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "lightgbm", "shap", "scikit-learn"]
        },
        "source_sha256": hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        "measurements": measurements,
        "median_seconds": {
            key: float(np.median([m[key] for m in measurements]))
            for key in measurements[0]
        },
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "selected_features": selector.selected_features_.tolist(),
        "shadow_cutoff": selector.sha_cutoff,
        "cv_df": selector.cv_df.to_dict(orient="list"),
    }


if __name__ == "__main__":
    result = run(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("case", "median_seconds", "peak_process_rss_kib")
            }
        )
    )
