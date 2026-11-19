"""Benchmark BoostAGroota source snapshots in fresh, single-worker processes.

Set PYTHONPATH to a source snapshot, then run:
python tests/boostagroota_benchmark.py tall OUTPUT.json
Cases: tall, wide, multiclass, forest, permutation. Three fits per process.
"""

import hashlib
import importlib.metadata
import json
import resource
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestRegressor

from arfs.feature_selection import allrelevant as a


def run(case):
    rows, features = {
        "tall": (3000, 24),
        "wide": (500, 300),
        "multiclass": (1000, 24),
        "forest": (500, 12),
        "permutation": (200, 6),
    }[case]
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        rng.normal(size=(rows, features)), columns=[f"x{i}" for i in range(features)]
    )
    X["category"] = pd.Categorical(rng.randint(0, 4, rows))
    y = 3 * X.x0 - 2 * X.x1 + rng.normal(size=rows)
    weight = np.where(X.x0 > 0, 4.0, 1.0)
    params = {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "random_state": 42}
    if case == "forest":
        estimator = RandomForestRegressor(**params)
    else:
        cls = lgb.LGBMClassifier if case == "multiclass" else lgb.LGBMRegressor
        estimator = cls(**params, num_leaves=7, min_child_samples=15, verbosity=-1)
    if case == "multiclass":
        y = pd.Series(np.digitize(y, [-1, 1]))
    selector = a.BoostAGroota(
        estimator,
        cutoff=1,
        iters=3 if case == "permutation" else 10,
        max_rounds=3,
        importance="pimp" if case == "permutation" else "shap",
        random_state=42,
    )
    X_before, y_before, w_before = X.copy(deep=True), y.copy(), weight.copy()
    params_before = estimator.get_params()
    legacy = hasattr(a, "_reduce_vars_sklearn")
    owner = a if legacy else a.BoostAGroota
    method_name = "_reduce_vars_sklearn" if legacy else "_run_round"
    method = getattr(owner, method_name)
    times, traces = [], []
    for _ in range(3):
        trace = {"rounds": []}

        def record_round(*args, trace=trace, **kwargs):
            result = method(*args, **kwargs)
            if legacy:
                stop, selected, table, threshold = result
                columns = table.feature.tolist()
                scores = table.filter(regex=r"^fscore\d+$").to_numpy()
                means = table.Mean.to_numpy()
            else:
                scores, names = result
                columns = names.tolist()
                means = scores.mean(axis=1)
                n = len(columns) // 2
                threshold = means[n:].max() / selector.cutoff
                keep = (means[:n] >= threshold) & (means[:n] > 0)
                selected = names[:n][keep]
                stop = len(selected) == 0 or len(selected) / n > 1 - selector.delta
            trace["rounds"].append(
                {
                    "columns": columns,
                    "scores": scores.tolist(),
                    "means": means.tolist(),
                    "threshold": float(threshold),
                    "selected": list(selected),
                    "stop": bool(stop),
                }
            )
            return result

        with patch.object(owner, method_name, record_round):
            start = perf_counter()
            selector.fit(X, y, sample_weight=weight)
            times.append(perf_counter() - start)
        trace.update(
            history=selector.sha_cutoff_df.to_numpy().tolist(),
            history_columns=selector.sha_cutoff_df.columns.tolist(),
            history_index=selector.sha_cutoff_df.index.tolist(),
            support=selector.support_.tolist(),
            ranking=selector.ranking_.tolist(),
            absolute_ranking=selector.ranking_absolutes_,
            selected=selector.selected_features_.tolist(),
        )
        traces.append(trace)
        assert estimator.get_params() == params_before
        assert not hasattr(estimator, "n_features_in_")
        assert_frame_equal(X, X_before)
        assert_array_equal(y, y_before)
        assert_array_equal(weight, w_before)
    for trace in traces[1:]:
        assert len(trace["rounds"]) == len(traces[0]["rounds"])
        for current, first in zip(trace["rounds"], traces[0]["rounds"]):
            for key in ("scores", "means", "threshold"):
                assert_allclose(current[key], first[key], rtol=0, atol=0)
            for key in ("columns", "selected", "stop"):
                assert current[key] == first[key]
        for key in trace.keys() - {"rounds"}:
            assert trace[key] == traces[0][key]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "case": case,
        "shape": list(X.shape),
        "times_seconds": times,
        "median_seconds": float(np.median(times)),
        "peak_process_rss_kib": peak_rss // 1024 if sys.platform == "darwin" else peak_rss,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "lightgbm", "shap", "scikit-learn"]
        },
        "source_sha256": hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        "settings": selector.get_params(deep=False) | {"estimator": repr(estimator)},
        "trace": traces[0],
    }


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(run(sys.argv[1]), indent=2) + "\n")
