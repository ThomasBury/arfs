"""Compare Leshy snapshots in fresh processes, with bounded single-worker fits.

Set PYTHONPATH to a source snapshot and run:
python tests/leshy_benchmark.py tall OUTPUT.json
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
    params = {"max_depth": 3, "n_jobs": 1, "random_state": 42}
    if case == "forest":
        estimator = RandomForestRegressor(**params)
    else:
        cls = lgb.LGBMClassifier if case == "multiclass" else lgb.LGBMRegressor
        estimator = cls(**params, num_leaves=7, min_child_samples=15, verbosity=-1)
    if case == "multiclass":
        y = pd.Series(np.digitize(y, [-1, 1]))
    selector = a.Leshy(
        estimator,
        n_estimators=40,
        max_iter=4 if case == "permutation" else 20,
        random_state=42,
        importance="pimp" if case == "permutation" else "shap",
    )
    X_before, params_before = X.copy(deep=True), estimator.get_params()
    times, traces = [], []
    test_decisions, assign_hits = a.Leshy._do_tests, a.Leshy._assign_hits
    for _ in range(3):
        trace = {"thresholds": [], "hits": [], "decisions": [], "seeds": []}

        def record_hits(hits, scores, threshold, trace=trace):
            trace["thresholds"].append(float(threshold))
            return assign_hits(hits, scores, threshold)

        def record_decisions(self, decisions, hits, iteration, trace=trace):
            result = test_decisions(self, decisions, hits, iteration)
            trace["hits"].append(hits.copy().tolist())
            trace["decisions"].append(result.copy().tolist())
            trace["seeds"].append(int(self.estimator_.random_state))
            return result

        with (
            patch.object(a.Leshy, "_assign_hits", staticmethod(record_hits)),
            patch.object(a.Leshy, "_do_tests", record_decisions),
        ):
            start = perf_counter()
            selector.fit(X, y, sample_weight=weight)
            times.append(perf_counter() - start)
        trace.update(
            history=selector.imp_real_hist.tolist(),
            support=selector.support_.tolist(),
            weak=selector.support_weak_.tolist(),
            ranking=selector.ranking_.tolist(),
            absolute_ranking=selector.ranking_absolutes_,
            selected=selector.selected_features_.tolist(),
        )
        traces.append(trace)
        assert estimator.get_params() == params_before
        assert_frame_equal(X, X_before)
    for trace in traces[1:]:
        for key in ["history", "thresholds"]:
            assert_allclose(trace[key], traces[0][key], rtol=0, atol=0, equal_nan=True)
        for key in [
            "hits",
            "decisions",
            "seeds",
            "support",
            "weak",
            "ranking",
            "absolute_ranking",
            "selected",
        ]:
            assert_array_equal(trace[key], traces[0][key])
    first = traces[0]
    first["history"] = [
        [None if np.isnan(value) else value for value in row] for row in first["history"]
    ]
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
        "trace": first,
    }


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(
        json.dumps(run(sys.argv[1]), indent=2, allow_nan=False) + "\n"
    )
