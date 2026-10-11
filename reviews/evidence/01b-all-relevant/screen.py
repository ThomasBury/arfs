"""Small reproducible selector screen; five seeds are not a calibration study.

Run against the unit 0 frozen wheel. The output directory is the only argument.
Each scenario uses independent training and test observations. Test outcomes are
never passed to selectors, relevance ranking, or model fitting.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import arfs
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, r2_score

from arfs.feature_selection import BoostAGroota, GrootCV, Leshy


def estimator(task, seed):
    cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    return cls(n_estimators=30, num_leaves=7, max_depth=3, min_child_samples=15,
               n_jobs=1, verbosity=-1, random_state=seed)


def selectors(task, seed):
    yield "Leshy", Leshy(estimator(task, seed), n_estimators=30, max_iter=20,
                         random_state=seed, importance="shap")
    for cutoff in (1, 4):
        yield f"BoostAGroota_cutoff{cutoff}", BoostAGroota(
            estimator(task, seed), cutoff=cutoff, iters=3, max_rounds=3)
    yield "GrootCV", GrootCV(objective="binary" if task == "classification" else "l2",
                            n_folds=3, n_iter=1, n_jobs=1,
                            lgbm_params={"num_leaves": 7, "max_depth": 3,
                                         "min_data_in_leaf": 15, "seed": seed})


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    installed = Path(arfs.__file__).parent
    hashes = {}
    for filename in ("feature_selection/allrelevant.py", "utils.py"):
        source = (repo / "src/arfs" / filename).read_bytes()
        wheel = (installed / filename).read_bytes()
        assert source == wheel, f"Wheel differs from checkout: {filename}"
        hashes[filename] = {"sha256": hashlib.sha256(wheel).hexdigest(),
                            "installed_path": str(installed / filename),
                            "matches_checkout": True}
    versions = dict(sorted((d.metadata["Name"], d.version)
                           for d in importlib.metadata.distributions()))
    environment = {"review_date": "2026-10-11", "commit": "03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2",
                   "python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "versions": versions,
                   "source_hashes": hashes, "seeds": list(range(5)),
                   "thread_limits": {k: os.environ.get(k) for k in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                      "NUMBA_NUM_THREADS", "LOKY_MAX_CPU_COUNT")},
                   "workaround": "setuptools==80.9.0 from unit 0; no ARFS source patch",
                   "environment_reused_from_unit_0": True}
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (output / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items()) + "\n")

    records = []
    with (output / "screen-log.txt").open("w") as log:
        for task in ("regression", "classification"):
            for scenario in ("signal", "null"):
                for seed in range(5):
                    rng = np.random.RandomState(seed)
                    X = pd.DataFrame(rng.normal(size=(1500, 8)), columns=[f"x{i}" for i in range(8)])
                    raw = (3 * X.x0 + 2 * X.x1 + rng.normal(size=len(X)) if scenario == "signal"
                           else pd.Series(rng.normal(size=len(X))))
                    y = (raw > 0).astype(int) if task == "classification" else raw
                    X_train, X_test = X.iloc[:300], X.iloc[300:]
                    y_train, y_test = y.iloc[:300], y.iloc[300:]
                    score = accuracy_score if task == "classification" else r2_score
                    dummy_cls = DummyClassifier if task == "classification" else DummyRegressor
                    dummy = dummy_cls().fit(X_train, y_train)
                    dummy_score = float(score(y_test, dummy.predict(X_test)))

                    def evaluate(columns):
                        if not columns:
                            return dummy_score
                        fitted = estimator(task, seed).fit(X_train[columns], y_train)
                        return float(score(y_test, fitted.predict(X_test[columns])))

                    univariate = list(X_train.corrwith(y_train).abs().nlargest(2).index)
                    baselines = {"dummy": dummy_score, "all": evaluate(list(X)),
                                 "univariate_top2": evaluate(univariate),
                                 "oracle": evaluate(["x0", "x1"]) if scenario == "signal" else dummy_score}
                    for name, selector in selectors(task, seed):
                        np.random.seed(seed)
                        random.seed(seed)
                        started = time.perf_counter()
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            selector.fit(X_train.copy(deep=True), y_train.copy())
                        selected = list(selector.selected_features_)
                        assert len(selected) == len(set(selected))
                        assert set(selected) <= set(X)
                        relevant = {"x0", "x1"} if scenario == "signal" else set()
                        record = {"task": task, "scenario": scenario, "seed": seed,
                                  "selector": name, "selected": selected,
                                  "true_selected": len(set(selected) & relevant),
                                  "false_selected": len(set(selected) - relevant),
                                  "test_score": evaluate(selected), "baselines": baselines,
                                  "selection_and_evaluation_seconds": time.perf_counter() - started}
                        assert np.isfinite(record["test_score"])
                        records.append(record)
                        print(task, scenario, seed, name, selected, flush=True)
    assert len(records) == 80
    (output / "screen.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
