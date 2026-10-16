"""Bounded R33 experiment, run explicitly; not a calibration or speed benchmark.

Run from a copied tests directory beside an installed wheel:
python -I tests/selection_screen.py OUTPUT_DIRECTORY
"""

import contextlib
import hashlib
import importlib.metadata
import json
import random
import sys
from functools import partial
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import log_loss, mean_squared_error

from arfs.feature_selection import BoostAGroota, GrootCV, Leshy
from arfs.preprocessing import OrdinalEncoderPandas

SEEDS = range(10)
SCENARIOS = ("signal", "null", "weak", "correlated", "mixed", "weighted", "interaction")
TRAIN_ROWS = 400
TEST_ROWS = 1000


def data(task, scenario, seed):
    """Independent rows; weights depend on covariates, never the response."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(TRAIN_ROWS + TEST_ROWS, 12)),
        columns=[f"x{i}" for i in range(12)],
    )
    signal = 3 * X.x0 + 2 * X.x1
    relevant = {"x0", "x1"}
    if scenario == "null":
        signal = np.zeros(len(X))
        relevant = set()
    elif scenario == "weak":
        signal = 0.35 * X.x0 + 0.25 * X.x1
    elif scenario == "correlated":
        X["x2"] = X.x0 + 0.1 * rng.normal(size=len(X))
        relevant.add("x2")  # A redundant proxy is relevant for all-relevant selection.
    elif scenario == "mixed":
        X["x1"] = pd.Categorical(np.where(X.x1 > 0, "high", "low"))
        X["x11"] = pd.Categorical(rng.integers(0, 20, len(X)).astype(str))
        signal = 3 * X.x0 + 2 * (X.x1 == "high")
    elif scenario == "interaction":
        signal = 3 * X.x0 * X.x1
    noise = rng.normal(size=len(X))
    y = pd.Series(signal + noise)
    if task == "classification":
        probability = 1 / (1 + np.exp(-np.asarray(signal)))
        y = pd.Series(rng.binomial(1, probability))
    weights = np.where(X.x0 > 0, 4.0, 1.0) if scenario == "weighted" else None
    return X, y, weights, relevant


def estimator(task, seed):
    cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    return cls(
        n_estimators=40,
        num_leaves=7,
        max_depth=3,
        min_child_samples=15,
        n_jobs=1,
        verbosity=-1,
        random_state=seed,
    )


def selectors(task, seed):
    yield (
        "Leshy",
        Leshy(
            estimator(task, seed),
            n_estimators=40,
            max_iter=20,
            random_state=seed,
            importance="shap",
        ),
    )
    for cutoff in (1, 4):
        yield (
            f"BoostAGroota_cutoff{cutoff}",
            BoostAGroota(
                estimator(task, seed),
                cutoff=cutoff,
                iters=3,
                max_rounds=3,
            ),
        )
    yield (
        "GrootCV",
        GrootCV(
            objective="binary" if task == "classification" else "l2",
            n_folds=3,
            n_iter=1,
            n_jobs=1,
            lgbm_params={
                "num_leaves": 7,
                "max_depth": 3,
                "min_data_in_leaf": 15,
                "seed": seed,
            },
        ),
    )


def prepare(X_train, X_test):
    encoder = OrdinalEncoderPandas(return_pandas_categorical=True)
    return encoder.fit_transform(X_train), encoder.transform(X_test), encoder


def loss(task, y, prediction, weights=None):
    if task == "classification":
        return float(log_loss(y, prediction, sample_weight=weights, labels=[0, 1]))
    return float(mean_squared_error(y, prediction, sample_weight=weights))


def evaluate(task, seed, X_train, X_test, y_train, y_test, weights, columns):
    w_train, w_test = (
        (None, None)
        if weights is None
        else (
            weights[: len(X_train)],
            weights[len(X_train) :],
        )
    )
    if columns:
        model = estimator(task, seed).fit(
            X_train[columns], y_train, sample_weight=w_train
        )
        query = X_test[columns]
    else:
        cls = DummyClassifier if task == "classification" else DummyRegressor
        model = cls().fit(X_train, y_train, sample_weight=w_train)
        query = X_test
    prediction = (
        model.predict_proba(query) if task == "classification" else model.predict(query)
    )
    result = loss(task, y_test, prediction, w_test)
    assert np.isfinite(result) and result >= 0
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seeds": list(SEEDS),
        "tasks": ["regression", "classification"],
        "scenarios": SCENARIOS,
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "features": 12,
        "metric": "MSE / binary log loss, lower is better",
        "weights": "1 or 4 from x0 sign; supplied to fitting and held-out loss",
        "settings": {
            task: {name: repr(s) for name, s in selectors(task, 0)}
            for task in ("regression", "classification")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "python": sys.version,
    }
    # Write the fixed protocol before observing outcomes; there is no tuning loop.
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    with (output / "fit-log.txt").open("w") as log:
        for task in protocol["tasks"]:
            for scenario in SCENARIOS:
                for seed in SEEDS:
                    X, y, weights, relevant = data(task, scenario, seed)
                    X_train, X_test, _ = prepare(
                        X.iloc[:TRAIN_ROWS], X.iloc[TRAIN_ROWS:]
                    )
                    y_train, y_test = y.iloc[:TRAIN_ROWS], y.iloc[TRAIN_ROWS:]
                    assert X_train.index.intersection(X_test.index).empty
                    # Numeric codes here serve only this deliberately simple baseline.
                    numeric = X_train.apply(
                        lambda s: (
                            s.cat.codes
                            if isinstance(s.dtype, pd.CategoricalDtype)
                            else s
                        )
                    )
                    top2 = list(numeric.corrwith(y_train).abs().nlargest(2).index)

                    score = partial(
                        evaluate, task, seed, X_train, X_test, y_train, y_test, weights
                    )

                    baselines = {
                        "dummy": score([]),
                        "all": score(list(X)),
                        "univariate_top2": score(top2),
                        "known_signal": score([c for c in X if c in relevant]),
                    }
                    for name, selector in selectors(task, seed):
                        # BoostAGroota/GrootCV use NumPy's global shuffle stream.
                        np.random.seed(seed)
                        random.seed(seed)
                        with (
                            contextlib.redirect_stdout(log),
                            contextlib.redirect_stderr(log),
                        ):
                            selector.fit(
                                X_train.copy(deep=True),
                                y_train.copy(),
                                sample_weight=None
                                if weights is None
                                else weights[:TRAIN_ROWS],
                            )
                        selected = list(selector.selected_features_)
                        assert len(selected) == len(set(selected))
                        assert set(selected) <= set(X)
                        transformed = selector.transform(X_test)
                        assert list(transformed) == selected
                        assert transformed.index.equals(X_test.index)
                        true_count = len(set(selected) & relevant)
                        false_count = len(set(selected) - relevant)
                        records.append(
                            {
                                "task": task,
                                "scenario": scenario,
                                "seed": seed,
                                "selector": name,
                                "selected": selected,
                                "true_selected": true_count,
                                "false_selected": false_count,
                                "false_discovery_fraction": false_count
                                / max(len(selected), 1),
                                "relevant_recall": true_count / len(relevant)
                                if relevant
                                else None,
                                "loss": score(selected),
                                "baselines": baselines,
                            }
                        )
                    # Retain every completed seed even if a later fit fails.
                    (output / "results.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    print(task, scenario, seed, "passed", flush=True)
    assert len(records) == 2 * len(SCENARIOS) * len(SEEDS) * 4


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
