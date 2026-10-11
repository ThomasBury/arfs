"""Small seeded screen of the unmodified selector; not a calibration study.

Usage: python screen.py OUTPUT.json
Data generation, selection and all refits use independent train/test samples.
Test responses are used only to evaluate final predictions.
"""
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arfs.feature_selection.lasso import LassoFeatureSelection


def make_data(seed, family, signal):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(1180, 6)), columns=[f"x{i}" for i in range(6)])
    w = rng.uniform(.25, 2, len(X))
    eta = (.8 * X.x0 - .5 * X.x1).to_numpy() if signal else np.zeros(len(X))
    if family == "gaussian":
        y = 2 + eta + rng.normal(size=len(X)) / np.sqrt(w)
    elif family == "poisson":
        y = rng.poisson(w * np.exp(.6 + eta)) / w
    else:
        y = rng.binomial(1, 1 / (1 + np.exp(.3 - 2 * eta))).astype(float)
    return X.iloc[:180], X.iloc[180:], y[:180], y[180:], w[:180], w[180:]


def objective(family):
    return {"gaussian": sm.families.Gaussian, "poisson": sm.families.Poisson,
            "binomial": sm.families.Binomial}[family]()


def design(X, names):
    return np.column_stack([np.ones(len(X)), X[list(names)].to_numpy()])


def fit_glm(X, y, w, names, family):
    return sm.GLM(y, design(X, names), var_weights=w, family=objective(family)).fit()


def loss(y, mu, w, family):
    return float(objective(family).deviance(y, mu, var_weights=w) / w.sum())


def main(destination):
    results = []
    for seed in range(5):
        for family in ["gaussian", "poisson", "binomial"]:
            for signal in [True, False]:
                X, Xt, y, yt, w, wt = make_data(seed, family, signal)
                truth = ["x0", "x1"] if signal else []
                marginal = sorted(X.columns, key=lambda name: fit_glm(X, y, w, [name], family).deviance)[:2]
                baselines = {}
                for label, names in [("all", list(X)), ("dummy", []),
                                     ("oracle", truth), ("univariate_top2", marginal)]:
                    model = fit_glm(X, y, w, names, family)
                    baselines[label] = {"features": names,
                                        "test_deviance_per_weight": loss(yt, model.predict(design(Xt, names)), wt, family)}
                for score in ["bic", "deviance"]:
                    row = {"seed": seed, "family": family, "signal": signal,
                           "score": score, "truth": truth, "baselines": baselines}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            selector = LassoFeatureSelection(family=family, score=score,
                                                             n_iterations=7, n_jobs=1).fit(X, y, w)
                            names = [str(name) for name in selector.selected_features_ if name != "Intercept"]
                            selected_glm = fit_glm(X, y, w, names, family)
                            row.update({"features": names, "alpha": float(selector.best_estimator_.alpha),
                                        "intercept_selected": "Intercept" in selector.selected_features_,
                                        "true_selected": len(set(names) & set(truth)),
                                        "false_selected": len(set(names) - set(truth)),
                                        "test_deviance_per_weight": loss(yt, selected_glm.predict(design(Xt, names)), wt, family),
                                        "wrapper_test_deviance_per_weight": loss(yt, selector.best_estimator_.predict(Xt), wt, family)})
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                    row["warnings"] = dict(Counter(f"{type(w.message).__name__}: {w.message}" for w in caught))
                    results.append(row)
                    Path(destination).write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
                    print(seed, family, signal, score, row.get("features", row.get("error")), flush=True)
    assert len(results) == 60


if __name__ == "__main__":
    main(sys.argv[1])
