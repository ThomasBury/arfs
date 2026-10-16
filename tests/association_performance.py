"""Manual wheel benchmark; run outside the checkout with controlled native threads.

Example: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -I association_performance.py results.json
Not collected by pytest. Compare saved values as well as timings between wheels.
"""

import hashlib
import json
import os
import platform
import sys
import warnings
from importlib import metadata
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from arfs import association as a


def main(output):
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        assert os.environ.get(name) == "1", f"Set {name}=1 before starting Python"
    rng = np.random.default_rng(20261016)
    numeric = pd.DataFrame(rng.normal(size=(2000, 24)))
    categorical = pd.DataFrame(
        rng.integers(0, 40, size=(2000, 10)).astype(str), columns=list("abcdefghij")
    )
    mixed = pd.concat([categorical, numeric.iloc[:, :10]], axis=1)
    weights = rng.uniform(0.2, 2, len(numeric))
    workloads = {
        "pearson-matrix": (a.wcorr_matrix, numeric, {"method": "pearson"}),
        "spearman-matrix": (a.wcorr_matrix, numeric, {"method": "spearman"}),
        "pearson-series": (a.wcorr_series, numeric, {"target": 0, "method": "pearson"}),
        "spearman-series": (a.wcorr_series, numeric, {"target": 0}),
        "theil-matrix": (a.theils_u_matrix, categorical, {}),
        "theil-series": (a.theils_u_series, categorical, {"target": "a"}),
        "ratio-matrix": (a.correlation_ratio_matrix, mixed, {}),
        "ratio-series": (a.correlation_ratio_series, mixed, {"target": 0}),
    }
    record = {
        "python": platform.python_version(),
        "versions": {
            n: metadata.version(n)
            for n in ["arfs", "numpy", "pandas", "scipy", "joblib"]
        },
        "module": str(Path(a.__file__).resolve()),
        "sha256": hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        "seed": 20261016,
        "numeric_shape": list(numeric.shape),
        "categorical_shape": list(categorical.shape),
        "categories": 40,
        "threads": {
            n: os.environ.get(n)
            for n in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "LOKY_MAX_CPU_COUNT",
            ]
        },
        "results": {},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, (fn, data, kwargs) in workloads.items():
            for jobs in [1, 2]:
                fn(data, sample_weight=weights, n_jobs=jobs, **kwargs)
                times = []
                for _ in range(5):
                    start = perf_counter()
                    result = fn(data, sample_weight=weights, n_jobs=jobs, **kwargs)
                    times.append(perf_counter() - start)
                values = (
                    result.val.to_numpy()
                    if isinstance(result, pd.DataFrame)
                    else result.to_numpy()
                )
                key = f"{name}/{jobs}"
                record["results"][key] = {
                    "seconds": times,
                    "median": float(np.median(times)),
                    "range": [min(times), max(times)],
                    "values": values.tolist(),
                    "labels": repr(list(zip(result.row, result.col)))
                    if isinstance(result, pd.DataFrame)
                    else repr(result.index.tolist()),
                }
                print(key, record["results"][key]["median"], flush=True)
    Path(output).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
