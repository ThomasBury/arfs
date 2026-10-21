"""Manual bounded Gower benchmark, with a small dense-reference self-check.

Run each variant in a fresh Linux process to compare whole-process peak RSS:
    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python tests/gower_sampling_performance.py --rows 2000 --kind mixed
Add --dense for the previous square-matrix implementation. No timing assertions.
"""

import argparse
import hashlib
import importlib.metadata
import json
import resource
import statistics
import time

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from test_gower_sampling import dense_sample

from arfs.sampling import sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument(
        "--kind", choices=["numeric", "mixed", "nominal"], default="mixed"
    )
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()
    if not 100 <= args.rows <= 4000:
        parser.error("this bounded benchmark requires 100 <= rows <= 4000")
    rng = np.random.default_rng(42)
    n_numeric = {"numeric": 10, "mixed": 8, "nominal": 0}[args.kind]
    X = pd.DataFrame(
        rng.normal(size=(args.rows, n_numeric)),
        columns=[f"x{i}" for i in range(n_numeric)],
    )
    for i in range(10 - n_numeric):
        X[f"c{i}"] = rng.choice(["a", "b", "c", "d"], size=args.rows)
    small = X.head(20)
    assert_frame_equal(sample(small, n=5), dense_sample(small, n=5))
    calculate = dense_sample if args.dense else sample
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        result = calculate(X, n=100)
        timings.append(time.perf_counter() - start)
        assert result.shape == (100, 10)
    print(
        json.dumps(
            {
                "method": "dense" if args.dense else "condensed",
                "rows": args.rows,
                "kind": args.kind,
                "seconds": timings,
                "median_seconds": statistics.median(timings),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "retained_distance_bytes": 8 * args.rows**2
                if args.dense
                else 4 * args.rows * (args.rows - 1),
                "output_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(result).values.tobytes()
                ).hexdigest(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scipy", "scikit-learn")
                },
            }
        )
    )


if __name__ == "__main__":
    main()
