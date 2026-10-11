"""Bounded distance allocation measurements and notebook source evidence."""

import json
import time
import tracemalloc
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from arfs.sampling import gower_matrix

records = []
for n in [100, 200, 400]:
    X = pd.DataFrame({"x": np.linspace(0., 10., n),
                      "c": pd.Series(np.where(np.arange(n) % 2, "a", "b"), dtype=object)})
    tracemalloc.start()
    started = time.perf_counter()
    result = gower_matrix(X)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.shape == (n, n) and result.dtype == np.float32
    assert result.nbytes == 4 * n * n
    assert np.isfinite(result).all() and np.allclose(result, result.T)
    records.append({"rows": n, "seconds": elapsed, "matrix_bytes": result.nbytes,
                    "tracemalloc_peak_bytes": peak})

repo = Path(__file__).resolve().parents[3]
notebook = json.loads((repo / "docs/notebooks/arfs_large_data_sampling.ipynb").read_text())
cells = {str(i): "".join(cell["source"]) for i, cell in enumerate(notebook["cells"])
         if i in [0, 14, 15, 16, 17]}
assert "data_dum = data.sample(n=10_000)\ndata_dum = data.copy()" in cells["15"]
Path(sys.argv[1]).write_text(json.dumps({
    "measurements": records,
    "limits": "Single measurements, no throughput claim; tracemalloc is not process RSS; clustering not timed",
    "projected_matrix_bytes_only_not_allocated": {str(n): 4 * n * n for n in [10_000, 20_640, 100_000]},
    "notebook_cells": cells,
}, indent=2) + "\n")
