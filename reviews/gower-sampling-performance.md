# Exact Gower sampling optimization

**Status: Implemented locally on `perf/gower-condensed-sampling`, based on
`efc51fb` on updated main.** This is the separately authorized optimization
following C06; C06's original documentation-only handoff remains historical.

## Change and boundaries

`sample(method="gower")` keeps its existing imputation, equal feature weights,
exact complete linkage and synthetic column means/modes. Its distance calculation
now combines SciPy's native Manhattan distances on range-normalized numeric
features with Hamming mismatch counts on encoded nominal features. Constant
numeric features contribute zero and remain in the denominator. Infinite
numeric values and overflowing ranges still fail before clustering. Numeric
normalization works on a copy, preserving representative values and caller data.

Only unique row pairs are retained. SciPy complete linkage consumes that
condensed vector directly, avoiding the old square matrix and the two large
triangular-index arrays created by scikit-learn's conversion. A small heap-based
tree cut preserves the existing cluster IDs and output row order for an
unchanged merge tree, without using private dependency APIs.

The public `gower_matrix`, nearest-neighbor and isolation functions retain their
executable code. In particular, feature weights and pairwise missingness still
use per-pair scaling, including tiny weights when dominant features are missing.
No dependency, public parameter, approximation or runtime row cap was added.

The distance definition is unchanged, but native floating-point reduction order
can change nearly tied distances and therefore cluster membership. Exact
categorical mismatch counts are recovered before combining contributions, to
avoid introducing rounding differences for purely categorical ties. The docs
and changelog ask users to recheck downstream results.

## Measured full-sampler results

These measurements include distance preparation, clustering, tree cutting and
representative aggregation. Each row compares fresh Linux processes running
three sequential calls, using their median elapsed time and whole-process peak
resident memory (RSS, including imports and temporary allocations). Numeric,
mixed and nominal cases have 10/0, 8/2 and 0/10 numeric/nominal features,
respectively. All use seed 42 and return 100 representatives.

| Input rows | Features | Dense → condensed median (s) | Speedup | Dense → condensed peak RSS (MiB) |
| ---: | --- | ---: | ---: | ---: |
| 2,000 | Numeric | 1.309 → 0.230 | 5.7× | 285.6 → 217.4 |
| 2,000 | Mixed | 1.229 → 0.224 | 5.5× | 285.8 → 217.7 |
| 2,000 | Nominal | 2.085 → 0.212 | 9.9× | 283.4 → 217.0 |
| 4,000 | Numeric | 4.879 → 0.732 | 6.7× | 571.3 → 317.1 |
| 4,000 | Mixed | 4.972 → 1.192 | 4.2× | 570.8 → 317.3 |
| 4,000 | Nominal | 7.263 → 0.637 | 11.4× | 569.7 → 316.0 |

All six before/after output hashes match, including cluster IDs and row order.
The dense reference in [the regression suite](../tests/test_gower_sampling.py)
reproduces the original imputation, public Gower calculation, scikit-learn
complete linkage and aggregation. Both variants run against the same source
and dependencies; the benchmark ran before the concurrent validation jobs.

Retained distance values fall from 32,000,000 to 15,992,000 bytes at 2,000 rows,
and from 128,000,000 to 63,984,000 bytes at 4,000 rows. This is separate from
whole-process peak memory. Storage remains quadratic: `4 * N * (N - 1)` bytes,
about 400 MB / 10 GB / 40 GB at 10,000 / 50,000 / 100,000 rows, before
clustering and other temporary memory. Those large sizes were calculated, not
allocated. Direct `gower_matrix` calls still return `8 * n_x * n_y` bytes.

Environment: Linux x86-64, Intel i5-7300U, Python 3.12.7, NumPy 2.0.2,
pandas 2.2.3, SciPy 1.13.1 and scikit-learn 1.6.1. From the repository root,
with uv 0.12.15 and a system Pandoc executable installed, create a disposable
environment from the committed lockfile. The docs group supplies Sphinx for
the documentation check below:

```sh
env UV_PROJECT_ENVIRONMENT=/tmp/arfs-gower-repro \
  uv sync --locked --python 3.12.7 --group test --group docs
```

Then reproduce a benchmark pair in two fresh processes:

```sh
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/arfs-gower-repro/bin/python \
  tests/gower_sampling_performance.py --rows 4000 --kind mixed --dense
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/arfs-gower-repro/bin/python \
  tests/gower_sampling_performance.py --rows 4000 --kind mixed
```

Use `--kind numeric` or `--kind nominal` and `--rows 2000` for the other cases.
The script bounds its workload to 100–4,000 rows and checks a small dense
reference before timing. It asserts no performance threshold. Raw records stay
in `/tmp/arfs-gower-native/benchmark.jsonl`.

## Validation and limits

The maintained suite adds 50 cases for dense-reference outputs and cluster IDs,
all cluster counts in tied trees, numeric constants and extreme scales,
interleaved mixed features, nullable columns, category identity, input ownership,
invalid numeric values and condensed-only clustering. The integer-column caller
regression now checks the new condensed path against its scalar reference.

- Full locked Python 3.12 source suite: **2,095 passed**, 561 warnings, 255.93 s.
- Affected suites on locked Python 3.9.25: **238 passed**, 21.64 s, using
  NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1, scikit-learn 1.6.1 and pytest 8.3.5.
- Affected suites on newer Python 3.12.7: **238 passed**, 24.24 s, using
  NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1 and pytest 9.1.1.
- Strict HTML build passes with Sphinx 6.2.1 and notebook execution off.
  Rendered API, workflow and notebook pages contain the revised memory numbers.
- Ruff 0.17.0 lint and formatting, whitespace and local report links pass.
  AST comparison confirms existing functions other than `sample` have unchanged
  executable bodies. Notebook comparison confirms one Markdown cell and one code
  comment changed; executable tokens, metadata and saved outputs are unchanged.

Rerun the full suite in the recreated locked environment:

```sh
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/arfs-gower-native/mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0 \
  /tmp/arfs-gower-repro/bin/python -m pytest tests \
  -q --tb=short -p no:cacheprovider
```

Recreate the compatibility environments separately:

```sh
env UV_PROJECT_ENVIRONMENT=/tmp/arfs-gower-repro-py39 \
  uv sync --locked --python 3.9.25 --group test
uv venv --python 3.12.7 /tmp/arfs-gower-repro-newer
uv pip install --python /tmp/arfs-gower-repro-newer/bin/python \
  -r .github/requirements/runtime-newer-py312.txt
```

Run pytest with the same thread/source settings using
`/tmp/arfs-gower-repro-py39/bin/python` or
`/tmp/arfs-gower-repro-newer/bin/python` and the four files
`tests/test_gower_sampling.py`, `tests/test_gower_distances.py`,
`tests/test_sampling_outputs.py` and `tests/test_nearest_neighbors.py`.
The HTML command uses `/tmp/arfs-gower-repro/bin/python -m sphinx
-W --keep-going -b html docs /tmp/arfs-gower-native/html`, with `PYTHONPATH=src`
and the same headless plotting/thread settings. Logs and rendered HTML stay
under `/tmp/arfs-gower-native/`.

No huge allocations, full installed-wheel matrix, optional-backend execution,
full notebook execution or downstream predictive-quality study was performed.
The observed speedups apply to these synthetic workloads; they do not establish
performance or statistical superiority over random sampling.
