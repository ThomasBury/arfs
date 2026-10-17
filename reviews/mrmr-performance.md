# Measure MRMR before changing its metric or storage

Keep Theil’s U as the default. On Adult, Cramér’s V took **0.842 s** versus
**0.601 s** at the five-fit median (1.40×, +0.241 s). The separate warmed profile
identifies contingency-table construction as its main cost. This is an optional
metric optimization candidate if a real Cramér’s V workload needs it, rather than
a reason to change the default. No predictive-quality comparison was performed.

The wide cases measure storage scaling separately. Their results and the two
explicitly skipped 20,000-feature cases are recorded below. No production code,
public API, statistical default, or storage layout changed in this phase.

## Public data: Adult income classification

Source: Becker and Kohavi (1996), [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult),
[DOI](https://doi.org/10.24432/C5XW20). Only the archive’s `adult.data` training
file was used, with SHA-256
`5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`.
Reading `?` as missing and dropping incomplete rows leaves **30,162 of 32,561
rows**, with 14 predictors: eight pandas categorical columns and six numeric
columns. `fnlwgt` remains a numeric predictor; no sample weights were supplied.
The target is categorical income; select 10 predictors without encoding them.

The comparison changes only the existing redundancy callback:

```python
partial(association_series, nom_nom_assoc=cramer_v, n_jobs=1, normalize=False)
```

The other arm uses default MRMR. Both keep default bounded classification
relevance, Spearman numeric redundancy, correlation ratio for mixed pairs,
`numpy.mean` aggregation, the 0.001 floor, and comparison across domains.
`cramer_v` is the current bias-corrected implementation. It uses the generic
callback path; default Theil’s U uses the existing prepared categorical kernel.

Each metric has one warm-up and five unprofiled fits in its own fresh process.
Separate fresh processes each warm up once and profile one more fit. Data
loading and input hashing are outside fit timing. The timed repetitions also
include selector construction. Do not compare profiled durations to unprofiled
durations as speed ratios: profiling adds overhead.

| Redundancy | Individual unprofiled fits (s) | Median (s) | Range (s) |
| --- | --- | ---: | --- |
| Default Theil’s U | 0.586436, 0.592451, 0.615129, 0.601090, 0.956449 | 0.601090 | 0.586436–0.956449 |
| Cramér’s V callback | 0.889931, 0.836822, 0.889327, 0.827639, 0.842417 | 0.842417 | 0.827639–0.889931 |

All five fits and the separate profile reproduced each metric’s selected order:

- Theil’s U: `education-num`, `sex`, `fnlwgt`, `capital-loss`, `capital-gain`,
  `race`, `marital-status`, `workclass`, `native-country`, `occupation`.
- Cramér’s V: `education-num`, `sex`, `fnlwgt`, `capital-loss`, `capital-gain`,
  `age`, `race`, `hours-per-week`, `workclass`, `native-country`.

Eight predictors overlap; order also differs. These are different redundancy
criteria, so later pairs evaluated differ too. The timing difference is for the
complete workflows, not identical pair workloads. It does not establish better
selection or predictive quality.

Adult’s accounted input is 1,728,679 bytes, its retained redundancy table is
2,945 bytes, and the timing workers peak at 310.45 MiB (Theil) and 310.81 MiB
(Cramér). This narrow input does not put meaningful pressure on table storage.

The warmed Cramér’s V profile took **1.263 s**; default MRMR’s separate profile
took **0.861 s**. Cumulative times below include nested work and must not be added:

| Function | Calls | Cumulative seconds | Share of profiled Cramér fit |
| --- | ---: | ---: | ---: |
| `cramer_v` | 21 | 0.480431 | 38.0% |
| pandas `crosstab` | 21 | 0.303939 | 24.1% |
| pandas `pivot_table` (inner implementation) | 21 | 0.266 | 21.1% |
| SciPy `chi2_contingency` | 21 | 0.015622 | 1.2% |

`crosstab` accounts for 63.3% of `cramer_v` cumulative time. Its pivot/grouping
and factorization work is the useful place to investigate if this optional
metric becomes important. The current workflow still fits in under a second at
both unprofiled medians, so there is no runtime pass/fail gate or immediate
optimization requirement. The existing callback also evaluates self-pairs;
this benchmark preserves that behavior.

## Synthetic data: memory scaling

The fixed generator uses `numpy.random.default_rng(20261016)`, 2,000 rows,
float64 standard-normal numeric predictors, and, in mixed cases, exactly 10%
categorical predictors with 20 integer levels stored as pandas categoricals.
The binary classification target is the sign of the first numeric predictor
plus independent standard-normal noise. Default MRMR selects 20 predictors.
These inputs establish scaling, not production runtime or predictive quality.

| Input | Features | Input (MiB) | Retained table (MiB) | Peak RSS (GiB) | Fit (s) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Numeric | 1,000 | 15.26 | 7.71 | 0.404 | 20.13 | Complete |
| 10% categorical | 1,000 | 13.99 | 7.71 | 0.395 | 18.56 | Complete |
| Numeric | 5,000 | 76.30 | 191.13 | 1.009 | 138.53 | Complete |
| 10% categorical | 5,000 | 69.96 | 191.13 | 0.944 | 144.63 | Complete |
| Numeric | 10,000 | 152.59 | 763.73 | 2.098 | 242.86 | Complete |
| 10% categorical | 10,000 | 139.92 | 763.73 | 2.004 | 223.10 | Complete |
| Numeric | 20,000 | — | — | — | — | Skipped: preflight |
| 10% categorical | 20,000 | — | — | — | — | Skipped: preflight |

All six completed wide fits stayed below the safety cutoff. At 10,000 features,
the table is about 0.75 GiB, compared with 2.10 GiB numeric and 2.00 GiB mixed
process peaks. A table-only accounting would substantially understate total
memory. These runs did not demonstrate actual budget pressure at a tested size.

The provisional total worker-process budget is **4 GiB**, including imports,
inputs, allocator retention, and temporary allocations. The supervisor polls
Linux RSS/high-water memory every 50 ms, stops at 90% of the budget (3.6 GiB),
and stops when machine/cgroup headroom falls below 512 MiB. Preflight estimates
`8*p**2 + 3*2000*p*8 + 512 MiB`, modeling mixed inputs as float64 too. This is a
heuristic, not an upper bound: the 10,000-feature numeric peak was 2.10 GiB
versus a 1.69 GiB forecast. Runtime monitoring is still necessary, and sampling
cannot guarantee protection against an arbitrarily fast allocation spike.

Input bytes mean `X.memory_usage(index=True, deep=True).sum()` plus the same
accounting for `y`. Retained table bytes use the DataFrame’s same accounting,
including its row index but excluding its column index and other selector state.
Peak RSS comes from Linux `ru_maxrss` through the end of measurement, cross-checked
against `/proc` high-water readings; it includes the scientific stack and benchmark
hashing. It is not a Python-only allocation trace, and it excludes the small
supervisor process. Each wide case runs once in a fresh worker; it is not a
five-repetition timing comparison.

MRMR allocates the entire eligible-feature square, even when selecting only 20.
The values alone require `8*p**2` bytes: 7.63 MiB at 1,000 features, 190.73 MiB
at 5,000, 762.94 MiB at 10,000, and **2.98 GiB at 20,000**. The last size leaves
only 1.02 GiB within the 4 GiB budget for everything else. The preflight estimate
is 4.37 GiB, beyond the 3.6 GiB safety cutoff, so neither 20,000-feature case
is launched. Their input/table measurements and fit times are unavailable;
the theoretical table size is not an observed memory failure.

Do not redesign storage from these skipped cases alone. The measured table
growth is quadratic, but peak RSS also includes substantial linear input/copy
cost and imports. Revisit storage when a representative workload reaches actual
memory pressure and a measured allocation breakdown attributes it to this table.
The narrow Adult workload cannot settle that decision.

## Reproduction and verification

The manual script is [tests/mrmr_performance.py](../tests/mrmr_performance.py);
it adds no dependencies and is not collected by pytest. The local raw
measurements in `evidence/mrmr-performance/measurements.json` contain individual fits,
input/target hashes, selections, source and script hashes, lock hash, threadpool
details, memory, settings, and skip reasons. The cumulative profile listings are
`evidence/mrmr-performance/adult-cramer-profile.txt` and
`evidence/mrmr-performance/adult-theil-profile.txt`. These raw files stay ignored
and are not part of the PR; the commands below reproduce the measurements.

Measured source: `4fc3c7a52340a249a823a7982e0ec222df413fef`, ARFS 4.0.1,
CPython 3.12.7 on Linux x86-64, Intel i5-7300U. NumPy 2.0.2, pandas 2.2.3,
SciPy 1.13.1, scikit-learn 1.6.1, joblib 1.4.2; further dependency versions are
in the raw provenance. The source checkout was used via `PYTHONPATH=src` in the
existing project environment. This was a shared desktop run, without CPU
isolation; retain the slower Theil repetition rather than discarding it.
No test suites ran concurrently with the measurements.
After measurement, the runner’s forecast wording changed from “conservative”
to “heuristic” in its comment and skip message only. The raw provenance keeps
the measured script hash, delivered hash, and exact wording replacements;
all calculations and settings are unchanged.

The project environment had no pytest. Contracts used the existing
`/tmp/arfs-association/py312-locked/bin/python` environment, with the same Python,
NumPy, pandas, SciPy, and scikit-learn versions, importing the current checkout.
For another machine, use an environment with the project’s declared `test` group.

```sh
curl --fail --location https://archive.ics.uci.edu/static/public/2/adult.zip -o /tmp/adult.zip
unzip -p /tmp/adult.zip adult.data > /tmp/adult.data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
.venv/bin/python tests/mrmr_performance.py --self-check
.venv/bin/python tests/mrmr_performance.py /tmp/arfs-mrmr-new --adult-data /tmp/adult.data
/tmp/arfs-association/py312-locked/bin/python -m pytest tests/test_mrmr_ranking.py tests/test_association_kernels.py tests/test_statistical_arithmetic.py tests/test_observation_boundaries.py tests/test_category_identity.py tests/test_parallel_contracts.py -q --tb=short -p no:cacheprovider
```

Each fit uses `n_jobs=1`; native pools were checked to have one thread. The
self-check exercises both metrics twice, checks selected counts and rankings,
verifies input/target preservation with pandas assertions, regenerates the same
synthetic input, and forces categorical comparisons to prove the Cramér callback
executes. Full cases additionally hash inputs before/after fitting and assert
the selected count; repeated Adult fits assert identical selections. Existing
Theil rounding warnings are retained in local logs rather than changing the
production warning policy.

Verification: **502 contract tests passed** (218 warnings) in 74.86 s. The
benchmark self-check, Ruff 0.17.0 lint/format checks, source/evidence
reconciliation, local-link checks, and whitespace checks passed. The full
repository suite, documentation builds, and a new installed-wheel matrix were
not rerun for this benchmark-only phase.

Not measured: predictive quality, weighted Adult fits, larger categorical
cardinality, parallel execution, production workloads, or alternative storage.
