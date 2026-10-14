# R26 — Bounded isolation attempts

**R26 complete on 2026-10-14**, based on R25 commit `b352069`.
Branch: `fix/r25-r26-remediation`. This is the second and final phase of the
requested sequential R25 → R26 batch. Local phase commit: `5c631bf`.

## Changes and decisions

The sampler now fits its forest once and tries 20 random draws without replacement
at each distinct proper-subset size. Sizes are 5%, 10%, …, 95% of the input rows,
rounded up and capped at N-1; duplicate sizes for small inputs are tried once.
The first draw is at 5%, rounded up to at least one row. For 100 rows the schedule
is 5, 10, …, 95; for ten rows it is 1, 2, …, 9. There are at most 19 sizes and
380 attempts. The retry budget is per size, not a global 20-attempt limit.

The first KS **p-value >0.95** accepts the candidate. On exhaustion, all original
row positions are returned in order, so callers receive the full original data.
This avoids silently returning a rejected subset or dropping observations.
Empty/single-row isolation inputs return a copy unchanged without fitting or
running KS. The direct helper returns empty/[0] positional arrays respectively.
Existing feature-schema and observation-weight validation still apply; an empty
input uses no weights, and a singleton needs positive weight mass if supplied.
Gower's explicit tiny-input rejection remains the R25 contract.

`get_5_percent_splits` requires a positive integer length; length one returns an
empty integer array because it has no proper subset. Zero/negative/noninteger
and Boolean lengths raise a clear ValueError. The direct isolation helper checks
a two-dimensional predictor matrix with features before its identity branch.
No random_state, retry option, dependency or alternate sampling algorithm is added.
The existing NumPy/global forest random state remains in use.

Forest fit weights retain R25's positional validation and zero-row omission.
The candidate draws, scored population and KS comparison remain unweighted.
The API and the sampling notebook now explain that the subset overlaps the full
anomaly-score population, contrary to KS's independent-sample assumption. A high
p-value is failure to reject a score-distribution difference, not a 95% guarantee
of representativeness. Repeated screening and score matching do not establish
preservation of feature/target relationships or minority groups. Random subset
selection is not an outlier-removal rule. The native threshold is retained as a
heuristic; changing its statistical basis remains outside this repair.

Caller tracing covers get_5_percent_splits, isof_find_sample, isolation_forest,
sample, maintained/historical tests, API autodoc and notebook source. Only the
sampling notebook's relevant markdown cell changes; its other cells, metadata
and saved outputs are unchanged. Its presampling overwrite is still R32.
Historical review reports and scripts are preserved. New evidence remains local
and ignored under `reviews/evidence/`; the regression tests and handoffs are tracked.

## Validation

The 29 maintained R26 cases cover exact size schedules, strict threshold equality,
20 attempts per size, forced exhaustion with a 380-call guard, one forest fit,
input validation, tiny identities, original-row fallback and real constant/weighted
forest runs. On the old R25 installed wheel, the 28 safe cases produce **25 failures
and three passing controls**. The real nonconstant weighted replay case is excluded
from the old-wheel run because the old retry loop has no effective bound.

Affected source checks pass **218 tests** in the locked snapshot:

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  /tmp/arfs-r25-final/py312-locked/bin/python -m pytest \
  tests/test_isolation_sampling.py tests/test_sampling_outputs.py \
  tests/test_gower_distances.py tests/test_category_identity.py \
  -q --tb=short -p no:cacheprovider
```

All **79** sampling checks pass with the newer source snapshot using
`/tmp/arfs-r25-final/py312-newer/bin/python` and the first two test files above.
Installed-wheel full-suite results: **1256 / 1256 / 1240 passed** in
py39-locked / py312-locked / py312-newer. All **four** selected unchanged
historical retry/tiny-split checks pass unmasked in both Python 3.12 cells.
The matrix is Linux Python 3.9.25 locked and 3.12.7 locked/newer.
Locked: NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1, sklearn 1.6.1.
Newer: NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, sklearn 1.9.1.

```sh
uv build --python /tmp/arfs-r25-final/py312-locked/bin/python --out-dir /tmp/arfs-r26/dist
/tmp/arfs-r25-final/py312-locked/bin/python /tmp/arfs-r26/check_payloads.py
/tmp/arfs-r25-final/py312-locked/bin/python /tmp/arfs-r26/validate.py py39-locked
/tmp/arfs-r25-final/py312-locked/bin/python /tmp/arfs-r26/validate.py py312-locked
/tmp/arfs-r25-final/py312-locked/bin/python /tmp/arfs-r26/validate.py py312-newer
uv lock --check --python /tmp/arfs-r25-final/py312-locked/bin/python
uv tool run --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
uv tool run --from ruff==0.17.0 ruff check src/arfs/sampling.py tests/test_sampling_outputs.py tests/test_isolation_sampling.py
uv tool run --from ruff==0.17.0 ruff format --check src/arfs/sampling.py tests/test_sampling_outputs.py tests/test_isolation_sampling.py
/tmp/arfs-r25-final/py312-locked/bin/python -m unittest discover -s .github/scripts -p 'test_*.py'
git diff --check
```

The runner reuses R25's locked export or the checked-in newer requirements;
exact argv/cwd/exit codes and version/hash identities are in ignored local
`reviews/evidence/r26-isolation-attempts/`. Source/sdist/wheel payloads agree
(19 files), with no setuptools workaround. All 13 installed source-module hashes
and copied test bytes are checked. Dependency, lock, Ruff (331 total, zero new),
gate unit tests (two passed), changed lint/format and whitespace pass.
Notebook JSON parses, and a structural comparison confirms only its one markdown
cell changed. Context7's [SciPy KS documentation](https://github.com/scipy/scipy/blob/main/doc/source/tutorial/stats/comparing_two_samples.rst)
was consulted for the independent continuous-sample assumptions and p-value meaning.

Hosted CI, full docs/notebook execution, optional backends, other platforms,
rare-group/large-data sampling quality, throughput and calibration are not checked.
The bound counts attempts, not wall-clock time; forest fitting and KS still cost
work proportional to their inputs. Exhaustion can return the entire dataset.
Passing the sampler checks does not establish release readiness.

## Resume

Stop after the local R26 phase commit. Next is **R27 alone**, with R24 complete:
trace gower_topn/smallest_indices and their key/query/count/nonfinite-distance
contracts before implementing. R28–R30, R32 and R33 also remain pending; R31 is
already complete. No publication, version bump or release is part of this batch.
