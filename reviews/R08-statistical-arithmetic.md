# R08 — Weighted association and relevance arithmetic

**R08 complete on 2026-10-12. Next group: R09 alone.**
Based on `170b3e95fee635e8243ed8cdc740ecb80c4738ad` on `main`; changes remain
uncommitted. Pre-existing R04–R07 work and historical review evidence are
preserved. No commit, push, version bump or release was performed.

## Changes and decisions

The relevance APIs use **frequency weight mass**, preserving the existing
grouped ANOVA meaning. Weights are positional and never automatically
normalized. N is total retained weight, after the existing complete-case policy,
and zero-weight rows do not contribute groups or constant/nonconstant status.
Integer counts reproduce literal row replication. Fractional mass is accepted
for scoring, but is not literal replication or a claim of calibrated inference.
Relative/exposure weights remain accepted by association coefficients; they do
not give scale-invariant relevance scores. This is a deliberate API/statistical
choice, documented in `association.py` and MRMR's fit weight parameter.

Numeric regression now uses Pearson correlation and
`F = r² / (1 - r²) * (N - 2)`. It rejects N <= 2. The original response length
does not count dropped observations. Roundoff above squared correlation one
cannot produce negative F scores. Constant predictors/responses produce NaN
with `force_finite=False` and zero with its default True; perfect nonconstant
dependence produces infinity or maximum float, respectively. These are score
outputs, not returned p-values.

Grouped ANOVA uses `F = (SS_between / (K - 1)) / (SS_within / (N - K))`, with K
positive-mass groups. Its degrees of freedom follow the
[standard ANOVA calculation](https://www.itl.nist.gov/div898/handbook/prc/section4/prc433.htm).
It rejects fewer than two groups and N <= K rather than returning negative or
undefined-degree statistics. The direct API preserves its one-element array:
constant observations return NaN with the existing constant warning; distinct
constant group levels return infinity. Categorical regression's relevance
wrapper gives predictors constant on retained positive-weight rows zero; direct
ANOVA still rejects their undefined one-group comparison. Numeric classification
keeps the grouped ANOVA contract, including rejection of a one-class response.

ANOVA and correlation ratio share one small centered-sums helper. It first
subtracts an observed origin, computes group means and between-group squared
deviations, and accumulates within-group residual squares directly. Neither
calculation subtracts large uncentered squares or total-minus-between variance.
Correlation ratio keeps zero for constant observations and a one-element array
in its grouped helper. Adding 1e10 to observations leaves the reference results
unchanged; a separate near-perfect example retains within-group variation even
when between-group variation is 4e20. This cannot recover variation already lost
when the caller's floating-point inputs were created.

Single-worker Pearson matrices use centered cross-products and sums of squares
without dividing by `sum(weights)-1`; that correction cancels in correlation.
Normalized relative weights now work and agree with pairwise/two-worker paths.
The shared matrix-to-long formatter explicitly drops missing entries after
stacking, preserving its observed lower-triangle output with pandas 3's
[new stack behavior](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html).
Both Series and reset-index dataframe paths are covered. All formatter callers
were traced; its only internal caller is this optimized Pearson path.

Theil's U preserves `U(constant, y)=1` as its zero-entropy association convention,
while honoring both scalar and dataframe output. Matrix construction and the
collinearity selector therefore process constant categorical inputs. Categorical
classification relevance explicitly gives constant predictors or a constant
response zero, regardless of `force_finite`. Its existing transform
`U² / (1 - U²) * (N - 2)` now uses retained frequency mass and rejects N <= 2;
it is documented as a ranking heuristic without an asserted F null law. Perfect
nonconstant dependence retains infinity/maximum-float behavior. R09 owns the
mixed relevance scale and quotient ranking decisions.

All four response-augmentation paths use one shared validation helper. It keeps
the response dtype and positional alignment, copies the predictor frame, and
appends underscores to a colliding response name until it is free. Unnamed,
categorical, numeric and repeated `target`/`target_` collisions preserve every
predictor and caller input. Categorical classification scores retain the original
column-index dtype after worker concatenation, including mixed-type column
indexes under pandas 3. The misleading shared warning that constant columns
had been removed was deleted; that helper did not remove any columns. One
obsolete Ruff allowance was removed with it.

Caller tracing covered every direct/grouped association and relevance function,
matrix/series and parallel dispatch, MRMR's default relevance, the collinearity
selector, their tests and API documentation. Source changes reach
`association.py` and MRMR's weight documentation; shared parallel behavior and
R09 ranking arithmetic were preserved. No dependency, public option, export or
compatibility layer was added.

## Validation and reproduction

Evidence is in [r08-statistical-arithmetic](evidence/r08-statistical-arithmetic/).
The named historical cases reproduce against the installed R07 locked wheel:
**10 failed, 1 passed, 31 deselected**. The initial 83 maintained R08 cases against
that old wheel record **60 failed, 23 passed**. Its association source hash matches
the saved pre-edit module. Maintained tests have no expected-failure markers;
historical diagnostics and logs remain unchanged. The first installed-wheel run
passed both locked cells but exposed eight newer failures: four masked matrix
entries and four categorical index-dtype mismatches. The saved failure log,
environment and initial wheel hashes identify that interim implementation.
After repair, all 85 focused R08 checks pass in the newer source snapshot. The
two additional direct reshape cases fail on the old newer R07 wheel.

The [maintained tests](../tests/test_statistical_arithmetic.py) include independent
sklearn Pearson-F and SciPy row-replicated ANOVA references, hand-calculated
weighted sums, fractional mass without automatic normalization, rejection of
insufficient mass, retained-row degrees of freedom, zero-mass categories,
constant and perfect cases, response-name/input preservation, MRMR eligibility,
real collinearity fits and one/two-worker agreement.

| Final installed R08 wheel | Full maintained suite | Historical R08 cases and Pearson control |
| --- | --- | --- |
| Python 3.9.25 locked; pandas 2.2.3 / NumPy 2.0.2 | 413 passed, 124 warnings | Not repeated on 3.9 |
| Python 3.12.7 locked; pandas 2.2.3 / NumPy 2.0.2 | 413 passed, 127 warnings | 12 passed, 30 deselected |
| Python 3.12.7 newer; pandas 3.0.6 / NumPy 2.5.4 | 397 passed, 115 warnings | 12 passed, 30 deselected |

R08 adds 85 maintained cases in each cell. Both locked cells use SciPy 1.13.1,
scikit-learn 1.6.1 and pytest 8.3.5; the newer cell uses SciPy 1.18.1,
scikit-learn 1.9.1 and pytest 9.1.1. All historical cases selected here run with
`--runxfail`, without masking their result. Earlier R09 findings remain open.

All three runtime environments were newly created before the first wheel run,
then the rebuilt final wheel was installed with `--reinstall-package arfs`.
Exact package versions and interpreters match R07's corresponding snapshots;
`uv pip check` passes, setuptools is absent, and every recorded module hash
matches the current checkout. All 19 packaged source/data payloads match the
checkout, sdist and final wheel. Copied tests match the maintained suite. The
initial environment recorder encountered a Matplotlib font-cache log before
its JSON output; rerunning after cache creation fixed the recorder without
changing runtime dependencies or suppressing warnings.

Per-cell command JSON and [commands.txt](evidence/r08-statistical-arithmetic/commands.txt)
record exact commands and settings. Full tests run outside the checkout with
isolated Python, Agg rendering and two-worker tests enabled:

```sh
cd /tmp/arfs-r08/run-py312-locked
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arfs-r08/mpl-py312-locked \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0 \
  /tmp/arfs-r08/py312-locked/bin/python -I -m pytest tests \
  -q --tb=short -p no:cacheprovider
```

Validation used uv 0.12.15. `uv lock --check`, Ruff 0.17.0's baseline gate
(419 existing diagnostics, zero new), gate unit tests (2 passed), new-test
lint/format and `git diff --check` pass. Ten unrelated pre-existing tracked
diffs remain byte-for-byte unchanged. Existing model/tag/precision warnings
and constant-Pearson division warnings remain visible.

Hosted CI, complete docs/notebooks, optional backends, Arrow string storage,
performance, full sklearn compliance, extreme floating-point ranges beyond
these reference cases and broad statistical calibration were not checked.
No mixed-type significance or release-readiness claim is made. No R08 policy
decision remains unresolved; R09 owns the mixed score/ranking decision.

## Resume

The next group is **R09 alone**, dependent on completed R08. Read `1A F1/F2` in
[the statistical report](01a-statistical-foundations.md) and its evidence, then
trace mixed relevance and default redundancy through MRMR. R09 must choose a
nonnegative common relevance scale, remove invalid normalization, use absolute
unnormalized redundancy and define empty/equal/perfect candidate behavior.
R08 settles frequency mass and primitive constant/output behavior; it does not
settle the mixed ranking scale or claim calibrated mixed-type significance.
Stop at this phase boundary; R09–R33 and release readiness remain pending.
