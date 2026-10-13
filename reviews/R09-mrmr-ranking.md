# R09 — Coherent MRMR ranking

**R09 complete on 2026-10-12. Next group: R10 → R11.**
Based on `170b3e95fee635e8243ed8cdc740ecb80c4738ad` on `main`; changes remain
uncommitted. Pre-existing R04–R08 work and historical review evidence are
preserved. No commit, push, version bump or release was performed.

## Changes and decisions

The user chose **F/(1+F)** for both mixed relevance wrappers. It replaces the
separate numeric/categorical z-scores with the same nonnegative [0, 1] map.
The score of a predictor no longer depends on how many predictors of its type
are present. Singletons, equal scores and perfect scores remain eligible.
Zero maps to zero, infinity/maximum float maps to one, and very large finite
scores can round to one. This is a bounded ranking heuristic, not a common
mixed-type effect size, a p-value, or calibrated significance. It intentionally
changes selected features. Frequency weight mass, retained-row policies,
primitive F arithmetic and integer replication remain as recorded in R08.
The primitive `f_cont_*` and `f_cat_*` APIs still return their raw scores.

`force_finite=True` fills undefined score NaNs with zero in the mixed wrappers;
False preserves them. Infinity maps to one in both modes. R08's existing
one-class ANOVA rejection remains in effect: a numeric classification predictor
with a one-class response raises rather than inventing a valid ANOVA test.
Categorical-only classification retains its primitive constant-response zero.

Default MRMR redundancy now requests `association_series(normalize=False)` and
uses the existing absolute magnitudes and 0.001 denominator floor. Signed
associations [1, .8, -1] therefore retain magnitudes [1, .8, 1]. The fixed-score
counterexample selects `a`, then `b`, before `neg`. Aggregate redundancy exactly
one retains the existing infinite-denominator/zero-quotient rule; duplicates are
postponed, not removed. With mean aggregation, a later partial aggregate can be
less than one even for a candidate duplicating one already selected feature.
The first selected feature retains the historical infinite MRMR ranking score
and zero recorded redundancy; that display convention is not significance.

Ties follow input column order, including custom scores returned in a different
order. Zero-relevance predictors remain eligible and MRMR fills the requested
count on null/constant data. Missing/NaN custom scores are unavailable. Empty
candidate sets, zero input columns and a zero requested count produce an empty
fitted ranking, a boolean support mask and a zero-column transform. Zero-count
and zero-column requests skip relevance computation after observation-boundary
validation, so one usable observation also permits an empty selection. The requested
count stays unchanged across fitting; fitted count and input feature metadata
are recorded separately, so a fit with no candidates can be followed by a
normal fit. Negative/noninteger/boolean requested counts are rejected.

Custom relevance must be a Series with unique, known predictor labels and
finite nonnegative available scores. Negative or infinite custom scores are
rejected explicitly, rather than reversing the quotient's redundancy penalty.
Missing labels remain allowed and unavailable. These contracts are recorded in
MRMR and the two mixed-wrapper docstrings and exercised by maintained tests.

Caller tracing covered both mixed wrappers, all four primitive relevance
branches, default/custom MRMR callbacks, selection/ranking/support/transform,
association and parallel dispatch, maintained tests and the MRMR notebook.
Only `association.py`, `feature_selection/mrmr.py`, the new maintained test file
and remediation handoffs change in this phase. No dependency, public option,
export, backend or compatibility layer was added.

## Validation and reproduction

Evidence is in [r09-mrmr-ranking](evidence/r09-mrmr-ranking/).
Before edits, all five named historical F1/F2 cases fail unmasked against the
installed R08 locked wheel (**5 failed, 37 deselected**). The first 41 maintained
R09 cases also fail on that wheel (**41 failed**); nine later callback/undefined/empty-request
checks extend the final file to 50 cases. The old wheel's association and
MRMR module hashes match the saved pre-edit checkout. Historical scripts and
logs remain unchanged; the maintained file has no expected-failure markers.

The [maintained tests](../tests/test_mrmr_ranking.py) check the exact common map
against primitive scores, integer replication, predictor-composition invariance,
one/two-worker agreement, original input preservation, fixed-score signed
redundancy ranking, perfect/constant/undefined scores, input-order ties, empty
and refitted selections, return_scores output, masks/transforms/metadata,
callback validation, and five prespecified seeds for regression/classification
strong-signal and mixed-null screens. The screens do not establish calibration,
held-out utility or superiority on weak/correlated signals.

| Fresh installed R09 wheel | Full maintained suite | Historical R09 cases and signal/null controls |
| --- | --- | --- |
| Python 3.9.25 locked; pandas 2.2.3 / NumPy 2.0.2 | 463 passed, 171 warnings | Not repeated on 3.9 |
| Python 3.12.7 locked; pandas 2.2.3 / NumPy 2.0.2 | 463 passed, 179 warnings | 11 passed, 31 deselected |
| Python 3.12.7 newer; pandas 3.0.6 / NumPy 2.5.4 | 447 passed, 146 warnings | 11 passed, 31 deselected |

R09 adds 50 maintained cases per cell. Both locked cells use SciPy 1.13.1,
scikit-learn 1.6.1 and pytest 8.3.5; newer uses SciPy 1.18.1, scikit-learn 1.9.1
and pytest 9.1.1. All three environments were freshly created from the recorded
dependency snapshots and the newly built wheel. `uv pip check` passes,
setuptools is absent, and all eight recorded installed module hashes match the
checkout. All 19 source/data payloads agree across checkout, sdist and wheel;
copied maintained tests match the checkout. Exact interpreter/dependency versions,
commands and output are recorded per cell. Eleven unrelated pre-existing tracked
diffs remain byte-for-byte unchanged.

An interim 46-case implementation also passed the full matrix (459/459/443).
Final inspection added four empty-request checks and a short circuit that skips
unused statistics for zero-count/zero-column requests, including one usable row.
The final wheel was rebuilt, reinstalled and the full matrix rerun; interim logs
and wheel/environment identities are retained separately.

Full tests run outside the checkout with isolated Python, Agg rendering and
two-worker tests enabled. For example:

```sh
cd /tmp/arfs-r09/run-py312-locked
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arfs-r09/mpl-py312-locked \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0 \
  /tmp/arfs-r09/py312-locked/bin/python -I -m pytest tests \
  -q --tb=short -p no:cacheprovider
```

[commands.txt](evidence/r09-mrmr-ranking/commands.txt) and per-cell command JSON
record the exact reproduction, build, install, provenance, full-suite and
historical commands. Validation used uv 0.12.15 and Ruff 0.17.0.
`uv lock --check`, Ruff's baseline gate (419 existing diagnostics, zero new),
gate unit tests (2 passed), new-test lint/format and `git diff --check` pass.
Existing model/tag/precision and constant-statistic warnings remain visible.

Hosted CI, complete docs/notebooks, optional backends, performance, broad sklearn
compliance and statistical calibration were not checked. No R09 policy decision
remains unresolved; no release-readiness claim is made.

## Resume

The next group is **R10 → R11**, sequentially, with R05/R07 prerequisites already
complete. Read `1B F1/F2` for Leshy support/history and `1B F3/F4` for weighted
signed permutation importance, in [the all-relevant report](01b-all-relevant.md),
its historical evidence, `allrelevant.py` and the established selector tests.
Complete/validate R10 and make its separate local commit before starting R11,
preserving all earlier uncommitted work. R11 must explicitly settle zero-total
importance behavior. Stop after that group; R12–R33 remain pending.
