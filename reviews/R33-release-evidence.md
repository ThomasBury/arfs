# R33 — Release evidence and remaining claims

**R33 complete on 2026-10-15**, based on `f49bf9d`, on local branch
`fix/r33-release-evidence`. R01–R33 are complete. This phase includes local
implementation and handoff commits; no push, tag or release is included.
The package remains 3.0.0 with changes under Unreleased.
Implementation commit: `aabb238`.

## Changes and scope

The installed-wheel CI matrix now covers locked dependencies on every declared
Python minor (3.9–3.12), exact direct runtime minima on 3.9, and the separately
pinned newer snapshot on 3.12. The new minimum snapshot retains the locked 3.9
transitive/test pins and changes SHAP from 0.47.1 to its declared floor, 0.47.0.
No runtime dependency bounds or supported Python range change.

The maintained [selection screen](../tests/selection_screen.py) records its
settings before fitting, keeps evaluation rows untouched and reports all seeds.
Three small tests check training-only category learning and weighted regression/
classification loss against integer replication. The full screen runs explicitly;
the ordinary suite runs only those small boundary checks. A one-line
`MANIFEST.in` includes the helper and test-package initializer alongside the
regression module in the sdist, following setuptools' native
[manifest rules](https://github.com/pypa/setuptools/blob/main/docs/userguide/miscellaneous.rst).
The first build exposed their omission; the final build repairs it.

README, method descriptions and the workflow guide distinguish tested snapshots
from arbitrary allowed combinations, remove categorical-encoding superiority
claims and state the optional-backend, artifact and statistical limits.
Historical review evidence is unchanged. The approved runtime lint cleanup and
one additional missing-backend repair are described below.

Caller tracing covered installed-wheel CI, both dependency snapshots and project
metadata, the three shadow selectors and their support/transform outputs, fitted
ordinal encoding, weighted scoring, public method claims and Sphinx entry points.
New raw evidence remains local and ignored under
`reviews/evidence/r33-release-evidence/`. Its `lint-cleanup/` directory contains
the authoritative final artifacts, environments and results. Initial R33 evidence
is preserved separately and is superseded by the refreshed checks.

## Runtime and documentation gates

The six fresh wheel environments and their exact commands/distributions are
recorded in the final evidence directory. All full suites pass, with no skips or
expected failures. The additional missing-backend regression passes in all cells.

| Cell | NumPy / pandas / sklearn / SHAP / LightGBM | Full installed suite |
| --- | --- | --- |
| Python 3.9.25 locked | 2.0.2 / 2.2.3 / 1.6.1 / 0.47.1 / 4.6.0 | 1462 passed |
| Python 3.9.25 exact direct minima | 2.0.2 / 2.2.3 / 1.6.1 / 0.47.0 / 4.6.0 | 1462 passed |
| Python 3.10.18 locked | 2.0.2 / 2.2.3 / 1.6.1 / 0.47.1 / 4.6.0 | 1462 passed |
| Python 3.11.10 locked | 2.0.2 / 2.2.3 / 1.6.1 / 0.47.1 / 4.6.0 | 1462 passed |
| Python 3.12.7 locked | 2.0.2 / 2.2.3 / 1.6.1 / 0.47.1 / 4.6.0 | 1462 passed |
| Python 3.12.7 newer | 2.5.4 / 3.0.6 / 1.9.1 / 0.53.0 / 4.7.0 | 1446 passed |

There are 8,756 successful test executions across the six cells. The newer
snapshot has 16 fewer Copy-on-Write parameter cases because pandas 3 always uses
Copy-on-Write; these are not skipped tests.

All environments install the newly built wheel outside the checkout, pass
dependency checks, agree with source module hashes and contain no setuptools
workaround. All 19 source/sdist/wheel code/data payloads agree. The lock check
passes, including metadata for the dynamically versioned local project. All
eleven direct dependency versions exactly match their declared floors in the
minimum cell. The ordinary persistence suite includes fresh-process loads on new rows,
including categories, binary/multiclass tasks and external offset guards.

All six suites use the rebuilt post-cleanup wheel. Another clean installation
passes **25** package-resource and new regression tests extracted from its sdist.
The final sdist's helper, initializer and regression bytes match source. Final
artifact hashes and installed payload/test/documentation identities are recorded
in `artifact-verification.json`; smoke commands are in `final-smoke-commands.json`.
No initial-wheel result substitutes for a changed runtime module's final validation.

Strict HTML builds using `-W --keep-going` pass with zero Sphinx warnings in both
Python 3.12 dependency snapshots. The locked build uses a fresh wheel-plus-docs-only
installation. Both import the installed package without setuptools/pkg_resources.
Four unchanged historical optional-backend boundary checks pass unmasked in each
3.12 cell, as does the historical reduced-feature time-series regression; their
missing-backend behavior is not backend execution. Both snapshots also freshly
execute R32's same five bounded workflows: **90 code cells and 13 figures per
snapshot**, using temporary named kernels and the same declared local data
substitutions/source-cell scopes. All 26 PNGs are byte-identical to the previously
inspected R32 figures. All 17 notebook schemas validate and tutorial pages render.
Fresh ledgers, source hashes and comparisons are under `lint-cleanup/locked/`,
`lint-cleanup/newer/` and `notebook-verification.json`.

## Fixed selection experiment

Each snapshot runs 560 fits: two tasks × seven scenarios × ten seeds × four
configurations. Seeds are 0–9, with 400 training and 1,000 independent evaluation
rows, and 12 predictors. There is no search over settings or seeds. The two
snapshots produce identical selected sets and held-out losses in all 560 paired
records; they are not 20 independent seeds. The final screen runs against fresh
post-cleanup wheel installations. Every one of their 19 installed code/data
payloads matches source and the rebuilt wheel. All 1,120 final records exactly
match the corresponding pre-cleanup selections, held-out losses and baselines.
The refresh adds no independent seeds. Protocols, source identities and the
before/after comparison are retained in final evidence.

Strong regression uses `3*x0 + 2*x1 + N(0,1)`; weak regression uses coefficients
0.35 and 0.25. Classification draws Bernoulli responses with the corresponding
logistic probability. Null responses are independent of predictors. Correlated
data add `x2 = x0 + 0.1*N(0,1)` and count that marginally informative proxy as
relevant. Mixed data replace x1 with a binary category and add a 20-level noise
category. Interaction data use `3*x0*x1`. Weighted data retain the strong signal
and use weights 4 for x0 > 0 and 1 otherwise, in training and evaluation.

Ordinal encoding learns categories on each outer training split only and retains
categorical dtype. No evaluation values or outcomes enter preprocessing,
selection, internal validation or model fitting. This is an independent outer
holdout experiment, not a nested-CV study; GrootCV's internal folds operate on
that outer training frame. Future tuning/CV must refit preprocessing and selection
inside each outer training fold.

Leshy uses SHAP, 40 trees and 20 iterations, retaining its default percentile 90,
alpha 0.05, two-step correction and confirmed-only selection. BoostAGroota uses SHAP, three
iterations and at most three reduction rounds, separately at cutoff 1 and the
default cutoff 4. GrootCV uses three folds, one repeat, cutoff 1, its native
20-round early stopping and 10,000-round ceiling. The common LightGBM settings
are seven leaves, depth three, minimum leaf size 15 and one worker. Both NumPy
and Python shuffle streams are reset per fit; Leshy/LightGBM receive the seed.
These are bounded settings, not validation of every default or importance mode.

Evaluation refits the same 40-tree learner on each selected set, using a dummy
predictor for an empty set. Baselines are the dummy, all features, unweighted
absolute-correlation top two (with category codes for this baseline only), and
the known-relevant predictors. All fitted models receive the training weights
when present. Loss is weighted/unweighted mean squared error for regression or
binary log loss for classification, with lower values better. Baselines are not
used to choose selector settings. See scikit-learn's
[leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) and
[dummy estimator guidance](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators),
consulted through Context7.

The following null results show the number of seeds selecting any noise variable
and the mean selected count. Under the global null every selection is false.

| Configuration | Regression: any / mean count | Classification: any / mean count |
| --- | --- | --- |
| Leshy | 5/10 / 0.9 | 4/10 / 0.5 |
| BoostAGroota, cutoff 1 | 6/10 / 1.2 | 7/10 / 1.5 |
| BoostAGroota, cutoff 4 | 10/10 / 9.1 | 10/10 / 9.0 |
| GrootCV | 7/10 / 1.4 | 3/10 / 1.6 |

All configurations recover both strong additive signals in every seed and task,
but also select noise. Mean strong-signal false-discovery fractions are
0.217–0.731 in regression and 0.133–0.726 in classification. Ten seeds cannot
calibrate the procedure's error rate; these observed null selections also do not
support a blanket 5% false-discovery claim. Keeping more relevant predictors at
cutoff 4 comes with many more noise selections in these cases.

The next table gives mean relevant recall and held-out loss divided by dummy
loss, averaged over ten seeds. A ratio below 1 improves on the dummy. It reports
the difficult cases, rather than treating strong-signal recovery as sufficient.

| Task / scenario | Leshy recall / loss ratio | Boost cutoff 1 | Boost cutoff 4 | GrootCV |
| --- | --- | --- | --- | --- |
| Regression / weak | 1.00 / 0.906 | 1.00 / 0.908 | 1.00 / 0.917 | 1.00 / 0.906 |
| Regression / interaction | 0.45 / 0.818 | 0.65 / 0.786 | 1.00 / 0.773 | 1.00 / 0.485 |
| Classification / weak | 0.55 / 1.030 | 0.65 / 1.026 | 1.00 / 1.031 | 0.50 / 1.028 |
| Classification / interaction | 0.05 / 1.024 | 0.15 / 1.011 | 0.95 / 0.939 | 0.20 / 1.006 |

All-feature loss ratios for those rows are 0.913, 0.813, 1.028 and 0.926.
Full per-seed losses, all four baselines, paired differences, seed ranges and
noise selections for every scenario are retained in `selection-summary.json`
and both `screen-*/results.json` files. Correlated/mixed/weighted cases recover
every labelled relevant predictor here, but still select noise. None establishes
superiority over other encodings, weighted fits or alternative selectors.

## Approved lint cleanup and release decision

The user explicitly approved the broader cleanup after the initial automatic
approval rejection. No further approval for that work is needed. Direct
`ruff check .` now passes with zero diagnostics; CI runs it directly. The old
count baseline and both of its scripts are removed. Historical review evidence
is unchanged.

The reviewed fixes sort imports, remove unused imports/assignments and obsolete
future imports, simplify equivalent expressions, and enumerate GrootCV folds
without changing their order or thresholds. Encoder dtype defaults are immutable
tuples; caller-supplied options retain their values. Plot styling preserves its
values and missing-stage coloring uses `pd.isna` explicitly.

The documented Ruff runtime-typing setting retains evaluable `Union`/`Optional`
annotations on Python 3.9. Native collection annotations are valid there. All
112 functions/methods in the four affected annotated modules pass
`typing.get_type_hints` on Python 3.9 and newer 3.12. Seven local rule exceptions
preserve real compiled-backend import probes, third-party error wrappers and the
existing unsupported-SHAP-output ValueError. Their reasons are recorded in code;
no broad rule or file ignore is added. See Ruff's
[runtime-typing documentation](https://docs.astral.sh/ruff/settings/#lint_pyupgrade_keep-runtime-typing),
consulted through Context7.

The private FastTreeSHAP helper now raises its missing-backend ImportError
before cloning or training. Its new regression fails on the original installed
wheel and passes on the repaired wheel; the public fallback paths are preserved.
All 17 notebooks retain markdown, saved outputs, metadata and cell order. Code
changes remove unused/duplicate imports and three non-displaying expressions.

All post-cleanup installed-wheel suites, fixed selection screens, strict HTML
builds and bounded fresh-kernel workflows pass. R33's local gates are complete;
there is no remaining RXX implementation batch. The intentional API/helper
removals warrant a major release when delivery is requested. Keep version 3.0.0
and Unreleased until then: publishing, tagging and a version bump remain outside
this implementation request. Hosted CI and review belong to that separate delivery.

## Reproduction

The validation scripts, requirement exports, complete command records, artifact
hashes and logs remain under `/tmp/arfs-r33-cleanup/` and final local ignored
evidence. These commands create fresh run directories; use unused output paths
for reruns. The selection screen output directory must not already exist.

```sh
uv build --python /tmp/arfs-r33/docs-locked/bin/python --out-dir /tmp/arfs-r33-cleanup/dist
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/check_payloads.py
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/validate.py py39-locked
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/validate.py py39-minimum
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/validate.py py310-locked
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/validate.py py311-locked
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/validate.py py312-locked
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/validate.py py312-newer
/tmp/arfs-r33/py312-locked/bin/python /tmp/arfs-r33-cleanup/final_smoke.py
# From the copied run directory, with the thread limits recorded by validate.py:
/tmp/arfs-r33-cleanup/py312-locked/bin/python -I tests/selection_screen.py OUTPUT_DIRECTORY
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/arfs-r33-cleanup/docs-locked/bin/python -I -m sphinx -b html -W --keep-going \
  docs /tmp/arfs-r33-cleanup/html-locked
JUPYTER_PATH=/tmp/arfs-r33-cleanup/share/jupyter \
  /tmp/arfs-r33-cleanup/docs-locked/bin/python -I /tmp/arfs-r33-cleanup/execute_examples.py locked
uv lock --check --python /tmp/arfs-r33/py312-locked/bin/python
uv tool run --from ruff==0.17.0 ruff check .
git diff --check
```

The full notebook-execution gaps in R32 remain explicit. No GPU, live optional
backend, other platform, large-data throughput, cross-version artifact load or
hosted CI/Read the Docs run is certified by this phase.
