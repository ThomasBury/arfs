# Separate tree binning and grouping, then refresh the tutorials

## Scope and working agreement

Implement one phase per session and stop. Preserve unrelated MRMR work and the
completed R01–R33 queue and historical reports. No new dependencies, version bump,
publication, Azoic changes, quantile strategies, monotonic pooling, support floors,
mutable mapping overrides, shadow-selection redesign, or portfolio benchmark.
This is a breaking change before release; existing discretizers must be refitted.

## Phase 1 — Split the API and establish correct behavior

**Status: Complete**

- Keep TreeDiscretizer's constructor and numeric output options (`raw`,
  `n_bins_max`, `num_bins_as_category`). `all`, `numerical`, and None select
  eligible numeric columns; others pass through. Explicit categorical selections
  and `categorical` direct users to TreeGrouper.
- Add `TreeGrouper(group_features="all", n_groups=10, boost_params=None,
  raw=False, task="regression")`, selecting category/object/string/Boolean
  columns. Explicit numeric codes require categorical dtype.
- Both support weighted fit, transform, feature names, copied DataFrames, fitted
  array names/width, and preserved index/column order/passthrough columns.
- Fit one native LightGBM tree per selected feature on all positive-weight rows,
  without changing the general GradientBoosting wrapper. Share only necessary
  schema, validation, parameter and fitting helpers; no base-class framework.
  Preserve objective/iteration/leaf alias protection and constructor parameters.
- Validate finite numeric regression targets and numeric/Boolean binary {0,1}
  targets (constants valid). Validate positional weights and all inputs before
  dropping zero-weight rows; require positive mass. Learn vocabularies only from
  retained rows.
- Preserve exact numeric thresholds, right-closed intervals, leaf identity,
  missing routes, constants/all-missing columns. Native nominal splits must be
  independent of arbitrary label order; ordered categories allow contiguous
  groups in declared order. Missing/unseen levels use native missing routes.
- Freeze output category vocabularies; nominal groups unordered and ordinal
  groups ordered. `raw=True` gives native responses, including probabilities.
  Keep numeric fitted maps; `tree_dic` now contains Boosters (no `.model`). Move
  categorical leaf labels into TreeGrouper. No mixed compatibility facade.
- Update docstrings, API reference, README, introduction, methods and Unreleased
  migration notes. Migrate existing preprocessing, Lasso and MRMR notebook calls:
  group categoricals then bin numerics, explicitly forwarding weights to each.
  Remove noise-as-NaN/Inf and group/fused-Lasso claims; clear changed cell outputs.

**Acceptance:** focused runtime/documentation tests pass; migrated examples
execute; strict documentation build passes; Phase 2 is identified as next.

**Implementation record (2026-10-17):**

- Split the estimators and replaced the discretizer's wrapper with direct native
  one-tree training; the general GBM wrapper is unchanged. Fit validates response
  and positional weights before omitting zero-weight rows. Row-sampling aliases
  are overridden to honor the all-row contract. Ordered codes start at one so
  `zero_as_missing` cannot merge distant observed ordinal levels.
- Added frozen category vocabularies/labels, native missing routes, full output
  feature names and numeric-only selection. Updated existing partition/schema
  tests, added independent numeric/categorical native references, relabeling,
  ordered contiguity, validation, zero-weight, batch, serialization and pipeline
  weight-forwarding checks. Existing numeric fitted map names remain.
- Fixed a necessary shared-validator defect exposed by the MRMR call: pandas
  categorical interval `isin([inf])` tests containment. Comparing object values
  instead preserves valid unbounded bins while still rejecting actual missing
  or infinite outcomes. Category-identity regression tests cover both cases.
- Updated API/docstrings, README, introduction, methods and Unreleased breaking
  migration notes. Migrated the three notebook call sites and cleared changed
  and downstream dependent outputs. The Lasso construction uses factor bins.
  No new dependencies or version changes. Unrelated MRMR source, tests, review
  records and the existing `docs/workflows.rst` edits were preserved.

**Verification environment:** current sources via `PYTHONPATH=src`, Linux,
CPython 3.12.7. Reused disposable environments without changing dependencies:

| Snapshot | Interpreter prefix | LightGBM | NumPy | pandas | sklearn | SciPy | statsmodels | pytest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Locked | `/tmp/arfs-association-delivery/py312-locked` | 4.6.0 | 2.0.2 | 2.2.3 | 1.6.1 | 1.13.1 | 0.14.4 | 8.3.5 |
| Newer | `/tmp/arfs-association-delivery/py312-newer` | 4.7.0 | 2.5.4 | 3.0.6 | 1.9.1 | 1.18.1 | 0.15.0 | 9.1.1 |

Documentation environment: `/tmp/arfs-r33-delivery/docs-locked`, CPython 3.12.7,
Sphinx 6.2.1, nbsphinx 0.9.2 and nbformat 5.10.4. The project's `.venv` lacks
pytest, so verification used these existing disposable environments.

**Verification results:**

- With `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg`,
  run either snapshot's `bin/python -m pytest tests/test_tree_partitions.py
  tests/test_transformer_inputs.py tests/test_pandas_writes.py
  tests/test_documentation_workflows.py -q --tb=short`: locked **198 passed**;
  newer **194 passed**. pandas 3 runs only its supported copy-on-write setting,
  accounting for the different count. Logs: `/tmp/arfs-tree-preprocessing-focused-final.log`
  and `/tmp/arfs-tree-preprocessing-newer.log`. The final numeric native reference
  explicitly omits zero-weight rows (verified in the final locked run).
- Expanded checks also cover observation boundaries, category identity, MRMR
  ranking and association kernels. One initial run had 548 passes and two new
  test-fixture errors (Theil U needs a categorical target); after correcting
  that fixture, `bin/python -m pytest tests/test_category_identity.py -q --tb=short`
  passed **52 tests**, including both interval-value cases.
- The exact migrated notebook cells execute on bounded 240-row local fixtures:
  preprocessing cells 18/22, Lasso cell 47, MRMR cell 33. MRMR uses a declared
  local synthetic substitute for its still-remote tutorial loader and excludes
  its artificial identifier. These are construction checks, not held-out
  comparisons or complete fresh-kernel notebook executions.
- `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
  /tmp/arfs-r33-delivery/docs-locked/bin/python -m sphinx -W --keep-going -b html
  docs /tmp/arfs-tree-preprocessing/html-phase1`: **passed**, including the final
  rebuild after clearing affected outputs. `nbsphinx_execute="never"` remains.
  Log: `/tmp/arfs-tree-preprocessing-build-final.log`.
- `uv tool run --offline ruff check .` (Ruff 0.17.0), changed-Python formatting,
  `uv lock --check`, `git diff --check`, local-link tests and nbformat validation
  of all three changed notebooks: **passed**. Dependencies/lockfile unchanged.
- Broader locked run: same environment with `MKL_NUM_THREADS=1
  NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1`, `bin/python -m pytest tests -q
  --tb=short`: **1,649 passed, 2 failed** in 213.95 seconds. Its test collection
  preceded the categorical-target fixture correction above. The only failures
  were those two new interval tests. After the correction, `bin/python -m pytest
  tests --lf -q --tb=short` passed **2 tests**. The complete
  suite was not rerun after that fixture-only correction. Logs:
  `/tmp/arfs-tree-preprocessing-runtime.log` and
  `/tmp/arfs-tree-preprocessing-runtime-rerun.log`.

**Remaining limits:** Full installed-wheel/Python compatibility matrix, full
fresh-kernel notebook executions, refreshed saved outputs, held-out teaching
comparisons, rendered presentation inspection and the workflow execution ledger
remain Phase 2 work. Runtime examples emit existing sklearn-tag deprecations,
association-rounding and GLM convergence warnings; passing finite-output checks
does not establish model quality or inference validity.

## Phase 2 — Refresh and verify the teaching material

**Status: Complete under revised scope; manual notebook execution pending**

**User scope update (2026-10-17):** Implement the tutorial refresh, including a
carefully designed artificial-data check. The user will rerun the notebooks
manually. Leave saved outputs clear; full fresh-kernel notebook execution and
inspection of executed tables/figures are explicitly deferred to that manual
handoff. Automated computational-cell checks and strict documentation builds
remain part of this implementation. This overrides the original requirement to
save freshly executed notebooks in this session.

Keep the three tutorial entry points and use notebook-safe minimal edits.

- Preprocessing: teach numeric, nominal, ordinal and composition behavior; fitted
  maps, constants, missing/unseen levels, stable categories and a category-renaming
  counterexample. Recommend factors/encoding for separate bin effects, rather
  than midpoints. Compare dummy, ordinary encoded and grouped/binned regression
  with weighted MSE on identical untouched evaluation rows.
- Lasso: train-only preprocessing/design/selection/prediction; compare ordinary
  and grouped/binned encoding with the same split and learner. Treat groups/bins
  as factors; design-column selection is not joint original-variable selection.
  Seed stochastic cells locally. Generate Poisson counts from exposure times
  expected rate, then divide by exposure. Explain invalid post-selection p-values.
- MRMR: replace affected remote Titanic workflow with local synthetic
  classification fixture, excluding its artificial identifier; retain local
  regression. Compare all-feature, ordinary MRMR and grouping/binning plus MRMR
  on one split/learner using binary log loss. Explain changed relevance and
  amplified training associations; claim no calibrated errors or general gain.
- Use bounded local fixtures, local RNGs, one native thread, 75/25 seed-42 splits,
  classification stratification, settings frozen before evaluation and no seed
  selection. Keep learned steps inside training folds; future tuning must search
  the whole pipeline. Call fixture weights illustrative observation weights.
  Report results even when transformed pipelines lose.
- Update workflow execution ledger and reconcile API/tutorial/migration prose.
  Save fresh executed outputs, inspect HTML/figures. Keep Sphinx execution off.

**Original acceptance:** three notebooks execute top-to-bottom in fresh kernels,
outputs match narrative, rendered presentation inspected, execution scope documented.

**Revised acceptance:** implemented notebooks with clear outputs, known-truth and
training-isolation computational checks, strict builds and the installed-wheel
matrix pass. Full notebook execution/output inspection is the user's explicit
manual handoff, not a completed check.

**Implementation record (2026-10-17):**

- Rebuilt the preprocessing walkthrough around a designed 4,320-row factorial
  process, crossing 60 dose values, six nominal channels, six ordered stages and
  two replicates. Outcomes follow three numeric plateaus, two nominal groups
  that are not contiguous under alphabetical codes, and three contiguous ordinal
  groups, with independent Gaussian outcome noise. Independent numeric/category
  noise, constants, all-missing columns and explicitly defined missing effects
  exercise limits. The generating mean is deliberately additive and piecewise
  constant; it favors these representations and does not test pure interactions.
- The main notebook checks known thresholds/groups, ordinal contiguity, category
  relabeling, native missing/unseen routing and frozen category vocabularies.
  A counterexample compares native grouping against trees on arbitrary numeric
  codes. It prints fitted maps and supplies truth-versus-fitted plot cells.
- Added weighted dummy/ordinary/Ridge-partition comparisons, ordinary/partitioned
  Gaussian Lasso comparisons, and all-feature/ordinary-MRMR/partitioned-MRMR
  classification comparisons. Every arm shares its split, evaluation rows and
  downstream learner. Supervised weights are forwarded to each step; scores
  never gate success or select settings/seeds. No winner is asserted.
- Lasso retains its separate linear and Poisson illustrations, uses local RNGs
  and 75/25 seed-42 splits, generates counts from exposure times expected rate,
  and treats groups/bins as factors. Its comparison uses a training-BIC search;
  future CV must search the whole pipeline. Explained selected design columns,
  unpenalized refits, post-selection p-value limits and convergence warnings.
- MRMR retains a local regression example, replaces remote Titanic reads with
  the existing local classification fixture, and explicitly drops its artificial
  identifier before all predictive comparisons. Explained changed relevance,
  training associations, missing policies and calibration/performance limits.
- Cleared all saved outputs as requested; retained the linear/Poisson section's
  cell order and the same three tutorial entry points. No notebook kernels or
  figure cells were run. Tests use stable cell tags for computational checks,
  including spies that reject non-training rows or incorrectly forwarded weights,
  factor design checks, independently recomputed losses and Poisson generation.
  Tests allow outputs to be saved by a later manual run.
- Updated the workflow execution ledger, README tutorial link and Unreleased
  notes. Installed-wheel CI now copies CHANGELOG.md too, because the README's
  migration link is checked outside the checkout. Existing MRMR source/workflow
  changes and Phase 1 production changes are preserved. A single sampling-test
  fixture now requests `to_numpy(copy=True)` before mutating its weights; pandas
  3 correctly exposed the former view as read-only. The MRMR guard assertion and
  production implementation are unchanged. No Phase 2 production
  code, dependency, lock, version or publication changes were needed.

**Verification results:**

- Initial locked source-cell checks: **19 passed**, using
  `/tmp/arfs-association-delivery/py312-locked/bin/python -m pytest
  tests/test_documentation_workflows.py -q --tb=short` with `PYTHONPATH=src`,
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  NUMBA_NUM_THREADS=1 MPLBACKEND=Agg LOKY_MAX_CPU_COUNT=1`.
  Log: `/tmp/arfs-tree-phase2/focused-locked.log`. This executes tagged
  computational cells, not complete notebooks, kernels or figures.
- Strict locked and newer Python 3.12 Sphinx builds: **passed**, using the Phase 1
  documentation environments and `PYTHONPATH=src ... bin/python -m sphinx -W
  --keep-going -b html docs /tmp/arfs-tree-phase2/html-{locked,newer}`. Final
  rebuild logs: `/tmp/arfs-tree-phase2/html-locked-final.log` and
  `/tmp/arfs-tree-phase2/html-newer-final.log`. Notebook execution remains off.
- All three nbformat schemas validate and all saved outputs are clear. Ruff
  0.17.0 checks/formatting, `uv lock --check` and `git diff --check` pass.
- Built an sdist and wheel using `uv build --offline --out-dir
  /tmp/arfs-tree-phase2/dist`; exported locked requirements with `uv export
  --locked --group test --no-emit-project --output-file
  /tmp/arfs-tree-phase2/locked.txt`. The six CI snapshot cells run in clean
  `/tmp/arfs-tree-phase2/py*` environments, installed offline from the wheel and
  requirements, with `uv pip check` and installed source-hash checks. Tests run
  outside the checkout from copied sources under `/tmp/arfs-tree-phase2/run` with
  `bin/python -I -m pytest tests -q --tb=short -p no:cacheprovider` and the CI
  one-thread environment. Commands, version/import manifests and test logs are
  under `/tmp/arfs-tree-phase2/<cell>-{commands.json,imports.log,tests.log}`.
  Matrix progress/results: `/tmp/arfs-tree-phase2/matrix-progress.log` and
  `/tmp/arfs-tree-phase2/matrix-results.json`.

Final notebook-source checks (including the exact fitted-threshold assertion)
passed **19 tests in each of the six installed-wheel environments**. They used
`bin/python -I -m pytest tests/test_documentation_workflows.py -q --tb=short
-p no:cacheprovider` outside the checkout against final copied tutorial sources
under `/tmp/arfs-tree-phase2/final-run`. Logs:
`/tmp/arfs-tree-phase2/<cell>-final-tutorials.log`; summary:
`/tmp/arfs-tree-phase2/final-tutorial-results.json`. The broader suite uses the
same computational cells; this final focused pass also covers the last
boundary assertion and source-only cleanup added while that suite was running.

Both documentation environments use Sphinx 6.2.1 and nbsphinx 0.9.2; nbformat is
5.10.4 (locked) / 5.11.1 (newer). Runtime versions and broad-suite results:

| Snapshot | CPython | LightGBM | NumPy | pandas | sklearn | Passed tests |
| --- | --- | --- | --- | --- | --- | --- |
| py39-locked | 3.9.25 | 4.6.0 | 2.0.2 | 2.2.3 | 1.6.1 | 1656 |
| py39-minimum | 3.9.25 | 4.6.0 | 2.0.2 | 2.2.3 | 1.6.1 | 1656 |
| py310-locked | 3.10.18 | 4.6.0 | 2.0.2 | 2.2.3 | 1.6.1 | 1656 |
| py311-locked | 3.11.10 | 4.6.0 | 2.0.2 | 2.2.3 | 1.6.1 | 1656 |
| py312-locked | 3.12.7 | 4.6.0 | 2.0.2 | 2.2.3 | 1.6.1 | 1656 |
| py312-newer | 3.12.7 | 4.7.0 | 2.5.4 | 3.0.6 | 1.9.1 | 1640 |

Wheel SHA256: `f9ffaf20f19dbdefd6bf9309f81fbe03fdb0d5984a4743173bded85cfa87cf25`.

The initial newer full suite had **1,639 passes and one failure** in the
pre-existing MRMR sampling test's setup, before reaching its intended assertion:
pandas 3 returned a read-only view for weights that the test mutates. Requesting
an explicit copy fixes the fixture without weakening the test. Its sampling-test
rerun is recorded in `/tmp/arfs-tree-phase2/py312-newer-sampling-fixed.log`; the
complete newer-suite rerun is in `/tmp/arfs-tree-phase2/py312-newer-tests-fixed.log`:
**1,640 passed** in 173.18 seconds. The sampling suite separately passed **21
checks** on that snapshot. The corrected guard test also passed on all five
older snapshots (`<cell>-guard-fixed.log`). Final matrix summary:
`/tmp/arfs-tree-phase2/matrix-final-results.json` (**all six passed**). The raw
initial `matrix-results.json` intentionally retains the original failed run.

**Manual handoff:** Run all three notebooks top-to-bottom in fresh locked and
newer Python 3.12 kernels, save outputs, inspect the displayed mappings/tables,
truth-versus-fitted curves and loss figures, then update the workflow ledger.
Exact nbconvert commands are in `docs/workflows.rst`. These full runs and visual
checks were deliberately not done here at the user's request. Existing sklearn
and GLM/association warnings do not establish convergence, calibration or model
quality; no predictive gain or valid post-selection inference is claimed.

## Verification and completion

Extend existing tests: independent native positive-weight one-tree references;
nominal relabeling invariance and ordered contiguity; invalid targets; positional
weights, zero-weight invariance and input/parameter immutability; exact thresholds,
missing/unknown routes, constants, stable vocabularies; refit/clone, arrays after
named fitting, schemas/names, pipeline weight forwarding and serialization.
Tutorial checks cover isolation, finite predictions/loss and consistent factors.

Phase 1 runs focused tests. Phase 2 runs the full installed-wheel CI compatibility
matrix, the three notebooks and strict Sphinx HTML with locked and newer Python
3.12 snapshots. Run Ruff, notebook schema/local links, lock consistency and
`git diff --check`. Record commands, versions, results and limitations here.

Correct partitions and green tests do not establish predictive improvement,
calibration or valid post-selection inference.

## Resume point

Phases 1 and 2 implementation are complete under the user's revised scope, with
uncommitted changes based on `4fc3c7a52340a249a823a7982e0ec222df413fef`. All six
local installed-wheel configurations and both strict Python 3.12 documentation
builds pass. **Next action belongs to the manual notebook handoff:** restart each
kernel, run all three tutorials, save outputs, inspect tables/figures and update
the execution ledger. Do not present these manual checks as already completed.

The unrelated MRMR implementation remains intact; the only test adjustment in
that track requests a writable copy for its mutated fixture array on pandas 3.
No commit, version bump, publication or memory update was performed. Stop here.
