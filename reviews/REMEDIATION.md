# ARFS remediation handoff

**Status: R01–R33 complete. C01–C04 complete; C05–C06 pending. Tree preprocessing, MRMR sampling and the confirmed plan are being delivered separately; the next package release is deferred.**
The [confirmed follow-up plan](#confirmed-follow-up-plan) below is separate from
that historical delivery. **C01–C04 are complete; C05–C06 are pending; next follow-up: C05.**
This handoff follows [review 07](07-final-assessment.md) of
`03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`. The review queue is complete; this is
a separate repair backlog. By default, a request to implement the next phase
authorizes one phase, its affected callers and regression checks, then a stop
for review. A larger batch or concurrent work requires an explicit request;
the guidance below does not authorize implementation.

Keep one distribution. Preserve unrelated changes and historical review
evidence. Prefer existing helpers and supported native APIs; no compatibility
layer, new backend, packaging split, or broad rewrite is required by this plan.
Names below are phase IDs, not commit promises or a release schedule.

## Tree preprocessing and MRMR delivery — 2026-10-17

The user authorized topical Conventional Commits, a dedicated branch, a PR,
Codex bot review, fixes for actionable comments, merge and branch removal.
Branch: `feat/tree-groups-mrmr-sampling`, based on `4fc3c7a` on main. The feature
commits are `391cf54` (optional MRMR sampling and measured benchmarks) and
`da411b6` (separate native numeric bins/category groups and revised tutorials).
The confirmed C01–C06 plan is a separate documentation commit; none of its
repairs is included in this delivery. CodeRabbit is not a merge gate.

Raw measurements/profiles were moved to the ignored
`evidence/mrmr-performance/` and `evidence/mrmr-sampling/` directories. Reports
describe those files as local evidence and retain reproduction commands; the
raw files are not added to the PR. Existing tracked historical evidence is
unchanged. The three notebooks intentionally retain clear outputs; their full
manual execution and rendered-output inspection remain the user's handoff.

**Release decision: defer the version bump, GitHub release and PyPI upload.**
GitHub's latest release is 4.0.1. The TreeDiscretizer API changes require a new
major release, but confirmed correctness/diagnostic phases C01–C04 remain open.
Merge this delivery as unreleased work, complete those phases and the manual
notebook handoff, then validate the final release candidate before selecting
the next major version. C05/C06 remain separate work; this delivery does not
authorize their implementation. Keep `__version__`, CITATION and release tags
at 4.0.1; the changelog keeps these changes under Unreleased.

The configured PyPI Trusted Publisher is not a publication or a passing release
gate. At release time, verify its exact repository/workflow/environment identity.
Use a protected GitHub environment and the PyPA publishing action, with OIDC
`id-token: write` limited to the publishing job and no long-lived PyPI token.
Build once from the merged tagged commit, test the distributions, and publish
the same distributions/checksums to the GitHub release and PyPI. Do not rebuild
different artifacts between destinations or publish on ordinary branch pushes.
See the [PyPI publisher guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [PyPA workflow guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).
No publication workflow is added or triggered by this delivery.

Fresh delivery validation is recorded below. Final hosted CI/Codex review
results will be recorded in the PR discussion before merge. The historical
source reports retain their original validation scope and do not substitute
for final-head hosted checks.

- Fresh locked Python 3.12 source suite: **1,656 passed** (510 warnings), 249.01 s.
- Fresh newer Python 3.12 installed-wheel suite, outside the checkout with
  isolated Python: **1,640 passed** (305 warnings), 288.86 s. Dependency checks
  pass; the wheel's runtime/data payloads match both source and sdist.
- Strict HTML builds pass in locked and newer documentation environments.
  Notebook execution stays off. All three changed notebook schemas validate
  and their outputs remain clear. Ruff 0.17.0, lock and whitespace checks pass.
- Runtime files match the recorded six-cell Phase 2 wheel. The only notebook
  change from its final validation copy is a Markdown clarification about group
  capacity; computational cells and tests match. Fresh hosted checks still run
  on the final PR head rather than treating those older results as current CI.
- New delivery logs, recovery diff, distributions and checksums stay under
  `/tmp/arfs-pr-tree-mrmr/`. No runtime payload, test or notebook was changed
  while organizing commits; added delivery changes are documentation only.

## Confirmed follow-up plan

Recorded on 2026-10-17 against working-tree source at
`4fc3c7a52340a249a823a7982e0ec222df413fef` (ARFS 4.0.1), including the existing
uncommitted MRMR and preprocessing work. This is a plan, not implementation
authorization or a claim that these repairs are complete. Preserve R01–R33 and
their historical evidence. Implement one C phase per request, verify it, record
its handoff here, and stop. Do not reorder or absorb the separate
[tree preprocessing](tree-preprocessing.md) work.

Only confirmed defects and measured costs from the report assessment are
included. Targeted breaking changes are allowed, with migration notes and no
compatibility shim. Dependencies, statistical defaults, and unrelated source
remain unchanged except where a phase explicitly specifies a correction.

### Evidence and limits

The current-source check passed **309 tests** across shadow contracts,
all-relevant aggregation, importance selection, MRMR ranking/sampling, and Gower
distances. Additional in-memory probes demonstrated:

- Both BoostAGroota and GrootCV report `[2, 1]` for support `[True, False]`.
- VariableImportance classification defaults reach training as `objective='rmse'`
  and `stratified=False`; zero-total GrootCV importance becomes NaN.
- Leshy changes its integer seed to a RandomState and changes the supplied
  estimator's tree count and seed. MRMR cloning succeeds, but changing task and
  n_jobs leaves its default callbacks bound to regression and the old job count.
- MRMR with `return_scores=True` raises AttributeError in `fit_transform`.
- Categorical missing-value filling raises TypeError; an invalid missing-value
  policy silently fills zeros. Importing the affected preprocessing/selector
  submodules resets NumPy's global seed; bare `import arfs` does not.
- BoostAGroota's plot can color a rejected feature blue and a selected feature
  gray. GrootCV's default five-fold split of 100 rows with five minority labels
  produces two validation folds with no minority observations.

The environment was Python 3.12.7, NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1,
LightGBM 4.6.0, SHAP 0.47.1, and pytest 8.3.5, importing `src` through PYTHONPATH.
The maintained-suite command was:

```sh
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/arfs-report-check-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
  /tmp/arfs-association/py312-locked/bin/python -m pytest \
  tests/test_shadow_contracts.py tests/test_allrelevant_aggregation.py \
  tests/test_importance_selection.py tests/test_mrmr_ranking.py \
  tests/test_mrmr_sampling.py tests/test_gower_distances.py \
  -q --tb=short -p no:cacheprovider
```

Verify or recreate that disposable environment before reuse. The ad hoc probes
were not saved as executable evidence; each repair must add its maintained
regression before claiming completion. The full suite, optional backends,
installed-wheel matrix, and statistical calibration were not rerun for this
assessment. Passing existing tests does not disprove the newly reproduced bugs.

### C01 — Task, ranking, and zero-importance correctness

**Status: Complete on `fix/c01-task-ranking-zero-importance` (based on `a8d8672` on `main`).** Original report items 3, 4, and the zero-total part of 5.

- In [allrelevant.py](../src/arfs/feature_selection/allrelevant.py), assign rank 1
  to selected BoostAGroota/GrootCV features and rank 2 to rejected features.
- In [variable_importance.py](../src/arfs/feature_selection/variable_importance.py),
  use regression, binary, or multiclass defaults from task when no objective is
  supplied. Merge defaults into a copy of caller parameters, honor explicit
  objectives, derive stratification from the resolved objective, and resolve
  num_class for multiclass training without mutating caller dictionaries.
- Make SHAP/native zero-total normalization finite in BoostAGroota and GrootCV.
  Entirely uninformative fits select nothing. Reject nonfinite importance;
  preserve signed permutation normalization and nonzero threshold equality.
- Verify binary/multiclass default training and objective overrides, rank/support
  agreement, all-zero and mixed-zero iterations/folds, finite histories, and
  unchanged permutation-score behavior.

#### C01 handoff — 2026-10-17

- **Changes and behavior changes:**
  - `BoostAGroota.fit` and `GrootCV.fit` in [allrelevant.py](../src/arfs/feature_selection/allrelevant.py) now set `ranking_ = np.where(self.support_, 1, 2)`, assigning rank 1 to selected features and rank 2 to rejected features, aligned with `feature_names_in_` and `support_`, without changing selected feature sets.
  - `_compute_varimp_lgb` in [variable_importance.py](../src/arfs/feature_selection/variable_importance.py) resolves `objective` on a copied parameter dictionary from `task` (`regression -> rmse`, `classification -> binary`, `multiclass -> multiclass`) when `lgb_kwargs` is `None` or omits `objective`, while preserving explicit objectives and other caller keys. Validation-split `stratified` is derived from whether the resolved objective is a recognized classification objective (`binary`, `multiclass`, `softmax`, `multiclassova`, `multiclass_ova`, `ova`, `ovr`), rather than from `task`. For classification objectives, labels are validated and converted to integers (`bool -> astype(int)`, integral finite floats to `int`, rejecting non-numeric/string/non-integral/non-finite values and targets with `< 2` distinct classes), resolved `binary` with $K > 2$ classes is promoted to `multiclass` with `num_class = K` and a `UserWarning` (raising `ValueError` if an explicit `num_class`/`num_classes` conflicts with $K$), inferred multiclass requires labels `{0..K-1}`, explicit `num_class = n` requires labels in `{0..n-1}`, and stratification is pre-checked before predictor encoding or training. Caller dictionaries remain unmutated.
  - `_reduce_vars_sklearn`, `_compute_importance`, `_merge_importance_df`, `_reduce_vars_lgb_cv`, and the shared importance helpers (`_get_shap_imp`, `_get_shap_imp_fast`, `_get_perm_imp`, `_get_imp`) in [allrelevant.py](../src/arfs/feature_selection/allrelevant.py) validate finite importance values and totals (removing `_reduce_vars_sklearn`'s `except ValueError:` handler so nonfinite errors propagate), map zero-total iterations/folds to finite zeros that remain in the mean/median aggregation, and require strictly positive summary importance (`Mean > 0` in `BoostAGroota`, `Med > 0` in `GrootCV`) in addition to `>=` threshold comparison so uninformative fits yield finite zero summaries/cutoffs and select nothing. Signed permutation normalization and nonzero threshold equality are unchanged.
  - Updated affected docstrings in [allrelevant.py](../src/arfs/feature_selection/allrelevant.py) and [variable_importance.py](../src/arfs/feature_selection/variable_importance.py), method notes in [Methods overview.rst](../docs/Methods%20overview.rst), and [CHANGELOG.md](../CHANGELOG.md).
- **Commands, environments, and results:**
  - Pre-fix regression verification on `a8d8672`: **57 failed, 111 passed** across `tests/test_allrelevant_aggregation.py` and `tests/test_importance_selection.py`, reproducing inverted `[2, 1]` rankings, `FloatingPointError` / `NaN` on zero-total iterations/folds, swallowed `ValueError` on nonfinite importances, missing `objective` / `num_class`, and unstratified classification splits.
  - Focused post-fix caller suites on source (`PYTHONPATH=src`, `/tmp/arfs-association/py312-locked/bin/python` — Python 3.12.7, NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0, SHAP 0.47.1, pytest 8.3.5):
    ```sh
    env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
      MPLCONFIGDIR=/tmp/arfs-report-check-mpl OPENBLAS_NUM_THREADS=1 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
      /tmp/arfs-association/py312-locked/bin/python -m pytest \
      tests/test_allrelevant_aggregation.py tests/test_importance_selection.py \
      tests/test_permutation_importance.py tests/test_shadow_contracts.py \
      tests/test_observation_boundaries.py tests/test_category_identity.py \
      tests/test_leshy_support.py tests/test_allrelevant.py \
      tests/test_plot_contracts.py tests/test_benchmark_contracts.py \
      -q --tb=short -p no:cacheprovider
    ```
    Result: **642 passed**.
  - Full maintained source suite (`PYTHONPATH=src`, Python 3.12.7 locked):
    ```sh
    env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
      MPLCONFIGDIR=/tmp/arfs-report-check-mpl OPENBLAS_NUM_THREADS=1 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0 \
      /tmp/arfs-association/py312-locked/bin/python -m pytest tests \
      -q --tb=short -p no:cacheprovider
    ```
    Result: **1,807 passed** (518 warnings, 201.95 s), including 151 new C01 regression cases.
  - Installed-wheel validation outside the checkout in `/tmp/opencode/arfs-c01/run` with `python -I`, `uv pip check`, no `setuptools` installed, all 19 source/sdist/wheel files (including 3 data payloads) and all 16 installed module SHA256 hashes verified:
    - `py39-locked` (Python 3.9.25, NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0, SHAP 0.47.1, pytest 8.3.5): **1,718 passed** (519 warnings, 298.33 s).
    - `py312-locked` (Python 3.12.7, NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0, SHAP 0.47.1, pytest 8.3.5): **1,718 passed** (518 warnings, 291.65 s).
    - `py312-newer` (Python 3.12.7, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, LightGBM 4.7.0, SHAP 0.53.0, pytest 9.1.1): **1,702 passed** (313 warnings, 306.79 s; the 16-test difference is pandas 2.2 vs 3.0 `copy_on_write` parameterization).
  - Static and lock checks: `uv tool run --from ruff==0.17.0 ruff check .` (0 diagnostics), `uv lock --check`, and `git diff --check` all pass.
- **Remaining limits and next phase:**
  - Estimator ownership, `MRMR.fit` return contracts, and module/selector RNG threading remain in **C02**. Missing-value policy and stratified `GrootCV` folds remain in **C03**. Multi-round `BoostAGroota` history/plotting, quiet modes, and documentation examples remain in **C04**. MRMR storage/reduction and Gower memory documentation remain in **C05–C06**.
  - Optional `fasttreeshap` execution, full notebook reruns, and the remaining Python 3.9 minimum / 3.10 / 3.11 cells were not rerun in this session. Next confirmed follow-up: **C02**.

### C02 — Estimator ownership, fit contracts, and randomness

**Status: Complete locally on `fix/c02-estimator-contracts-randomness` (based on `026b7a1` on `main`).** Confirmed parts of original items 6, 8, and 13.

- Give Leshy an owned estimator clone and separate runtime RNG/importance state.
  Fit and optional-backend fallback must preserve constructor parameters and the
  caller's estimator. Use integer seeds for its existing learner-seed updates.
  Preserve BoostAGroota's already-cloned learner ownership and supplied seed;
  resolve its optional-backend fallback without overwriting importance.
- Resolve MRMR's default callbacks during fit, storing resolved functions in
  fitted attributes. Recompute them after set_params; preserve custom callbacks
  and the existing row-sampling contract.
- Remove return_scores and always return self from MRMR.fit. Scores remain in
  selected_features_, relevance_, redundancy_, and ranking_. Update existing
  sampling/ranking tests and examples that unpack the tuple; add migration notes.
- Remove module-level np.random.seed calls. Add random_state=None to
  BoostAGroota and isolation sampling, and random_state=2652124 to GrootCV.
  Thread selector RNGs into shadows, default GrootCV folds, and isolation
  fitting/draws. Preserve the fixed internal holdout procedure; do not introduce
  refreshed splits or override BoostAGroota's learner seed.
- Verify parameter and caller ownership, clone and refit behavior, updated task
  and n_jobs callbacks, pipelines, import-time RNG preservation in fresh
  processes, and repeatable integer-seeded selection/sampling. None retains
  normal sklearn RNG semantics; do not claim that fitting with None never
  consumes global randomness.

#### C02 handoff — 2026-10-18

- **Changes and migration:**
  - Leshy configures an owned `estimator_` clone, keeps its runtime RNG in
    `random_state_`, and draws integer learner seeds. The caller's estimator,
    including an already-fitted model, and constructor parameters are preserved.
    Importance helpers continue fitting their own clones. Leshy and BoostAGroota
    resolve optional-backend fallback in `importance_` on each fit without
    replacing the requested `importance` parameter.
  - MRMR resolves default callbacks into `relevance_func_` and `redundancy_func_`
    at fit time using the current `task` and `n_jobs`. Custom callbacks and the
    shared relevance/redundancy sample are preserved. Remove `return_scores`;
    `fit` always returns self. Read `selected_features_`, `relevance_`,
    `redundancy_` and `ranking_` after fitting instead of unpacking a tuple.
  - Removed the module-level NumPy seed resets in preprocessing, base and
    unsupervised selectors. Added `random_state=None` to BoostAGroota and the
    isolation sampling entry points, and `random_state=2652124` to GrootCV.
    Runtime RNGs control shadows, default repeated folds, and isolation fitting
    and draws. Integer seeds replay; supplied RandomState instances advance;
    None can consume NumPy's global stream. BoostAGroota retains its learner's
    seed, which must be set separately for reproducible stochastic training.
    Custom GrootCV splitters retain their RNG policy; fixed internal holdouts
    and GrootCV learner parameters are unchanged.
  - Updated affected ranking, sampling, benchmark and shadow tests, the reusable
    selection screen, method/API notes, changelog and two notebook calls. Refit
    and recheck selections after these RNG changes; there is no compatibility
    shim for `return_scores`.
- **Validation:**
  - Final full maintained Python 3.12.7 locked source suite: **1,864 passed**
    (520 warnings, 183.57 s). Log: `/tmp/arfs-c02-full-final.log`.
  - The initial new regression file reproduced **14 failures, 3 passes** before
    implementation. It now contains 32 cases covering ownership, clone/refit,
    callbacks after `set_params`, pipelines, optional-backend resolution,
    custom/default folds, integer replay, None/RandomState semantics, and fresh
    process import-time RNG preservation.
  - Focused source/caller suites initially passed **210 tests**. The expanded
    C02/sampling suites pass **134 tests** each on Python 3.9.25 locked and
    Python 3.12.7 newer dependencies. Updated benchmark contracts pass **20 tests**.
    Existing benchmark assertions that expected learner/seed mutation were
    revised to check preserved caller state and owned runtime state.
  - Source validation uses `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`,
    `MPLBACKEND=Agg`, `MPLCONFIGDIR=/tmp/arfs-c02-mpl`,
    `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
    `NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2`, with `PYTHONHASHSEED=0` for the
    full suite. Commands:
    ```sh
    /tmp/arfs-association/py312-locked/bin/python -m pytest tests \
      -q --tb=short -p no:cacheprovider
    # Repeat the focused command with py39-locked and py312-newer:
    /tmp/arfs-association/py39-locked/bin/python -m pytest \
      tests/test_estimator_randomness.py tests/test_mrmr_sampling.py \
      tests/test_isolation_sampling.py tests/test_sampling_outputs.py \
      -q --tb=short -p no:cacheprovider
    ```
  - Verified source imports and versions: locked cells use NumPy 2.0.2,
    pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0, SHAP 0.47.1 and pytest
    8.3.5. The newer Python 3.12 cell uses NumPy 2.5.4, pandas 3.0.6,
    scikit-learn 1.9.1, LightGBM 4.7.0, SHAP 0.53.0 and pytest 9.1.1.
  - Strict locked HTML build passes with source imports, Agg and one-thread
    BLAS/OpenMP:
    `/tmp/arfs-r33-delivery/docs-locked/bin/python -m sphinx -W --keep-going
    -b html docs /tmp/arfs-c02-html`. Notebook execution stays disabled; both
    edited notebook schemas validate. Ruff 0.17.0, `uv lock --check` and
    `git diff --check` pass. Logs stay in `/tmp/arfs-c02-*.log`.
- **Limits and next phase:** installed-wheel/release matrices, real optional
  FastTreeSHAP/CatBoost/XGBoost execution, full notebook reruns and repeated-seed
  statistical calibration were not rerun. C03 remains next: missing-value policy
  and classification folds. C04 diagnostics/quiet modes/examples and C05–C06
  memory work remain pending. Git delivery and Codex bot review are tracked in
  the PR; no release is included.

##### C02 Codex review follow-up — 2026-10-18

Codex identified that GrootCV's lazy repeated-fold generator shared its RNG
with shadow permutations. Later repeats could therefore change when the feature
count changed despite the same fold seed. Default folds now receive a copy of
the initial RNG state, keeping their full sequence independent of shadows.
Custom splitters retain their own policy. The expanded three-repeat regression
also refits with fewer features; before the fix it failed for default folds
and passed for custom folds (**1 failed, 1 passed**).

Also addressed CodeRabbit's related fit-state finding with an explicit
`support_` check in MRMR's support accessor. Early callback resolution no
longer makes `get_support()` mistake a rejected fit for a completed fit.
Both invalid-count and invalid-sample-size cases reproduced `AttributeError`
before the fix and now require `NotFittedError`.

After the fixes, locked Python 3.12 source checks passed **168 tests** across
the RNG/aggregation/shadow suites and **99 tests** across the RNG/MRMR
ranking/sampling suites. Ruff 0.17.0 and whitespace checks pass. Logs:
`/tmp/arfs-c02-codex-fold-green.log` and `/tmp/arfs-c02-review-fit-green.log`.

### C03 — Missing-value handling and classification folds

**Status: Complete on `fix/c03-missing-values-classification-folds` (based on `199bc93` on updated `main`).** Confirmed parts of original items 2 and 7.

- In [association.py](../src/arfs/association.py), accept only drop, fill, or
  None. Keep numeric zero filling; fill nominal missing values with a
  collision-free missing category, supporting category/string/Boolean inputs
  without changing caller data. Update the missing-value policy documentation.
- Keep documented listwise deletion and response/weight validation. Do not
  silently replace it with featurewise deletion or change degrees of freedom.
- Use RepeatedStratifiedKFold for recognized GrootCV classification objectives
  and retain RepeatedKFold for regression. Respect supplied splitters and their
  precedence. Reject infeasible default stratification with an actionable error.
- Verify invalid policies, category dtype/missingness variants, sentinel
  collisions, input preservation, balanced/imbalanced labels, insufficient class
  counts, objective aliases, custom splitters, and unchanged regression folds.

#### C03 handoff — 2026-10-18

- The shared association input check now accepts only `"drop"`, `"fill"`, or
  `None`, rejecting invalid policies with `ValueError`. Filling copies the
  frame, keeps numeric zero filling, and converts nominal columns with missing
  values to categorical columns with a collision-free missing level. Existing
  category order and unused levels are preserved; nullable strings/Booleans,
  mixed nominal identities and entirely missing nominal columns are supported.
  Complete columns retain their dtypes. Listwise deletion, positional response
  and weight alignment, validation and frequency-weight degrees of freedom are
  unchanged. Updated public policy docstrings, method notes and changelog.
- GrootCV chooses `RepeatedStratifiedKFold` from the resolved classification
  objective, including existing aliases and objectives from `lgbm_params`;
  explicit `objective` retains precedence. Default stratification requires two
  observed classes and at least `n_folds` rows per class, otherwise an error
  explains how to reduce `n_folds` or supply `folds`. Regression retains
  `RepeatedKFold` and the copied fold RNG state from C02. Supplied splitters,
  lists, iterators and generators take priority and retain their own policy;
  splitters receive data when counting their splits.
- Added 92 maintained regressions in
  [test_missing_value_policy.py](../tests/test_missing_value_policy.py) and
  [test_groot_classification_folds.py](../tests/test_groot_classification_folds.py).
  The final tests against an untouched temporary source copy of `199bc93`
  reproduced **80 failures, 12 passes**. The native rare-class test verifies
  that every five-fold validation set receives one of five minority rows.
- Validation on checkout source, Python 3.12.7:
  - Locked snapshot (NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, LightGBM
    4.6.0, SHAP 0.47.1, pytest 8.3.5): full suite **1,958 passed**
    (529 warnings, 308.20 s).
  - Existing newer snapshot (NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
    LightGBM 4.7.0, SHAP 0.53.0, pytest 9.1.1): C03 plus observation-boundary,
    category-identity, estimator-randomness and aggregation suites
    **443 passed** (80 warnings, 132.26 s). This was not a fresh resolution.
  - Strict HTML documentation build with notebook execution disabled,
    Ruff 0.17.0, `uv lock --check` and `git diff --check` pass.

Reproduction from the checkout root:

```sh
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/arfs-c03-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0
/tmp/arfs-association/py312-locked/bin/python -m pytest tests -q --tb=short -p no:cacheprovider
/tmp/arfs-association/py312-newer/bin/python -m pytest \
  tests/test_missing_value_policy.py tests/test_groot_classification_folds.py \
  tests/test_observation_boundaries.py tests/test_category_identity.py \
  tests/test_estimator_randomness.py tests/test_allrelevant_aggregation.py \
  -q --tb=short -p no:cacheprovider
/tmp/arfs-association/docs-locked/bin/sphinx-build -W --keep-going -b html docs /tmp/arfs-c03-html
uv tool run --from ruff==0.17.0 ruff check .
uv lock --check
git diff --check
```

Logs stay in `/tmp/arfs-c03-{baseline,full,newer,docs}.log`; generated HTML
stays in `/tmp/arfs-c03-html`. Verify these disposable environments before reuse.
No dependency, version or historical report changes were made; other
statistical defaults are unchanged. Refit and recheck classification selections
because the default folds change; invalid missing-value policies now fail
instead of silently filling.
The installed-wheel/Python matrix, optional backends, notebook execution and
statistical calibration were not rerun. Commit and PR delivery are tracked
separately; no release was made.
**C04 is next**; C04–C06 remain outside this implementation.

##### C03 Codex review follow-up — 2026-10-18

[PR #73](https://github.com/ThomasBury/arfs/pull/73) review found that refitting
with a consumed iterator reached an undefined `shadow_names`. The shared CV
reducer now rejects empty or exhausted folds with `ValueError`, explaining how
to use a reusable list/splitter or a new iterator. No constructor parameter is
replaced or cached. Public docs state that iterators are consumed by fitting.

Extended the existing custom-fold regression to refit every supported form and
added four empty-input cases (list, iterator, generator and splitter). Before
the fix, these checks gave **6 failures, 2 passes**; both reusable forms still
refit with the same splits. C03 now has **96 regression cases**.

After the fix, Python 3.12 source checks passed **216 tests** across
`test_groot_classification_folds.py`, `test_allrelevant_aggregation.py`,
`test_estimator_randomness.py` and `test_shadow_contracts.py` on the locked
snapshot. All **46 classification-fold tests** pass on the existing newer
snapshot. Commands use the same environment and pytest options recorded above.
Strict HTML documentation, Ruff 0.17.0 and whitespace checks pass. Logs stay in
`/tmp/arfs-c03-review-{before,locked,newer,docs}.log`. The full local source suite
above predates this guard; hosted wheel checks run again on the follow-up commit.

### C04 — Truthful diagnostics, quiet modes, and examples

**Status: Complete locally on `fix/c04-diagnostics-quiet-examples` (based on `4c8b4c2` on `main`).** Confirmed plotting/scale issues in item 5, unwanted output
in item 15, and documentation defects in item 16.

- Stop overwriting BoostAGroota histories across reduction rounds. Make
  sha_cutoff_df contain the final round only, matching mean_shadow. Plot those
  candidates against that round's cutoff and color by selected feature names.
  Derive the complete-feature ranking_absolutes_ from first-round means; document
  both diagnostic meanings and their changed table scope.
- Disable tqdm and routine print/early-stopping messages through existing
  verbose, silent, and verbose_eval controls, including GradientBoosting's
  routine split/save/load output. Keep warnings visible and parameter names intact.
- Repair Leshy's ndarray/Boruta example and make_fs_summary's invalid
  VariableImportance arguments. Document callable association defaults and remove
  the nonexistent collinearity_summary_ claim rather than adding a new feature.
- Keep the collinearity diagonal zero: setting it to one made the current
  elimination routine discard both features of a zero-association matrix.
  Retain canonical exports from arfs.feature_selection.
- Verify multiple elimination rounds, selected/rejected box colors, matching
  history/cutoff scales, complete first-round ranking, quiet stdout/stderr, and
  bounded executable examples. Check collinearity keeps unrelated features.

#### C04 handoff — 2026-10-18

- **Changes and migration:** BoostAGroota no longer merges and overwrites
  histories from different reduction rounds. `sha_cutoff_df` contains only
  final-round normalized iteration scores, with real candidates first and
  their shadows second, matching `mean_shadow`. Its plot shows those candidates
  and colors boxes by selected names, including integer feature names.
  Consumers expecting every input feature in this table must adapt.
  `ranking_absolutes_` orders all input features by first-round means, with
  input-order ties; it is separate from selected/rejected `ranking_`.
- **Quiet controls:** Leshy's `verbose<=0`, BoostAGroota/GrootCV's `silent=True`,
  and GradientBoosting's `verbose_eval<=0` now suppress routine prints and
  progress bars. GrootCV threads `silent` to its early-stopping callback.
  VariableImportance inherits the GBM fix through its existing `verbose`
  argument. Split/save/load and custom-metric messages follow the same GBM
  control. Warnings remain visible; supplied learners retain their own logging
  controls. Public parameter names, training and selection rules are unchanged.
- **Examples and contracts:** Leshy's example now creates a small local
  DataFrame instead of importing Boruta or reading absent CSV files. The
  summary example uses supported `VariableImportance(lgb_kwargs=...)` options
  in a bounded local pipeline. Corrected callable association defaults and
  removed the nonexistent `collinearity_summary_` claim. The existing zero
  diagonal and canonical `arfs.feature_selection` exports are preserved.
- **Regression evidence:** The initial 20 C04 checks reproduced **12 failures,
  8 passes** before implementation. The final `tests/test_diagnostics.py` has
  **24 cases**, covering one to three rounds, two cutoff scales, complete
  first-round ranking, stable ties/refit state, selected/rejected plot colors,
  quiet/noisy stdout and stderr, visible warnings, executable docstrings and
  unrelated-feature retention under both collinearity methods.
- **Validation:** The full maintained source suite passed **1,986 tests**
  (534 warnings, 225.13 s), including all 24 C04 cases, on the locked Python
  3.12.7 snapshot (NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1,
  LightGBM 4.6.0, SHAP 0.47.1, pytest 8.3.5). All **24 C04 cases**
  also pass on the existing newer Python 3.12 snapshot (NumPy 2.5.4, pandas 3.0.6,
  scikit-learn 1.9.1, LightGBM 4.7.0, SHAP 0.53.0, pytest 9.1.1).
  The strict HTML documentation build, Ruff 0.17.0, `uv lock --check` and
  whitespace checks pass. Documentation builds keep notebook execution off.

Reproduction from the checkout root:

```sh
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/arfs-c04-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 PYTHONHASHSEED=0
/tmp/arfs-association/py312-locked/bin/python -m pytest tests -q --tb=short -p no:cacheprovider
/tmp/arfs-association/py312-newer/bin/python -m pytest tests/test_diagnostics.py -q --tb=short -p no:cacheprovider
/tmp/arfs-association/docs-locked/bin/python -m sphinx -b html -W --keep-going docs /tmp/arfs-c04-html
uv tool run --from ruff==0.17.0 ruff check .
uv lock --check
git diff --check
```

Source and newer-snapshot test logs stay in `/tmp/arfs-c04-source-tests.log`
and `/tmp/arfs-c04-newer-tests.log`; generated HTML stays in
`/tmp/arfs-c04-html`. Verify these disposable environments before reuse.
The installed-wheel/Python matrix, optional backends, full notebook execution
and statistical calibration were not rerun. No dependency, version, historical
report or release changes are included. PR delivery and bot review are tracked
in the pull request. **C05 is next**;
C05–C06 remain outside this implementation.

### C05 — MRMR storage and built-in reduction costs

**Status: Pending.** Confirmed parts of original item 9.

- Replace the full redundancy square with columns allocated only for selected
  features, reducing table storage from O(p²) to O(p k). Vectorize recognized
  mean/max reducers; retain the existing path for arbitrary callable reducers.
- Preserve frequency weights, floors, perfect-association handling, domain
  restrictions, stable ties, custom callbacks, row sampling, and selected order.
- Compare against existing hand-checked ranking tests and small dense references
  for mean/max/custom reducers. Rerun the existing
  [MRMR benchmark](mrmr-performance.md), recording runtime and peak process memory
  separately from retained-table memory. Its recorded 10,000-feature table is
  about 764 MiB; no 10–50× speedup is promised.
- Do not add persistent rank/category caches or rewrite Cramér's V in this phase.
  Those are separate optimization proposals, not necessary for the storage fix.

### C06 — Document exact Gower sampling's memory cost

**Status: Pending.** Confirmed scaling in original item 11; documentation only.

- Document dense float64 distance storage as 8 N² bytes: approximately 20 GB at
  50,000 rows and 80 GB at 100,000, before clustering and other temporary memory.
  State that exact Gower sampling is unsuitable when this exceeds the caller's
  memory budget. Do not imply that output sample size bounds distance storage.
- Add no row cap, memory-budget parameter, or approximate algorithm, following
  the user's explicit choice. Preserve per-pair weight scaling: normalizing all
  weights once can underflow valid small weights when dominant features are
  missing. The existing extreme-weight/missingness regression must remain green.
- Verify arithmetic and documentation references without allocating huge arrays.

### Changes from the proposed plan

Only confirmed repairs and measured costs remain. The methodology experiment
phase, fresh iteration splits/learner seeds, changed SHAP evaluation rows,
featurewise deletion, universal ndarray support, replacement importance engine,
and unsupported speed claims are excluded. Independent fold shadows, the zero
collinearity diagonal, existing exports, and valid LightGBM n_jobs=0 behavior
are retained. This avoids treating design choices as proven bugs or introducing
regressions while fixing demonstrated failures.

## R33 delivery and 4.0.0 release — 2026-10-16

The user authorized staging, topical Conventional Commits, a dedicated branch,
a pull request, Codex review, fixes for its findings, merge, branch removal and
a GitHub release with a version bump/tag when appropriate. Branch:
`fix/r33-release-evidence`, based on merged R32 (`dbbd3c5`). The original R33
commits `aabb238` and `fa08538` become `c047a03` and `fb277ea`. The changelog and
handoff conflicts preserve both R32 review fixes and all earlier delivery notes.

**Release decision: prepare 4.0.0.** No RXX phase remains. The intentional public
API/helper removals and incompatible legacy GBM artifacts warrant a major release.
`664a055` updates the package/citation version and release date, moves the repairs
out of Unreleased and adds migration notes. The entire lock remains unchanged:
local version metadata is dynamic, and dependency pins/hashes are preserved.
After current-head validation, hosted CI and Codex review pass, merge this PR,
tag its merged commit `4.0.0` and publish a GitHub release with the validated wheel,
source archive and checksums. No PyPI publication workflow is configured.

[PR #66](https://github.com/ThomasBury/arfs/pull/66) delivers this phase. On
`541d7ab`, hosted push/PR checks and Codex review pass with no findings. The final
handoff-only commit still needs its own hosted checks and Codex re-review before
merge. CodeRabbit is not a gate. No new raw evidence is tracked; it stays local
and ignored under `evidence/r33-delivery/`.

The fresh 4.0.0 wheel passes **1473 tests in five cells / 1457 in newer 3.12**:
Python 3.9 locked, 3.9 exact direct minima, 3.10 locked, 3.11 locked, 3.12 locked
and 3.12 newer. Local and hosted totals agree, with no skips or expected failures:
**8822 local full-suite executions**. All eleven direct dependency floors match
in the minimum cell. Source/sdist/wheel code/data payloads, every installed
payload and all copied test/documentation fixtures match current source.

Both fresh strict HTML builds pass with zero Sphinx warnings against the 4.0.0
wheel, without a setuptools workaround. Each snapshot freshly executes the same
five bounded workflows plus R32's three commented regression examples:
**93 code cells and 13 figures** per snapshot, including added setup/assertions
and the three no-op cells. All 26 figures are byte-identical to the previously
inspected R33 outputs. All 17 notebook schemas validate. The data substitutions,
omitted cells and remaining full-notebook gaps stay explicit in the ledgers.

All 112 runtime function/method annotations evaluate on Python 3.9 and newer 3.12.
The shipped-sdist helper/resource smoke suite passes 25 tests in another clean
installation. Four unchanged optional-backend boundary checks pass unmasked in
each 3.12 snapshot, as does the historical reduced-feature time-series check.
Direct Ruff reports zero diagnostics; its former count baseline and scripts
are gone. Lock metadata/pins, whitespace and all recorded command exits pass.

The existing fixed experiment retains its 1,120 measured fits; it is not rerun or
counted as additional seeds. Every runtime payload matches its measured wheel
apart from the package version and R32's class-frequency lookup. The screen
script and dependency versions are identical. In both snapshots, all 70 binary
training sets contain labels 0/1 and leave the old and new imbalance tests off,
so that lookup changes none of the experiment's parameter dictionaries. The
shared helper is called once before GrootCV's folds; the other selectors do not
use it. This checked boundary supports retaining the recorded measurements;
it does not establish calibration or support outside their original protocol.
The final candidate's six fresh suites separately exercise the repaired imbalance
branch and native fits. The notebook/platform/backend gaps remain explicit.

## R33 completion — 2026-10-15

The [R33 handoff](R33-release-evidence.md) records the expanded six-cell
installed-wheel matrix, exact-minimum snapshot, strict documentation builds,
1,120 fixed-protocol selector fits and narrowed support/statistical claims.
The user explicitly approved broader lint cleanup. Direct Ruff reports zero
diagnostics, and the temporary baseline and scripts are removed. Final rebuilt
wheel suites pass **1462 in five cells / 1446 in newer 3.12**; 25 shipped-sdist
tests pass in another clean installation. All 1,120 refreshed selector records
exactly match pre-cleanup selections, losses and baselines. Both strict HTML
builds and five bounded fresh-kernel workflows per snapshot pass; all 26 figures
are byte-identical to the inspected R32 outputs. The missing FastTreeSHAP raise
has an old-wheel failure/new-wheel success regression. Final evidence is under
`evidence/r33-release-evidence/lint-cleanup/`. There is no next repair batch.
Keep version 3.0.0/Unreleased; do not publish or bump it without a delivery request.
R01–R32 historical records are unchanged.
Implementation commit: `aabb238` on `fix/r33-release-evidence`, based on `f49bf9d`.

## R32 delivery — 2026-10-16

The user authorized staging, topical Conventional Commits, a dedicated branch,
a pull request, Codex bot review, fixes for its comments, merge and branch removal.
Branch: `fix/r32-documentation-workflows`, based on merged R30 (`8856009`). Only
the original R32 commits were rebased: `3f61981` becomes `48126e0` (documentation
workflows), and `f49bf9d` becomes `5a6e585` (validation handoff). The single status
line conflict was resolved without removing earlier delivery records. Original
hashes in the [R32 handoff](R32-documentation-workflows.md) describe the completed
local phase. This delivery includes R32 only; raw evidence remains local and
ignored, with no historical evidence added or removed.

The initial PR head `c7691ca` matches the recorded installed-wheel source,
test and documentation bytes. Both hosted push/PR matrices pass at that head:
**1458 / 1458 / 1442 tests** on Python 3.9 locked / Python 3.12 locked /
Python 3.12 newer. Its notebook sources match the recorded five-workflow
fresh-kernel ledgers in both snapshots: 90 code cells and 13 figures each.
Runtime code ASTs match R30 after docstrings are removed; runtime pins/hashes
are unchanged. Strict HTML builds, the lint gate and its two tests pass.

[PR #65](https://github.com/ThomasBury/arfs/pull/65) delivers R32. Codex reported
one valid P2 on `c7691ca`: regression's optional FastTreeSHAP examples still ran
without the backend. All three model cells are now commented out, their saved
outputs (including the traceback) are cleared, and the notebook/guide explain
how to opt in. Three new cases fail before the repair; all **11 documentation
checks** pass afterward in each installed-wheel snapshot. No runtime code changes.

Fresh locked/newer regression kernels include all three repaired cells alongside
the original bounded scope: **15 code cells and two figures per snapshot**,
without FastTreeSHAP installed. The optional cells produce no output; all four
figures are byte-identical to the earlier runs. The previously executed source
cells remain unchanged. Both post-review strict HTML builds pass with zero
Sphinx warnings, all 17 notebook schemas validate, and lint has zero new
diagnostics. New evidence is ignored under `evidence/r32-pr-review/`. The full
notebook/backend execution gaps in the R32 handoff remain. Current-head hosted
CI and Codex re-review must finish before merge. CodeRabbit is not a gate.

Codex's second review on `da0c716` found a valid P2 in the imbalance claim:
`_set_lgb_parameters` looked up label 0's frequency rather than the largest
frequency. The shared helper now uses `y_freq_table.iloc[0]`; the strict >80%
unweighted threshold is unchanged. Caller tracing covered GrootCV.fit,
_reduce_vars_lgb_cv, native training and all helper references. Six parameter
cases check both binary majority labels, integer labels without 0 at the private
helper boundary, and exact-80% controls; two real GrootCV fits verify the flag
reaches native LightGBM with either binary majority. Before the fix, four fail
and four pass as controls; all eight pass afterward. This fixes the frequency
lookup, without promising native training on arbitrary integer label sets.

A new sdist/wheel has matching source code/data payloads. Fresh full-suite
installed-wheel runs are in progress in all three snapshots; raw logs and
identities are ignored under `evidence/r32-imbalance-review/`. Both strict HTML
builds pass against this repaired wheel with zero Sphinx warnings. The original
execution scopes remain bounded; the earlier optional-cell sources are unchanged.
The final full-suite and Codex re-review results will be recorded on PR #65.

**Release decision: defer the version bump, tag and GitHub release.** GitHub's
latest release and this checkout remain `3.0.0`. R33 is implemented on a separate
local branch but is not part of this delivery or remote `main`. Deliver R33 and
check its final hosted results before choosing the next version; the intentional
API removals warrant a major release. Keep changes under Unreleased. R33 is the
next separate delivery.

## R30 delivery — 2026-10-16

The user authorized staging, topical Conventional Commits, a dedicated branch,
a pull request, Codex bot review, fixes for its comments, merge and branch removal.
Branch: `fix/r30-dataset-fixtures`, based on merged R29 (`9d35340`). Only the
original R30 commits were rebased: `db2f7f7` becomes `1bcfb87` (dataset fixtures),
and `0164300` becomes `7fe3ce5` (validation handoff). The original hashes in the
[R30 handoff](R30-dataset-fixtures.md) describe the completed local phase.
This delivery includes R30 and its affected callers/tests only. New evidence
remains local and ignored; no historical evidence is added or removed.

All 16 source module files and 34 maintained test files match the recorded R30
installed-wheel runs: **1450 / 1450 / 1434 passed** on Python 3.9 locked /
Python 3.12 locked / Python 3.12 newer. The handoff records 25 regression cases,
including exact row counts, repeatability, caller RNG preservation and a
network-free Titanic family fixture. The 68 affected source tests pass again
for delivery; the lint gate reports zero new diagnostics and its two tests pass.
Hosted CI and Codex review must finish on the PR before merge. CodeRabbit is
not a gate.

**Release decision: defer the version bump, tag and GitHub release.** GitHub's
latest release and this checkout remain `3.0.0`. R32 and R33 are implemented
on separate local branches, but are not part of this delivery or remote `main`.
Deliver those PRs and check their final hosted results before choosing the next
version; the intentional API removals warrant a major release. Keep changes
under Unreleased. R32 is the next separate delivery.

## R29 delivery — 2026-10-16

The user authorized staging, topical Conventional Commits, a dedicated branch,
a pull request, Codex bot review, fixes for its comments, merge and branch removal.
Branch: `fix/r29-plot-contracts`, based on `605cbff` on remote `main`.
The existing commits are `24db954` (plot contracts) and `56f4f6c` (validation
handoff). This delivery includes R29 and its affected callers/tests only.
New evidence remains local and ignored; no historical evidence is added or removed.

All 16 source module files and 33 maintained test files match the recorded R29
installed-wheel runs: **1425 / 1425 / 1409 passed** on Python 3.9 locked /
Python 3.12 locked / Python 3.12 newer. The [R29 handoff](R29-plot-contracts.md)
records the 53 regressions, visual checks, commands and limits. Hosted CI and
Codex review must finish on the PR before merge. CodeRabbit is not a gate.

**Release decision: defer the version bump, tag and GitHub release.** GitHub's
latest release and this checkout remain `3.0.0`. R30, R32 and R33 are already
implemented on separate local branches, but are not part of this delivery or
remote `main`. Deliver those PRs and check their final hosted results before
choosing the next version; the intentional API removals warrant a major release.
Keep changes under Unreleased. R30 is the next separate delivery.

## R28 delivery — 2026-10-14

The user authorized staging, topical Conventional Commits, the dedicated branch,
a pull request, Codex bot review, fixes for its comments, merge and branch removal.
Branch: `fix/r28-parallel-contracts`, based on `81dc6c9` on remote `main`.
The phase commits are `2181790` (parallel contracts) and `53c3699` (handoff).
This delivery includes completed R28 and its affected callers/tests only.

The source and maintained tests match the final rebuilt installed-wheel matrix:
**1372 / 1372 / 1356 passed** on Python 3.9 locked / Python 3.12 locked /
Python 3.12 newer. All 67 R28 regressions and 12 selected historical parallel
checks pass in the recorded cells. Package/source/test hashes, dependency/lock
checks, Ruff (330 total, zero new) and two gate unit tests are recorded in
[the R28 handoff](R28-parallel-contracts.md). New evidence remains local and
ignored; no historical evidence is removed or added to this delivery.
Hosted CI and Codex review are checked on the PR before merge. CodeRabbit is
not a delivery gate.

**Release decision: defer the version bump, tag and GitHub release.** The
checkout and latest published GitHub release remain `3.0.0` (verified during
this delivery). R29–R30, R32 and R33 still cover plot/selection correctness,
dataset fixtures, runnable documentation and final release evidence. Keep
these API repairs under Unreleased in CHANGELOG.md and choose the next version
after the remaining API changes and R33 checks are complete. No tag is needed
for this independently reviewable phase. Next remains **R29 alone**, with
R03/R10/R17 complete.

## R25–R27 delivery — 2026-10-14

The user authorized staging, topical Conventional Commits, a dedicated branch,
a pull request, Codex bot review, fixes for its comments, merge and branch removal.
Branch: `fix/r25-r27-remediation`, based on `34641a2` on remote `main`.
The five existing phase/handoff commits retain their Conventional Commit titles
and the independently validated R25, R26 and R27 histories. This delivery
includes those completed phases only; R28 remains the next separate repair.

[PR #61](https://github.com/ThomasBury/arfs/pull/61) delivers the changes. Codex's
first review on `7361d4d` reported one valid P2: zero-weight categories changed
ordinal-code spacing before the forest omitted their rows. The repair learns
category vocabularies from positive-weight rows only and maps unseen score-only
categories to -1 with native Index.get_indexer. Four new regressions reproduce
all failures on the old wheel and pass after the fix, including exact native
positive-row score equality with zero-weight rows physically removed. The draws
and KS screen remain unweighted, and returned rows retain original values/dtypes.

Final package source and all maintained test bytes match the post-review rebuilt
wheel matrix: **1305 passed** on Python 3.9 locked, **1305 passed** on Python 3.12
locked and **1289 passed** on Python 3.12 newer. All six selected historical R27
checks pass unmasked in both Python 3.12 cells; 217 affected source tests pass
in each dependency snapshot. All 13 module hashes and 19 source/sdist/wheel
payloads agree, with no setuptools workaround. Dependency/lock, Ruff (331 total,
zero new), gate unit tests, format, links and whitespace checks pass.

The original R27 phase commands/identities remain in [R27](R27-nearest-neighbors.md).
Post-review commands use the same dependency snapshots and locked export:

```sh
uv build --python /tmp/arfs-r27/py312-locked/bin/python --out-dir /tmp/arfs-r25-r27-review/dist
/tmp/arfs-r27/py312-locked/bin/python /tmp/arfs-r25-r27-review/check_payloads.py
/tmp/arfs-r27/py312-locked/bin/python /tmp/arfs-r25-r27-review/validate.py py39-locked
/tmp/arfs-r27/py312-locked/bin/python /tmp/arfs-r25-r27-review/validate.py py312-locked
/tmp/arfs-r27/py312-locked/bin/python /tmp/arfs-r25-r27-review/validate.py py312-newer
```

Exact commands, environment identities, hashes and logs are local under ignored
`reviews/evidence/r25-r27-delivery/`; no historical evidence is deleted or added
to this delivery. Initial hosted Ruff and both push/PR installed-wheel matrices
pass on `7361d4d`. The review fix needs current-head hosted CI and Codex re-review
before merge; those results are recorded on PR #61. CodeRabbit is not a gate.

**Release decision: defer the version bump, tag and GitHub release.** The checkout
and latest published GitHub release remain `3.0.0` (verified during delivery).
R28–R30, R32 and R33 still include parallel-label/worker contracts, plots,
dataset fixtures, runnable documentation and final release evidence. The changes
remain under Unreleased in CHANGELOG.md. These independently validated repairs
are ready for a PR now; choose the next version only after the remaining API
changes and R33 checks are complete. No further repair phase or release is part
of this delivery. Next remains **R28 alone**, with R06/R08 complete.

## R13–R24 and R31 delivery — 2026-10-14

The user authorized staging, topical Conventional Commits, a dedicated branch,
a pull request, Codex bot review, fixes for its comments, merge and branch removal.
Branch: `fix/r13-r24-remediation`, based on `64b0ce7` on remote `main`.
The 25 existing phase commits already have topical Conventional Commit titles;
they retain the phase histories and validation evidence below. This delivery
includes completed R13–R24 and R31 only; it does not start the next repair group.

The branch's package source and tests match the final R24 installed-wheel
validation: **1177 passed** on Python 3.9 locked, **1177 passed** on Python 3.12
locked and **1161 passed** on Python 3.12 newer. All 28 historical R24 distance
checks pass unmasked in both 3.12 cells. Source/artifact hashes, dependency checks,
lock validation, Ruff (**390 total, zero new**) and gate unit tests (**two passed**)
are recorded in [R24](R24-gower-distances.md).

[PR #60](https://github.com/ThomasBury/arfs/pull/60) delivers these changes. Hosted
Ruff and both push/PR installed-wheel matrices pass on f05cdba, with **1177 / 1177 /
1161** tests in the PR matrix. Codex's first review completed with two P2 reports.
Both fail to reproduce in the supported snapshots: check_cv already chooses KFold
for continuous binomial proportions, and binary_logloss is a metric name that
native LightGBM rejects as an objective before prediction. Real weighted fractional
CV matches independent native GLM folds, fractional grid search passes for frames/
arrays, and native binary labels/probabilities are correct. The exact checks and
results are in [delivery evidence](evidence/r13-r24-delivery/commands.txt).
The review threads receive these dispositions and Codex is asked to reconsider;
no package-source or established-test changes are needed for these reports.
CodeRabbit is not a delivery gate.

**Release decision: defer the version bump and GitHub release.** The checkout
and latest published release remain `3.0.0`. R25–R30, R32 and R33 remain pending,
including sampling weights/output/retries, nearest neighbors, parallel labels,
plots, dataset fixtures, runnable documentation and final release evidence.
The API changes in this branch remain under Unreleased in CHANGELOG.md; choose
the next version after those blockers and R33 checks are complete. No tag or
release is part of this delivery. The next recorded group remains **R25 → R26**,
conditional on its weighting, tiny-input and exhaustion policies.

## R04–R12 delivery — 2026-10-13

The user authorized staging, topical Conventional Commits, a dedicated branch,
a pull request, Codex bot review, fixes for its comments, merge and branch removal.
Branch: `fix/r04-r12-remediation`, based on `170b3e9` on remote `main`.
Earlier phase notes retain their historical uncommitted status; the commits
below now include those completed repairs and their dependencies.

| Scope | Commit |
| --- | --- |
| R04 installed-wheel runtime, lock and baseline tests | `e1c836f` |
| R05/R06/R08 observation boundaries, category identity and weighted statistics | `00447ec` |
| R07 owned encoding and faithful categorical shadows | `e2c9de3` |
| R09 coherent MRMR ranking and reduced Ruff allowances | `92af366` |
| R10 Leshy support/history | `acf99fc` |
| R11 weighted signed permutation scoring | `f68e482` |
| R12 explicit objective/output handling and shadow aggregation | `dda0b5b` |

The complete branch uses the R12 installed-wheel validation: **533 passed** on
Python 3.9 locked, **533 passed** on Python 3.12 locked and **517 passed** on
Python 3.12 newer. Its package-source hashes match the tested R12 wheel.
R12's handoff records the exact commands, versions and remaining check limits.
Hosted CI and Codex review are checked separately during PR delivery.

**Release decision: defer the version bump and GitHub release.** The latest
published release and checkout version are `3.0.0`. R13–R33 still include
release blockers: GLM scoring/design/likelihood, preprocessing, training and
persistence contracts, sampling, documentation workflows and final release
evidence. Passing repaired-phase tests does not settle those defects. Keep an
Unreleased changelog entry and choose the next version once the remaining API
and statistical changes and R33 checks are complete. No tag or release is
created as part of this delivery. Next implementation group remains R13 alone.

## How a phase becomes complete

1. Reconfirm the source revision, trace the affected callers, and reproduce the
   named failure from its linked evidence. If intervening changes fix it, verify
   that fact instead of implementing a duplicate repair.
2. Resolve only the explicit policy choices needed by the phase. Do not invent
   statistical semantics to make an assertion pass. A phase with such a choice
   remains incomplete until the choice is recorded in its API/docs and tests.
3. Make the smallest complete change. Promote meaningful reference cases into
   the established test suite, removing their expected-failure marker there.
   Preserve the historical diagnostic scripts/logs. A repair's test must fail on
   the old behavior and pass unmasked on the new behavior.
4. Run the phase's checks and affected callers in the frozen and newer snapshots
   where applicable. Packaging changes additionally need a newly built wheel,
   installed outside the checkout without the setuptools workaround. Record any
   inherited blocker rather than patching the environment silently.
5. Update this handoff with result, validation, and remaining dependencies; stop.
   “Green phase” means its own regression checks pass. It never implies that the
   whole package is green while other listed release blockers remain.

## Session batching and handoffs

Keep one phase per session by default. If the user requests a batch, combine at
most two small, related phases, complete and validate each sequentially, and
record each outcome separately. Stop at a phase boundary if a new API or
statistical decision, unresolved failure, or wider scope makes the batch too
large. Session boundaries and repair phases are not published releases.

R01–R33 are complete. The explicitly requested R02/R03, R10/R11,
R18/R31 and R25/R26 batches were implemented and validated sequentially.
R33's separate final release-evidence phase is complete. There is no remaining
implementation group; the recorded scope limits remain in its handoff.

| Phases | Session recommendation | Reason |
| --- | --- | --- |
| R01 | Complete; do not repeat | Installed-package checks are recorded in the R01 handoff |
| R02 + R03 | Complete sequential batch; do not repeat | Results and checks are recorded in the [compatibility handoff](R02-R03-compatibility.md) |
| R04 | Complete; do not repeat | Baseline, lock, and installed-wheel CI are recorded in the [R04 handoff](R04-runtime-baseline.md) |
| R05 | Complete; do not repeat | Boundary policy and checks are recorded in the [R05 handoff](R05-observation-boundaries.md) |
| R06 | Complete; do not repeat | Category identity, dtype agreement and installed-wheel checks are recorded in the [R06 handoff](R06-category-identity.md) |
| R07 | Complete; do not repeat | Owned encoding, categorical shadow roles and installed-wheel checks are recorded in the [R07 handoff](R07-shadow-contracts.md) |
| R08 | Complete; do not repeat | Frequency arithmetic and installed-wheel checks are recorded in the [R08 handoff](R08-statistical-arithmetic.md) |
| R09 | Complete; do not repeat | Bounded mixed relevance and coherent ranking are recorded in the [R09 handoff](R09-mrmr-ranking.md) |
| R10 → R11 | Complete sequential batch; do not repeat | [R10](R10-leshy-support.md) and [R11](R11-permutation-importance.md) record support/history consistency and the agreed signed normalization policy |
| R12 | Complete; do not repeat | [R12](R12-all-relevant-aggregation.md) records resolved objectives/output axes and the user-chosen median/mean shadow rules |
| R13 | Complete; do not repeat | [R13](R13-glm-scoring.md) records negative response-scale deviance, BIC direction and independent weighted folds |
| R14 | Complete; do not repeat | [R14](R14-glm-design.md) records fitted design/schema, input/support metadata, current family/link and fit return |
| R15 | Complete; do not repeat | [R15](R15-glm-likelihood.md) records zero-row omission, corrected criterion dimensions and truthful unpenalized refits |
| R16 | Complete; do not repeat | [R16](R16-threshold-contracts.md) records unique names, aligned callbacks, strict equality, empty stages, stable ties and fitted metadata |
| R17 | Complete; do not repeat | [R17](R17-importance-selection.md) records the crossing prefix, zero/nonfinite policy, one fit, equal-row explanation population and multiclass/backend repair |
| R18 → R31 | Complete sequential batch; do not repeat | [R18](R18-ordinal-schema.md) and [R31](R31-legacy-helpers.md) record full encoder schema, owned inverse inputs and the approved legacy-helper removals |
| R19 | Complete; do not repeat | [R19](R19-tree-partitions.md) records exact thresholds/leaf IDs, copied one-tree parameters, native missing routes and raw categorical/binary outputs |
| R20 | Complete; do not repeat | [R20](R20-transformer-inputs.md) records reusable options, array/dictionary preparation, midpoint scalar names and Patsy row/intercept policy |
| R21 | Complete; do not repeat | [R21](R21-gbm-contracts.md) records explicit split rejection, strict prediction schemas, objective aliases, categorical/offset boundaries and successful-refit state |
| R22 | Complete; do not repeat | [R22](R22-gbm-persistence.md) records versioned wrapper artifacts, coherent validated loads, legacy-file rejection and staged writes |
| R23 | Complete; do not repeat | [R23](R23-benchmark-contracts.md) records training-only encoding, array/name support, supported selectors and fresh configurations |
| R24 | Complete; do not repeat | [R24](R24-gower-distances.md) records pooled float64 ranges, cross/self distances, pairwise missingness and feature-weight/input contracts |
| R25 → R26 | Complete sequential batch; do not repeat | [R25](R25-sampling-outputs.md) and [R26](R26-isolation-attempts.md) record row-weight rejection/forwarding, original outputs, tiny identities and bounded full-data fallback |
| R27 | Complete; do not repeat | [R27](R27-nearest-neighbors.md) records actual queries, key exclusion/deduplication, finite-only stable ranking and count/schema boundaries |
| R28 | Complete; do not repeat | [R28](R28-parallel-contracts.md) records original labels, stable ordering, typed empty results and native worker counts capped to useful work |
| R29 | Complete; do not repeat | [R29](R29-plot-contracts.md) records feature coordinates, membership colors, fitted cutoffs, display limits and singleton clustering |
| R30 | Complete; do not repeat | [R30](R30-dataset-fixtures.md) records isolated reproducible draws, exact row counts and the correct Titanic alone flag |
| R32 | Complete; do not repeat | [R32](R32-documentation-workflows.md) records clean strict HTML builds, bounded fresh-kernel execution, optional dependency policy and remaining notebook gaps |
| R33 | Complete; do not repeat | [Final release evidence](R33-release-evidence.md), zero-diagnostic Ruff, six-cell wheel matrix and bounded calibration/support limits |

### Recorded groups for next-batch requests

When the user asks to "implement the next batch of RXX", read the current phase
statuses in this file and use these groups. **R01–R33 are complete; no recorded
group remains.** The three two-phase groups below are already complete.
R10/R11 are complete; their separate handoffs record support/history consistency,
weighted permutation scoring and the user-chosen absolute-total normalization.
R12 is complete: GrootCV uses maximum shadow medians across folds/repeats;
BoostAGroota uses maximum shadow means across iterations, both divided by cutoff.
R13 is complete: GLM scores are negative response-scale deviance; internal BIC
selection still minimizes BIC, and weighted CV clones each fold with positional
array/frame slicing and task-aware splits. R14 is complete: one fitted design
aligns named schemas, preserves supplied constants and separates input/support
metadata from generated-intercept output; fit uses current family/link and
returns self. R15 is complete: zero-weight GLM rows are omitted before fitting,
scoring and tuning; AIC/BIC use selected design rank and retained row count;
nonfinite search criteria fail clearly. The unpenalized-refit contract replaces
the near-ridge promise. R16 is complete: unique feature names, labelled statistic
alignment, fit-time parameter validation, valid empty collinearity stages, input-order
ties and fitted feature counts/guards. Strict equality and named projection are
preserved. R17 is complete: selection includes the crossing positive feature,
zero total selects nothing, nonfinite contributions fail and model output determines
multiclass axes. The user chose one fit/removal of n_iterations and equal validation-row
importance averaging with weighted training/loss. R18 → R31 is complete: full
encoder metadata describes passthrough output, inverse owns writable codes, and
current parameters apply at refit. The user chose removal of both public legacy
helpers without shims. R19 is complete: numeric thresholds and leaf IDs define
partitions, native missing routes and fit-time one-tree parameters are retained,
and raw binary output is probabilities. R20 is complete: constructor options survive
refits, arrays use fitted names/width, dictionaries retain their keys, midpoint
scalar names work, and Patsy honors NA_action/intercept choices. Explicit drop
removes predictor rows only. R21 is complete: group/stratified conflicts fail,
named prediction schemas are strict, native category choices and objective aliases
are honored, multiclass offsets retain their output axis, and successful refits
replace fitted state. R22 is complete: versioned wrapper metadata survives loading,
legacy Booster-only files are rejected, loaded state resets coherently and staged
writes protect existing models before commit. Model/PNG replacement remains two
separate operations. R23 is complete: encoders fit training rows only, arrays get
stable names, GrootCV comparisons fail clearly and each supported comparison uses
a fresh selector/model with copied seed state. R24 is complete: pooled float64
ranges, actual cross/self indexing, pairwise available-feature weighting and NaN
for undefined pairs, with validated input/mask/feature-weight contracts.
R25 → R26 is complete: Gower rejects row weights and missing-only columns,
returns synthetic means/modes with the original feature schema; isolation
forwards fit weights, omits zero-weight fit rows and returns original rows.
Distinct rounded-up 5%–95% proper subsets get 20 attempts each; tiny inputs
return unchanged and exhaustion returns full data. KS acceptance remains an
unweighted heuristic. R27 is complete: require an explicit one-row query,
score one distance per candidate, exclude the query key/missing candidate IDs,
keep the nearest finite-distance row per eligible ID, and reject insufficient
counts. Zero counts return empty arrays; ties keep candidate order. R28 is complete:
parallel chunks preserve original labels, use native negative worker counts capped
to useful work, return typed empty results and sort with stable ties. Serial Pearson
now shares the pairwise row layout. R29 is complete: plots retain feature
coordinates, color fitted membership, honor display limits and terminal cutoffs,
and bypass singleton clustering. R30 is complete: dataset augmentation and
generators use local seeded draws, odd sizes retain every row and Titanic
marks only zero-relative passengers alone. R32 is complete: strict HTML builds
use the docs extra without obsolete/mandatory optional backends, repaired examples
pass bounded clean-kernel checks and remaining notebook gaps are recorded.
R33 is complete: the final handoff records the six-cell support matrix,
zero-diagnostic Ruff gate, repeated-seed held-out screens and remaining limits.
There is no next implementation group.
A request for the next batch authorizes that one group, its affected callers,
tests, local phase commits and handoff update; stop after
it. It does not authorize the following group or Git publication/release unless
the user also requests those actions.

| Later two-phase group | Prerequisites | Why it fits one session | Decision to settle before the affected phase |
| --- | --- | --- | --- |
| R10 → R11 (complete) | R05/R07 complete | Leshy support/history and weighted permutation importance share selector code | Settled: absolute-total normalization, all-zero stays zero, only positive mean permutation scores eligible |
| **R18 → R31 (complete)** | R06/R18 complete | Encoder schema repairs and related legacy helpers share dtype/schema checks | Settled: remove both public legacy helpers without compatibility shims |
| **R25 → R26 (complete)** | R05 and R24 complete | Sampling output repairs and bounded isolation retries share one module | Settled: reject Gower row weights, forward isolation fit weights, omit zero-weight fit rows, tiny identity, 20 attempts per distinct size and full-data fallback |

R10 → R11, R18 → R31 and R25 → R26 are complete in separate sequential
local phase commits, with decisions recorded in their handoffs. R27–R32 are also
complete, as is R33. No group remains. These are sequential groups, not parallel agent assignments.

For an unnamed next-batch request, continue from the lowest-numbered Pending
phase in the ordered table. If it is R10, R18, or R25, use its recorded pair once
the prerequisites pass; skip any member already complete. Otherwise implement
that phase alone. Do not silently jump to a later ready group. When R31 has
already been completed with R18, skip it when reaching its numerical position.

Keep R07, R08, R09, R12–R17, R19–R24, R27–R30, R32, and R33 as **one phase per
session**; the ranges here never mean a combined batch. In particular, keep
the statistical repairs, GLM chain, tree partition repairs, training/persistence
contracts, and final release assessment independent. R31 runs alone if R18 is
already complete or the pair had to stop after R18.

Start each group in a fresh session and cap it at two phases. Trace all affected
callers before edits; previous caller audits do not replace tracing a new
change. Complete, validate, and make a separate local commit for the first phase
before starting the second. Settle and record required API/statistical choices
in the relevant docs and tests; do not invent them to make a check pass. If a
choice remains unresolved, a failure cannot be explained, or the scope widens,
stop at the phase boundary and record exactly what remains.

At the end, update the phase statuses and name the exact next group, its
prerequisites, validation commands/results, and unresolved decisions. Preserve
earlier evidence and do not carry implementation into another group in the same
session. A completed group is not a release: versioning/publication still needs
a separate delivery request even after the local R33 gates pass.

When concurrent work is explicitly requested, these tracks are candidates for
separate branches/worktrees and sessions after their prerequisites pass. Complete
one phase per session within each track; do not treat a whole track as one batch.

| Track | Internal order | Prerequisites outside the track |
| --- | --- | --- |
| GLM | R13 → R14 → R15 | R03, R05 |
| Preprocessing | R18 → R19 → R20 | R02, R05, R06 |
| Sampling | R24 → R25 → R26; R24 → R27 | R05, R06 |
| Training and persistence | R21 → R22 | R05, R06 |

These candidates follow the phase dependencies, not a completed file-overlap
audit. Before concurrent edits, trace callers and agree which branch owns each
affected file. Keep shared validation, dtype, weight, and association helpers
under one owner; settle their contracts before dependent work. If tracks need
the same helper or file, serialize that work and recheck callers after integration.
Separate branches do not eliminate shared behavioural dependencies.

At every phase boundary, add a short result entry to this handoff containing:

- Phase status, branch, and commit (or an explicit note that changes are uncommitted).
- What changed and any API or statistical decisions made.
- Exact validation commands, environment versions, results, and checks skipped.
- Remaining blockers, the next phase, and the files/evidence needed to resume.

A fresh session should read this handoff, check Git status and the recorded
commit, then inspect only the relevant findings, callers, and checks. Verify or
recreate disposable environments rather than assuming earlier `/tmp` paths still
exist. Preserve historical review evidence and stop for review at the requested
phase or batch boundary.

## R05 result — 2026-10-11

R05 is complete on `main`, based on `170b3e9`; changes remain uncommitted with
pre-existing R04 work preserved. The [handoff](R05-observation-boundaries.md)
records the target/weight boundary policy, affected callers and exact commands.
Weights remain unnormalized; frequency versus relative/exposure semantics stay
with R08/R15/R25. Missing responses are rejected rather than imputed.

`python -I -m pytest tests -q --tb=short -p no:cacheprovider` outside the checkout
passes on the installed wheel: Python 3.9.25 locked **236 passed**, Python 3.12.7
locked **236 passed**, Python 3.12.7 newer **232 passed**. The suite includes 172
new boundary/reference cases. All three dependency/provenance checks pass;
`uv lock --check`, Ruff 0.17.0's baseline gate (zero new diagnostics), gate unit
tests (2 passed), and `git diff --check` pass. Historical scripts and failure
logs remain intact; the newer raw inferred-string case is still a known R06
failure, with explicit-object R05 cases green. Hosted CI, docs, optional backends,
performance and broad statistical calibration were not checked.

**Next group: R06 alone**, with R01/R05 complete. Resume from `1A F13`, `03 F4`,
`05 C1`, `06 F1` and the shared dtype/encoding callers. R06–R33 remain pending;
there is no release-readiness claim or authorization to start another group.

## R06 result — 2026-10-11

R06 is complete on `main`, based on `170b3e9`; changes remain uncommitted with
pre-existing R04/R05 work preserved. The [handoff](R06-category-identity.md)
records the shared categorical dtype rule, original-value factorization, missing
code -1, first-occurrence/pandas-equality semantics and affected callers.
Correlation ratio and categorical ANOVA no longer sort mixed original labels.
The helper's in-place ownership remains explicit; R07 owns selector/shadow repair.

`python -I -m pytest tests -q --tb=short -p no:cacheprovider` outside the checkout
passes on fresh installed-wheel environments: Python 3.9.25 locked **286 passed**,
Python 3.12.7 locked **286 passed**, Python 3.12.7 newer **282 passed**, including
50 new R06 cases. Named historical checks and controls pass unmasked in both
3.12 snapshots (**10 passed** each). All three dependency/provenance checks,
source/wheel payload checks, `uv lock --check`, Ruff 0.17.0's baseline gate
(zero new diagnostics), gate unit tests (2 passed), new-test lint/format and
`git diff --check` pass. Exact commands, versions and evidence are in the handoff.
Hosted CI, docs, optional backends, performance and statistical calibration were
not checked; existing model/tag/precision warnings remain visible.

**Next group: R07 alone**, with R02/R06 complete. Resume from `1B F5/F6/F10`
and the selector encoding/ownership, reduction, fold and shadow callers.
R07–R33 remain pending; there is no release-readiness claim or authorization to
start another group.

## R07 result — 2026-10-12

R07 is complete on `main`, based on `170b3e9`; changes remain uncommitted with
pre-existing R04–R06 work preserved. The [handoff](R07-shadow-contracts.md)
records owned selector/shared-importance encoding, categorical identities across
reductions and folds, matching original/shadow model roles, preserved Leshy
extension dtypes, collision-free names and explicit GrootCV real-feature
bookkeeping. R06's category equality and missing code -1 remain in effect.
R10–R12 statistical/support repairs remain separate; no R07 decision is unresolved.

`python -I -m pytest tests -q --tb=short -p no:cacheprovider` outside the checkout
passes on fresh installed-wheel environments: Python 3.9.25 locked **328 passed**,
Python 3.12.7 locked **328 passed**, Python 3.12.7 newer **312 passed**. R07 adds
42 cases on pandas 2.2 and 30 on pandas 3 (Copy-on-Write combinations differ).
Historical R07 cases and shared shadow/fold controls pass unmasked in both 3.12
snapshots (**9 passed** each). Dependency/provenance and source/wheel payload
checks, `uv lock --check`, Ruff 0.17.0's baseline gate (zero new diagnostics),
gate unit tests (2 passed), new-test lint/format and `git diff --check` pass.
Exact commands, versions, warnings and evidence are in the handoff. Hosted CI,
complete docs, optional backends, performance, visual image review and statistical
calibration were not checked. Programmatic GrootCV plot/label checks pass.

**Next group: R08 alone**, with R05/R06 complete and R03 available for newer
NumPy. Resume from `1A F3–F7/F11/F12` and the association/relevance callers.
R08 must explicitly settle weight meaning/degrees of freedom and constant/output
policies before its arithmetic repairs. R08–R33 remain pending; there is no
release-readiness claim or authorization to start another group.

## R08 result — 2026-10-12

R08 is complete on `main`, based on `170b3e9`; changes remain uncommitted with
pre-existing R04–R07 work preserved. The [handoff](R08-statistical-arithmetic.md)
records frequency weight mass and retained degrees of freedom, Pearson linear
F, centered grouped sums, constant relevance and Theil's U output, normalized
Pearson matrices, pandas 3 output consistency and collision-free responses.
Integer weights reproduce row replication; fractional mass defines scores only.
Categorical U-to-F remains a labelled ranking heuristic without an F null claim.

`python -I -m pytest tests -q --tb=short -p no:cacheprovider` outside the checkout
passes on the final installed wheel: Python 3.9.25 locked **413 passed**,
Python 3.12.7 locked **413 passed**, Python 3.12.7 newer **397 passed**.
R08 adds 85 cases per cell. Named historical cases and a Pearson control pass
unmasked in both 3.12 snapshots (**12 passed** each). Dependency/provenance and
source/sdist/wheel payload checks, `uv lock --check`, Ruff 0.17.0's baseline gate
(zero new diagnostics), gate unit tests (2 passed), new-test lint/format and
`git diff --check` pass. The newer interim output failures and final evidence
are preserved in the handoff. Hosted CI, complete docs/notebooks, optional
backends, performance and broad statistical calibration were not checked.

**Next group: R09 alone**, with R08 complete. Resume from `1A F1/F2` and the
mixed relevance/default redundancy callers in `association.py` and `mrmr.py`.
R09 must settle a nonnegative common scale and empty/equal/perfect ranking
behavior. R09–R33 remain pending; no release-readiness claim is made.

## R09 result — 2026-10-12

R09 is complete on `main`, based on `170b3e9`; changes remain uncommitted with
pre-existing R04–R08 work preserved. The [handoff](R09-mrmr-ranking.md) records
the user-agreed bounded F/(1+F) scale, absolute unnormalized redundancy, stable
input-order ties, empty/constant/perfect candidate behavior and validated custom
relevance. Frequency-weight primitive contracts and one-class ANOVA rejection
remain as in R08. Mixed relevance is a ranking heuristic without a calibrated
significance or common effect-size claim; large F scores can round to one.

Full maintained suites pass outside the checkout on fresh installed-wheel
environments: py39-locked **463 passed**, py312-locked **463 passed**, py312-newer **447 passed**.
R09 adds 50 maintained cases per cell. Historical F1/F2 and signal/null controls
pass unmasked in both 3.12 snapshots (**11 passed** each). Dependency/provenance,
source/sdist/wheel payload, `uv lock --check`, Ruff baseline (zero new diagnostics),
gate unit tests (2 passed), new-test lint/format and `git diff --check` pass.
Exact commands, versions and evidence are in the handoff. Hosted CI, full
docs/notebooks, optional backends, performance and statistical calibration were
not checked; no R09 policy decision remains unresolved.

**Next group: R10 → R11**, sequentially, with R05/R07 complete. Resume from
`1B F1–F4`, Leshy support/history and shared importance callers in
`allrelevant.py`. R11 must settle zero-total importance behavior. R10–R33 remain
pending; no release-readiness claim or start of another group is authorized.

## R10 result — 2026-10-12

R10 is complete on `main`, committed separately as `acf99fc`. Earlier R04–R09
work remains uncommitted and preserved. The [handoff](R10-leshy-support.md)
records support-mask ownership, actual selected counts with keep_weak on/off,
all actual history rows for tentative medians/rejected ranking, and unchanged
strict tentative-median comparison. The 17 promoted cases pass without xfails;
the R09 wheel fails 11 of them and both historical F1/F2 counterexamples.

Fresh installed-wheel full suites pass: py39-locked **480 passed**, py312-locked
**480 passed**, py312-newer **464 passed**. Both 3.12 historical/control runs
pass unmasked (**6 passed** each). Dependency/provenance, source/sdist/wheel
payloads, lock, Ruff baseline (zero new diagnostics), gate unit tests (2 passed),
new-test lint/format and diff checks pass. Commands/versions are in the handoff.
Hosted CI, full docs/notebooks, optional backends, performance, visual plot
review and statistical calibration were not checked. R11 followed sequentially.

## R11 result — 2026-10-12

R11 is complete on `main`, committed separately as `f68e482`. The
[handoff](R11-permutation-importance.md) records aligned validation weights at
the shared scorer and the user-chosen absolute-total normalization for signed
permutation scores. Cancellation is well defined, all-zero stays zero, and
BoostAGroota requires a positive mean permutation score as well as the existing
shadow comparison. Native/SHAP normalization and Leshy's signed decision rule
remain as before; R12 still owns fold/shadow aggregation. Earlier R04–R09 work
and historical evidence are preserved; shared remediation/queue edits remain
uncommitted with those earlier updates.

Fresh installed-wheel full suites pass: py39-locked **496 passed**, py312-locked
**496 passed**, py312-newer **480 passed**, including 16 new R11 cases per cell.
Historical F4 and controls pass unmasked (**5 passed** in both 3.12 snapshots).
The old F3 script's diagnostic equality with unweighted scores is preserved;
the maintained replacement compares directly with the real weighted sklearn
reference. Both historical F3/F4 cases and 12 of the 16 maintained cases fail
on the installed R10 wheel. Dependency/provenance, source/sdist/wheel payloads,
lock, Ruff baseline (zero new diagnostics), gate unit tests (2 passed), new-test
lint/format and diff checks pass. Exact commands, versions and remaining gaps
are in the handoff. Hosted CI, full docs/notebooks, optional backends,
performance, visual plot review and statistical calibration were not checked.

**Next group: R12 alone**, with R07 complete. Resume from `1B F7–F9` and
GrootCV objective/output-axis handling, then choose and document exact
fold/shadow statistics for GrootCV and BoostAGroota. R12–R33 remain pending.
The R10 → R11 batch is complete; stop here. No publication or release is authorized.

## R12 result — 2026-10-13

R12 is complete on `main`, committed separately as `dda0b5b` from `f68e482`.
Earlier R04–R09 work is preserved; shared queue and
remediation edits remain uncommitted with earlier updates. The
[handoff](R12-all-relevant-aggregation.md) records shape-based SHAP aggregation,
resolved objective/parameter ownership, and the user-chosen rules: GrootCV takes
feature medians across folds/repeats and compares with maximum shadow median
divided by cutoff; BoostAGroota takes iteration means and compares with maximum
shadow mean divided by cutoff. Equality is retained; R11's positive mean
permutation requirement remains. GrootCV ranking and plot order use medians.
These repairs can change selected sets; no R12 decision remains unresolved.

Fresh installed-wheel full suites pass: py39-locked **533 passed**, py312-locked
**533 passed**, py312-newer **517 passed**, including 37 new R12 cases per cell.
Both 3.12 historical/control runs pass unmasked (**10 passed** each). The R11
wheel fails all four named historical cases and 35 of the 37 maintained cases.
Dependency/provenance, source/sdist/wheel payloads, lock, Ruff baseline
(zero new diagnostics), gate unit tests (2 passed), new-test lint/format and
diff checks pass. All twelve unrelated tracked diff sections are unchanged.
Exact commands and versions are in the handoff/evidence. Hosted CI, full
docs/notebooks, optional backends, performance, visual image inspection and
statistical calibration were not checked. No release-readiness claim is made.

**Next group: R13 alone**, with R03/R05 complete. Resume from `1C F2–F4/F9/F12`,
GLM response-scale deviance/scoring, weighted CV/grid search and the historical
Poisson signal counterexample. R14 design/intercept and R15 likelihood/refit
decisions stay separate. R13–R33 remain pending; stop after this R12 group.

## R13 result — 2026-10-13

R13 is complete on `main`, committed separately as `28b9529` from the clean
`64b0ce7` checkout. The [handoff](R13-glm-scoring.md) records negative total
response-scale deviance, higher-is-better public/CV scores, retained BIC
minimization, task-aware native folds, positional array/frame slicing and a
fresh clone per fold. Variance weights, unit scale, mean total fold loss and
unshuffled order remain explicit. Direct score callers must handle the new
negative sign. R14 design/parameter and R15 likelihood/refit repairs stay separate.

Fresh installed-wheel full suites pass: **py39-locked 566 passed;
py312-locked 566 passed; py312-newer 550 passed**, including 33 maintained R13
cases. Those cases fail 29 times on the unchanged R12 wheel; repaired source
and affected boundary/API callers pass **217** tests. Fourteen historical R13
cases and controls pass unmasked in both 3.12 snapshots. The seeded Poisson
counterexample now chooses alpha=1 with independent mean fold deviance 31.7024,
versus 236.7310 for the former null-model choice. Dependency/provenance checks,
eight source/installed hashes, 19 source/sdist/wheel payloads, copied tests,
`uv lock --check`, Ruff 0.17.0 baseline (**417 total, zero new**), gate unit
tests (**two passed**), new-test lint/format, local links and `git diff --check`
pass. Exact commands and Python/dependency versions are in the handoff/evidence.
Hosted CI, full docs/notebooks, other platforms, optional backends, performance,
full sklearn compliance and statistical calibration were not checked.

**Next group: R14 alone**, with R13 complete. Resume from `1C F5/F7/F8/F10`,
`EnetGLM.fit/predict`, selector fit/transform, constants, input/support metadata,
current family/link parameters and summary integration. Settle the fitted
intercept/input-schema contract before edits. R15's zero-weight likelihood,
finite criteria and refit policy remain unresolved and separate. R14–R33 are
pending; no R13 decision remains unresolved. Stop after this completed R13 group.

## R14 result — 2026-10-13

R14 is complete on `main`, committed separately as `3e3e7e3` from the clean
`fb942fc` checkout. The [handoff](R14-glm-design.md) records one fitted design
across prediction/transformation, preserved supplied constants under both
intercept modes, collision-free generated intercept names, named-schema alignment
and rejection of missing/extra/duplicate inputs, positional arrays, input/support
versus coefficient-design metadata, current family/link resolution and fit return.
A selected generated intercept remains in output and remains penalized; input
metadata and support no longer include it. `fit_intercept=False` disables only
automatic addition. No R14 API choice remains unresolved.

Fresh installed-wheel full suites pass: **py39-locked 636 passed;
py312-locked 636 passed; py312-newer 620 passed**, including 70 maintained R14
cases. Those cases produce **67 failures and three passes** on the verified
R13 wheel; source and affected R05/R13/API callers pass **287** tests. Original
R14 diagnostics reproduce nine failures before repair; all **17** selected
historical checks/controls pass unmasked in both repaired 3.12 snapshots.
Dependency/provenance checks, eight installed/source hashes, 19 source/sdist/wheel
payloads, copied tests, `uv lock --check`, Ruff 0.17.0 baseline (**417 total,
zero new**), gate unit tests (**two passed**), new-test lint/format, handoff links
and `git diff --check` pass. Real scaled pandas pipeline feature names and rendered
summary HTML agree with input support. Exact commands, versions and results are
in the handoff/evidence. Hosted CI, full docs/notebooks, other platforms, optional
backends, performance, full sklearn compliance and statistical calibration were
not checked.

**Next group: R15 alone**, with R13/R14 complete. Resume from `1C F6/F11`,
`EnetGLM.fit`, likelihood/BIC attributes and `grid_search_cv` candidate handling.
Decide zero-weight likelihood row omission versus explicit rejection, reject
all-invalid candidate criteria, document regularized selection followed by
unpenalized refit and remove the near-ridge promise. Preserve R13 scoring and R14
fitted design. R15–R33 remain pending; stop after this completed R14 group.

## R15 result — 2026-10-13

R15 is complete on `main`, committed separately as `51d44f2` from the clean
`2577289` checkout. The [handoff](R15-glm-likelihood.md) records the chosen
zero-weight policy: omit aligned rows before GLM fitting/scoring/tuning and
before direct GLM CV splits. Positive weights stay unnormalized, responses and
weights still validate before omission, and generic CV keeps its existing fold
mass guards. Any nonfinite candidate criterion now fails with score/alpha
context; backend exceptions propagate without silent candidate dropping.

Independent likelihood checks exposed the backend refit's extra parameter/row
in information-criterion bookkeeping. Wrapper AIC/BIC now use returned
log-likelihood, retained observation count and selected design rank, counting
mean parameters without dispersion. Raw backend summaries/results remain
unchanged. The documented contract is regularized selection followed by
unpenalized refitting; the near-ridge claim is removed, pure ridge/invalid
mixtures fail clearly, and post-selection/variance-weight likelihood limits
are explicit. API/method prose and two notebook explanatory sentences agree;
notebook code, outputs and other cells are preserved. No R15 policy choice
remains unresolved.

Fresh installed-wheel full suites pass: **py39-locked 688 passed;
py312-locked 688 passed; py312-newer 672 passed**, including 52 maintained R15
cases. Those cases produce **41 failures and 11 passes** on the verified R14
wheel; source and affected R05/R13/R14/API callers pass **339** tests. Original
R15 likelihood/near-ridge diagnostics reproduce two failures before repair;
all **12** selected historical checks/controls pass unmasked in both repaired
3.12 snapshots. The old near-ridge expectation remains historical, replaced
by independent checks of the truthful refit contract. Dependency/provenance,
eight installed/source hashes, 19 source/sdist/wheel payloads, copied tests,
`uv lock --check`, Ruff 0.17.0 baseline (**417 total, zero new**), gate unit
tests (**two passed**), affected/new-test lint, new-test format, notebook
preservation, handoff links and `git diff --check` pass. Exact commands, versions
and results are in the handoff/evidence. Hosted CI, full docs/notebooks, other
platforms, optional backends, performance, full sklearn compliance and
statistical calibration were not checked.

**Next group: R16 alone**, with R06/R08 complete. Resume from `03 F5–F9/F12`,
`BaseThresholdSelector` in `base.py`, threshold selectors in `unsupervised.py`,
callback label alignment, parameter validation, collinearity edge cases, stable
ties and fitted metadata. Settle equality wording explicitly and preserve named
projection if retained. R16–R33 remain pending; stop after this completed R15 group.

## R16 result — 2026-10-13

R16 is complete on `main` in local commit `2fa18e8`, based on clean `985b0a3`.
The [handoff](R16-threshold-contracts.md) records duplicate-name rejection at
fit/transform, aligned and validated callback statistics, current fit-time
parameters, fitted feature counts and explicit support guards for all five
selectors. Named projection remains permissive for extra/reordered columns
and omitted rejected columns; missing selected names still raise KeyError.
Strict threshold equality is preserved and documented. Unique/cardinality
thresholds require integers >= 1; missingness/collinearity require finite real
cutoffs in [0, 1]. No R16 policy choice remains unresolved.

Collinearity handles zero/one candidates with correctly labelled zero matrices
and boolean support while retaining existing observation/weight validation.
Full matrices and tied elimination follow fitted input order. The shared name
guard also reaches VariableImportance's fit/transform and metadata, leaving its
importance arithmetic, iteration and optional-backend repairs to R17. API
prose and the Unreleased changelog agree; historical evidence is preserved.
Three repaired Ruff allowances are removed, with none added.

Fresh installed-wheel suites pass outside the checkout without setuptools:
**py39-locked 772 passed; py312-locked 772 passed; py312-newer 756 passed**.
All **29** selected historical checks/controls pass unmasked in each 3.12
snapshot. The 84 maintained R16 cases produce **73 failures and 11 passes**
on the verified R15 wheel; final source/affected callers pass **321** tests.
Dependency/provenance checks, eleven installed/source hashes, 19 source/sdist/
wheel payloads and copied tests agree. `uv lock --check`, Ruff 0.17.0's gate
(**414 total, zero new**), gate unit tests (**two passed**), new test/evidence
lint and format, local links and `git diff --check` pass. Exact commands,
versions and logs are in the handoff/evidence. Hosted CI, full docs/notebooks,
full sklearn compliance, other platforms, optional backends, performance and
statistical calibration were not checked.

**Next group: R17 alone**, with R03/R05/R06 complete. Resume from
`03 F1–F3/F11/F13`, `variable_importance.py` and its training/contribution callers.
Settle crossing-prefix/zero-total selection, weighted explanation population
and whether repeated identical fits should remain; test array multiclass and
missing-backend errors. R17–R33 remain pending. Stop after this completed R16 group.

## R17 result — 2026-10-13

R17 is complete on `main` in local commit `6f44345`, based on clean `1e88c54`.
The [handoff](R17-importance-selection.md) records the user-chosen one fit and
removal of `n_iterations`, with no compatibility shim or new resampling mode.
Training/validation loss retain observation weights; global importance averages
validation rows equally, as the user chose. `verbose` uses native training-metric
logging. None resolves the same default model parameters without mutable defaults.
No R17 policy choice remains unresolved.

Cumulative selection retains the crossing positive feature, excludes zeros,
selects nothing at cutoff zero or zero total, and retains every positive feature
at cutoff one despite rounding. Current finite [0, 1] thresholds validate at fit.
Native output blocks exclude their bias columns and infer output count without
pandas-only target calls. Native and optional outputs reuse the existing mean
absolute/sum-over-outputs helper. Invalid native shapes/nonfinite contributions
and totals fail clearly; missing fasttreeshap fails before encoding/training.
API/changelog and one notebook paragraph agree. Notebook code, outputs,
metadata and all other cells are preserved. Seven Ruff allowances are removed;
none are added. Plot correctness remains R29.

Fresh installed-wheel full suites pass outside the checkout without setuptools:
**py39-locked 821 passed; py312-locked 821 passed; py312-newer 805 passed**.
All **24** selected historical assertions/controls pass unmasked in both 3.12
snapshots after an explicitly recorded disposable adaptation removing exactly
two retired `n_iterations=1` arguments; original scripts/logs remain unchanged.
Obsolete repeated-fit/iteration-average assertions are excluded, with the new
one-fit contract tested directly. Original checks reproduce **11 failures and
13 passes** before repair. The 49 maintained R17 cases produce **36 failures
and 13 passes** on the verified R16 wheel; final source/affected callers pass
**354** tests. Dependency/provenance, eleven installed/source hashes, 19 source/
sdist/wheel payloads, copied tests, `uv lock --check`, Ruff 0.17.0's gate
(**407 total, zero new**), gate unit tests (**two passed**), new/affected test
and evidence lint/format, notebook preservation, links and `git diff --check`
pass. Exact commands, versions and evidence are in the handoff. Hosted CI,
full docs/notebooks, full sklearn compliance, other platforms, installed optional
backend equivalence, performance and statistical calibration were not checked.

**Next group: R18 → R31**, with R06 complete. Read `02 F8/F13/F14`,
`OrdinalEncoderPandas` and its full schema/inverse/output-dtype callers for R18.
Complete, validate and commit R18 before R31. Before R31, decide whether to
retain or remove public `cat_var` and `validate_pandas_input`, tracing callers,
exports and docs from `02 F15` and `06 F10`. Stop at that phase boundary if the
API decision or wider scope remains unresolved. R18–R33 remain pending.
Stop after this completed R17 group; no following-group work or publication is authorized.

## R18 result — 2026-10-13

R18 is complete in local commit `7df5daa` on `main`, based on clean `37cf6b2`.
[R18](R18-ordinal-schema.md) records the
owned native categorical encoder, full passthrough names/count, fitted-order
schema, current dtype/unknown/missing parameters, nullable categorical input and
non-mutating inverse. Native inverse receives writable owned codes to avoid the
pandas 3 read-only-array failure. Decoded label dtypes are inferred by pandas;
passthrough dtypes stay intact. No separate mapping or compatibility layer is added.

Fresh installed-wheel suites pass: **Python 3.9 locked 856; Python 3.12 locked
856; Python 3.12 newer 840**. All **7** original historical ordinal checks pass
unmasked in both 3.12 cells. The 35 maintained cases fail on the verified R17
wheel (**32 failed, 3 passed**). Exact commands, versions, provenance, interim
pandas 3 failure, final payload hashes and check limits are in the handoff and
[evidence](evidence/r18-ordinal-schema/commands.txt). Dependency/provenance,
19 source/sdist/wheel payloads, lock, Ruff (**403 total, zero new**, four removed
allowances), gate unit tests (**2 passed**), new-test/evidence lint/format, local
links and whitespace pass. Hosted CI, full docs/notebooks, full sklearn compliance,
optional numerical backends, other platforms, performance and calibration are skipped.

**Next phase in this authorized batch: R31**, with R06/R18 complete. The user
explicitly chose **remove both public legacy helpers** (`cat_var` and
`validate_pandas_input`), without compatibility shims. Trace exports/docs, remove
the dead benchmark comment, add removal/migration checks and commit R31 separately.
Stop after R31; the next group then is R19 alone, with R05/R06/R18 complete.
No following phase, publication, version bump or release is authorized.

## R31 result — 2026-10-14

R31 is complete in local commit `ffb7388` on `main`, based on the completed R18
commit `7df5daa`.
The user explicitly chose **remove both public legacy helpers**, without shims.
[R31](R31-legacy-helpers.md) records removal of `cat_var` and
`validate_pandas_input`, the dead benchmark comment and cat_var-only tqdm import.
There were no active internal callers or explicit exports/docs listings. The
Unreleased changelog and handoff explain the encoder/array migration and the
intentional tuple/code-policy differences. Every retained function/class AST is
unchanged by R31; no unrelated unused helper is removed.

Fresh installed-wheel suites pass: **Python 3.9 locked 862; Python 3.12 locked
862; Python 3.12 newer 846**. Six maintained removal/migration cases report
**2 failed, 4 passed** on the verified R18 wheel. The historical missing-category
and declared-array failures reproduce before removal; original reports/scripts
remain intact. The original preprocessing diagnostic module imports cat_var at
load, so it is not recollected after the intentional removal; maintained ordinal
and removal suites run unmasked, with no injected helper. Exact commands,
versions, provenance, payloads and check limits are in the handoff and
[evidence](evidence/r31-legacy-helpers/commands.txt). Dependency/provenance,
19 source/sdist/wheel payloads, lock, Ruff (**397 total, zero new**, six removed
allowances), gate unit tests (**2 passed**), test/evidence lint/format, local links
and whitespace pass. Hosted CI, full docs/notebooks, full sklearn compliance,
optional numerical backends, other platforms, performance and calibration are skipped.

**R18 → R31 batch complete. Next group: R19 alone**, with R05/R06/R18 complete.
Read `02 F1/F2/F4/F5/F9` and TreeDiscretizer's fit/transform, actual tree partitions,
collision/missing-value handling and copied fit-time parameters. R19–R30 and
R32–R33 remain pending. Stop here; no next phase, publication, version bump or
release is authorized by this batch. Passing it does not establish release readiness.

## R19 result — 2026-10-14

R19 is complete in local commit `6af7415` on `main`, based on clean `132e0e0`. [R19](R19-tree-partitions.md)
records exact fitted numerical thresholds and leaf-ID intervals, distinct
categorical leaf groups, removal of temporary prediction-column writes, current
copied one-tree/task/bin parameters and aliases, shared categorical encoding in
both branches, and native missing/unknown routes. All-missing numeric input has
one (-inf, inf] leaf interval. Raw binary output is native probabilities. Labels
and fitted maps change deliberately: leaf_interval_dic_ replaces tree_imputer,
and categorical maps use leaf IDs with unambiguous readable labels. Refit old
saved discretizers; no fitted-state compatibility shim is added. R20's remaining
input/state/Patsy/midpoint decisions are not treated as complete.

Fresh installed-wheel full suites pass: **Python 3.9 locked 922; Python 3.12 locked
922; Python 3.12 newer 906**. All **14** adapted historical tree checks pass
unmasked in both 3.12 cells. The 60 maintained cases report **49 failed, 11
passed** on the verified R31 wheel; historical before checks reproduce **11
failures and 3 controls**. The disposable historical copy removes only the cat_var
import retired by R31; its diff/hashes and unchanged assertions are recorded.
Original evidence remains intact. The initial repaired wheel passed its suites,
then a caller probe caught an unnamed categorical-column regression; explicit
column naming fixes it, with None/integer-name controls included in the final run.

Exact commands, versions, provenance and limits are in the handoff and
[evidence](evidence/r19-tree-partitions/commands.txt). Source/affected callers
pass **338 tests**; newer tree/pandas source checks pass **65**. All dependency
checks, 12 installed module hashes, 19 source/sdist/wheel payloads, lock, Ruff
(**397 total, zero new**, unchanged allowances), gate unit tests (**2 passed**),
new-test/evidence lint/format, local links and whitespace pass. Hosted CI, full
docs/notebooks, full sklearn compliance, optional numerical backends, other
platforms, performance, calibration and cross-process artifacts are skipped.
The existing wrapper's 80/20 split remains; no improved held-out performance,
optimal binning or group/fused-penalty equivalence is claimed.

**Next group: R20 alone**, with R02/R18/R19 complete. Read `02 F6/F7/F10–F12`,
TreeDiscretizer's resolved bin_features and array normalization, midpoint scalar
names/refits, and Patsy formula/NA_action/intercept/row alignment. R20–R30 and
R32–R33 remain pending; skip completed R31 at its numerical position. Stop after
R19; no next phase, publication, version bump or release is authorized by this
request. Passing this phase does not establish release readiness.

## R20 result — 2026-10-14

R20 is complete on `main`, based on clean `e030fde`, in local phase commit
`cff3c59`. The [handoff](R20-transformer-inputs.md) records reusable constructor
options, fitted bin_features_/cat_features_/cols_/formula_, shared fit/transform
input preparation, scalar midpoint names and Patsy NA_action/intercept behavior.
Named tree inputs require the full fitted schema/order; arrays use the fitted
names and width. The existing intercept default is retained; explicit Patsy
drop affects predictor rows only, leaving y/weight alignment with the caller.

Fresh installed-wheel full suites pass **975 / 975 / 959** on Python 3.9 locked,
Python 3.12 locked and Python 3.12 newer. All **16** adapted historical cases pass
unmasked in both 3.12 cells. Source/caller checks pass **391** tests. The **53**
maintained cases show **36 failed, 17 passed** before repair on the verified R19
wheel; historical before checks show 11 failures/five controls. Python 3.12's
inherited Patsy traceback-formatting crash blocks complete before-maintained
reports; Python 3.9 supplies that evidence. Original historical evidence is intact.
All 12 installed-module hashes and 19 source/sdist/wheel payloads match; fresh
installs use no setuptools. Dependencies, lock, Ruff 0.17.0 (396 total, zero new),
gate unit tests (two), new-test/evidence lint/format, local links and whitespace
pass. The resolved preprocessing C414 allowance is removed. Exact commands,
versions, interim harness corrections and check limits are recorded in the handoff.
Hosted CI, full docs/notebooks/sklearn compliance, optional numerical backends,
other platforms, performance/calibration, cross-process artifacts and implicit
fit_transform evaluation-scope repair are not checked.

**Next group: R21 alone**, with R05/R06 complete. Read `04 F1–F5/F8/F9`,
GradientBoosting and its split/prediction/category/offset/refit callers. Record
any necessary API choices before repairing those contracts; R22 persistence
follows R21. R21–R30 and R32–R33 remain pending; R31 is complete. Stop after
R20; passing this phase is not a release-readiness claim.

## R21 result — 2026-10-14

R21 is complete in local commit `6204a3f` on `main`, based on clean `d51c567`.
[R21](R21-gbm-contracts.md) records explicit rejection of group/stratified
conflicts, full ordered named prediction schemas and positional arrays, native
name normalization, multiclass objective aliases, actual categorical support and
explicit disable, finite multiclass offset matrices, and successful-refit state.
Failed fits preserve the previous model's contract. Category/Boolean input and
explicit numeric codes use native handling; unsupported object/string/time types
receive conversion errors. No learned encoder or combined split is introduced.
New-row offset predictions still require an external baseline on the raw scale.

Fresh installed-wheel full suites pass: **Python 3.9 locked 1031; Python 3.12
locked 1031; Python 3.12 newer 1015**. All **39** selected unchanged historical
checks pass unmasked in both 3.12 cells. The 56 maintained cases report **45
failed, 11 passed** on the verified R20 wheel; original selected counterexamples
reproduce **15 failures** before repair. Rejection policies supersede historical
successful-fit expectations and are checked in maintained tests. Exact commands,
versions, provenance, payloads and limits are in the handoff and
[evidence](evidence/r21-gbm-contracts/commands.txt). Affected source/callers pass
**389 tests** before the last ordinary-load schema control; full installed suites
include it. Dependency checks, 12 installed-module hashes, copied tests, 19
source/sdist/wheel payloads, lock, Ruff (**395 total, zero new**, one removed
allowance), gate unit tests (**2 passed**), test/evidence lint/format, local links
and whitespace pass. Hosted CI, docs/notebooks, optional backends, other platforms,
performance, calibration and cross-process artifact repairs remain unchecked.

**Next group: R22 alone**, with R21 complete. Read `04 F10`, GradientBoosting's
save/load and the fresh-process reload evidence. Persist offset/category/original-
schema metadata and reset loaded state; decide legacy Booster-only handling
without guessing the baseline contract. Keep trusted-input/type/error boundaries
explicit and verify safe-write failures. R22–R30 and R32–R33 remain pending;
R31 is complete. Stop here; no following phase, publication or release is included.

## R22 result — 2026-10-14

R22 is complete in local commit `4bd3681` on `main`, based on clean `4d0e240`.
[R22](R22-gbm-persistence.md) records a versioned joblib dictionary with Booster,
offset guard, category names and original fitted schema. Loading validates the
artifact before replacing fitted state, clears validation data/curves and retains
constructor options for the next fit. Legacy Booster-only files raise TypeError:
refit and save again because their baseline contract is unknown. New-row offset
predictions still require an external baseline on the raw scale. Joblib remains
trusted-input only; metadata checks do not make unknown pickle files safe.

Save stages serialization and optional PNG rendering on the destination filesystem
and syncs the model file before replacement. Pre-commit failures preserve existing
files. Model replacement is the commit point; a later PNG replacement failure
propagates with the new model saved and the old PNG retained. This separate-file
limit is explicit and tested; no joint transaction or power-loss durability is
claimed. Existing filenames still overwrite and default paths remain relative.

Fresh installed-wheel full suites pass: **Python 3.9 locked 1068; Python 3.12
locked 1068; Python 3.12 newer 1052**. All **11** selected unchanged historical
checks pass unmasked in both 3.12 cells. The 37 maintained cases report **33
failed, 4 passed** on the verified R21 wheel; original counterexamples reproduce
**3 failures**. Eight isolated-process cases exercise ordinary/offset regression,
binary, multiclass-alias and category predictions on new rows. Affected source/
callers pass **194 tests**, excluding those eight installed-only cases. Exact
commands, versions, provenance and limitations are in the handoff and
[evidence](evidence/r22-gbm-persistence/commands.txt). Dependency checks, 12 module
hashes, copied tests, 19 source/sdist/wheel payloads, lock, Ruff (**395 total, zero
new**), gate unit tests (**2 passed**), test/evidence lint/format, links and
whitespace pass. Hosted CI, docs/notebooks, optional backends, other platforms,
cross-version loading, hostile artifacts, concurrent writers, interruption/power-
loss recovery, performance and calibration remain unchecked.

**Next group: R23 alone**, with R07/R10–R12/R18/R21 complete. Read
`04 F11/F12/F14/F15`, sklearn_pimp_bench, compare_varimp and their encoder/selector/
model callers. Use training-only encoding, settle array/name support, reject
unsupported GrootCV comparisons and construct fresh configurations for each run.
R23–R30 and R32–R33 remain pending; R31 is complete. Stop here; no following
phase, publication, version bump or release is included.

## R23 result — 2026-10-14

R23 is complete in local commit `ff8faa8` on `main`, based on clean
`9a6a8b7`. [R23](R23-benchmark-contracts.md) records training-only encoder fitting,
retained ndarray input with x0/x1/... names, preserved DataFrame model names and
text-based highlighting. Held-out-only categories receive NaN unknown codes;
the estimator must support missing values when these occur. Weighted training,
validation scoring and classification stratification are retained.

compare_varimp accepts Leshy/BoostAGroota and rejects GrootCV before training.
Each model/method pair gets a freshly constructed selector and cloned estimator;
caller objects remain unchanged. Integer seeds are reused; RandomState instances
are copied at their current state, including advanced state from fitted templates.
None and BoostAGroota's global shadow RNG keep their existing behavior. Leshy.fit
refreshes backend flags for direct set_params/refits as well. This prints feature
selections/plots, not common held-out predictive-performance comparisons.

Fresh installed-wheel full suites pass **1088 / 1088 / 1072** in Python 3.9 locked /
Python 3.12 locked / Python 3.12 newer. Five unchanged historical checks pass
unmasked in both 3.12 cells. Original targeted checks reproduce **four failures**
on the R22 wheel; 20 finalized maintained cases report **19 failed, one passed**
there. Affected source/callers pass **316 tests**. Exact commands, versions and
provenance are in the [handoff](R23-benchmark-contracts.md) and
[evidence](evidence/r23-benchmark-contracts/commands.txt). All copied tests,
12 installed source hashes, 19 source/sdist/wheel payloads, dependency checks,
lock, Ruff (**395 total, zero new**), gate unit tests (**two passed**), changed-test/
evidence lint/format, links and whitespace checks pass. Historical evidence is
unchanged; GrootCV's old acceptance expectation is replaced by rejection tests.

**Next group: R24 alone**, with R06 complete. Read `05 F1–F5`, gower_matrix and
sampling/nearest-neighbor callers. Settle missing/no-comparable-pair behavior,
repair stable numeric ranges and cross/self matrices, and validate input masks
and distance feature weights with independent scalar references. R24–R30 and
R32–R33 remain pending; R31 is complete. Hosted CI, docs/notebooks, optional
backends, other platforms, performance and calibration remain unchecked.
Stop here; no following phase, publication, version bump or release is included.

## R24 result — 2026-10-14

R24 is complete in local commit `5325b7f` on `main`, based on clean
`4e50f38`. [R24](R24-gower-distances.md) records the user-chosen NaN result for
pairs without comparable positive-weight features, including entirely missing
diagonals. Missing features are omitted separately for each pair. Observed
constant columns contribute zero and retain their denominator weights. Numeric
ranges retain pooled X/Y observations, using float64 direct spans and centered
scaling; infinities and overflowing spans fail clearly. Pairwise weight scaling
prevents overflow and keeps tiny-weight-only pairs defined.

Cross-matrices compute every pair, even with equal row counts; symmetry is used
only for actual self-comparisons. Arrays, named reference Series, nullable dtypes,
empty row axes and category masks have explicit validated contracts. DataFrame
schema/order and nominal identity are preserved. Feature weights are positional
column weights, not observation weights. The sample caller passes a Boolean
category mask so integer names are safe. Sampling output/observation weights,
isolation retries and key/query/nonfinite-neighbor selection remain R25–R27.
Float64 output doubles the former float32 storage per entry and remains dense/
quadratic for self-distances, using row-sized feature scratch.

Fresh installed-wheel full suites pass **1177 / 1177 / 1161** in Python 3.9 locked /
Python 3.12 locked / Python 3.12 newer. All **28** unchanged historical R24 checks
pass unmasked in both 3.12 cells; the R23 locked wheel reports **17 failed, 11
passed**. All **89** final maintained cases report **75 failed, 14 passed** on that
verified before wheel. Affected source/dtype callers pass **139 tests**. Exact
commands, versions and provenance are in the [handoff](R24-gower-distances.md) and
[evidence](evidence/r24-gower-distances/commands.txt). All copied tests,
13 installed source hashes, 19 source/sdist/wheel payloads, dependency checks,
lock, Ruff (**390 total, zero new**, five allowances removed), gate unit tests
(**two passed**), changed-test/evidence lint/format, links and whitespace pass.
Historical evidence remains unchanged. Hosted CI, docs/notebooks, optional
backends, other platforms, large matrix performance and calibration are unchecked.

**Next recorded group: R25 → R26**, with R05/R24 complete, conditional on settling
representative weighting, tiny-input and exhaustion policies. Read `05 F6–F10`,
sample, isolation_forest, isof_find_sample and get_5_percent_splits. Complete and
commit R25 before starting R26; stop at the phase boundary if scope or choices
remain unresolved. R25–R30 and R32–R33 remain pending; R31 is complete.
Stop here; no following phase, publication, version bump or release is included.

## R25 — Sampling outputs complete — 2026-10-14

[R25](R25-sampling-outputs.md) records the first phase of the requested R25 → R26
batch, based on `34641a2`, on local branch `fix/r25-r26-remediation`; this entry
is included in the R25 phase commit. Gower rejects row weights and entirely
missing columns, imputes numeric columns separately and preserves feature names
without overwriting `label`. Isolation forwards positional fit weights, omits
zero-weight fit rows and returns original selected rows. Its draws and KS screen
remain unweighted. The R05 zero-weight rule also resolves a native sklearn 1.9
weighted-subsampling failure; all three final suites were rerun after that repair.

Fresh installed-wheel full suites pass **1227 / 1227 / 1211** in Python 3.9 locked,
3.12 locked and 3.12 newer. All 11 selected historical output checks pass unmasked
in both 3.12 cells; 50 maintained checks produce 43 failures/seven controls on
the old wheel. Affected source checks pass 189 tests. Artifact/source/test hashes,
dependency checks, lock, Ruff (331 total, zero new), two gate unit tests, changed
lint/format and whitespace pass. New evidence remains local/ignored; historical
reports and tracked evidence are preserved. Exact commands/versions are in R25.
Hosted CI, docs/notebooks, optional backends, other platforms, scaling and
sampling calibration are unchecked.

Continue to **R26 only** after the R25 commit. The recorded batch defaults are
empty/single isolation identity, distinct rounded-up 5%–95% proper-subset sizes,
20 attempts per size and full original data on exhaustion. Preserve the >0.95
p-value threshold as a heuristic, without representativeness/outlier guarantees.
No later phase, push, PR, version bump or release is included.

## R26 — Bounded isolation complete — 2026-10-14

[R26](R26-isolation-attempts.md) completes the requested sequential R25 → R26
batch on `fix/r25-r26-remediation`. R25 commit: `b352069`; R26 commit: `5c631bf`.
No following phase or publication is included. The sampler tries 20 draws per
rounded-up distinct 5%–95% proper-subset size, at most 380 draws, fitting one
forest. Empty/single inputs return unchanged; exhaustion returns full original
data in order. Input and fit-weight validation remain explicit. API/notebook
prose corrects the p-value interpretation and outlier-removal claim. No weighted
KS or representativeness guarantee is added.

Fresh installed-wheel full suites pass **1256 / 1256 / 1240** in Python 3.9
locked, 3.12 locked and 3.12 newer. The four selected historical retry/tiny-split
checks pass unmasked in both 3.12 cells. The 29 maintained cases include bounded
rejection, exact schedules/strict acceptance, tiny/invalid inputs, original-row
fallback and real constant/weighted replay runs. On the old R25 wheel, 28 safe
cases produce 25 failures/three controls; the potentially unbounded real
nonconstant case is excluded there. Affected source checks pass 218; all 79
sampling checks pass with newer dependencies. Source/artifact/test hashes,
dependency checks, lock, Ruff (331 total, zero new), two gate unit tests, changed
lint/format, notebook structure, local links and whitespace pass. Exact commands,
versions and limits are in R26. New evidence remains local and ignored.

Hosted CI, complete docs/notebooks, optional backends, other platforms, large-data
throughput, rare-group sampling and calibration are unchecked. The retry bound
counts attempts, not wall-clock time, and exhaustion may return the full dataset.

**Next recorded group: R27 alone**, with R24 complete. Trace gower_topn and
smallest_indices; settle actual query/key exclusion, unique eligible candidates,
count bounds and insufficient/nonfinite-distance behavior. Stop after this batch;
no push, PR, version bump or release is included. R27–R30 and R32–R33 remain
pending; R31 is complete.

## R27 — Query-aware nearest neighbors complete — 2026-10-14

[R27](R27-nearest-neighbors.md) runs alone, based on local R25/R26 handoff
`fcb811e`, on branch `fix/r27-nearest-neighbors`. Local phase commit: `45c3f98`.
The lookup uses the actual one-row query, allocates one N-by-1 distance vector,
excludes the query ID and missing candidate IDs, then retains the nearest finite
row per eligible key. Series queries keep candidate dtype inference. Counts are
nonnegative integers; zero returns empty arrays, and insufficient eligible counts
raise. Nonfinite values are excluded, ties retain original candidate order, and
inputs/indexes are preserved. Feature masks/weights exclude the key column.
Candidate-first shared Gower row calculations and stable O(N log N) sorting
remain a documented shortcut; no new backend or public parameter is added.

Fresh installed-wheel full suites pass **1301 / 1301 / 1285** in Python 3.9 locked,
3.12 locked and 3.12 newer. All six selected unchanged historical query/key/count
checks pass unmasked in both 3.12 cells; the old wheel reports four failures/two
controls there. The 45 maintained cases produce 40 failures/five controls on the
old R26 wheel. Affected source checks pass 213 in each dependency snapshot.
Source/artifact/test hashes, dependency checks, lock, Ruff (331 total, zero new),
two gate unit tests, changed lint/format, local links and whitespace pass.
Exact commands, versions and limits are recorded in R27. Historical reports and
tracked evidence are preserved; new evidence stays local and ignored.

Hosted CI, complete docs/notebooks, optional backends, other platforms and
large-data throughput are unchecked. Linear distance storage does not establish
large-data throughput. Missing/undefined matches are no longer selected as valid
neighbors. This phase does not establish release readiness.

**Next recorded group: R28 alone**, with R06/R08 complete. Trace parallel_matrix_entries,
parallel_df and their association callers. Define original label preservation,
worker-count semantics, result order and empty/no-pair results without introducing
a new execution framework. Stop after R27; no next phase, push, PR, version bump
or release is included. R28–R30 and R32–R33 remain pending; R31 is complete.

## R28 — Parallel labels and useful workers complete — 2026-10-14

R28 is complete, based on clean `main` commit `81dc6c9`, on local branch
`fix/r28-parallel-contracts`. Phase commit: `2181790`. The
[handoff](R28-parallel-contracts.md) records original mixed/tuple/MultiIndex labels,
stable descending results, fresh matrix indexes, typed empty results and native
joblib worker semantics capped to nonempty work. Duplicate caller CPU-count
rewrites are removed. Serial Pearson retains its matrix calculation and now
returns the same distinct-pair directions as the parallel path, including NaN
for undefined constant-column correlations.

All 67 maintained phase tests pass; the exact final tests produce 62 failures
and five controls on the pre-R28 wheel. Affected source checks pass 508 in each
Python 3.12 snapshot. Fresh installed-wheel full suites pass **1372 / 1372 /
1356** in Python 3.9 locked / 3.12 locked / 3.12 newer. All 12 selected
historical parallel checks pass unmasked in both 3.12 cells. All 13 installed
module hashes, copied test bytes and 19 source/sdist/wheel payloads agree;
no setuptools workaround is used. Dependency/lock/export checks, Ruff (330
total, zero new), two gate unit tests, focused lint/format and whitespace pass.
The four inherited association.py formatting differences remain unchanged.
New evidence stays local and ignored under the user's existing rule.

**Next recorded group: R29 alone**, with R03/R10/R17 complete. Repair plot
coordinates/selection membership, terminal thresholds/plot limits and singleton
clustering. Stop after R28 and its local handoff; no R29 implementation, push,
PR, version bump or release is included. Hosted CI, complete docs/notebooks,
optional backends, other platforms and large-data performance remain unchecked.
R29–R30 and R32–R33 remain pending; R31 is complete.

## R29 — Plot contracts complete — 2026-10-15

R29 is complete, based on merged R28 `main` commit `605cbff`, on local branch
`fix/r29-plot-contracts`, implementation commit `24db954`. The
[handoff](R29-plot-contracts.md) records feature/subplot coordinates, actual Leshy
membership colors, fitted VariableImportance cutoffs,
terminal thresholds, full-data cumulative shares with display limits and singleton
association clustering. It preserves and completes the R29 changes already in the
starting checkout. Historical reports and evidence are unchanged.

All **53** maintained R29 regressions fail on the unchanged pre-R29 installed
wheel and pass on the repaired wheel. The full installed-wheel suites pass
**1425 / 1425 / 1409** tests on Python 3.9 locked / Python 3.12 locked /
Python 3.12 newer. Eight selected unchanged historical checks pass unmasked in
both 3.12 cells. Affected source checks pass **294 tests** with locked 3.12.
Eight controlled rendered examples were visually inspected on the newer wheel.
All 19 source/sdist/wheel payloads and 13 installed module hashes agree; copied
tests match current source. Dependency/lock checks, Ruff (330 total, zero new),
two gate tests, new-test formatting and whitespace pass. Inherited source
formatting differences are unchanged. The first locked 3.12 wheel-test process
ended with status 143 without a result; its complete rerun passes.

**Next recorded group: R30 alone**, with R04 complete. Use local RNG for fixtures,
generate exact odd/even sizes and correct Titanic's alone flag. R30, R32 and R33
remain pending. Hosted CI, full notebooks/docs, optional interactive backends,
other platforms and large-feature performance were not checked. No push, PR,
version bump or release is included.

## R30 — Local dataset fixtures complete — 2026-10-15

R30 is complete, based on R29 handoff commit `56f4f6c`, on local branch
`fix/r30-dataset-fixtures`, implementation commit `db2f7f7`. The
[handoff](R30-dataset-fixtures.md) records local
seeded draws across both generators and Titanic/cancer augmentation, reproducible
weights, exact odd/even row counts and Titanic's zero-relative alone flag. Existing
even-size features/targets remain identical to the R29 wheel at four checked sizes.
Historical reports and evidence are unchanged.

All **25** maintained R30 regressions pass in every installed-wheel cell; the
unchanged R29 wheel produces **13 failures and 12 passing controls**. Affected
source checks pass **68 tests**. Full installed-wheel suites pass **1450 / 1450 /
1434** on Python 3.9 locked / Python 3.12 locked / Python 3.12 newer. All three
selected unchanged historical RNG/odd-size checks pass unmasked in both 3.12
cells. Titanic feature construction uses a network-free fixture covering zero,
one and two relatives; no live OpenML download was needed.

All 19 source/sdist/wheel payloads and 13 installed module hashes agree; copied
tests match current source. Dependency/lock checks, Ruff (330 total, zero new),
two gate tests, new-test formatting and whitespace pass. The two inherited
utils.py formatting differences are unchanged. No new dependency or seed option
is added. These fixture checks do not establish selection calibration.

**Next recorded group: R32 alone**, with R01–R31 complete. Repair and execute the
documentation workflows and record optional dependency/claim boundaries. R32 and
R33 remain pending. Hosted CI, full docs/notebooks, optional backends, other
platforms and large-data performance were not checked. No push, PR, version bump
or release is included.

## R32 — Documentation workflows complete — 2026-10-15

R32 is complete, based on R30 handoff `0164300`, on local branch
`fix/r32-documentation-workflows`, implementation commit `3f61981`. The
[handoff](R32-documentation-workflows.md)
records early classification encoding, regression benchmark imports, optional
comparison imports, preserved sampling limits, reduced time-series model inputs,
current local links and qualified method/weight/backend claims.

Local/hosted configuration now uses the project docs extra and system Pandoc;
unused Python pandoc/sphinx-tabs requirements and mandatory FastTreeSHAP are
removed. HTML renders stored outputs without executing notebooks. Clean strict
HTML builds pass with zero Sphinx warnings in both Python 3.12 snapshots, including a
separate docs-extra-only wheel installation without setuptools. API docstring
markup repairs and native autodoc settings remove inherited build warnings;
runtime ASTs and all runtime requirement pins/hashes remain unchanged.

All **8** maintained regressions fail on the unchanged R30 source snapshot and
pass in every installed-wheel cell. CI copies documentation fixtures alongside
tests. Full wheel suites pass **1458 / 1458 / 1442** on Python 3.9 locked / Python
3.12 locked / Python 3.12 newer; the unchanged historical time-series regression
passes unmasked in both 3.12 cells. All package payloads, recorded installed module
hashes, test and documentation bytes agree. Ruff reports **326 total, zero new**,
with four repaired allowances removed; lock/dependency, gate, format and
whitespace checks pass.

Two full notebooks (basic/preprocessing), classification/regression excerpts and
the exact reduced-input time-series cells pass in fresh kernels in both 3.12
snapshots: **90 code cells and 13 figures per snapshot**. Classification explicitly
substitutes local cancer data for remote Titanic. Sampling's 10,000-row bound is
checked without the large distance allocation. All 17 notebook schemas and HTML
pages validate/render; the handoff separately lists unexecuted cells/notebooks.
Generated figures, the workflow page and an executed notebook preview were
visually inspected. One newer build was interrupted (143); its complete rerun
passes. No runtime change was made for that interruption.

**Next recorded group: R33 alone**, with R01–R32 complete. Final release/support
and calibration evidence remain pending. No full execution of all notebooks,
live OpenML, optional/GPU backend matrix, Windows or hosted build is claimed.
No R33 implementation, push, PR, version bump, tag or release is included.

## Ordered phases

R01–R33 are **Complete**. No next implementation group remains. The
[R01 handoff](R01-installed-package.md) records changes, decisions, exact checks,
environment identities, inherited failures, and the resume point. Changes are
committed as `6b675fd` on `fix/r01-installed-package` at checkpoint base `0270aca`;
at that handoff no next phase had started. The user subsequently authorized PR delivery.
The [R02/R03 handoff](R02-R03-compatibility.md) records the completed compatibility
batch, based on `a4b7ff6` on `fix/r02-r03-dependency-compatibility`. Its source
repairs are committed as `36f1ee6` (R02) and `65f894a` (R03). A version bump and
GitHub release remain separate delivery work; R33's local gates are now complete.
The [R04 handoff](R04-runtime-baseline.md) records the green installed-wheel
baseline on Linux Python 3.9.25 (locked) and 3.12.7 (locked/newer), dependency
policy, checks, and limits. R04 changes are uncommitted on `main`, based on
`170b3e9`. The [R05 handoff](R05-observation-boundaries.md) records completed
observation-boundary validation, promoted regressions and installed-wheel checks.
R05 changes are also uncommitted on `main`, preserving R04 work. The
[R06 handoff](R06-category-identity.md) records completed category identity and
dtype agreement, with changes uncommitted on `main` preserving R04/R05 work.
The [R07 handoff](R07-shadow-contracts.md) records completed owned encoding and
faithful shadow roles, with changes uncommitted on `main` preserving R04–R06 work.
The [R08 handoff](R08-statistical-arithmetic.md) records completed weighted
association/relevance arithmetic, uncommitted on `main` preserving R04–R07 work.
The [R09 handoff](R09-mrmr-ranking.md) records completed bounded relevance and
MRMR ranking, uncommitted on `main` preserving R04–R08 work. The
[R10 handoff](R10-leshy-support.md) and [R11 handoff](R11-permutation-importance.md)
record the completed sequential batch in separate local commits. Earlier
R04–R09 changes remain uncommitted; the commits rely on that checkout state.
Those uncommitted notes are historical; the R04–R12 delivery above records
their merged commits. [R13](R13-glm-scoring.md) is complete in local commit
`28b9529` on `main`, based on `64b0ce7`. [R14](R14-glm-design.md) is complete
in local commit `3e3e7e3` on `main`, based on `fb942fc`. [R15](R15-glm-likelihood.md)
is complete in local commit `51d44f2` on `main`, based on `2577289`.
[R16](R16-threshold-contracts.md) is complete in local commit `2fa18e8` on `main`,
based on `985b0a3`. [R17](R17-importance-selection.md) is complete in local commit
`6f44345` on `main`, based on `1e88c54`. [R18](R18-ordinal-schema.md) is complete
in local commit `7df5daa`, based on `37cf6b2`. [R31](R31-legacy-helpers.md) is complete
in local commit `ffb7388`, based on `7df5daa`. [R19](R19-tree-partitions.md) is complete
in local commit `6af7415`, based on `132e0e0`. [R20](R20-transformer-inputs.md) is complete
in local commit `cff3c59`, based on `e030fde`. [R21](R21-gbm-contracts.md) is complete
in local commit `6204a3f`, based on `d51c567`. [R22](R22-gbm-persistence.md) is complete
in local commit `4bd3681`, based on `4d0e240`. [R23](R23-benchmark-contracts.md) is
complete in local commit `ff8faa8`, based on `9a6a8b7`. [R24](R24-gower-distances.md)
is complete in local commit `5325b7f`, based on `4e50f38`.
[R25](R25-sampling-outputs.md) is complete in local commit `b352069`, based on
`34641a2`; [R26](R26-isolation-attempts.md) is complete in local commit `5c631bf`,
based on `b352069`. Both use branch `fix/r25-r26-remediation`.
[R27](R27-nearest-neighbors.md) is complete in local commit `45c3f98`, based
on `fcb811e`, on branch `fix/r27-nearest-neighbors`.
[R28](R28-parallel-contracts.md) is complete in local commit `2181790`, based
on `81dc6c9`, on branch `fix/r28-parallel-contracts`.
[R29](R29-plot-contracts.md) is complete on `fix/r29-plot-contracts`, based on
`605cbff`, implementation commit `24db954`.
[R30](R30-dataset-fixtures.md) is complete on `fix/r30-dataset-fixtures`, based
on R29 handoff `56f4f6c`, implementation commit `db2f7f7`.
[R32](R32-documentation-workflows.md) is complete on `fix/r32-documentation-workflows`,
based on R30 handoff `0164300`, implementation commit `3f61981`.
[R33](R33-release-evidence.md) completes local release evidence on
`fix/r33-release-evidence`, based on `f49bf9d`, implementation commit `aabb238`.
Hosted CI/review and publication
remain separate delivery work. An ID such
as `1A F3` means finding 3 in the linked original report. The complete ID-to-phase
mapping follows the table.

| Phase | Smallest complete outcome | Depends on | Required validation |
| --- | --- | --- | --- |
| **R01 — Usable installed package (Complete)** | Replace `pkg_resources` resource access with the standard library, include the housing description in distributions, bind the existing missing export. Decide whether the unconditional Styler return remains supported; if so, declare Jinja2. Sources: [00 F1/F3/F4](00-baseline.md), [03 F14](03-selector-contracts.md) | None | Build sdist then wheel from clean source; clean install with no historical setuptools; import all 16 modules, bind every `__all__` name, load bundled Boston/cancer/housing outside checkout; summary reaches its styling path. Verify resource lifetime when an API needs a real path |
| **R02 — Owned pandas column writes (Complete)** | Replace read-only shadow-array mutation and dtype-incompatible midpoint assignment, preserving values, dtypes, index and caller input. [00 F2](00-baseline.md), [02 F3](02-preprocessing.md) | R01 | Frozen with Copy-on-Write both off/on, newer pandas; numeric/category shadow values and real/shadow shape; interval helper/estimator and tree-to-midpoint pipeline |
| **R03 — Supported dependency APIs (Complete)** | Use the current sklearn keyword, Styler map, boxplot keyword, documented SciPy distance namespace, and scalar NumPy conditions. Avoid wrappers because supported minima already have the replacements. [1C F1](01c-weighted-glm-lasso.md), [03 F10](03-selector-contracts.md), [04 F13](04-training-and-benchmarks.md), [05 C2](05-sampling-and-parallel.md), [06 F11](06-utilities-and-workflows.md) | R01 | Small GLM entry fit; real summary HTML; real permutation plot; association clustering; scalar constant/nonconstant ANOVA in both snapshots. Numerical defects left to R08/R13 must not be mistaken for compatibility success |
| **R04 — Reproducible baseline and runtime gate (Complete)** | Correct the constant-test threshold/comparison and missing synthetic `nice_guys` column; add checks that cannot pass by selecting everything. Choose supported Python/dependency cells, regenerate/review the matching lock, and add installed-wheel import/resource/export/tests to CI. [00 F5–F8](00-baseline.md) | R01–R03 | Existing suite passes in chosen baseline cells; fixture asserts its intended signal/noise/category schema. A deliberately select-all implementation fails a meaningful selection test. Lock check succeeds; CI runs from installed artifacts. Begin with Linux/Python 3.12 frozen/newer and add an actual oldest-supported interpreter check before claiming the full range |
| **R05 — Valid observations at boundaries (Complete)** | Validate one-dimensional finite targets and aligned, finite nonnegative weights with positive retained/split mass. Do not invent missing responses. Apply empty-sample and zero-mass-group handling consistently through statistics and training; remove the overwritten duplicate converter. [1A F8–F10](01a-statistical-foundations.md), [1B F11](01b-all-relevant.md), [04 F6/F7](04-training-and-benchmarks.md) | R01 | Lists/arrays/Series, wrong lengths/dimensions, negative/nonfinite/all-zero weights, no complete rows, zero-weight added category, and custom `handle_na=None`; unchanged data and aligned rows. This does not choose frequency versus relative weighting |
| **R06 — Category identity and dtype agreement (Complete)** | Reuse one pandas-aware dtype rule and encode distinct original values without string collisions. Cover boolean, object/category, explicit strings and inferred strings. [1A F13](01a-statistical-foundations.md), [03 F4](03-selector-contracts.md), [05 C1](05-sampling-and-parallel.md), [06 F1](06-utilities-and-workflows.md) | R01/R05 | Distinguish `1` from `'1'`, missing values from literal `'nan'`/`'None'`; series/matrix and worker agreement; nullable-string cardinality; all shared encoder callers |
| **R07 — Faithful shadow controls (Complete)** | Encode owned selector frames, carry categorical identity through reductions, give shadows matching model roles, and generate collision-free labels. [1B F5/F6/F10](01b-all-relevant.md) | R02/R06 | No mutation after real BoostAGroota/GrootCV fits; original/shadow categorical positions in successive rounds and folds; numeric-valued categories in Leshy; input names already matching generated labels |
| **R08 — Weighted association/relevance arithmetic (Complete)** | Choose weight meaning and degrees of freedom; implement the documented linear statistic, constant handling, output shape, stable centered sums, scale-cancelling Pearson arithmetic and collision-free response augmentation. [1A F3–F7/F11/F12](01a-statistical-foundations.md) | R05/R06; R03 for newer NumPy | Independent Pearson/F/ANOVA references, retained-row degrees of freedom, integer replication or relative-scale invariance as appropriate, normalized weights, constant/zero-mass categories, large translations, predictor named `target`, one/two-worker equivalence |
| **R09 — Coherent MRMR ranking (Complete)** | Remove the two invalid score normalizations; choose a nonnegative common relevance scale and use absolute unnormalized redundancy. Define empty/equal/perfect candidate behavior. [1A F1/F2](01a-statistical-foundations.md) | R08 | Mixed singleton predictors remain eligible; no redundant negative-copy preference; hand ranking with fixed relevance; empty/constant/perfect scores; strong-signal/null and mixed-type screens. Do not claim calibrated mixed-type significance |
| **R10 — Leshy support/history consistency (Complete)** | Let support computation own the mask and use all actual history rows. [1B F1/F2](01b-all-relevant.md) | R07 | Confirmed-only, tentative-only, mixed and empty selections with `keep_weak` on/off; mask/count/transform agreement; two-iteration median and rejected ranking |
| **R11 — Weighted signed permutation importance (Complete)** | Forward validation scoring weights; normalize permutation scores by absolute total, keep all-zero vectors zero and require positive mean evidence. [1B F3/F4](01b-all-relevant.md) | R05/R07 | Real weighted sklearn reference; negative-total and zero-total fixed vectors; both Leshy and BoostAGroota callers; no promotion of negative scores as positive evidence |
| **R12 — Explicit all-relevant aggregation (Complete)** | Resolve GrootCV objective/output-axis handling, then choose the intended fold/shadow statistic for GrootCV and BoostAGroota and align prose. [1B F7–F9](01b-all-relevant.md) | R07 | Binary-to-multiclass and objective-alias real fits; exact hand examples with fold columns only; documented cutoff axes/order. Calibration remains a separate claim |
| **R13 — Correct GLM scoring and independent CV (Complete)** | Compute response-scale deviance once, expose a higher-is-better sklearn score while minimizing BIC correctly, pass stratification targets, slice documented arrays, clone each fold. [1C F2–F4/F9/F12](01c-weighted-glm-lasso.md) | R03/R05 | Independent Poisson/binomial/Gamma losses, real external GridSearchCV, binary fold splits, array/frame parity, stateful-estimator isolation; rerun the Poisson signal counterexample |
| **R14 — Stable GLM design and parameters (Complete)** | Use one fitted intercept/input-column contract across fit/predict/transform; derive the family/link at fit from current parameters; return self. [1C F5/F7/F8/F10](01c-weighted-glm-lasso.md) | R13 | Single-row prediction, existing constants, `fit_intercept=False`, reorder/missing/extra/duplicate names, arrays and support metadata, `set_params` vs fresh construction, chained fit/predict and summary integration |
| **R15 — Honest likelihood and refit contract (Complete)** | Decide zero-weight likelihood handling, reject all-invalid candidate criteria, describe regularized selection followed by unpenalized refit, remove the near-ridge promise. [1C F6/F11](01c-weighted-glm-lasso.md) | R13/R14 | Zero-weight equivalence to omission or clear rejection; all-nonfinite search fails; independent weighted Gaussian/Poisson BIC comparisons; documented coefficient/refit behavior and post-selection inference limit |
| **R16 — Threshold selector identity/state (Complete)** | Reject duplicate names or adopt consistent positions, align labelled callbacks, validate parameters at fit, handle zero/one-feature collinearity, stabilize ties and fitted metadata. Resolve equality wording explicitly. [03 F5–F9/F12](03-selector-contracts.md) | R06/R08 | Masks/names/output agree, reordered callback labels, invalid `set_params`, empty pipeline stages, multiple hash seeds, correct feature counts and NotFittedError. Preserve legitimate named projection if retained as the API |
| **R17 — Truthful cumulative importance selection (Complete)** | Keep the crossing feature, reject nonfinite contributions, define zero-total selection, handle array multiclass targets, and raise the optional-backend error. Use one fit unless actual reproducible resampling is required. [03 F1–F3/F11/F13](03-selector-contracts.md) | R03/R05/R06 | Hand `[.6,.3,.1]` thresholds, constant-response real fit, threshold 1, invalid contributions, three-class array/Series parity, iteration behavior and missing fastshap error before wasted training |
| **R18 — Ordinal encoder full schema (Complete)** | Keep categorical encoder validation separate from full passthrough metadata, copy inverse input, honor changed output dtype. [02 F8/F13/F14](02-preprocessing.md) | R06 | Mixed categorical/numeric `get_feature_names_out` and pandas `set_output`; unknown/missing values, input preservation, inverse, clone/refit and `set_params(output_dtype=...)` |
| **R19 — Actual fitted tree partitions (Complete)** | Derive bins from real thresholds/leaf membership, avoid temporary-column collisions, derive copied one-tree parameters at fit, apply encoding in raw mode, define missing/all-missing behavior. [02 F1/F2/F4/F5/F9](02-preprocessing.md) | R05/R06/R18 | Equal-valued leaves, binary probabilities vs leaf IDs, values between observed extrema and fitted splits, `x`/`x_g`, category named `missing_added`, all-missing column, parameter aliases and native weighted one-tree reference |
| **R20 — Reusable transformer inputs and Patsy rows (Complete)** | Separate constructor options from fitted state; normalize documented fit/transform inputs identically; honor Patsy `NA_action` and intercept choice; treat scalar midpoint name as one name. [02 F6/F7/F10–F12](02-preprocessing.md) | R02/R18/R19 | Refit different schemas, ndarray fit_transform, named dict formula, midpoint second fit/scalar name, Patsy raise/drop branches and row alignment. Document that a transformer cannot also drop y/weight rows |
| **R21 — GBM prediction and split contracts (Complete)** | Reject or explicitly prioritize incompatible group/stratification options; validate prediction schemas and objective aliases; state real categorical support/disable behavior; define multiclass offset input; reset fitted state. [04 F1–F5/F8/F9](04-training-and-benchmarks.md) | R05/R06 | Disjoint groups, reorder/replacement columns, label/probability shapes across aliases, explicit category disable, object/string/time error or fitted mapping, offset shapes, refits clearing flags/validation data/curves |
| **R22 — Persist the wrapper contract (Complete)** | Save/load required wrapper metadata with the Booster and reset state coherently. Define legacy Booster-only file handling without guessing an unknown offset contract. [04 F10](04-training-and-benchmarks.md) | R21 | Fresh-process ordinary/category/offset round trips, external baseline required for new rows, loading into previously fitted wrapper, invalid/missing artifact errors; trusted-input boundary stays explicit. Exercise safe write failure behavior before claiming durable persistence |
| **R23 — Comparisons that run what they label (Complete)** | Fit encoders on training rows only; settle documented array/name support; reject incompatible GrootCV comparisons; construct fresh selector/model combinations. [04 F11/F12/F14/F15](04-training-and-benchmarks.md) | R07/R10–R12/R18/R21 | Held-out-only category excluded from vocabulary; integer display names, array policy, real fitted estimator/method tracing, seed/backend state reset, weighted scoring retained |
| **R24 — Valid Gower distances (Complete)** | Use stable direct numeric ranges, distinguish cross/self matrices, handle pairwise missingness explicitly, normalize documented input/masks, validate feature weights. [05 F1–F5](05-sampling-and-parallel.md) | R06 | Hand scalar reference, translation/scale invariance, equal-sized different frames, missing/constant features, no-comparable-pair policy, arrays/Series/extension dtypes/sparse errors, valid feature-weight rescaling |
| **R25 — Sampling returns the intended data (Complete)** | Separate row weights from distance feature weights; choose weighted representative semantics or reject them; impute by column, avoid `label` collision, raise unknown methods and return original isolation-selected rows. [05 F6–F8/F10](05-sampling-and-parallel.md) | R05/R24 | N≠P and N=P weights, per-column means, original column/index/dtype schema, an existing `label`, selected categorical values, clear unsupported-method/weight errors |
| **R26 — Bounded isolation attempts (Complete)** | Advance retry state, bound size progression and define exhaustion/tiny-input behavior. Correct KS-p-value interpretation in API/example prose. [05 F9 and statistical limits](05-sampling-and-parallel.md) | R25 | Deterministic forced rejection reaches bounded termination; draw sizes follow the documented rule; 0/1/10-row inputs handled explicitly. A terminating sampler still has no representativeness guarantee |
| **R27 — Query-aware nearest neighbors (Complete)** | Use the actual one-row query, exclude its key, deduplicate eligible keys once, validate n/query and define insufficient-candidate/nonfinite behavior. [05 F11/F12](05-sampling-and-parallel.md) | R24 | Query changes result; duplicate IDs; enough/too few unique candidates; n=0/negative/too large; optional-query policy; only a query-by-candidate distance vector allocated |
| **R28 — Parallel labels and bounded useful work (Complete)** | Chunk original label objects without NumPy coercion; settle result order/empty schema and worker-count semantics, avoiding empty tasks. [05 F13 and resource boundaries](05-sampling-and-parallel.md) | R06/R08 | Integer/string and tuple/MultiIndex labels with one/two workers; propagated callback error; no-pair and empty work; supported negative counts; worker cap observed without a new execution framework |
| **R29 — Plots match the data and selection (Complete)** | Keep feature/subplot coordinates separate, color actual Leshy membership, include terminal importance thresholds, respect plot limits and linear feature positions, bypass singleton clustering. [06 F2–F7](06-utilities-and-workflows.md) | R03/R10/R17 | Render and inspect string/integer/single-panel target plots; masks disagreeing with importance order; threshold 1/one/no feature, plot_n limit, first cumulative point, input/figure ownership |
| **R30 — Honest local dataset fixtures (Complete)** | Use local RNG consistently, generate exactly the requested odd/even row count, correct Titanic's alone flag. [06 F8/F9/F15](06-utilities-and-workflows.md) | R04 | Caller RNG stream unchanged; reproducible features/targets/weights; strong/noise construction checks; odd size; network-free 0/1/2-relative fixture. Do not introduce a new data service |
| **R31 — Retain or remove legacy helpers deliberately (Complete)** | User chose removal of public `cat_var` and `validate_pandas_input`, without shims. Remove the dead benchmark comment and cat_var-only import; document encoder/array migration. [02 F15](02-preprocessing.md), [06 F10](06-utilities-and-workflows.md) | R06/R18 | Absent public helpers/exports; supported encoder keeps missing distinct, handles newer strings and owns inputs; active callers and retained function/class ASTs rechecked |
| **R32 — Reproducible documentation workflows (Complete)** | Repair classification/regression setup, sampling overwrite, time-series reduced-feature calls, README filename and local Sphinx source. Reconcile docs-extra/RTD/fastshap policy; qualify method/weight/backend claims after preceding choices. [05 F14](05-sampling-and-parallel.md), [06 F12–F14](06-utilities-and-workflows.md), [07 F1](07-final-assessment.md) | Relevant R01–R31 fixes for each example | Actual Sphinx build and local links; bounded clean-kernel basic/classification/regression/preprocessing examples; exact reduced-feature model input; size limit maintained. Record data substitutions, excluded optional cells, and remaining notebooks separately |
| **R33 — Release evidence and remaining claims (Complete)** | Run the repaired package gates and a justified support matrix. Reassess selection claims with meaningful held-out baselines and document remaining optional/performance limits; narrow unsupported claims instead of promising unbounded validation. | All release-blocking phases above | Fresh source/wheel/lock metadata agree; no workaround; established suite and promoted regressions pass; repeated-seed signal/null, weak/correlated/mixed/weighted cases appropriate to claims; fold-local preprocessing/selection; clean-process artifacts; docs build; clean Ruff without the temporary baseline. Optional backends/platforms require their own tests before claiming support |

R21's group/schema safeguards are complete, with R05/R06 prerequisites retained.
R24–R33 are complete. No recorded implementation group remains. Do not parallelize edits to
shared dtype/weight helpers without agreeing their contract first.

## Complete finding-to-phase map

This map keeps every historical ID visible without copying its explanation.
Cross-cutting side effects are tested in the affected phase as described above.

| Report | Finding IDs → phase |
| --- | --- |
| [00](00-baseline.md) | F1/F3/F4 → R01; F2 → R02; F5/F6/F7/F8 → R04 |
| [1A](01a-statistical-foundations.md) | F1/F2 → R09; F3/F4/F5/F6/F7/F11/F12 → R08; F8/F9/F10 → R05; F13 → R06 |
| [1B](01b-all-relevant.md) | F1/F2 → R10; F3/F4 → R11; F5/F6/F10 → R07; F7/F8/F9 → R12; F11 → R05 |
| [1C](01c-weighted-glm-lasso.md) | F1 → R03; F2/F3/F4/F9/F12 → R13; F5/F7/F8/F10 → R14; F6/F11 → R15 |
| [02](02-preprocessing.md) | F1/F2/F4/F5/F9 → R19; F3 → R02; F6/F7/F10/F11/F12 → R20; F8/F13/F14 → R18; F15 → R31 |
| [03](03-selector-contracts.md) | F1/F2/F3/F11/F13 → R17; F4 → R06; F5/F6/F7/F8/F9/F12 → R16; F10 → R03; F14 → R01 |
| [04](04-training-and-benchmarks.md) | F1/F2/F3/F4/F5/F8/F9 → R21; F6/F7 → R05; F10 → R22; F11/F12/F14/F15 → R23; F13 → R03 |
| [05](05-sampling-and-parallel.md) | F1/F2/F3/F4/F5 → R24; F6/F7/F8/F10 → R25; F9 → R26; F11/F12 → R27; F13 → R28; F14 → R32; C1 → R06; C2 → R03 |
| [06](06-utilities-and-workflows.md) | F1 → R06; F2/F3/F4/F5/F6/F7 → R29; F8/F9/F15 → R30; F10 → R31; F11 → R03; F12/F13/F14 → R32 |
| [07](07-final-assessment.md) | F1 → R32 |

## Decisions that must remain explicit

| Decision | Phase/owner and safe boundary |
| --- | --- |
| Supported Python/dependency range | R04/R33; six installed-wheel cells cover Linux x86_64 snapshots, not every allowed combination/platform. Do not copy the Python-3.12-only newer pins onto earlier interpreters |
| Frequency counts vs relative/exposure weights | R05/R08/R15/R25; validation can be shared, statistical meaning cannot be silently normalized away. Gower feature weights are a different dimension |
| Mixed relevance scale and significance | R08/R09; Theil-U-to-F arithmetic is not evidence of an F null distribution. Keep a ranking heuristic labelled as such unless justified |
| Shadow threshold rule and error control | R10–R12/R33; decide axes/mean/median/max explicitly. Correct binomial arithmetic or signal recovery is not whole-procedure calibration |
| Threshold equality/empty output | R16/R17/R24; keep undefined/no-positive-mass distinct from zero effect. Document any intentional change in selected sets |
| GLM intercept/likelihood/refit | R14/R15; fit-time design must govern inference. Weight-valid prediction loss does not automatically justify every likelihood/BIC or post-selection p-value |
| Weighted global explanations | R17 complete: the user chose equal averaging over validation rows, with weights retained in training and validation loss. API/docs and native references record that population |
| Named schemas, optional features, legacy files/helpers | R01/R16/R21/R22/R31/R32; record permissive projection vs strict schema, styling/backend install policy, and old artifact/public-API behavior |
| Sampling meaning | R24–R27; cluster representatives are synthetic rows, isolation returns sampled original rows. KS acceptance is not 95% representativeness or proven outlier removal |

## Accounted-for validation gaps

The [coverage ledger](evidence/07-final-assessment/coverage.md) and review 07
remain the detailed source of truth. R33 owns the release decision, not a promise
to support every possible configuration. Specifically:

- Python/platform claims require actual installed-wheel runs, including exact
  minima if advertised. Optional GPU/CUDA, XGBoost/CatBoost, fasttreeshap and
  HoloViews/Panel need separate execution or clearly stated limits.
- Full sklearn compliance and arbitrary transformer feature lineage require
  targeted estimator/pipeline checks after state/schema repair. Same-environment
  ordinary model reload is not cross-version/hostile-artifact safety.
- Small existing screens are insufficient for calibrated false-discovery claims,
  weak/correlated/interacting signals or weighted/categorical superiority. Use
  prespecified repeated seeds and training-only preprocessing with untouched
  evaluation data; keep statistical interpretation separate from smoke success.
- Large Gower matrices, large-data throughput, rare-group sampling, quota-aware
  parallel performance, interrupted writes, and remote dataset outages remain
  unmeasured. Test them only to the scale/security guarantees being claimed;
  keep quadratic-storage and trusted-pickle limits explicit.
- Strict local HTML builds and the recorded five bounded workflows pass in both
  snapshots. Full notebook execution and hosted builds remain outside that scope.
  Optional benchmarks are not required dependencies for ordinary examples without
  an explicit reason.

No implementation, split, dependency migration, commit, push, or release is
authorized merely by this handoff. Findings assigned to pending phases remain open;
the historical review reports retain their original observations.
