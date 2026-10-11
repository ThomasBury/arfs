# 07 — Compatibility, package boundaries, and final assessment

**The queued review is complete. Release blockers remain. Keep one distribution.**
The source still contains the installation, data-integrity, statistical, and API
failures demonstrated in reviews 00–06. Completing this review is not approval
to release or a claim that every supported configuration has been tested.

Reviewed commit: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11.
Only review reports, evidence, and queue/plan status changed. Package source,
existing tests, dependencies, lockfile, CI, earlier reports, and the project
`.venv` were preserved. Nothing was committed, pushed, or published.

The [coverage ledger](evidence/07-final-assessment/coverage.md) accounts for all
16 source modules, their public workflows, both dataset resources, and all 17
notebook entry points. The [remediation handoff](REMEDIATION.md) orders repairs
into independently verifiable phases. All earlier findings remain open.

## Evidence and what was repeated

The source revision has not moved. Both saved installed-wheel environments still
match **all 16 source files byte for byte**. The new [frozen audit](evidence/07-final-assessment/frozen-audit.json)
and [newer audit](evidence/07-final-assessment/current-audit.json) record paths,
SHA-256 hashes, complete installed distribution metadata, import statements,
public definitions, history, and bounded import measurements. Both use Python
3.12.7/Linux and retain `setuptools==80.9.0`, unit 00's import workaround.

Unit 00's clean-install, wheel-through-sdist build, exports, resources, suite,
and stale-lock results remain applicable and were reused, not relabelled as new
successful checks. The existing suite was 13 passed/2 failed frozen and 7
passed/8 failed newer, **after** the workaround. Its unmodified clean installs
failed collection. No package-wide test/build/lock rerun was justified by a
source change here. Unit 06's failed documentation build attempts also remain
authoritative: Sphinx was absent, and no HTML build completed.

New work consists of source/dependency ownership and history inspection,
cross-version dependency resolution, import-cost probes, missing optional-backend
boundaries, and an exact-cell time-series counterexample. [Five checks](evidence/07-final-assessment/checks.py)
produce **4 passed/1 strict expected failure in each snapshot**. With markers
disabled, each has **4 passed/1 failed**. Logs: [frozen](evidence/07-final-assessment/locked-checks.txt),
[frozen unmasked](evidence/07-final-assessment/locked-failures.txt),
[newer](evidence/07-final-assessment/wheel-checks.txt),
[newer unmasked](evidence/07-final-assessment/wheel-failures.txt).
These passing checks establish actionable HoloViews/Panel import errors and
explicit Leshy/BoostAGroota fastshap fallback; no optional backend actually ran.

## Compatibility matrix

ARFS declares Python `>=3.9,<3.13` and lower-bounded runtime dependencies.
[Resolver evidence](evidence/07-final-assessment/resolution.json) contains exact
commands, selected pins, stderr, exit codes, and elapsed times. The new resolution
queries do not install packages or regenerate `uv.lock`. They use Linux and
`--no-build`, requiring usable wheels. [uv's documented target-version resolution](https://docs.astral.sh/uv/concepts/resolution/)
checks a chosen Python version; it does not execute the resolved package.

| Cell | New resolution / metadata result | Runtime evidence and limit |
| --- | --- | --- |
| Python 3.9, declared runtime | Online resolution succeeds: NumPy 2.0.2, pandas 2.3.3, sklearn 1.6.1, SHAP 0.49.1 | Not executed; no 3.9 interpreter installed. Initial offline attempt lacked a cached Matplotlib wheel; that was a cache restriction, not an impossible dependency range |
| Python 3.10, declared runtime | Resolves: NumPy 2.2.6, pandas 2.3.3, sklearn 1.7.2, SHAP 0.49.1 | Interpreter available, environment not installed/tested in this review |
| Python 3.11, declared runtime | Resolves: NumPy 2.4.6, pandas 3.0.6, sklearn 1.9.1, SHAP 0.51.0 | Interpreter available, environment not installed/tested; 3.12 results cannot certify it |
| Python 3.12, frozen saved wheel | Metadata-compatible near-minimum dependency snapshot | Executed across reviews, with many demonstrated defects and the import workaround. SHAP is 0.47.1, so this is not an exact test of every declared minimum |
| Python 3.12, saved newer wheel | New runtime resolution succeeds and selects the same eight headline numerical/model versions as the saved snapshot | Existing installed snapshot executed; no freshly installed resolution. Numerous allowed-version failures below |
| Python 3.10, project `docs` extra only | Resolves 105 packages, including Sphinx 6.2.1 | Not installed/rendered; dependency resolution is not a documentation build |
| Python 3.10, hosted-docs combined requirements | Wheel-only resolution stops because fasttreeshap has no usable wheel | Source build/native ABI behavior untested. RTD allows source builds, so this is **not proof that RTD cannot install** |
| Python 3.13/3.14 | Excluded by ARFS metadata | Not proposed or tested as supported versions |
| Other operating systems/architectures | No new resolution or runtime matrix | Untested, including native LightGBM/SHAP dependencies |

There is no demonstrated unsatisfiable **base** range: all four declared Python
minor versions resolve. The complete saved newer snapshot cannot be copied onto
3.9–3.11: its NumPy 2.5.4, SciPy 1.18.1, and SHAP 0.53.0 require Python >=3.12.
That is an impossible *specific pin combination*, not a defect in ARFS's lower
bounds. Older interpreters need different pins, as the resolution shows.

Allowed dependency versions nonetheless break public workflows:

| Dependency change | Established failure / source location | Original evidence |
| --- | --- | --- |
| Runtime without historical setuptools | `utils.py:10` imports missing `pkg_resources`, blocking 14/16 imports | [00 F1](00-baseline.md) |
| pandas protected views | `allrelevant.py:1693` shuffles a read-only array; also triggered on frozen pandas with Copy-on-Write | [00 F2](00-baseline.md) |
| pandas dtype changes / strings | Midpoint and `cat_var` assignments; inconsistent extension/inferred-string handling in associations/Gower | [02 F3/F15](02-preprocessing.md), [03 F4](03-selector-contracts.md), [05 F4/C1](05-sampling-and-parallel.md) |
| pandas Styler removal | `summary.py:69`, `applymap` absent after adding Jinja2 | [03 F10/F14](03-selector-contracts.md) |
| sklearn keyword removal | `lasso.py:449`, `force_all_finite` rejected | [1C F1](01c-weighted-glm-lasso.md) |
| NumPy scalar operation | `association.py:1461–1462`, scalar `where/nonzero` rejected | [05 C2](05-sampling-and-parallel.md) |
| Matplotlib boxplot keyword | `benchmark.py:93–96`, `labels` rejected | [04 F13](04-training-and-benchmarks.md) |
| SciPy namespace exposure | `association.py:2119`, `sch.distance` absent | [06 F11](06-utilities-and-workflows.md) |

Use supported APIs already available at the declared minima before considering
compatibility wrappers. Constraints may document genuinely unsupported versions;
they must not be used to claim correctness for older versions with the same
statistical defects. The exact support policy remains a maintainer decision.

## Installation, optional features, and release gates

- The wheel metadata and `arfs.__version__` say **3.0.0**; `uv.lock` records
  **2.4.0**. Its dependency pins are useful historical evidence, but unit 00's
  failing `uv lock --check` means it is not a matching project lock.
- Build requirements are not runtime requirements. The `pkg_resources`
  workaround does not repair clean installs. The omitted housing description
  and unbound `BaseThresholdSelector` export remain separate failures
  ([00 F1/F3/F4](00-baseline.md)).
- Jinja2 is an actual missing runtime dependency for the unconditional styled
  summary return ([03 F14](03-selector-contracts.md)). Direct imports of joblib
  and Patsy are not declared by ARFS, but installed metadata explicitly brings
  them through sklearn and statsmodels, respectively. Those are transitive
  ownership concerns, **not newly demonstrated missing imports**. If retained
  as directly used APIs, make that dependency ownership explicit when editing
  metadata; no new implementation dependency is needed.
- HoloViews/Panel imports are lazy and fail descriptively. Leshy/BoostAGroota
  warn and change `importance='fastshap'` to `'shap'` when unavailable.
  VariableImportance instead has the proven missing-`raise` error (03 F13).
  GrootCV imports fasttreeshap during explanation after fitting and wraps
  failures in `RuntimeError`; actual backend outputs remain untested.
- `docs/requirements.txt` has older runtime floors and requires fasttreeshap,
  while the project docs extra comments it out. RTD installs **both** requirement
  sources using Python 3.10; project floors still apply, so those older floors
  do not prove an old-runtime build. They do add a native backend not required
  by the base package. Git commit `dabb7da` removed the backend from project
  installation; docs and examples retained it. Resolve this policy explicitly,
  keeping an optional comparison out of the ordinary docs build unless needed.
- `.github/workflows/ruff.yml` is the sole workflow and runs Ruff only. It does
  not validate imports, wheel resources, exports, runtime behavior, docs, or
  statistical claims. This review did not run or certify the Ruff gate.
  Existing all-relevant assertions also accept selecting every feature
  ([00 F7/F8](00-baseline.md)); a green lint or current suite would be insufficient.

## Package boundaries and maintainability

**Keep one `arfs` distribution.** Its demonstrated users combine filtering,
associations/MRMR, all-relevant selection, preprocessing and plotting in the same
tabular modelling workflows. GLM users also depend on preprocessing and summary
contracts. Splitting packages would not repair the semantics those users share.

The recorded imports show concrete coupling: MRMR and collinearity use
associations; associations use parallel execution and dtype utilities;
VariableImportance and TreeDiscretizer use GradientBoosting; the benchmark uses
selectors and preprocessing. Sampling has no other source caller and is a more
isolated candidate, but still reaches `utils` for three predicates. There is no
observed independent consumer, incompatible release cadence, or measured install
saving that requires a new sampling distribution.

There is a measurable import cost to revisit *inside* the package:

| Fresh-process import | Frozen median seconds / process peak MiB | Newer median seconds / process peak MiB |
| --- | ---: | ---: |
| `arfs` | <0.001 / 30.1 | <0.001 / 30.4 |
| `arfs.parallel` | 0.491 / 74.8 | 0.447 / 71.2 |
| `arfs.association` | 1.696 / 173.8 | 2.739 / 185.1 |
| `arfs.feature_selection.unsupervised` | 2.423 / 276.4 | 3.780 / 227.7 |

Three fresh interpreters per cell, warm filesystem cache, Linux `ru_maxrss`,
native threads limited to one; these are whole-process peaks, not marginal
allocations, wheel sizes, or general throughput estimates. Importing even the
unsupervised submodule executes its parent namespace and loads SHAP, LightGBM,
statsmodels, Patsy and plotting. Frozen SHAP also loads Numba. Associations reach
LightGBM through `utils`, even for hand-calculated statistics.

If startup or a smaller installation becomes a concrete requirement, first
separate lightweight helpers from eager backend/resource imports, and examine
optional extras within the existing distribution. Measure an actual user's
import and installation benefit before changing dependency policy. A split would
require resolving shared helper ownership, preserving/revising public imports,
coordinating versions and tests, and migrating examples. No quantified benefit
currently outweighs that cost; no new package architecture is proposed.

Deletion/reuse opportunities are narrower and already evidenced:

- The first `utils.validate_sample_weight` definition is overwritten by an
  identical second implementation at import time (`utils.py:230,242`). Delete
  that duplicate when fixing the actual validation boundary; keeping one
  converter alone would not make negative or nonfinite weights valid.
- `cat_var`, `validate_pandas_input`, `is_xgboost`, and the preprocessing
  `highlight_discarded` have no internal source/test consumers; two modules
  define `is_list_of_str`. Consider removal/reuse after an explicit public-API
  decision. Public visibility and autodoc mean “unused locally” is not proof
  of no external users.
- Prefer the existing encoder after its contracts are repaired to another
  category policy. Reuse one clear dtype rule across callers. Stop deterministic
  repeated importance fits that do identical work (03 F3), unless actual
  resampling is deliberately required.
- The last 40 non-merge source-touching commits contain many version/SHAP/API
  adjustments. Excluding the broad Ruff formatting commit, no file pair occurs
  together more than once; the largest pair set comes from a typo sweep.
  This sparse, mostly single-file history does not establish independent product
  boundaries or justify a split. Exact commits/pairs are retained in the audits.

## New finding from the remaining workflow inspection

**F1 — The time-series post-selection prediction plot still uses every original
feature. P2; high confidence; small remedy.** Source:
[arfs_timeseries.ipynb](../docs/notebooks/arfs_timeseries.ipynb), zero-based cell 10.
The cell creates `X_fs` without `random_num`/`holiday` and splits that frame, then
fits and predicts on `X`. Consequently the displayed predictions do not show
the reduced feature set. Cell 11's separate metric call does use `X_fs`.

The new check executes the **exact unchanged cell** with 60 local rows, a real
TimeSeriesSplit, and an argument-recording model stub. Both fit and predict see
`['signal','random_num','holiday']` instead of `['signal']` in both snapshots.
No remote data, real forecast performance, or optional backend result is claimed.

Smallest remedy: use `X_fs` for both calls, then keep this argument/schema
regression check and run the real excerpt during notebook repair. Separately,
selection over the full series followed by manually chosen column removal is
not evidence of fold-local feature-selection performance; evaluate selection
inside training folds before making that claim.

## Consolidated priorities and unresolved decisions

| Shared problem | Consequence and priority | Original owners |
| --- | --- | --- |
| Installation and API compatibility | P0 import/shadow blockers; P1 resource/export/summary and current dependency failures | 00; 1C F1; 02 F3/F15; 03 F10/F14; 04 F13; 05 C1/C2; 06 F11 |
| Lost input identity or invalid observations | P1 altered caller categories, generated-name collisions, wrong prediction columns, group leakage, fabricated responses and invalid weights | 1A F8/F10/F12/F13; 1B F5/F6/F10/F11; 1C F5/F8; 02 F2/F8/F10; 03 F4–F6; 04 F1/F2/F4–F10; 05 F4–F8/F10/F13; 06 F1 |
| Wrong selection/calculation | P1 negative/reversed scores, constant selection, wrong GLM loss/search direction, tentative promotion, cumulative cutoff, distorted bins/distances/neighbors | 1A; 1B F1–F4/F7–F9; 1C F2/F3/F6/F9; 02 F1/F4/F5; 03 F1/F2; 05 F1–F3/F11/F12 |
| State, runtime and misleading presentation | P1/P2 unbounded sampling, stale fit/load state, irrelevant repeated fits, false comparison/plot labels, broken workflows | 1C F7/F10/F12; 02 F6/F7/F9/F11–F14; 03 F3/F7–F13; 04 F9–F15; 05 F9/F14; 06; 07 F1 |
| Missing regression/release discipline | P1 defects survive tests and lint-only CI; support/statistical claims exceed evidence | 00 F5–F8 and all review limitations |

The [handoff](REMEDIATION.md) maps **every original finding ID**, dependencies,
and relevant tests to repair phases; this table groups causes rather than
recounting each symptom as a new defect. Fixes must retain explicit decisions:
frequency versus relative weights and degrees of freedom; mixed relevance scale
and Theil-U heuristic meaning; GrootCV/BoostAGroota shadow aggregation; threshold
equality; zero-importance/no-comparable-data behavior; GLM intercept and zero-weight
likelihood; weighted global explanations; sampling row versus feature weights;
legacy model metadata and public API removal. No statistical choice was silently
made by this assessment.

## Remaining checks and review completion

| Unresolved check | Evidence limit and required follow-up |
| --- | --- |
| Full Python/platform support | Four resolutions, runtime only on Linux/Python 3.12.7. After fixes, install/test the chosen oldest-supported and current Python/dependency cells, plus any platforms the release claims |
| Full docs/notebooks | No completed Sphinx build or complete notebook. Repair launchers/setup and optional prerequisites; run bounded clean-kernel examples and a real docs build before calling them working |
| Statistical validity | Existing small strong-signal/null screens and hand references do not establish error control, mixed-type calibration, correlated/weak/interacting signal recovery, weighted/categorical stability, or superiority to baselines |
| Temporal/grouped evaluation | Group-plus-stratification leak proven; no complete time-series selection evaluation. Keep preprocessing/selection inside training folds and use an untouched evaluation set |
| Estimator contracts and persistence | No full sklearn estimator suite, generic feature expansion lineage, all-transformer serialization or cross-version model compatibility. Trusted ordinary GBM round trips passed; offset metadata failed |
| Optional/native backends | No GPU/CUDA, XGBoost/CatBoost/Boruta comparison, fasttreeshap numeric equivalence, or actual HoloViews/Panel render. Import/fallback checks are narrower |
| Resources/security/performance | Gower scaling limited to 100–400 rows; import costs are bounded probes. No large-data throughput, isolated RSS attribution, rare-group sampling study, hostile artifact, interrupted save, remote outage or live OpenML check |

These gaps are accounted for, assigned follow-up in the handoff, and remain
limits on release claims. There is no unowned source module or pending review
item left. Earlier reports remain historical evidence; their old “next unit”
text is superseded by the current queue. **Review complete; remediation not
started; release blocked.**

## Reproduction

From the repository root, use the saved interpreters (their exact installed
requirements/metadata are in the audit JSON and earlier snapshots):

```sh
review_repo="$PWD"
review_evidence="$review_repo/reviews/evidence/07-final-assessment"
for review_mode in locked wheel; do
  review_python="/tmp/arfs-review-03f67d0/$review_mode-env/bin/python"
  "$review_python" "$review_evidence/audit.py" "$review_repo" \
    "$review_evidence/$review_mode-rerun-audit.json"
done
/tmp/arfs-review-03f67d0/locked-env/bin/python "$review_evidence/resolve.py"
mkdir -p /tmp/arfs-final-checks
cd /tmp/arfs-final-checks
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arfs-final-checks/mpl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for review_mode in locked wheel; do
  review_python="/tmp/arfs-review-03f67d0/$review_mode-env/bin/python"
  "$review_python" -m pytest "$review_evidence/checks.py" -q -rx --tb=short \
    -c /dev/null --rootdir="$review_evidence" -p no:cacheprovider
  "$review_python" -m pytest "$review_evidence/checks.py" -q --runxfail --tb=short \
    -c /dev/null --rootdir="$review_evidence" -p no:cacheprovider
done
```

Unmasked checks intentionally exit 1. The resolver uses online metadata after
one offline attempt; its output can change with new releases/cache state and
is not a frozen replay. Audit imports run in temporary directories with `-I`.
If the saved environments are gone, follow unit 00's audited-wheel recreation
instructions, not a silent substitution of the published package.

Not checked: the unexecuted cells and broader guarantees listed above. No source
fix was implemented; the documented release and statistical risks remain.
