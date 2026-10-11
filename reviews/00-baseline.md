What this repo does: ARFS selects predictors for tabular classification and regression. It also provides weighted associations, GLMs, preprocessing, sampling, training, plotting, and datasets. This review assumes CPU use with categorical predictors and sample weights.

This is unit 0 of the staged review: the reproducible baseline. It reviews commit `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2` on 2026-10-10. No library code, existing tests, dependencies, lockfile, or CI configuration changed. All installations and test runs used disposable directories under `/tmp/arfs-review-03f67d0`; the existing `.venv` was inspected without installing anything into it.

## Results

Clean installation succeeds, but importing the feature-selection namespace fails. Adding `setuptools==80.9.0` to the disposable environments gets past that failure and exposes a pandas compatibility defect and two failing baseline tests. The extra setuptools installation is a diagnostic workaround, not a passing clean-install result or a proposed permanent dependency.

| Environment | Dependencies | Before workaround | Existing suite after workaround |
| --- | --- | --- | --- |
| Existing `.venv` | Existing installed versions | No pytest; feature-selection import fails | Not run; environment preserved |
| Clean editable source | Current resolution | 14 of 16 modules fail; collection has 2 errors | 7 passed, 8 failed |
| Clean installed wheel | Exactly the same versions as clean source | Same failures | 7 passed, 8 failed |
| Clean installed wheel | Frozen runtime/test dependencies from `uv.lock` | Same failures | 13 passed, 2 failed |

All 15 existing tests ran in each diagnostic environment. The current source and wheel runs fail the same eight tests. Six failures are in BoostAGroota/GrootCV shadow construction; the remaining two also fail with frozen dependencies. All six Leshy tests pass, as does the missing-value filter test. With frozen dependencies, all 12 all-relevant selector tests pass. These outcomes do not establish statistical reliability.

Both the wheel and source distribution built from a clean Git archive. The wheel was built through the source distribution, using setuptools 84.0.0. Tests were copied unchanged to an outside directory, and wheel module paths resolve to `site-packages`, not the checkout. `uv lock --check` fails because the committed lockfile needs updating.

Raw evidence: [environment versions, import traces, exports, dataset checks, and small counterexamples](evidence/00-baseline/environments.json), [test and collection output](evidence/00-baseline/runs.txt), [build output](evidence/00-baseline/build.txt), and [distribution contents, metadata, and SHA-256 hashes](evidence/00-baseline/distributions.json).

## Environment and declared usage

The interpreter was CPython 3.12.7 on Linux x86-64, glibc 2.39, kernel 6.8.0-142-generic. The machine has an Intel Core i5-7300U at 2.60 GHz and four logical CPUs. The installer was uv 0.12.15. The package declares Python `>=3.9,<3.13`; this unit tested only Python 3.12.7 on Linux.

| Dependency | Existing / frozen baseline | Current source and wheel |
| --- | --- | --- |
| ARFS | 3.0.0 | 3.0.0 |
| LightGBM | 4.6.0 | 4.7.0 |
| matplotlib | 3.9.4 | 3.11.2 |
| NumPy | 2.0.2 | 2.5.4 |
| pandas | 2.2.3 | 3.0.6 |
| scikit-learn | 1.6.1 | 1.9.1 |
| SciPy | 1.13.1 | 1.18.1 |
| seaborn | 0.13.2 | 0.13.2 |
| SHAP | 0.47.1 | 0.53.0 |
| statsmodels | 0.14.4 | 0.15.0 |
| tqdm | 4.67.1 | 4.70.1 |
| pytest | Absent in existing; 8.3.5 in frozen | 9.1.1 |
| pytest-cov | Absent in existing; 6.0.0 in frozen | 7.1.0 |

Full transitive versions are in the environment evidence. The [current requirements snapshot](evidence/00-baseline/current-requirements.txt) reproduces the dependency set without resolving newer versions. No clean runtime environment initially contains setuptools. A build-system requirement does not supply this runtime import.

The [README](../README.md) advertises `pip install arfs`, sklearn-compatible selectors, categorical predictors, sample weights, three importance methods, GLM/Lasso selection, and supervised discretization. Notebook imports confirm workflows combining these components; they were inventoried, not executed. GPU and fasttreeshap examples exist but were not exercised.

`arfs` itself exposes version `3.0.0`. The feature-selection namespace declares these 12 names in `__all__`: `BaseThresholdSelector`, `MissingValueThreshold`, `UniqueValuesThreshold`, `CardinalityThreshold`, `CollinearityThreshold`, `VariableImportance`, `make_fs_summary`, `Leshy`, `BoostAGroota`, `GrootCV`, `MinRedundancyMaxRelevance`, and `LassoFeatureSelection`. Only 11 are bound after import; finding 4 describes the missing name. Other modules have no explicit `__all__`. Their non-private, locally defined classes/functions and imports are recorded in the [inventory](evidence/00-baseline/inventory.json); that inventory does not promise stability for every visible name.

## Findings ledger

Severity describes consequence; confidence describes evidence. Effort and compatibility impact are estimates for the smallest remedy, not authorization to implement it.

### Must fix

1. **Clean installations cannot import most of ARFS.** P0; high confidence; small effort; no intended public API change. Locations: [utils.py](../src/arfs/utils.py):10, 595, 615, 619; [pyproject.toml](../pyproject.toml):1–3, 21–32.

   What this is: dataset paths use `pkg_resources.resource_filename`, imported when utilities load. The feature-selection namespace, associations, GBM, preprocessing, sampling, and benchmarking reach this import eagerly.

   Problem: `import arfs.feature_selection` raises `ModuleNotFoundError: No module named 'pkg_resources'` after a successful install. The same root cause blocks 14 of 16 module imports in all three clean environments; only `arfs` and `arfs.parallel` import successfully. Existing-environment failure is therefore reproduced as a packaging/runtime defect, not explained away by that environment's missing test tools.

   Fix: replace the three resource-path calls and their shared import with standard-library resource access, preserving file lifetime where a real path is needed. Adding unrestricted setuptools would not solve current installations: setuptools removed `pkg_resources` in version 82. See the [official removal notice](https://setuptools.pypa.io/en/latest/history.html#v82-0-0).

   If skipped: ordinary feature-selection and supporting imports remain unusable without an undeclared historical dependency.

2. **BoostAGroota and GrootCV fail when pandas protects array views from writes.** P0; high confidence; small effort with categorical checks; preserve input, index, and dtype behavior. Location: [allrelevant.py](../src/arfs/feature_selection/allrelevant.py):1678–1707, especially 1693.

   What this is: `_create_shadow` copies the frame and shuffles each column through `np.random.shuffle(X_shadow[c].values)`. Its three call sites are BoostAGroota's reduction at line 1780 and GrootCV's training/validation construction at lines 2303–2304.

   Problem: a three-row numeric frame raises `ValueError: array is read-only` on pandas 3.0.6. The same frame succeeds on pandas 2.2.3 by default and fails when only `mode.copy_on_write=True` changes. All four BoostAGroota and both GrootCV tests encounter this failure with current dependencies. pandas documents [read-only views under Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html); ARFS permits these pandas versions.

   Fix: generate a shuffled column and assign it back to the shadow frame instead of mutating its underlying view. Check categorical dtype, non-default index, unchanged input, and preserved column values before adopting the change.

   If skipped: two of the three all-relevant selectors fail on ordinary numeric data, even after the import workaround.

3. **The housing loader needs a file omitted from both distributions.** P1; high confidence; small effort; no intended API change. Locations: [pyproject.toml](../pyproject.toml):69–71 and [utils.py](../src/arfs/utils.py):599–642.

   What this is: `load_data("housing")` opens a bundled description before reading the data. The loader expects `dataset/descr/housing.rst`, while package-data configuration names `arfs.dataset.description`.

   Problem: after the import workaround, source loading returns 20,640 rows and eight predictors; both wheel environments raise `FileNotFoundError`. Archive inspection confirms that `housing.zip` and `boston_bunch.joblib` ship, but `housing.rst` does not. Boston and cancer load in both installation modes.

   Fix: correct the package-data key and verify loading from a newly built wheel outside the checkout.

   If skipped: source examples can work while installed users cannot load the housing dataset.

4. **A declared public export does not exist.** P1; high confidence; small effort; additive compatibility impact. Location: [feature_selection/__init__.py](../src/arfs/feature_selection/__init__.py):1–27.

   What this is: `__all__` declares names available from the namespace. It includes `BaseThresholdSelector`, but the namespace never imports that class from `.base`.

   Problem: after the import workaround, `hasattr(arfs.feature_selection, "BaseThresholdSelector")` is false and `from arfs.feature_selection import *` raises `AttributeError`. The class exists in its defining module.

   Fix: import the existing class into the namespace and check every declared export.

   If skipped: consumers of the declared namespace contract still fail after finding 1 is repaired.

### Should fix

5. **The committed lockfile is not a valid frozen project baseline.** P1; high confidence; small effort after choosing dependency constraints; dependency changes need review. Locations: [uv.lock](../uv.lock):27–30 and [arfs/__init__.py](../src/arfs/__init__.py):3.

   What this is: the lockfile records ARFS 2.4.0, while source and freshly built metadata report 3.0.0.

   Problem: `uv lock --check` exits 1 with “The lockfile at `uv.lock` needs to be updated”. Exporting dependency pins with `--frozen --no-emit-project` allowed this review to test their runtime behavior, but does not establish that the whole project lock is current.

   Fix: regenerate and review the lock against the chosen support policy after resolving the runtime blockers. Preserve this snapshot as historical evidence.

   If skipped: locked development/CI commands cannot establish a matching project environment.

6. **Two baseline tests have broken inputs or expectations.** P1; high confidence; small effort; fixture changes can alter benchmark outputs. Locations: [test_featselect.py](../tests/test_featselect.py):37–61, [utils.py](../src/arfs/utils.py):883–923, and [base.py](../src/arfs/feature_selection/base.py):112–119.

   What this is: the uniqueness test intends to reject constants, and the cardinality test expects a generated string column.

   Problem: with `threshold=2`, the implementation rejects `var0`, `var4`, and `var10` because their unique counts are 2, 2, and 1. The test expects only `var10` and compares a multi-element NumPy array directly with a list, raising an ambiguous-truth-value error. Separately, the classification generator constructs the `nice_guys` label list but never assigns a column, so the cardinality test expects a feature absent from its input. Both failures occur with frozen and current dependencies.

   Fix: use threshold 1 and an array-aware comparison for the stated constant-removal test, add an explicit threshold-boundary test, and restore the intended generated column with a fixture assertion. Review threshold wording in the API unit before changing selector semantics.

   If skipped: the suite remains red independently of the current pandas incompatibility, and classification tests do not exercise the intended string predictor.

7. **All-relevant selector assertions accept selecting every feature.** P1; high confidence about the test gap; medium effort; test-only changes initially. Location: [test_allrelevant.py](../tests/test_allrelevant.py):17–267.

   What this is: all 12 tests assert a non-empty intersection between selected names and an expected list. The inventory retains each assertion.

   Problem: returning all input columns satisfies every selection assertion. The Boruta comparisons named in four tests are commented out. Tests do not constrain false selections, noise-only behavior, stability, or independent held-out performance. One Leshy test labeled regression also calls the classification generator at line 131.

   Fix: add independent, seeded reference checks and statistical screens in units 1A–1C, including false selections and the planned baselines. Start by correcting misleading test names/data and testing the actual claimed behavior.

   If skipped: passing selector tests can coexist with invalid selection behavior. No such statistical invalidity is proved by this baseline alone.

8. **Repository CI does not run a package or runtime check.** P1; high confidence; small initial effort; CI-only impact. Location: [ruff.yml](../.github/workflows/ruff.yml):1–8.

   What this is: the sole GitHub workflow runs Ruff. There is no test, build, or installed-wheel import step.

   Problem: none of the concrete import, dataset, export, or runtime failures above is exercised by that workflow. This review did not run Ruff or claim that current CI is green.

   Fix: begin with one Linux/Python 3.12 job that builds and installs the wheel outside the checkout, checks imports/datasets/exports, and runs the repaired tests. Unit 3 should choose the smallest additional Python/dependency matrix supported by evidence.

   If skipped: source changes and allowed dependency upgrades can break installation without a runtime CI check detecting them.

## Module-to-test map

The snapshot has 16 Python source files, 10,921 source lines, 15 active test functions, and 17 notebooks. Commented-out tests are excluded. File counts, locally defined names, exact test node IDs/assertions, and notebook import lines are in the inventory. Notebook import extraction records only matching lines, not full multiline import statements.

The frozen wheel suite executed 917 of 2,890 measured statements (31.73%). Coverage includes module loading, class/function definitions, and decorators; it is not a measure of statistical validity. The [coverage record](evidence/00-baseline/coverage.json) preserves per-module counts.

| Source module under `src/arfs/` | Existing tests / substantive use | Line execution |
| --- | --- | --- |
| `__init__.py` | Version assigned on import | 100% |
| `feature_selection/__init__.py` | Imported by both test files; no export contract assertion | 100% |
| `feature_selection/allrelevant.py` | `TestLeshy` (6), `TestBoostAGroota` (4), `TestGrootCV` (2) | 68% |
| `feature_selection/base.py` | Indirectly exercised by three threshold tests | 78% |
| `feature_selection/unsupervised.py` | Missing values, unique count, cardinality: one each; collinearity imported but not tested | 47% |
| `utils.py` | Synthetic data, LightForest wrappers, and internal selector helpers | 50% |
| `association.py` | No calculation assertions; import/decorator execution | 12% |
| `feature_selection/lasso.py` | No GLM, Lasso, or weighted-CV workflow test | 18% |
| `feature_selection/mrmr.py` | No MRMR fit test | 19% |
| `feature_selection/summary.py` | No summary workflow test | 27% |
| `feature_selection/variable_importance.py` | No importance-selector fit test | 16% |
| `gbm.py` | No wrapper-training test | 10% |
| `parallel.py` | No parallel-calculation test | 27% |
| `preprocessing.py` | No transformer fit/transform workflow test | 18% |
| `benchmark.py` | No existing test | 0% |
| `sampling.py` | No existing test | 0% |

The baseline's separate import, export, dataset, and shadow-helper probes are not included in these coverage numbers.

## Reproduction

These commands use the audited commit, Python 3.12.7, and disposable directories. Start in the checkout. They leave the original source and environment untouched. Choose an installed Python 3.12.7 executable if `.venv/bin/python` no longer points to it.

```sh
repo="$PWD"
audit_dir=$(mktemp -d /tmp/arfs-baseline.XXXXXX)
export UV_CACHE_DIR="$audit_dir/cache"
mkdir -p "$audit_dir/source" "$audit_dir/run"
git archive 03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2 | tar -x -C "$audit_dir/source"
cp -r "$audit_dir/source/tests" "$audit_dir/run/tests"

uv venv --python "$repo/.venv/bin/python" "$audit_dir/source-env"
uv pip install --python "$audit_dir/source-env/bin/python" \
  -r "$repo/reviews/evidence/00-baseline/current-requirements.txt" \
  -e "$audit_dir/source[test]"
uv build "$audit_dir/source" --out-dir "$audit_dir/dist"

uv venv --python "$repo/.venv/bin/python" "$audit_dir/wheel-env"
uv pip install --python "$audit_dir/wheel-env/bin/python" \
  -r "$repo/reviews/evidence/00-baseline/current-requirements.txt" \
  "$audit_dir/dist/arfs-3.0.0-py3-none-any.whl"

uv export --project "$audit_dir/source" --frozen --extra test \
  --no-emit-project --output-file "$audit_dir/locked-requirements.txt"
uv venv --python "$repo/.venv/bin/python" "$audit_dir/locked-env"
uv pip install --python "$audit_dir/locked-env/bin/python" \
  -r "$audit_dir/locked-requirements.txt" \
  "$audit_dir/dist/arfs-3.0.0-py3-none-any.whl"

uv lock --check --project "$audit_dir/source" --python "$repo/.venv/bin/python"
# Expected exit 1: stale lockfile. Continue with the checks below.
```

To repeat an unconstrained current resolution instead, omit the `-r` argument from the editable install, then capture its installed dependency versions for the wheel. New releases may produce different outcomes. The committed requirements file captures this run's resolved versions; it is not a new project support policy. Build dependencies remain unpinned in the project, so fresh archive hashes may differ; this run's wheel records setuptools 84.0.0.

```sh
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$audit_dir/mpl"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONHASHSEED=0
cd "$audit_dir/run"

for mode in source wheel locked; do
  "$audit_dir/$mode-env/bin/python" -I -c \
    'import arfs; print(arfs.__version__, arfs.__file__); import arfs.feature_selection'
  "$audit_dir/$mode-env/bin/python" -m pytest --collect-only -q \
    -p no:cacheprovider tests
done
# Every import fails on pkg_resources; every collection exits 2 with two errors.

for mode in source wheel locked; do
  uv pip install --python "$audit_dir/$mode-env/bin/python" 'setuptools==80.9.0'
  timeout 600 taskset -c 0 "$audit_dir/$mode-env/bin/python" -c \
    'import numpy as np, pytest, random; np.random.seed(20261010); random.seed(20261010); raise SystemExit(pytest.main(["tests", "-q", "-p", "no:cacheprovider"]))'
done
```

These commands intentionally return failures; do not use `set -e` across the diagnostic sequence. The original runs used CPU affinity 0 (frozen wheel), 1 (source), and 2 (current wheel). NumPy and Python random were seeded to 20261010 before pytest; ARFS imports/generators also reset NumPy's global seed internally. No repeat-seed reliability claim is made. Runs overlapped, and the frozen run included coverage, so their recorded 72.75, 43.01, and 37.65 seconds are execution records, not comparable performance measurements.

For coverage, add `--cov=arfs --cov-report=term --cov-report=json:coverage.json` to the frozen run's pytest argument list and set `COVERAGE_FILE` inside the temporary directory. No branch coverage was measured.

Run the recorded small counterexamples after the workaround:

```sh
"$audit_dir/locked-env/bin/python" \
  "$repo/reviews/evidence/00-baseline/reproduce_failures.py"
"$audit_dir/source-env/bin/python" \
  "$repo/reviews/evidence/00-baseline/reproduce_failures.py"

"$audit_dir/wheel-env/bin/python" -I -c \
  'from arfs.utils import load_data; print(load_data("housing").data.shape)'
"$audit_dir/source-env/bin/python" -I -c \
  'from arfs.utils import load_data; print(load_data("housing").data.shape)'
"$audit_dir/wheel-env/bin/python" -I -c \
  'import arfs.feature_selection as fs; print({name: hasattr(fs, name) for name in fs.__all__}); exec("from arfs.feature_selection import *")'
```

The reproduction script asserts the known fixture defects so an accidental change to the trigger fails visibly. It does not certify the algorithms. The housing commands respectively fail and return `(20640, 8)`; the export command reports the missing name and fails.

## Coverage boundaries and handoff

Inspected: package metadata and lock state; README and API module listings; notebook import usage; all active tests; all module imports and top-level public definitions; the failing helpers and their source callers; declared exports; local dataset loading; clean wheel/sdist contents; and the sole GitHub workflow. Runtime checks used the audited source and built wheel, not previously present `build/` or `dist/` outputs or the published PyPI release.

Unresolved: complete threshold-boundary semantics; categorical shadow behavior after any future repair; compatibility across all declared Python versions; the validity of statistical claims; and which dependency constraints the project intends to maintain. None is inferred from these test results.

The remediation backlog follows findings 1–8. Repair the import and shadow blockers first, then distribution/export failures, baseline tests/lock consistency, and runtime CI. Keep the statistical test expansion in the corresponding review units. No repair was applied in this session.

Recommended next review unit: **1A, statistical foundations**. Use the recorded frozen dependencies plus the explicitly documented import workaround in an isolated environment while reviewing weighted associations, relevance statistics, missing values, and MRMR scoring. Trace shared callers and start with hand-calculable references; do not treat the 12 passing selector smoke tests as statistical evidence. Units 1B–7 remain pending. One distribution remains the working assumption; no keep/split decision has yet been established.

Verdict: the baseline is reproducible and exposes release blockers; the package is not release-ready on the tested installations.

Not checked: statistical screens, full estimator contracts, notebook execution, documentation builds, remote dataset downloads, GPU/optional backends, serialization/security paths, a full version matrix, resource benchmarks, Git co-change history, downstream use, and package-separation costs. The source distribution was built and used to build the wheel; a separate end-user sdist installation was not run.
