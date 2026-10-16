# R32 — Reproducible documentation workflows

**R32 complete on 2026-10-15**, based on R30 handoff commit `0164300`.
Branch: `fix/r32-documentation-workflows`. Implementation commit: `3f61981`.
R32 runs alone, with R01–R31 complete.

## Changes and decisions

Classification defines its fitted ordinal encoding before the first Leshy fit.
Classification/regression import optional CatBoost/XGBoost dependencies only in
their comparison cells; regression imports sklearn_pimp_bench before using it.
The basic notebook uses current Styler.map and leaves its two FastTreeSHAP cells
commented out as explicit opt-in examples. The preprocessing introduction states
the estimator-specific meaning of category codes and training-only fitting.

The sampling notebook keeps its seeded 10,000-row presample instead of replacing
it with the full frame. Its dense float64 distance output alone is about 800 MB;
clustering/temporary storage needs more. The time-series post-selection plot now
fits and predicts with X_fs. Its manually reduced columns and full-series
selection remain exploratory, with outer-window selection required for unbiased
performance assessment. These repairs do not claim improved sampling or model
performance.

Both Sphinx launchers use docs/ as their source and the ignored _build directory
for output. conf.py imports the installed package version, without path hacks.
The HTML build renders saved outputs with nbsphinx_execute="never"; notebook
execution is a separate check. Saved outputs for edited code cells are cleared;
other historical outputs remain explicitly labelled as such in the workflow
guide. Fresh executed validation copies are retained in local evidence.

Read the Docs uses Python 3.12, a system Pandoc executable and the project's
single docs extra. The duplicate docs/requirements.txt, mandatory FastTreeSHAP
installation and theme-uninstall hook are removed. The unused Python pandoc
package and unused sphinx-tabs extension are removed from the extra. The latter
required pkg_resources and blocked a clean build. No setuptools workaround,
theme change or new dependency is added. The lock removes those packages and
pandoc's unused plumbum/ply dependencies; all runtime pins/hashes are unchanged.

The actual strict build exposed malformed API markup. Docstring-only repairs
remove redundant method tables, fix indentation and literal references, and
remove stale p-value promises from score-returning association helpers. The API
reference stops expanding duplicate package reexports and sklearn-generated
metadata-request helper descriptions. Automatic section labels cover top-level
headings; type preprocessing uses Napoleon's default. Detailed ARFS methods
remain documented. All package runtime ASTs match the starting revision after
docstrings are removed.

README/intro/method descriptions now state model/importance dependence, empirical
shadow rules, MRMR's ranking limits, unpenalized GLM refits, equal-row SHAP
summaries and optional-backend limits. The README preprocessing filename and an
additional broken comparison-notebook link are corrected. The new
[workflow guide](../docs/workflows.rst) records build and execution boundaries.

Caller tracing covered the six affected workflows, comparison imports, README
and toctree/notebook links, API autodoc pages, project docs metadata, the hosted
configuration and installed-wheel CI. Historical reviews/evidence are unchanged.
New evidence remains local and ignored under evidence/r32-documentation-workflows/.

## Validation

All **8** finalized documentation regressions fail against a faithful snapshot
of the unchanged `0164300` docs with the R30 installed wheel. They pass in all
three repaired installed-wheel cells. CI now copies docs, README and the logo
alongside tests so these checks actually run outside the checkout. They cover
dependency-free core imports, early categorical encoding, the exact presampling
bound before distance allocation, reduced model columns, both launcher paths
and local documentation links.

| Cell | Full installed suite | Unchanged historical time-series regression |
| --- | --- | --- |
| Python 3.9.25 locked | 1458 passed | Not rerun in this cell |
| Python 3.12.7 locked | 1458 passed | 1 passed unmasked |
| Python 3.12.7 newer | 1442 passed | 1 passed unmasked |

Fresh wheel environments have no setuptools. All 13 recorded installed module
hashes match source; copied tests and documentation fixtures match current bytes.
All 19 package source/sdist/wheel payloads agree. The full suite has no expected
failures or skipped documentation cases. The original review 07 time-series
check runs unchanged with --runxfail.

Strict HTML builds using ``-W --keep-going`` pass with **zero Sphinx warnings** in the
locked and newer snapshots. A separate clean wheel installation with the docs
extra alone also builds all API/tutorial pages with zero Sphinx warnings, without
setuptools, pkg_resources, Python pandoc, sphinx-tabs or FastTreeSHAP. The system
Pandoc version and installed distributions are recorded in evidence. One newer
HTML build ended with status 143 before reporting a result; its complete fresh
output-directory rerun passes. No runtime change was needed for that interruption.

The dependency/lock gates, Ruff (**326 total, zero new**), two gate unit tests,
focused formatting and whitespace pass. Four repaired Ruff allowances are
removed. The four edited package modules retain identical inherited formatter
differences; the AST and formatter comparisons are recorded separately.

## Clean-kernel execution ledger

Both Python 3.12 dependency snapshots pass the following five workflows. Each
uses a fresh explicitly named kernel, 120-second per-cell timeout, bounded native
threads/CPU affinity and declared setup/assertion cells. Existing estimator
hyperparameters in source cells are not rewritten. Each snapshot executes
**90 code cells** including the added checks and produces **13 figures**.

Cell indexes below are zero-based source JSON indexes; markdown and commented
cells retain their original positions.

| Notebook | Executed source scope | Data and boundary |
| --- | --- | --- |
| basic_feature_selection | All 57 source cells, top to bottom | Local 1,000-row regression fixture and local cancer data; FastTreeSHAP examples are comments, not executed backend checks |
| preprocessing | All 23 source cells, top to bottom | Local 1,000-row regression fixture; added five-tree weighted model check converts interval bins with IntervalToMidpoint first |
| arfs_classification | Cells 0–13 | Explicit local 569-row cancer substitute for remote Titanic; source setup/encoding stays unchanged; native and SHAP Leshy fits/plots run |
| arfs_regression | Cells 0–11 and 34 | Bundled Boston data; native Leshy and the repaired real permutation benchmark run |
| arfs_timeseries | Cells 6 and 10 | Local 60-row positive-response fixture with two temporal splits; real HistGradientBoostingRegressor fit/predict receive only the signal column |

The classification encoder additionally has a real categorical-input regression.
Sampling's exact 10,000-row bound is checked with a recording sample callable on
10,005 rows, without allocating a 10,000-square distance matrix. The distance
algorithm itself retains R24's independent tests.

Executed notebooks, HTML previews, PNGs, source hashes, exact selected/omitted
cells, data substitutions and added assertions are in the local
[locked ledger](evidence/r32-documentation-workflows/locked/execution-ledger.json)
and [newer ledger](evidence/r32-documentation-workflows/newer/execution-ledger.json).
All 17 notebook schemas validate and all 17 tutorial pages render. Figures from
both snapshots were visually inspected, with direct checks of the preprocessing
and temporal plots. The generated workflow page and executed preprocessing HTML
were also inspected in the browser. The temporary loopback preview server/tab
were closed.

## Remaining notebook execution

Classification, regression and time-series have only the bounded scopes above;
their remaining cells were not executed. The following notebooks were rendered
and schema-validated, but not executed in full:

| Notebook | Execution limit |
| --- | --- |
| association_and_feature_selection | Remaining association tutorial not run |
| arfs_large_data_sampling | Presampling call checked; full Gower/isolation and downstream selection not run |
| arfs_on_GPU | GPU/backend workload not run |
| arfs_shap_vs_fastshap | Optional compiled-backend comparison not run |
| arfs_grootcv_custom_params | Full custom-parameter tutorial not run |
| arfs_boruta_borutaShap_comparison | Optional comparisons not run; broken local link repaired |
| arfs_non_normal_loss_and_sample_weight | Full weighted/custom-loss tutorial not run |
| mrmr_feature_selection | Full MRMR tutorial not run |
| mrmr_fs_VS_arfs | Full method comparison not run |
| lasso_feature_selection | Full lasso tutorial not run |
| issue_categoricals | Full categorical investigation not run |
| issue_collinearity | Full collinearity investigation not run |

Successful examples are smoke/integration checks, not repeated-seed calibration,
held-out superiority or a supported optional-backend matrix. No live OpenML data,
Windows batch execution, hosted Read the Docs/CI build or large-data benchmark
was checked. Optional progress widgets can warn and fall back to text; no widget
dependency is added just for validation.

## Reproduction and resume

The exact argv/cwd/status records for wheel tests and saved kernel ledgers remain
in local evidence. Main commands:

```sh
uv build --python /tmp/arfs-r32/docs-locked/bin/python --out-dir /tmp/arfs-r32/dist
/tmp/arfs-r30/py312-locked/bin/python /tmp/arfs-r32/check_payloads.py
/tmp/arfs-r30/py312-locked/bin/python /tmp/arfs-r32/validate.py py39-locked
/tmp/arfs-r30/py312-locked/bin/python /tmp/arfs-r32/validate.py py312-locked
/tmp/arfs-r30/py312-locked/bin/python /tmp/arfs-r32/validate.py py312-newer
/tmp/arfs-r32/docs-locked/bin/python /tmp/arfs-r32/execute_examples.py locked
/tmp/arfs-r32/docs-newer/bin/python /tmp/arfs-r32/execute_examples.py newer
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make -C docs html \
  'SPHINXBUILD=/tmp/arfs-r32/docs-final/bin/python -m sphinx' \
  BUILDDIR=/tmp/arfs-r32/html-clean-install 'SPHINXOPTS=-W --keep-going'
/tmp/arfs-r32/docs-locked/bin/python /tmp/arfs-r32/check_final.py
uv lock --check --python /tmp/arfs-r30/py312-locked/bin/python
uv tool run --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
git diff --check
```

Context7's primary documentation was consulted for
[nbsphinx execution](https://github.com/spatialaudio/nbsphinx/blob/master/doc/never-execute.ipynb),
[Sphinx builds](https://www.sphinx-doc.org/en/master/man/sphinx-build.html),
[Read the Docs configuration](https://docs.readthedocs.com/platform/stable/config-file/v2.html)
and [NBClient execution](https://github.com/jupyter/nbclient/blob/main/docs/client.rst).

Stop after local R32 implementation/handoff commits. Next is **R33 alone**, with
R01–R32 complete: final release evidence, justified support/calibration checks and
remaining claim limits. No R33 implementation, push, PR, version bump, tag or
release is included. The package remains 3.0.0 pending that release decision.
