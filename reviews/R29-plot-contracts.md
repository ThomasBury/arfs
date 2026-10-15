# R29 — Plots match the data and selection

**R29 complete on 2026-10-15**, based on `main` commit
`605cbff` (merged R28). Branch: `fix/r29-plot-contracts`.
Local implementation commit: `24db954`.
R29 runs alone, with R03/R10/R17 complete. The starting checkout already
contained the R29 implementation and tests; this phase preserves and completes
that work.

## Changes and decisions

Target-versus-feature plots keep feature labels separate from subplot coordinates
and always use a two-dimensional axes array. String/integer names, single panels,
single rows and incomplete final rows use the correct data; unused axes are
hidden. Invalid column counts and featureless inputs fail before creating a
figure. Layout applies to the returned, caller-owned figure.

Leshy assigns box colors by feature name after sorting importance histories.
Confirmed features are blue, tentative features yellow and rejected features
gray. Tentative membership takes precedence when keep_weak also includes those
features in support_. Plot order never determines fitted membership.

VariableImportance slices only the displayed prefix to plot_n. Cumulative shares
still use all fitted features, and the title reports the full fitted selection
count. Markers use the last actually selected feature, including threshold 1
and rounded terminal shares. No feature cutoff is drawn for an empty selection
or one beyond the displayed prefix. The horizontal requested-share marker stays
visible. Feature coordinates and cumulative shares are linear; log controls only
the importance-value axis, with an all-zero vector kept linear. Automatic height
uses the displayed count with a two-inch minimum; explicit figsize takes priority.
Plot limits/density are validated before creating a figure, and plotting requires
fitted support metadata. Summary data is preserved.

The shared clustering helper returns a copy of a singleton association matrix,
retaining its original value, including NaN. Empty/nonsquare matrices raise a
deliberate ValueError. Static plotting validates/clusters before allocating its
figure, and preserves caller-supplied axes. Both static and optional interactive
clustering callers share the repair; the optional backend itself is untested.

Caller tracing covered plot_y_vs_X, all uses of cluster_sq_matrix, static and
interactive association plotting, CollinearityThreshold.plot_association, the
selector plots used by benchmarks, maintained tests and historical R29 evidence.
No estimator selection rule, export, dependency, backend or data fixture changes.
Historical reports and diagnostic scripts remain unchanged. New evidence is local
and ignored under reviews/evidence/r29-plot-contracts/.

## Validation

All **53** finalized R29 regression cases fail on the unchanged pre-R29 installed
wheel and pass in every repaired installed-wheel cell. Affected source checks pass
**294 tests** with Python 3.12 locked dependencies.

| Cell | Full installed suite | Selected unchanged historical checks |
| --- | --- | --- |
| Python 3.9.25 locked | 1425 passed | Not rerun in this cell |
| Python 3.12.7 locked | 1425 passed | 8 passed unmasked |
| Python 3.12.7 newer | 1409 passed | 8 passed unmasked |

The first locked 3.12 test process ended with status 143 before producing a result.
The complete rerun uses the same installed wheel and byte-identical tests and
passes; both attempts remain recorded. No package change was needed. Historical
importance fixtures set only a summary rather than fitted support metadata;
current importance regressions instead run the public fit with controlled scores.
The eight unchanged historical cases cover target plots, Leshy colors, singleton
clustering, signed annotations/ownership and empty-matrix rejection, with their
original expected-failure markers disabled using --runxfail.

Fresh environments install the rebuilt wheel outside the checkout, without
setuptools. All 13 installed module hashes match source, all copied test bytes
match the maintained suite and all 19 source/sdist/wheel payloads agree. Locked
NumPy/pandas/SciPy/Matplotlib are 2.0.2/2.2.3/1.13.1/3.9.4; newer versions are
2.5.4/3.0.6/1.18.1/3.11.2. Exact environment identities and command results remain
in local evidence.

Dependency checks, lock validation, exported requirement pins/hashes, Ruff
(**330 total, zero new**), two gate unit tests, focused formatting and whitespace
checks pass. Before/after formatter differences are identical in all four changed
source modules; inherited formatting/lint debt is preserved.

The regression suite covers string/integer labels, panel shapes, actual box colors
with and without keep_weak, selection/display cutoffs, thresholds 0/1, singleton
and zero-total importance, rounded terminal shares, invalid plot arguments,
fitted-state errors, input preservation, figure ownership, undefined singleton
associations and the public collinearity plotting caller.

Eight controlled examples were rendered from the newer installed wheel and
visually inspected: string/integer target panels, one panel, Leshy colors with
importance order disagreeing with membership, default log importance, truncated
full-data shares, zero-total importance and a singleton heatmap. Their data is
controlled to isolate display semantics; they do not claim calibration or signal
recovery. PNGs and the contact sheet remain in local evidence.

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
  PYTHONPATH=src /tmp/arfs-r28-final/py312-locked/bin/python -m pytest \
  tests/test_plot_contracts.py tests/test_dependency_apis.py \
  tests/test_shadow_contracts.py tests/test_allrelevant_aggregation.py \
  tests/test_importance_selection.py tests/test_leshy_support.py \
  tests/test_threshold_contracts.py -q --tb=short -p no:cacheprovider
uv build --python /tmp/arfs-r28-final/py312-locked/bin/python --out-dir /tmp/arfs-r29/dist
/tmp/arfs-r28-final/py312-locked/bin/python /tmp/arfs-r29/check_payloads.py
/tmp/arfs-r28-final/py312-locked/bin/python /tmp/arfs-r29/validate.py py39-locked
/tmp/arfs-r28-final/py312-locked/bin/python /tmp/arfs-r29/validate.py py312-locked
/tmp/arfs-r28-final/py312-locked/bin/python /tmp/arfs-r29/resume.py py312-locked
/tmp/arfs-r28-final/py312-locked/bin/python /tmp/arfs-r29/validate.py py312-newer
uv lock --check --python /tmp/arfs-r28-final/py312-locked/bin/python
uv export --locked --extra test --no-emit-project --output-file /tmp/arfs-r29/export-check.txt
uv tool run --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
uv tool run --from ruff==0.17.0 ruff check tests/test_plot_contracts.py
uv tool run --from ruff==0.17.0 ruff format --check tests/test_plot_contracts.py src/arfs/feature_selection/variable_importance.py
/tmp/arfs-r28-final/py312-locked/bin/python /tmp/arfs-r29/check_format.py
/tmp/arfs-r28-final/py312-locked/bin/python -m unittest discover -s .github/scripts -p 'test_*.py'
git diff --check
```

## Limits and resume

Hosted CI, full documentation/notebook execution, optional interactive/backends,
other platforms and large-feature performance were not checked. Existing
Matplotlib style-reset behavior remains; callers needing style isolation can use
Matplotlib's rc_context. Undefined singleton cells remain undefined and can emit
native annotation warnings. These rendering repairs do not establish statistical
selection calibration or package release readiness.

Stop after the local R29 phase and handoff. Next is **R30 alone**, with R04
complete: repair local RNG use, odd row counts and Titanic's alone flag.
R30, R32 and R33 remain pending. No push, PR, version bump or release belongs
to this request.
