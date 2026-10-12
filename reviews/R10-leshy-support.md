# R10 — Leshy support and history consistency

**R10 complete on 2026-10-12. Next phase in this batch: R11.**
Based on `170b3e9` on `main`, with pre-existing uncommitted R04–R09 work
preserved. The separate R10 local commit includes only this repair, its new
maintained tests and phase evidence. Earlier work stays uncommitted.

## Changes and decisions

Support calculation alone owns the selected mask. Relative ranking no longer
promotes tentative features when `keep_weak=False`. The fitted `n_features_`
counts the actual mask, including retained tentative features with `keep_weak=True`.
Confirmed features retain rank 1, tentative features rank 2 regardless of inclusion.
Both the tentative median comparison and rejected-feature ranking use every
actual iteration; the history begins empty and has no placeholder zero row.
Tentative median equality with the shadow median remains excluded.

Caller tracing covered fit/select_features, iteration history construction,
confirmed/tentative extraction, support/count/names, absolute and relative
ranking, get_support/transform, plotting consumers and existing selector tests.
Only the Leshy support/history implementation and its relevant API docstrings
change. R29 still owns plotting membership; R12 owns aggregation/calibration.
No dependency, export, option or compatibility layer was added.

## Validation

Evidence is in [r10-leshy-support](evidence/r10-leshy-support/).
Both historical F1/F2 checks fail unmasked on the installed R09 wheel (2 failed).
The 17 new maintained cases fail 11 times on that wheel, with 6 passing controls.
The source repair passes all 17 without expected-failure markers. Cases cover
confirmed-only, tentative-only, mixed, empty and no-rejected selections, both
keep_weak choices, mask/count/names/transform agreement, actual two-row history,
one-row and median-equality boundaries, and rejected ranks with/without tentative
features. Model outputs are fixed where tests isolate bookkeeping and arithmetic.
The existing real model suite is included in each installed-wheel run.

Fresh R10 wheel full maintained suites: **py39-locked 480 passed;
py312-locked 480 passed; py312-newer 464 passed**. Historical F1/F2 plus FDR,
binomial and constant-input controls pass unmasked in both 3.12 cells
(**6 passed** each). Python/dependency snapshots match R09: Python 3.9.25 and
3.12.7, locked pandas 2.2.3 / NumPy 2.0.2 / sklearn 1.6.1; newer pandas 3.0.6 /
NumPy 2.5.4 / sklearn 1.9.1. See per-cell environment JSON for complete versions.
All checks run outside the checkout, using isolated Python and a newly installed
wheel without setuptools. Installed module hashes match the checkout; all 19
source/sdist/wheel payloads agree. Twelve unrelated tracked diff sections are
unchanged. Historical scripts and logs remain intact.

`uv lock --check`, Ruff 0.17.0 baseline gate (zero new diagnostics), gate unit
tests (2 passed), new-test lint/format and `git diff --check` pass. Exact commands
are in [commands.txt](evidence/r10-leshy-support/commands.txt) and per-cell command
JSON. Existing model/tag/precision warnings remain visible.

Hosted CI, complete docs/notebooks, optional backends, performance, visual plot
review and statistical calibration were not checked. This is a green repair
phase, not release evidence. R11 follows sequentially; R12–R33 remain pending.
