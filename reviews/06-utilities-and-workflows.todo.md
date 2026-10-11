# 06 — Utilities, plots, datasets, and documentation workflows

**Status:** Reviewed; findings open.

**Outcome:** Completed on 2026-10-11 against
`03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`.
[Report](06-utilities-and-workflows.md) records 15 new findings, inherited
blockers, inspected plot outputs, local dataset checks, and bounded notebook
and documentation-build ledgers. Focused checks: frozen 9 passed/16 expected
failures; newer 7 passed/18 expected failures, with unmasked failure logs.
Sphinx was unavailable, so no completed documentation build is claimed.
No package fixes were implemented. Next: item 07. Evidence is in
[evidence/06-utilities-and-workflows/](evidence/06-utilities-and-workflows/).

**Purpose:** Review supporting functions and the user-facing examples that have
not yet received substantive checks, including whether plots and datasets tell
the truth and whether documented workflows run.

**Read first**

- [Queue and shared review procedure](README.md).
- Remaining utilities and dataset generators/loaders in [utils.py](../src/arfs/utils.py).
- Plotting functions in [association.py](../src/arfs/association.py),
  [benchmark.py](../src/arfs/benchmark.py), and selector modules under
  [feature_selection](../src/arfs/feature_selection/).
- [README](../README.md), [method overview](../docs/Methods%20overview.rst),
  [introduction](../docs/Introduction.rst), [API listings](../docs/arfs.rst),
  [selector API listings](../docs/arfs.feature_selection.rst), and
  [documentation configuration](../docs/conf.py).
- Use [unit 0's notebook inventory](evidence/00-baseline/inventory.json) to select
  representative notebook workflows without rereading every notebook initially.

**Review questions and checks**

1. Account for utility functions not substantively covered by earlier items:
   validation, dtype dispatch, category conversion, model detection, convenience
   estimators, and random generators. Trace actual callers. Check input mutation,
   invalid weights, global random-state effects, and stated synthetic signals.
2. Check local dataset schemas, feature/target alignment, descriptions, and
   loader options against installed artifacts. Cross-reference known packaging
   failures. Inspect network and serialized-resource paths and their error
   handling; use only trusted artifacts. Record any unexecuted remote downloads.
3. Render representative association/importance/data plots with a noninteractive
   backend and inspect them. Test signs, order, labels, selected/discarded
   features, small/empty results, one subplot, and figure ownership/cleanup.
   Check for misleading statistical labels, not just absence of exceptions.
4. Trace public README/API claims to implemented behavior and earlier evidence:
   categorical/weight support, scoring meaning, reproducibility, supported
   inputs, and the difference between ranking heuristics and calibrated tests.
   Identify undocumented limitations and stale examples without rewriting docs.
5. Execute bounded representative workflows from notebooks or equivalent exact
   extracted cells in a disposable directory. Prioritize basic selection,
   classification, regression, and preprocessing-to-model integration. Record
   cells run, data source, substitutions, failures, and skipped network/GPU
   requirements. Do not call a partial notebook run a full passing notebook.
6. Attempt the established documentation build if its environment is available;
   inspect local file references regardless. If dependency setup blocks a build,
   record the blocker and scope rather than altering project dependencies.

**Known findings to carry forward**

- [Unit 0](00-baseline.md): import/resource packaging defects and synthetic-data
  fixture failures. Do not rediscover them as new independent findings.
- [Units 1A](01a-statistical-foundations.md), [1B](01b-all-relevant.md),
  [1C](01c-weighted-glm-lasso.md), and [2](02-preprocessing.md): consult the
  relevant finding when an example fails or makes an unsupported claim.
- Items 03–05 own algorithmic reviews of their respective modules; this item
  checks remaining utilities, presentation, and integration.

**Deliverables and completion**

- Report: `reviews/06-utilities-and-workflows.md`.
- Evidence: `reviews/evidence/06-utilities-and-workflows/`, including inspected
  plot outputs and a compact workflow/build execution ledger.
- Account for remaining utility surfaces and key documented workflows. Explain
  unexecuted examples and inherited blockers. Follow the queue's completion/update
  procedure.
- Next queued item: [07 — final assessment](07-final-assessment.todo.md).

**Boundary:** No broad prose rewrite, generated documentation committed to the
site, dependency migration, or download of unnecessary large datasets. Review
artifacts stay under `reviews/`; temporary builds stay under `/tmp`.
