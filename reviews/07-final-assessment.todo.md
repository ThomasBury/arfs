# 07 — Compatibility, package boundaries, and final assessment

**Status:** Reviewed; findings open.

**Outcome:** Completed on 2026-10-11 against
`03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`.
[Report](07-final-assessment.md) and [remediation handoff](REMEDIATION.md) account
for every earlier finding, all 16 source modules, all 17 notebook entry points,
and unresolved support/statistical/performance checks. Both saved wheels match
all source files; Python 3.9–3.12 runtime requirements resolve, but runtime
evidence remains Linux/Python 3.12.7. New focused checks give 4 passes/1 expected
failure in each snapshot, with unmasked logs for the time-series plotting
counterexample. The hosted-docs wheel-only resolution is blocked by fasttreeshap;
no source-build or completed Sphinx-build success is claimed.

The queued review is complete, release blockers remain, and the recommendation
is to keep one distribution. No package fixes were implemented. Review evidence
is in [evidence/07-final-assessment/](evidence/07-final-assessment/).
There is no next review item; remediation requires a separate implementation
request, starting with R01 in the handoff.

**Purpose:** Close the review with an evidence-based account of compatibility,
remaining coverage, release blockers, maintainability, and whether package
boundaries should change. Review completion and release readiness are separate
conclusions.

**Read first**

- [Queue](README.md), then completed reports 0–6 for synthesis. This is the one
  item that deliberately reads across the reports instead of repeating their
  investigations.
- [Project metadata](../pyproject.toml), [lockfile](../uv.lock),
  [CI workflow](../.github/workflows/ruff.yml),
  [documentation requirements](../docs/requirements.txt), and
  [Read the Docs configuration](../.readthedocs.yml).
- [Unit 0's module inventory](evidence/00-baseline/inventory.json),
  [distribution evidence](evidence/00-baseline/distributions.json), and
  [environment results](evidence/00-baseline/environments.json).

**Review questions and checks**

1. Build a coverage ledger mapping every source module/public workflow to a
   completed review, concrete evidence, or an explicit remaining gap. Inspect
   any genuinely unowned surface before claiming the review is complete. Do not
   use line coverage as a substitute for semantic checks.
2. Assess the declared Python/dependency range against available evidence.
   Identify impossible resolutions, unsupported combinations, undeclared imports,
   optional-backend assumptions, and public API drift. Use a small justified
   compatibility matrix and record exactly which cells were executed, blocked,
   or untested; do not claim support for the entire range from one interpreter.
3. Reuse unit 0's clean-install, source/wheel, export, and resource results when
   still applicable. Rerun affected checks only if the source or environment
   changed or a remaining question requires them. Reconcile the lockfile,
   version metadata, CI gates, test assertions, and actual release requirements.
4. Assess maintainability from real import/call dependencies, duplicated or
   unused paths, measured resource/import costs where relevant, and Git co-change
   history where informative. Recommend deletion/reuse before abstractions or
   new dependencies. Keep evidence separate from preferences.
5. Evaluate the one-distribution working assumption. Identify users, dependency
   costs, cross-module coupling, migration costs, and a concrete benefit before
   recommending any split. “Many modules” or file length alone is insufficient.
   If evidence does not justify a split, explicitly recommend keeping one package.
6. Consolidate findings by shared root cause and priority, linking back to the
   original evidence. Produce an ordered remediation backlog with atomic,
   independently verifiable phases and meaningful regression checks. Preserve
   unresolved statistical/API decisions instead of choosing them silently.

**Deliverables and completion**

- Report: `reviews/07-final-assessment.md`.
- Evidence, if new checks are needed: `reviews/evidence/07-final-assessment/`.
- Remediation handoff: `reviews/REMEDIATION.md`, with links to original findings,
  dependencies between repairs, and validation required for each phase. Do not
  duplicate every original finding's prose.
- State explicitly whether **the review is complete**, whether **release
  blockers remain**, and which support/performance/statistical questions remain
  unresolved. Do not mark this item Reviewed if earlier items are still pending
  or material review gaps remain unaccounted for.
- Update this plan and the queue. Set the queue's next-review line to “None —
  queued reviews complete” only when the completion criteria are met. Stop;
  remediation needs a separate implementation request.

**Boundary:** This item proposes repairs and package decisions. It does not
change source, split distributions, regenerate the lockfile, redesign CI, commit,
push, or publish a release.
