# ARFS review queue

This is the entry point when the user asks **“grab the next item to review”**.
Read this file, then the first **Pending** item's plan. Read older reports only
where that plan points to a relevant finding. Do not reconstruct the queue from
scratch or repeat completed reviews.

This queue was reconstructed on 2026-10-11 from the completed reports and the
remaining source modules. The original full plan was not saved. The order below
is the continuation plan; older reports saying that later scope is unknown are
historical handoffs.

**Next item: None — queued reviews complete.**

[Final assessment](07-final-assessment.md): review complete, release blockers
remain; keep one distribution. [Remediation](REMEDIATION.md) is a separate repair
backlog: [R01](R01-installed-package.md), [R02/R03](R02-R03-compatibility.md),
[R04](R04-runtime-baseline.md), [R05](R05-observation-boundaries.md),
[R06](R06-category-identity.md), [R07](R07-shadow-contracts.md),
[R08](R08-statistical-arithmetic.md), [R09](R09-mrmr-ranking.md),
[R10](R10-leshy-support.md), [R11](R11-permutation-importance.md),
[R12](R12-all-relevant-aggregation.md) and [R13](R13-glm-scoring.md) are complete.
**R14 alone is next**, with R13 complete. Follow the
[session batching and handoff guidance](REMEDIATION.md#session-batching-and-handoffs):
repair fitted GLM design/intercept and parameter state in R14.
Future "implement the next batch" requests follow the
[recorded groups](REMEDIATION.md#recorded-groups-for-next-batch-requests).
Later concurrent tracks require completed prerequisites and a
check for shared files; the default remains one phase per session.

| Order | Review item | Status | Read |
| --- | --- | --- | --- |
| 0 | Installation, packaging, and test baseline | Reviewed; findings open | [Report](00-baseline.md) |
| 1A | Statistical foundations and MRMR | Reviewed; findings open | [Report](01a-statistical-foundations.md) |
| 1B | All-relevant selectors | Reviewed; findings open | [Report](01b-all-relevant.md) |
| 1C | Weighted GLM/Lasso selection | Reviewed; findings open | [Report](01c-weighted-glm-lasso.md) |
| 2 | Preprocessing and transformers | Reviewed; findings open | [Report](02-preprocessing.md) |
| 3 | Selector contracts and remaining selectors | Reviewed; findings open | [Report](03-selector-contracts.md), [Plan](03-selector-contracts.todo.md) |
| 4 | GBM training, prediction, persistence, and benchmarks | Reviewed; findings open | [Report](04-training-and-benchmarks.md), [Plan](04-training-and-benchmarks.todo.md) |
| 5 | Sampling, distances, and parallel execution | Reviewed; findings open | [Report](05-sampling-and-parallel.md), [Plan](05-sampling-and-parallel.todo.md) |
| 6 | Utilities, plots, datasets, and documentation workflows | Reviewed; findings open | [Report](06-utilities-and-workflows.md), [Plan](06-utilities-and-workflows.todo.md) |
| 7 | Compatibility, package boundaries, and final assessment | Reviewed; findings open | [Report](07-final-assessment.md), [Plan](07-final-assessment.todo.md), [Remediation](REMEDIATION.md) |

“Reviewed” means the investigation and report are complete. It does **not** mean
the findings have been fixed. At queue creation, no review finding has been
remediated. Existing reports are the authoritative finding lists; remediation
is a separate task and requires an implementation request.

**How to execute one item**

1. Read the first pending plan and applicable project instructions. Check Git
   status and the current commit. The existing reports reviewed
   `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`; if the source has moved, record the
   new target and recheck only the assumptions needed for this item.
2. Review exactly that item's scope. Trace callers and shared helpers before
   reporting a defect. Reuse existing evidence for known findings; distinguish
   inherited blockers from newly demonstrated failures.
3. Add the specified report and focused evidence. Preserve source code, existing
   tests, dependencies, lockfile, CI, previous reports, and unrelated work.
   Review authorization permits review artifacts, not repairs.
4. Use concrete counterexamples and independent references where possible.
   Record versions, module paths/hashes, commands, results, and limitations.
   Strict expected failures may document known defects; also show the unmasked
   failures. A passing smoke test is not evidence of statistical validity.
5. Finish the report with severity, source locations, reproduction, the smallest
   suggested remedy, and remaining uncertainty. Record skipped or blocked checks
   explicitly; do not claim they passed.
6. Update the item's **Status** and **Outcome** in its `.todo.md` file, this
   table, and the **Next item** line. Keep the plan file as a handoff record and
   add a link to its completed report. If the investigation is incomplete, leave
   it Pending and record the exact resume point. Stop after this one item.

**Reusable environment**

The earlier reviews used these disposable installed-wheel environments:

- Frozen dependencies: `/tmp/arfs-review-03f67d0/locked-env/bin/python`.
- Newer dependency snapshot: `/tmp/arfs-review-03f67d0/wheel-env/bin/python`.

Verify that they still exist and that the installed modules match the current
review target. They use Python 3.12.7 and retain `setuptools==80.9.0` solely as
unit 0's import workaround. They are not proof of clean-install success, and the
“newer” snapshot is not a fresh resolution. Do not install into the project's
existing `.venv` during a review.

Environment-recording examples and pins are in
[the preprocessing evidence](evidence/02-preprocessing/environment.py),
[frozen versions](evidence/02-preprocessing/frozen-environment.json), and
[newer versions](evidence/02-preprocessing/current-environment.json). Adapt the
module hash list to the item being reviewed. If the disposable environments are
gone, use [unit 0's reproduction instructions](00-baseline.md) to recreate the
audited target in new disposable directories; do not substitute the published
PyPI package silently.

Run installed-wheel checks outside the checkout with a temporary working
directory. Existing evidence uses these environment settings:

```sh
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
```

Set `MPLCONFIGDIR` to a writable temporary directory. Review checks live in the
item's evidence folder, separate from the established test suite. A typical run
is:

```sh
"$review_python" -m pytest "$evidence/checks.py" \
  -q -rx --tb=short -c /dev/null --rootdir="$evidence" -p no:cacheprovider
```

Here `review_python` is the verified interpreter and `evidence` is the absolute
path to this item's evidence directory. The item plans name their output paths;
these reports and evidence directories are created when the review is executed.

**When the overall review is complete**

All queued items must be reviewed, or explicitly removed from scope by the user,
and unit 7 must account for unresolved checks and findings. Completing the review
does not establish release readiness. Keep one distribution as the working
assumption until the final assessment supplies evidence for a different choice.
