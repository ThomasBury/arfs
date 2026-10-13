# ARFS remediation handoff

**Status: R01–R16 complete; R17–R33 pending. Next implementation group: R17 alone.**
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

R01–R16 are complete. The explicitly requested R02/R03 and R10/R11 batches
were implemented and validated sequentially; R04–R09 were completed alone.
Run **R17 alone** next, with R03/R05/R06 complete. Later phases use the recorded grouping
rules below; phases outside the three conditional pairs remain separate.

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
| R17 | Run alone next | R03/R05/R06 complete; repair cumulative importance selection and its training/output contracts |

### Recorded groups for next-batch requests

When the user asks to "implement the next batch of RXX", read the current phase
statuses in this file and use these groups. **The next batch is R17 alone.**
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
preserved. R17's R03/R05/R06 prerequisites are complete.
A request for the next batch authorizes that one group, its affected callers,
tests, local phase commits and handoff update; stop after
it. It does not authorize the following group or Git publication/release unless
the user also requests those actions.

| Later two-phase group | Prerequisites | Why it fits one session | Decision to settle before the affected phase |
| --- | --- | --- | --- |
| R10 → R11 (complete) | R05/R07 complete | Leshy support/history and weighted permutation importance share selector code | Settled: absolute-total normalization, all-zero stays zero, only positive mean permutation scores eligible |
| **R18 → R31** | R06 complete | Encoder schema repairs and related legacy helpers share dtype/schema checks | Retain or remove the public legacy helpers |
| **R25 → R26** | R05 and R24 complete | Sampling output repairs and bounded isolation retries mostly share one module | Define weighting, tiny-input, and exhaustion behavior |

R18 → R31 is the earliest eligible two-phase group after R06 and may be selected
explicitly at that point. R10 → R11 has already been completed.
R25 → R26 is larger and remains conditional on its scope and policy decisions
being settled. These are sequential groups, not parallel agent assignments.

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
callers before edits; these candidate pairs have not had a complete caller/scope
audit. Complete, validate, and make a separate local commit for the first phase
before starting the second. Settle and record required API/statistical choices
in the relevant docs and tests; do not invent them to make a check pass. If a
choice remains unresolved, a failure cannot be explained, or the scope widens,
stop at the phase boundary and record exactly what remains.

At the end, update the phase statuses and name the exact next group, its
prerequisites, validation commands/results, and unresolved decisions. Preserve
earlier evidence and do not carry implementation into another group in the same
session. A completed group is not a release: versioning/publication still waits
for the remaining release blockers and R33 evidence.

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

## Ordered phases

R01–R16 are **Complete**; R17–R33 are **Pending**. The
[R01 handoff](R01-installed-package.md) records changes, decisions, exact checks,
environment identities, inherited failures, and the resume point. Changes are
committed as `6b675fd` on `fix/r01-installed-package` at checkpoint base `0270aca`;
at that handoff no next phase had started. The user subsequently authorized PR delivery.
The [R02/R03 handoff](R02-R03-compatibility.md) records the completed compatibility
batch, based on `a4b7ff6` on `fix/r02-r03-dependency-compatibility`. Its source
repairs are committed as `36f1ee6` (R02) and `65f894a` (R03). A version bump and
GitHub release are deferred until the remaining release blockers satisfy R33.
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
based on `985b0a3`. R17 alone is next, with R03/R05/R06 complete.
Hosted CI for R13–R16 and remaining release evidence
are still pending. An ID such
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
| **R17 — Truthful cumulative importance selection** | Keep the crossing feature, reject nonfinite contributions, define zero-total selection, handle array multiclass targets, and raise the optional-backend error. Use one fit unless actual reproducible resampling is required. [03 F1–F3/F11/F13](03-selector-contracts.md) | R03/R05/R06 | Hand `[.6,.3,.1]` thresholds, constant-response real fit, threshold 1, invalid contributions, three-class array/Series parity, iteration behavior and missing fastshap error before wasted training |
| **R18 — Ordinal encoder full schema** | Keep categorical encoder validation separate from full passthrough metadata, copy inverse input, honor changed output dtype. [02 F8/F13/F14](02-preprocessing.md) | R06 | Mixed categorical/numeric `get_feature_names_out` and pandas `set_output`; unknown/missing values, input preservation, inverse, clone/refit and `set_params(output_dtype=...)` |
| **R19 — Actual fitted tree partitions** | Derive bins from real thresholds/leaf membership, avoid temporary-column collisions, derive copied one-tree parameters at fit, apply encoding in raw mode, define missing/all-missing behavior. [02 F1/F2/F4/F5/F9](02-preprocessing.md) | R05/R06/R18 | Equal-valued leaves, binary probabilities vs leaf IDs, values between observed extrema and fitted splits, `x`/`x_g`, category named `missing_added`, all-missing column, parameter aliases and native weighted one-tree reference |
| **R20 — Reusable transformer inputs and Patsy rows** | Separate constructor options from fitted state; normalize documented fit/transform inputs identically; honor Patsy `NA_action` and intercept choice; treat scalar midpoint name as one name. [02 F6/F7/F10–F12](02-preprocessing.md) | R02/R18/R19 | Refit different schemas, ndarray fit_transform, named dict formula, midpoint second fit/scalar name, Patsy raise/drop branches and row alignment. Document that a transformer cannot also drop y/weight rows |
| **R21 — GBM prediction and split contracts** | Reject or explicitly prioritize incompatible group/stratification options; validate prediction schemas and objective aliases; state real categorical support/disable behavior; define multiclass offset input; reset fitted state. [04 F1–F5/F8/F9](04-training-and-benchmarks.md) | R05/R06 | Disjoint groups, reorder/replacement columns, label/probability shapes across aliases, explicit category disable, object/string/time error or fitted mapping, offset shapes, refits clearing flags/validation data/curves |
| **R22 — Persist the wrapper contract** | Save/load required wrapper metadata with the Booster and reset state coherently. Define legacy Booster-only file handling without guessing an unknown offset contract. [04 F10](04-training-and-benchmarks.md) | R21 | Fresh-process ordinary/category/offset round trips, external baseline required for new rows, loading into previously fitted wrapper, invalid/missing artifact errors; trusted-input boundary stays explicit. Exercise safe write failure behavior before claiming durable persistence |
| **R23 — Comparisons that run what they label** | Fit encoders on training rows only; settle documented array/name support; reject incompatible GrootCV comparisons; construct fresh selector/model combinations. [04 F11/F12/F14/F15](04-training-and-benchmarks.md) | R07/R10–R12/R18/R21 | Held-out-only category excluded from vocabulary; integer display names, array policy, real fitted estimator/method tracing, seed/backend state reset, weighted scoring retained |
| **R24 — Valid Gower distances** | Use stable direct numeric ranges, distinguish cross/self matrices, handle pairwise missingness explicitly, normalize documented input/masks, validate feature weights. [05 F1–F5](05-sampling-and-parallel.md) | R06 | Hand scalar reference, translation/scale invariance, equal-sized different frames, missing/constant features, no-comparable-pair policy, arrays/Series/extension dtypes/sparse errors, valid feature-weight rescaling |
| **R25 — Sampling returns the intended data** | Separate row weights from distance feature weights; choose weighted representative semantics or reject them; impute by column, avoid `label` collision, raise unknown methods and return original isolation-selected rows. [05 F6–F8/F10](05-sampling-and-parallel.md) | R05/R24 | N≠P and N=P weights, per-column means, original column/index/dtype schema, an existing `label`, selected categorical values, clear unsupported-method/weight errors |
| **R26 — Bounded isolation attempts** | Advance retry state, bound size progression and define exhaustion/tiny-input behavior. Correct KS-p-value interpretation in API/example prose. [05 F9 and statistical limits](05-sampling-and-parallel.md) | R25 | Deterministic forced rejection reaches bounded termination; draw sizes follow the documented rule; 0/1/10-row inputs handled explicitly. A terminating sampler still has no representativeness guarantee |
| **R27 — Query-aware nearest neighbors** | Use the actual one-row query, exclude its key, deduplicate eligible keys once, validate n/query and define insufficient-candidate/nonfinite behavior. [05 F11/F12](05-sampling-and-parallel.md) | R24 | Query changes result; duplicate IDs; enough/too few unique candidates; n=0/negative/too large; optional-query policy; only a query-by-candidate distance vector allocated |
| **R28 — Parallel labels and bounded useful work** | Chunk original label objects without NumPy coercion; settle result order/empty schema and worker-count semantics, avoiding empty tasks. [05 F13 and resource boundaries](05-sampling-and-parallel.md) | R06/R08 | Integer/string and tuple/MultiIndex labels with one/two workers; propagated callback error; no-pair and empty work; supported negative counts; worker cap observed without a new execution framework |
| **R29 — Plots match the data and selection** | Keep feature/subplot coordinates separate, color actual Leshy membership, include terminal importance thresholds, respect plot limits and linear feature positions, bypass singleton clustering. [06 F2–F7](06-utilities-and-workflows.md) | R03/R10/R17 | Render and inspect string/integer/single-panel target plots; masks disagreeing with importance order; threshold 1/one/no feature, plot_n limit, first cumulative point, input/figure ownership |
| **R30 — Honest local dataset fixtures** | Use local RNG consistently, generate exactly the requested odd/even row count, correct Titanic's alone flag. [06 F8/F9/F15](06-utilities-and-workflows.md) | R04 | Caller RNG stream unchanged; reproducible features/targets/weights; strong/noise construction checks; odd size; network-free 0/1/2-relative fixture. Do not introduce a new data service |
| **R31 — Retain or remove legacy helpers deliberately** | Decide the public status of `cat_var` and `validate_pandas_input`; if retained, preserve missingness/dtype and accept documented arrays. Remove truly obsolete duplicates/local-unused helpers only with an explicit API decision. [02 F15](02-preprocessing.md), [06 F10](06-utilities-and-workflows.md) | R06/R18 | Direct public helper tests and exports/docs updated; missing ≠ category; newer strings; arrays; active callers rechecked. No compatibility shim by default |
| **R32 — Reproducible documentation workflows** | Repair classification/regression setup, sampling overwrite, time-series reduced-feature calls, README filename and local Sphinx source. Reconcile docs-extra/RTD/fastshap policy; qualify method/weight/backend claims after preceding choices. [05 F14](05-sampling-and-parallel.md), [06 F12–F14](06-utilities-and-workflows.md), [07 F1](07-final-assessment.md) | Relevant R01–R31 fixes for each example | Actual Sphinx build and local links; bounded clean-kernel basic/classification/regression/preprocessing examples; exact reduced-feature model input; size limit maintained. Record data substitutions, excluded optional cells, and remaining notebooks separately |
| **R33 — Release evidence and remaining claims** | Run the repaired package gates and a justified support matrix. Reassess selection claims with meaningful held-out baselines and document remaining optional/performance limits; narrow unsupported claims instead of promising unbounded validation. | All release-blocking phases above | Fresh source/wheel/lock metadata agree; no workaround; established suite and promoted regressions pass; repeated-seed signal/null, weak/correlated/mixed/weighted cases appropriate to claims; fold-local preprocessing/selection; clean-process artifacts; docs build. Optional backends/platforms require their own tests before claiming support |

R21's group/schema safeguards can be advanced immediately after R05/R06 if
training-wrapper users are the priority; its dependencies do not require waiting
for the intervening algorithm phases. Similarly independent R24–R28 work need
not await GLM repairs. The default next group is now R17 alone. Do not parallelize
edits to shared dtype/weight helpers without agreeing their contract first.

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
| Supported Python/dependency range | R04/R33; four successful resolutions are not runtime certification. Do not copy the Python-3.12-only newer pins onto earlier interpreters |
| Frequency counts vs relative/exposure weights | R05/R08/R15/R25; validation can be shared, statistical meaning cannot be silently normalized away. Gower feature weights are a different dimension |
| Mixed relevance scale and significance | R08/R09; Theil-U-to-F arithmetic is not evidence of an F null distribution. Keep a ranking heuristic labelled as such unless justified |
| Shadow threshold rule and error control | R10–R12/R33; decide axes/mean/median/max explicitly. Correct binomial arithmetic or signal recovery is not whole-procedure calibration |
| Threshold equality/empty output | R16/R17/R24; keep undefined/no-positive-mass distinct from zero effect. Document any intentional change in selected sets |
| GLM intercept/likelihood/refit | R14/R15; fit-time design must govern inference. Weight-valid prediction loss does not automatically justify every likelihood/BIC or post-selection p-value |
| Weighted global explanations | R17; ordinary mean absolute contributions currently average validation rows equally. Choose the population being summarized before changing that rule |
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
- Complete notebook execution and a real docs build remain outstanding even
  after selected cells pass. Optional benchmarks are not required dependencies
  for ordinary examples without an explicit reason.

No implementation, split, dependency migration, commit, push, or release is
authorized merely by this handoff. Findings assigned to pending phases remain open;
the historical review reports retain their original observations.
