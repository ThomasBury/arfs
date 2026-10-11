# ARFS remediation handoff

**Status: R01 complete; R02–R33 pending. Next implementation phase: R02.**
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

The recommended start is R01 alone, optionally R02/R03 together after R01, then
R04 alone. All remaining phases default to separate sessions.

| Phases | Session recommendation | Reason |
| --- | --- | --- |
| R01 | Alone | Establish a usable installed package before other repairs |
| R02 + R03 | Optional two-phase batch, sequentially, after R01 | Both repair dependency compatibility; inspect shared files and callers before combining |
| R04 | Alone | Establish the test baseline, dependency lock, and runtime CI |
| R05 | Alone | Shared observation and weight validation affects many callers |
| R06 | Alone | Shared categorical identity and dtype rules affect several algorithms |
| R08 and R09 | Separate sessions | Statistical arithmetic and ranking decisions need independent evidence |

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

## Ordered phases

R01 is **Complete**; R02–R33 are **Pending**. The
[R01 handoff](R01-installed-package.md) records changes, decisions, exact checks,
environment identities, inherited failures, and the resume point. Changes are
committed as `6b675fd` on `fix/r01-installed-package` at checkpoint base `0270aca`;
no next phase started. The user subsequently authorized PR delivery.
An ID such as `1A F3` means finding 3 in the linked original report. The complete
ID-to-phase mapping follows the table.

| Phase | Smallest complete outcome | Depends on | Required validation |
| --- | --- | --- | --- |
| **R01 — Usable installed package (Complete)** | Replace `pkg_resources` resource access with the standard library, include the housing description in distributions, bind the existing missing export. Decide whether the unconditional Styler return remains supported; if so, declare Jinja2. Sources: [00 F1/F3/F4](00-baseline.md), [03 F14](03-selector-contracts.md) | None | Build sdist then wheel from clean source; clean install with no historical setuptools; import all 16 modules, bind every `__all__` name, load bundled Boston/cancer/housing outside checkout; summary reaches its styling path. Verify resource lifetime when an API needs a real path |
| **R02 — Owned pandas column writes** | Replace read-only shadow-array mutation and dtype-incompatible midpoint assignment, preserving values, dtypes, index and caller input. [00 F2](00-baseline.md), [02 F3](02-preprocessing.md) | R01 | Frozen with Copy-on-Write both off/on, newer pandas; numeric/category shadow values and real/shadow shape; interval helper/estimator and tree-to-midpoint pipeline |
| **R03 — Supported dependency APIs** | Use the current sklearn keyword, Styler map, boxplot keyword, documented SciPy distance namespace, and scalar NumPy conditions. Avoid wrappers because supported minima already have the replacements. [1C F1](01c-weighted-glm-lasso.md), [03 F10](03-selector-contracts.md), [04 F13](04-training-and-benchmarks.md), [05 C2](05-sampling-and-parallel.md), [06 F11](06-utilities-and-workflows.md) | R01 | Small GLM entry fit; real summary HTML; real permutation plot; association clustering; scalar constant/nonconstant ANOVA in both snapshots. Numerical defects left to R08/R13 must not be mistaken for compatibility success |
| **R04 — Reproducible baseline and runtime gate** | Correct the constant-test threshold/comparison and missing synthetic `nice_guys` column; add checks that cannot pass by selecting everything. Choose supported Python/dependency cells, regenerate/review the matching lock, and add installed-wheel import/resource/export/tests to CI. [00 F5–F8](00-baseline.md) | R01–R03 | Existing suite passes in chosen baseline cells; fixture asserts its intended signal/noise/category schema. A deliberately select-all implementation fails a meaningful selection test. Lock check succeeds; CI runs from installed artifacts. Begin with Linux/Python 3.12 frozen/newer and add an actual oldest-supported interpreter check before claiming the full range |
| **R05 — Valid observations at boundaries** | Validate one-dimensional finite targets and aligned, finite nonnegative weights with positive retained/split mass. Do not invent missing responses. Apply empty-sample and zero-mass-group handling consistently through statistics and training; remove the overwritten duplicate converter. [1A F8–F10](01a-statistical-foundations.md), [1B F11](01b-all-relevant.md), [04 F6/F7](04-training-and-benchmarks.md) | R01 | Lists/arrays/Series, wrong lengths/dimensions, negative/nonfinite/all-zero weights, no complete rows, zero-weight added category, and custom `handle_na=None`; unchanged data and aligned rows. This does not choose frequency versus relative weighting |
| **R06 — Category identity and dtype agreement** | Reuse one pandas-aware dtype rule and encode distinct original values without string collisions. Cover boolean, object/category, explicit strings and inferred strings. [1A F13](01a-statistical-foundations.md), [03 F4](03-selector-contracts.md), [05 C1](05-sampling-and-parallel.md), [06 F1](06-utilities-and-workflows.md) | R01/R05 | Distinguish `1` from `'1'`, missing values from literal `'nan'`/`'None'`; series/matrix and worker agreement; nullable-string cardinality; all shared encoder callers |
| **R07 — Faithful shadow controls** | Encode owned selector frames, carry categorical identity through reductions, give shadows matching model roles, and generate collision-free labels. [1B F5/F6/F10](01b-all-relevant.md) | R02/R06 | No mutation after real BoostAGroota/GrootCV fits; original/shadow categorical positions in successive rounds and folds; numeric-valued categories in Leshy; input names already matching generated labels |
| **R08 — Weighted association/relevance arithmetic** | Choose weight meaning and degrees of freedom; implement the documented linear statistic, constant handling, output shape, stable centered sums, scale-cancelling Pearson arithmetic and collision-free response augmentation. [1A F3–F7/F11/F12](01a-statistical-foundations.md) | R05/R06; R03 for newer NumPy | Independent Pearson/F/ANOVA references, retained-row degrees of freedom, integer replication or relative-scale invariance as appropriate, normalized weights, constant/zero-mass categories, large translations, predictor named `target`, one/two-worker equivalence |
| **R09 — Coherent MRMR ranking** | Remove the two invalid score normalizations; choose a nonnegative common relevance scale and use absolute unnormalized redundancy. Define empty/equal/perfect candidate behavior. [1A F1/F2](01a-statistical-foundations.md) | R08 | Mixed singleton predictors remain eligible; no redundant negative-copy preference; hand ranking with fixed relevance; empty/constant/perfect scores; strong-signal/null and mixed-type screens. Do not claim calibrated mixed-type significance |
| **R10 — Leshy support/history consistency** | Let support computation own the mask and use all actual history rows. [1B F1/F2](01b-all-relevant.md) | R07 | Confirmed-only, tentative-only, mixed and empty selections with `keep_weak` on/off; mask/count/transform agreement; two-iteration median and rejected ranking |
| **R11 — Weighted signed permutation importance** | Forward validation weights at the shared importance call and preserve signed ordering when BoostAGroota normalizes. Explicitly decide the zero-total case. [1B F3/F4](01b-all-relevant.md) | R05/R07 | Real weighted sklearn reference; negative-total and zero-total fixed vectors; both Leshy and BoostAGroota callers; no promotion of negative scores as positive evidence |
| **R12 — Explicit all-relevant aggregation** | Resolve GrootCV objective/output-axis handling, then choose the intended fold/shadow statistic for GrootCV and BoostAGroota and align prose. [1B F7–F9](01b-all-relevant.md) | R07 | Binary-to-multiclass and objective-alias real fits; exact hand examples with fold columns only; documented cutoff axes/order. Calibration remains a separate claim |
| **R13 — Correct GLM scoring and independent CV** | Compute response-scale deviance once, expose a higher-is-better sklearn score while minimizing BIC correctly, pass stratification targets, slice documented arrays, clone each fold. [1C F2–F4/F9/F12](01c-weighted-glm-lasso.md) | R03/R05 | Independent Poisson/binomial/Gamma losses, real external GridSearchCV, binary fold splits, array/frame parity, stateful-estimator isolation; rerun the Poisson signal counterexample |
| **R14 — Stable GLM design and parameters** | Use one fitted intercept/input-column contract across fit/predict/transform; derive the family/link at fit from current parameters; return self. [1C F5/F7/F8/F10](01c-weighted-glm-lasso.md) | R13 | Single-row prediction, existing constants, `fit_intercept=False`, reorder/missing/extra/duplicate names, arrays and support metadata, `set_params` vs fresh construction, chained fit/predict and summary integration |
| **R15 — Honest likelihood and refit contract** | Decide zero-weight likelihood handling, reject all-invalid candidate criteria, describe regularized selection followed by unpenalized refit, remove the near-ridge promise. [1C F6/F11](01c-weighted-glm-lasso.md) | R13/R14 | Zero-weight equivalence to omission or clear rejection; all-nonfinite search fails; independent weighted Gaussian/Poisson BIC comparisons; documented coefficient/refit behavior and post-selection inference limit |
| **R16 — Threshold selector identity/state** | Reject duplicate names or adopt consistent positions, align labelled callbacks, validate parameters at fit, handle zero/one-feature collinearity, stabilize ties and fitted metadata. Resolve equality wording explicitly. [03 F5–F9/F12](03-selector-contracts.md) | R06/R08 | Masks/names/output agree, reordered callback labels, invalid `set_params`, empty pipeline stages, multiple hash seeds, correct feature counts and NotFittedError. Preserve legitimate named projection if retained as the API |
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
not await GLM repairs. The default next phase is now R02. Do not parallelize
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
