This review covers item **05: sampling, distances, and parallel execution** at
commit `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, on 2026-10-11.
It adds review evidence only. Library code, existing tests, dependencies, the
lockfile, CI, and earlier reports are unchanged.

**Result:** 14 findings in the scoped sampling and execution workflow, plus two
newly demonstrated compatibility blockers in its association/relevance callers.
Gower distances can be negative or assign zero distance to distinct observations;
nearest-neighbor lookup can ignore its query; isolation sampling has no effective
retry limit. Changing worker count can break valid column labels. These are open
findings, not repaired behavior.

**Evidence and environment**

The review reused the two disposable installed-wheel environments named in the
[queue](README.md). Both use Python 3.12.7 and the historical
`setuptools==80.9.0` import workaround from [unit 0](00-baseline.md). No package
was installed. The newer snapshot is an existing resolution, not a fresh one.

| Dependency | Frozen | Saved newer snapshot |
| --- | --- | --- |
| NumPy | 2.0.2 | 2.5.4 |
| pandas | 2.2.3 | 3.0.6 |
| SciPy | 1.13.1 | 1.18.1 |
| scikit-learn | 1.6.1 | 1.9.1 |
| joblib | 1.4.2 | 1.6.0 |

[Frozen identity and hashes](evidence/05-sampling-and-parallel/frozen-environment.json)
and [newer identity and hashes](evidence/05-sampling-and-parallel/current-environment.json)
verify that installed `sampling.py`, `parallel.py`, `association.py`, and
`utils.py` match the checkout byte for byte and were imported outside it.
The files also record the notebook hash, complete dependency versions, CPU
counts, and thread limits. Both CPU-count APIs reported four CPUs on this host.
Exact pins are saved beside those JSON files as `.requirements.txt` files.

The [79 checks](evidence/05-sampling-and-parallel/checks.py) produce:

| Environment | Strict diagnostic run | Unmasked run (`--runxfail`) |
| --- | --- | --- |
| Frozen | 43 passed, 36 expected failures | 43 passed, 36 failed |
| Newer | 36 passed, 43 expected failures | 36 passed, 43 failed |

The extra seven expected failures are one Gower inferred-string case (F4),
two association string-dispatch cases (C1), and four ANOVA relevance cases (C2).
Strict markers identify defects; they do not establish a green package gate.
See the [frozen strict log](evidence/05-sampling-and-parallel/frozen-checks.txt),
[frozen failures](evidence/05-sampling-and-parallel/frozen-failures.txt),
[newer strict log](evidence/05-sampling-and-parallel/current-checks.txt), and
[newer failures](evidence/05-sampling-and-parallel/current-failures.txt).

An independent scalar Gower reference uses absolute numeric differences divided
by their range, categorical mismatch indicators, and feature weights; missing
features are omitted for each pair. This follows the
[cluster package's documented Gower calculation](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/daisy.html).
The review explicitly assigns zero contribution to a constant numeric column.
For cross-frame checks it uses pooled ranges, matching ARFS's actual behavior,
so the reference-range documentation disagreement cannot cause F2's failure.
The hand-calculated three-row weighted matrix is independently asserted too.

**Actual sampling contract**

- `sample(..., method="gower", n=k)` clusters all supplied rows, then returns
  unweighted numeric means and categorical modes for each cluster. The rows are
  synthesized representatives, indexed by cluster names. They are not sampled
  original observations. Tied modes use the first encountered value.
- `sample(..., method="isoforest")` fits one isolation forest to encoded data,
  repeatedly draws row positions uniformly without replacement, and accepts a
  draw when its anomaly-score KS **p-value** exceeds 0.95. It does not explicitly
  filter outlier labels. `n` has no effect in this branch, consistent with the
  parameter documentation limiting it to Gower clustering.
- Neither method returns row weights or cluster masses. No conservation rule is
  promised. Treating each cluster mean as an equally weighted observation will
  generally change the empirical distribution. F6 concerns the provided
  `sample_weight` argument being misused or ignored, not an invented promise to
  return conserved mass.
- Tested calls preserve the caller's frame, including duplicate/non-default
  indexes. Isolation selection uses positional indexing correctly, although F10
  changes the values it returns. Gower intentionally replaces the original index.

**Findings**

P1 denotes wrong results, lost information, or failure in an ordinary supported
workflow. P2 denotes a narrower input, validation, or resource boundary.
Confidence is high for the demonstrated cases. Suggested remedies describe
future work only. Finding IDs match the strict expected-failure reasons, except
F14, which has a source assertion and bounded allocation evidence.

1. **F1 — Numeric Gower distances depend on the origin of measurement.** P1;
   small remedy. Location: [sampling.py](../src/arfs/sampling.py), lines 191–209
   and 293–300.

   A single numeric feature `[-3, -1]` gives distance **-1**, `[-3, 0]` gives
   **0**, and `[1e9, 1e9+1]` gives **0**. Each pair should have distance 1.
   The code divides by the column maximum, derives a signed relative range,
   and computes extrema after converting to float32. A negative maximum flips
   the distance sign; a zero maximum erases variation; float32 extrema collapse
   nearby large values. See `test_numeric_translation_invariance`.

   Use a float64 `max - min` range directly and handle constant columns
   explicitly. This shared repair reaches direct distances, Gower sampling,
   and nearest neighbors. Until then, input units or translation can change
   clustering and neighbor rankings. Wider precision/overflow limits were not
   characterized.

2. **F2 — Equal-sized query/reference frames are treated as a self-distance
   matrix.** P1; small remedy. Location: [sampling.py](../src/arfs/sampling.py),
   lines 231–249.

   With `X=[0,10]` and `Y=[1,3]`, the correct pooled-range matrix is
   `[[.1,.3],[.9,.7]]`; ARFS returns `[[.1,.3],[.3,.7]]`. Row/column count
   equality triggers triangular computation and mirroring even when the frames
   contain different observations. Rectangular frames pass the same reference.
   See `test_equal_size_cross_matrix` and `test_rectangular_gower_positions`.

   Only use symmetry for an actual self-comparison; otherwise compute all query
   and reference pairs. Also reconcile the Notes saying ranges come from X with
   the actual `concatenate((X,Y))` range calculation at line 180, and document
   the rectangular output shape. The evidence establishes pooled-range behavior,
   not which reference-range policy should be retained.

3. **F3 — Missing values corrupt otherwise computable distances.** P1; small
   to medium remedy. Location: [sampling.py](../src/arfs/sampling.py), lines
   191–209 and 288–305.

   Numeric NaNs propagate into distances, including diagonals. A missing
   categorical value compares unequal to itself, creating a nonzero diagonal.
   An entirely missing numeric column is converted into a zero contribution but
   remains in the denominator, diluting distances from available features.
   Three independent-reference cases reproduce these behaviors in
   `test_pairwise_missing_gower` with explicit object categories in both
   environments.

   Mask unavailable features separately for each pair and renormalize over
   usable positive-weight features; define the no-comparable-feature result.
   If missing input is intentionally unsupported, reject it explicitly instead
   of returning plausible distances. No ordinal or asymmetric-binary missingness
   contract was evaluated.

4. **F4 — Advertised Gower input forms fail before calculation.** P2, P1 for
   ordinary inferred strings in the newer environment; small remedy. Location:
   [sampling.py](../src/arfs/sampling.py), lines 134–170.

   Arrays with integer category positions or an empty category list access
   nonexistent `.columns`; a documented reference Series fails the two-value
   shape unpack. Automatic inference rejects pandas `category`, `Int64`, and
   `string` dtypes. In pandas 3 even `DataFrame({'x':['a','b']})` hits that
   failure. Sparse input raises `AttributeError` before reaching the promised
   sparse-input `TypeError`. The corresponding `test_array_category_index_masks`,
   `test_pandas_extension_dtype_auto`, `test_gower_reference_series`,
   `test_inferred_string_gower_auto`, and sparse check preserve diagnostics.

   Validate sparse/shape boundaries first, normalize a reference Series to one
   row, use `shape[1]` for positional masks, and use pandas-aware dtype checks.
   Validate mask length and named columns explicitly. Frame masks by names,
   booleans, or positions and array boolean masks pass the tested complete cases;
   arbitrary mask/container combinations were not exhaustively enumerated.

5. **F5 — Invalid feature weights can create negative or undefined distances.**
   P1; small remedy. Location: [sampling.py](../src/arfs/sampling.py), lines
   212–222 and 303–305.

   With one changing and one constant feature, weights `[-1,2]` produce distance
   -1. All-zero and NaN weights return NaNs instead of a deliberate validation
   error. Positive feature weights and positive common rescaling pass the
   independent reference. See `test_invalid_feature_weights_rejected`.

   Validate one finite, nonnegative weight per feature and positive usable mass.
   Rename/document this as feature weighting: the implementation indexes weights
   by columns, despite its docstring calling them sample weights. No importance-
   or frequency-weight interpretation is meaningful for this feature vector.

6. **F6 — Public sampling weights are misrouted or discarded.** P1; small
   remedy after choosing the supported contract. Location:
   [sampling.py](../src/arfs/sampling.py), lines 55, 62–64, and 74.

   Three row weights for a three-row/two-feature frame fail boolean indexing in
   the Gower branch. If rows happen to equal features, those weights are silently
   accepted as feature weights; `[1,9]` on rows `[0,0]` and `[10,20]` still returns
   the unweighted representative `[5,10]`. The isolation branch always calls
   `isof_find_sample(..., sample_weight=None)`. Boundary probes capture both
   paths; the direct `isof_find_sample` helper does forward weights to the forest.

   Do not pass row weights to the metric's feature-weight slot. Either define
   weighted representative behavior or explicitly reject unsupported weighting.
   Forward supported row weights in the isolation path and state whether its
   acceptance criterion is weighted. Even with forwarding, its KS comparison
   currently has no row-weight input; this review establishes no weighted
   representativeness guarantee.

7. **F7 — Numeric imputation uses the wrong column's mean.** P1; one-line
   remedy plus an all-missing policy. Location:
   [sampling.py](../src/arfs/sampling.py), line 52.

   For `a=[0,2,4]` and `b=[100,NaN,200]`, clustering to one representative returns
   `b=100.666667`, instead of 150 after per-column mean imputation. The scalar
   `X[num_cols].mean().iloc[0]` fills every numeric column. See
   `test_gower_sample_imputes_per_column`.

   Fill with the means indexed by column; define how entirely missing columns
   are treated before clustering. Otherwise column order and unrelated feature
   scales alter both distances and representatives. Categorical all-missing
   imputation was not separately characterized.

8. **F8 — Sampling leaks an internal column and mishandles API boundaries.**
   P1 for a `label` collision, P2 for the other cases; small remedy. Location:
   [sampling.py](../src/arfs/sampling.py), lines 60–77.

   Gower results acquire an all-NaN `label` column because the final reindex uses
   the augmented frame. An original numeric `label` feature is overwritten with
   cluster strings and mean aggregation raises `TypeError`; an original
   categorical `label` is replaced by cluster IDs. Separately, an unknown method
   returns `None` because `NotImplementedError(...)` is constructed but not raised.
   See the three schema checks and `test_unknown_sampling_method_rejected`.

   Group by external cluster labels, reindex against original feature names,
   and raise the unsupported-method error. Tested two-or-more-row inputs accept
   `n=len(df)` and reject larger requests. Empty/single-row Gower inputs raise
   `ValueError` from clustering; this review does not require an identity-return
   convention for those unsupported tiny cases.

9. **F9 — Isolation sampling never advances its retry counter.** P1; small
   remedy. Location: [sampling.py](../src/arfs/sampling.py), lines 424–425 and
   462–477.

   `iteration` is initialized to zero and never incremented. The advertised
   increase in sample size and iteration limit are unreachable; repeated KS
   rejection can run forever. A bounded probe substitutes rejection at the KS
   boundary and stops after 45 calls: all 45 draws still contain ten of 100 rows.
   This is a control-flow proof, not an observed hang rate for real datasets.

   Start at the intended size, increment attempts, bound size progression, and
   define exhaustion behavior. The code starts at `size[1]` (10% for 100 rows),
   despite the 5% documentation. For 0–10 rows, rounding the 5% step gives zero;
   tested public calls with 0, 1, and 10 rows raise `ZeroDivisionError` before
   fitting. Handle tiny inputs explicitly. No long-running rejection loop was
   allowed to run unbounded during review.

10. **F10 — Isolation sampling changes categorical values in returned rows.**
    P1; small remedy. Location: [sampling.py](../src/arfs/sampling.py), lines
    70–75.

    Categories are converted to string/category codes for the forest, and the
    sampler returns rows from that encoded copy. Selecting positions `[2,0]`
    from a frame containing `['a','a','b']` returns category values `[1,0]`, not
    `['b','a']`. A selected categorical target would likewise change meaning.
    `test_isoforest_sample_returns_original_rows` isolates selection from this
    deterministic output failure; position/index preservation passes separately.

    Keep the encoded frame for scoring and return `df.iloc[idx]` from the original
    data. The input itself is not mutated. Changing the encoding strategy is a
    separate statistical decision, not necessary to repair returned values.

11. **F11 — Key-aware nearest neighbors ignore the query and may exhaust the
    candidate array.** P1; small remedy. Location:
    [sampling.py](../src/arfs/sampling.py), lines 374–405.

    Both X and Y are assigned `data_x.drop(key)`, so the result is always based
    on the first candidate row and allocates a full candidate-by-candidate
    matrix. Candidates `(a,0),(b,5),(c,10)` queried with `(c,10)` return positions
    `[0,1]`, not `[1,0]` at distances `[.5,1]`. The query ID is never excluded.
    With five candidates having IDs `a,a,a,b,c` and `n=3`, the expanding search
    requests six rows and raises an out-of-bounds error despite three available
    distinct IDs. Both checks fail in both snapshots.

    Compute the one-query distance vector, remove the query's ID, sort once,
    and retain the first row per eligible ID. Define the insufficient-ID policy
    before slicing. With no key, the tested nearest-neighbor output uses correct
    row positions and includes the query itself if it occurs in the candidates;
    no automatic self-exclusion is advertised for that branch.

12. **F12 — Neighbor count and optional-query boundaries are inconsistent.**
    P2; small remedy. Location: [sampling.py](../src/arfs/sampling.py), lines
    310–331 and 372–407.

    `smallest_indices([.3,.1,.2], 0)` returns all three positions; `n=-1`
    silently returns two. Omitting the optional `data_y` in `gower_topn` raises
    `AttributeError`. The helper's prose says largest values, but it actually
    returns the smallest values in ascending order, as its name/caller require.

    Validate n, return empty arrays for zero, and either require a one-row query
    explicitly or implement a documented default. Requests beyond candidate
    count already raise `ValueError`; no clipping contract is assumed here.
    Ties return correct values but have no documented stable positional ordering.
    Nonfinite handling also needs an explicit policy: the helper substitutes 999
    for NaN and extreme finite values for infinities, so negative infinity wins;
    `gower_topn` converts NaNs to distance 1 before selection. These are recorded
    behaviors, not evidence that missing neighbors are meaningful matches.

13. **F13 — Parallel matrix chunks change column-label types.** P1; small
    remedy. Location: [parallel.py](../src/arfs/parallel.py), lines 49–52 and
    169–172; affected matrix callers in
    [association.py](../src/arfs/association.py), lines 287, 451, 676, 991, and 1398.

    `np.array_split(comb_list, n_jobs)` converts pairs to a homogeneous array.
    For columns `[0,'b','c']`, the integer 0 becomes string `'0'`, causing
    `KeyError` with two workers while one worker succeeds. Tuple/MultiIndex
    labels expand into arrays and also fail lookup. Actual joblib worker
    tracebacks are saved, not simulated dispatch failures.

    Split pair positions and use them to slice the original Python labels, or
    chunk the original list without dtype coercion. All-integer labels and mixed
    dtype-label combinations beyond the supplied fixtures were not exhaustively
    tested. Arithmetic agreement for ordinary string labels does not cover this
    execution-path failure.

14. **F14 — The large-data notebook discards its own presampling limit.** P2;
    one-line remedy. Location:
    [arfs_large_data_sampling.ipynb](../docs/notebooks/arfs_large_data_sampling.ipynb),
    code cell 15, file lines 348–350.

    `data_dum = data.sample(n=10_000)` is immediately overwritten by
    `data_dum = data.copy()`. The next comment still describes a 10,000-square
    matrix. The notebook's saved input shape is 20,640 rows; the float32 distance
    output alone is then 1,704,038,400 bytes instead of 400,000,000 bytes, before
    clustering and temporary storage. The overwrite is asserted in
    [scaling.py](evidence/05-sampling-and-parallel/scaling.py).

    Remove the overwrite and keep the prose explicit about quadratic storage.
    This does not make arbitrary large Gower clustering safe. No 20,640-square
    matrix or full notebook run was attempted in this review.

**Parallel results and resource boundaries**

Weighted and unweighted mixed association matrices agree by `(row,col)` with one
and two workers when categorical columns use object dtype. Association/correlation
series agree including ordering. Frozen-environment grouped relevance agrees
for regression and classification; the newer snapshot is blocked by C2 below.
Both worker modes propagate deliberately raised callable errors. Two workers
with just one pair and an explicitly selected threading backend preserve values.
Inputs retain duplicate indexes and remain unchanged in the checked calls.
These comparisons test execution, not the statistical correctness already
reviewed in [unit 1A](01a-statistical-foundations.md).

The matrix helper sorts by value only in serial mode; parallel results follow
pair/chunk order. Comparison by labels passes, but callers must not depend on a
stable rank order across worker counts. Empty column work returns an empty Series
for the tested callback. Empty matrix work raises `ValueError: No objects to
concatenate`, also reached by `wcorr_matrix` with one numeric column and its
default Spearman method. The generic association-matrix wrapper guards that
no-pair case. These boundaries should be made consistent when the helpers are
repaired; no existing documented empty-matrix schema was assumed.

Worker counts 0 and -2 fail while splitting, before joblib. ARFS handles -1
itself using `multiprocessing.cpu_count()`; it does not preserve joblib's
[negative-count semantics](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html).
A dispatch-only probe simulating eight CPUs and three features records eight
tasks, five empty. This can waste processes and memory, especially in
`wcorr_series`, which passes -1 directly. Some association wrappers cap -1 by
feature/pair count, but explicit positive counts can still create empty chunks.
The two CPU APIs agreed on this machine; container quota/affinity oversubscription
was not demonstrated. Using joblib's effective worker count and capping useful
work would address the risk without a new execution framework. Joblib also has
native controls for nested native-library threads; all review runs pinned those
threads to one.

Gower allocates a float32 `(len(X),len(Y))` matrix, concatenates both inputs, and
constructs numeric/categorical copies. For self-comparison, output storage is
exactly `4*N*N` bytes and distance work is O(N²P). Complete-linkage clustering
adds storage/work beyond that output. The key-aware neighbor bug also takes
this quadratic path unnecessarily.

The bounded [frozen measurements](evidence/05-sampling-and-parallel/frozen-scaling.json)
and [newer measurements](evidence/05-sampling-and-parallel/current-scaling.json)
use only 100, 200, and 400 rows, two features, and explicit object categories.
The output sizes are 40,000, 160,000, and 640,000 bytes: doubling rows quadruples
storage. Timing and Python allocation peaks are saved but are single observations,
not a throughput benchmark or process-RSS measurement. A 100,000-row output
would require **40,000,000,000 bytes**; that is a calculation, not an allocation
performed here.

**Compatibility blockers reached through the reviewed callers**

- **C1 — Inferred strings fail association type agreement in pandas 3.** P1;
  small remedy. At [association.py](../src/arfs/association.py), lines 600–605,
  `correlation_ratio` accepts only dtype equality with object/category/bool.
  pandas 3's new default string inference is documented in its
  [migration guide](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html).
  The dispatcher in [utils.py](../src/arfs/utils.py), line 154, currently includes
  inferred strings through pandas's object-selection compatibility behavior,
  then the statistic rejects them. `test_inferred_strings_association_matrix`
  passes on frozen dependencies and fails with both worker counts on newer ones.
  Use the same pandas-aware categorical classification throughout. This is a
  newly demonstrated caller failure related to the dtype inconsistency in
  [unit 03 F4](03-selector-contracts.md), not a new parallel arithmetic defect.
- **C2 — Scalar ANOVA checks fail on the saved newer NumPy.** P1; small remedy.
  At [association.py](../src/arfs/association.py), lines 1461–1462, `np.where` and
  `np.nonzero` are called on scalars. NumPy 2.0.2 warns; 2.5.4 raises `ValueError`.
  Both grouped regression and continuous-predictor classification relevance fail
  before a serial/parallel comparison can complete, weighted or unweighted.
  Four `test_parallel_relevance_agrees` cases isolate this with object categories.
  Handle the scalar condition directly while preserving constant-feature
  handling. Unit 1A's separate weighting, zero-mass-group, and cancellation
  findings remain open; this is an additional runtime blocker, not their repair.

**Statistical limits of isolation sampling**

The notebook calls this outlier removal and describes a KS statistic over 95%.
The source instead tests the **p-value** of a random subset's scalar anomaly
scores against the full score vector, which contains that subset. The
[SciPy 1.13.1 KS documentation](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.ks_2samp.html)
assumes independent samples from continuous distributions. A high p-value is
failure to reject a difference, not a 95% guarantee of representativeness.
Repeatedly choosing subsets until acceptance does not establish such a guarantee
either. Matching one-dimensional anomaly scores cannot establish preservation of
feature/target relationships, minority groups, or the full joint distribution.
These conclusions follow from the test inputs and interpretation; no empirical
false-acceptance rate is claimed.

A real constant-feature 40-row forest/sampler run returned four rows and replayed
exactly when resetting NumPy's global seed. The public API exposes no local
`random_state`; this narrow check does not establish deterministic behavior for
arbitrary data or version changes. Non-degenerate repeated-seed sampling quality,
rare-group preservation, and downstream feature-selection validity remain unmeasured.

**Reproduction and handoff**

From any directory, run the existing installed-wheel interpreters:

```sh
/tmp/arfs-review-03f67d0/locked-env/bin/python \
  /home/bsatom/Documents/arfs/reviews/evidence/05-sampling-and-parallel/run.py frozen
/tmp/arfs-review-03f67d0/wheel-env/bin/python \
  /home/bsatom/Documents/arfs/reviews/evidence/05-sampling-and-parallel/run.py current
```

[run.py](evidence/05-sampling-and-parallel/run.py) creates a temporary working
directory, limits native threads, checks installed/source identity, runs strict
and unmasked pytest with repository configuration/cache disabled, and measures
bounded allocations. Each subprocess has a 60-second limit. The isolation retry
probe separately stops after 45 KS calls. An unmasked pytest exit of 1 is
expected. Exact commands, working directories, and exit codes are in
[frozen runs](evidence/05-sampling-and-parallel/frozen-runs.json) and
[newer runs](evidence/05-sampling-and-parallel/current-runs.json).
If the disposable environments are gone, recreate the audited wheel using
unit 0's instructions and these recorded dependency pins; do not silently use
the published PyPI package or install into the repository's `.venv`.

Scope tracing covered every function in `sampling.py` and `parallel.py`, all
matrix/series/relevance callers of the parallel helpers, the sampling type-list
helpers, the large-data notebook's source, and the existing test inventory.
No source sampling callers outside this module were found; its demonstrated
external consumer is the notebook. Unit 1A's formula findings and unit 0's
packaging/import blockers remain open and were not counted again here.

Item **05 is reviewed; findings remain open**. The queue continues with
[06 — utilities and documentation workflows](06-utilities-and-workflows.todo.md).
This item stops here. Repairs should add focused normal regression tests for
the relevant evidence cases and recheck all shared callers.

Not checked: full notebook execution, large allocations/clustering, an extended
sampling-quality study, a platform/Python matrix, new dependency resolution,
or a package-wide test rerun. No release-readiness claim or library repair is
made; wrong distances, changed row values, and unbounded rejection remain risks.
