# Coverage ledger

Target: `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`, 2026-10-11.
“Reviewed” means inspected with the evidence and limits below, not defect-free.
The [source inventory](frozen-audit.json) lists every top-level public definition,
its line, imports, review owners, installed path, and source hash. An assertion
requires the ownership map to match **all 16 Python files** exactly. Both saved
wheel environments match those files. This is ownership coverage, not a claim
that every method or input combination has a runtime test.

## Source and public workflows

Paths in the first column are relative to `src/arfs/`.

| Module | Public workflow and owning review | Concrete evidence | Remaining boundary |
| --- | --- | --- | --- |
| `__init__.py` | Version/install identity: [00](../../00-baseline.md) | [Build and distribution metadata](../00-baseline/distributions.json); current hashes | Root import alone exercises no feature-selection workflow |
| `feature_selection/__init__.py` | Public exports: [00](../../00-baseline.md), [03](../../03-selector-contracts.md) | [Export probes](../00-baseline/environments.json), [contract checks](../03-selector-contracts/checks.py) | Missing `BaseThresholdSelector` export remains open |
| `association.py` | Entropy, U, Cramer's V, weighted Pearson/Spearman/ranks, ratio/ANOVA, mixed matrices/series, regression/classification relevance: [1A](../../01a-statistical-foundations.md); parallel/type paths: [05](../../05-sampling-and-parallel.md); reshape/cluster/static plots: [06](../../06-utilities-and-workflows.md) | [Independent calculations](../01a-statistical-foundations/checks.py), [worker checks](../05-sampling-and-parallel/checks.py), [rendering checks](../06-utilities-and-workflows/checks.py), [optional import checks](checks.py) | Interactive wrapper source and both missing-backend branches inspected; no actual Panel/HoloViews rendering. Decorators and matrix conversions traced through callers, not exhaustive standalone contracts. Mixed-type score calibration unresolved |
| `parallel.py` | Pair/column dispatch, worker count, labels, exceptions: [1A](../../01a-statistical-foundations.md), [05](../../05-sampling-and-parallel.md) | [One/two-worker references and counterexamples](../05-sampling-and-parallel/checks.py) | No distributed execution, container-quota benchmark, or exhaustive backend matrix |
| `feature_selection/mrmr.py` | Relevance/redundancy selection, target labels, strong-signal/null behavior: [1A](../../01a-statistical-foundations.md) | [Hand examples and seeded screens](../01a-statistical-foundations/checks.py) | Mixed relevance scale and weight meaning unresolved; no broad selection stability study |
| `feature_selection/allrelevant.py` | Leshy/BoostAGroota/GrootCV fit, shadows, support/rank, importance, fold/weight handling: [1B](../../01b-all-relevant.md); comparison/refit integration: [04](../../04-training-and-benchmarks.md); plots: [06](../../06-utilities-and-workflows.md) | [24 checks](../01b-all-relevant/checks.py), [80-fit screen](../01b-all-relevant/screen.json), [comparison checks](../04-training-and-benchmarks/checks.py), [plot checks](../06-utilities-and-workflows/checks.py), [fallback checks](checks.py) | Five-seed screen is numeric, SHAP, small settings, frozen dependencies. No formal error control, weighted/categorical screen, exhaustive custom folds, GPU or optional estimator/backend equivalence |
| `feature_selection/lasso.py` | Six GLM families, weighted CV/search, BIC/deviance, selector design, prediction/intercept/schema: [1C](../../01c-weighted-glm-lasso.md) | [References](../01c-weighted-glm-lasso/checks.py), [current entry failure](../01c-weighted-glm-lasso/current_checks.py), [60-attempt screen](../01c-weighted-glm-lasso/screen.json) | Newer runtime is blocked at the entry point. Arbitrary links, dispersion/power, zero-weight likelihood, post-selection inference not established |
| `preprocessing.py` | Ordinal encoding, dtype selector, legacy `cat_var`, tree discretization, midpoint APIs, Patsy: [02](../../02-preprocessing.md); integration: [06](../../06-utilities-and-workflows.md) | [40 checks](../02-preprocessing/checks.py), [workflow ledger](../06-utilities-and-workflows/frozen-workflow-ledger.json) | Every definition source-traced. Legacy styling helper has no internal caller. No generic expanded-feature lineage, exhaustive transform serialization or held-out binning benefit |
| `feature_selection/base.py` | Threshold callback, selected masks/names, transform: [03](../../03-selector-contracts.md) | [Contracts and ordered callback fixture](../03-selector-contracts/checks.py) | Complete sklearn check suite not run; strict full-schema rejection vs named projection remains a choice |
| `feature_selection/unsupervised.py` | Missing/unique/cardinality filters; collinearity elimination and plot callers: [03](../../03-selector-contracts.md), [06](../../06-utilities-and-workflows.md) | [Threshold/graph/reference checks](../03-selector-contracts/checks.py), [association plots](../06-utilities-and-workflows/checks.py) | Equality wording and ignored weights require explicit documentation; `method='correlation'` not exhaustively exercised |
| `feature_selection/variable_importance.py` | Native contributions, cumulative selection, categories/weights/tasks, repeated fitting and plots: [03](../../03-selector-contracts.md), [06](../../06-utilities-and-workflows.md) | [Selection references](../03-selector-contracts/checks.py), [plot boundaries](../06-utilities-and-workflows/checks.py) | Whether global importance averages a weighted population is unresolved; actual fasttreeshap output not tested |
| `feature_selection/summary.py` | Pipeline decision table, coloring, pandas Styler: [03](../../03-selector-contracts.md) | [Real pipeline plus styling overlay](../03-selector-contracts/frozen-styling-checks.txt), [newer failure](../03-selector-contracts/current-styling-failures.txt) | Name-preserving transforms only; arbitrary expanded/renamed feature lineage unestablished |
| `gbm.py` | Native training, splits/weights/objectives, predictions/raw/leaves, offsets, save/load, backend detection: [04](../../04-training-and-benchmarks.md) | [71 checks](../04-training-and-benchmarks/checks.py), [fresh-process reload](../04-training-and-benchmarks/reload.py) | Same-version trusted-joblib persistence only. No hostile/corrupt artifact, interrupted-write, cross-version, ranking/DART, sparse or GPU certification |
| `benchmark.py` | Weighted permutation, selector/method comparison, plotting/tick labels: [04](../../04-training-and-benchmarks.md), [06](../../06-utilities-and-workflows.md) | [Real/reference comparisons](../04-training-and-benchmarks/checks.py), [notebook ledger](../06-utilities-and-workflows/frozen-workflow-ledger.json) | Cross-estimator comparison is not a held-out predictive benchmark; all optional estimators not installed |
| `sampling.py` | Gower/direct/cross distances, nearest neighbors, cluster representatives, isolation sampling, helper boundaries: [05](../../05-sampling-and-parallel.md) | [79 checks](../05-sampling-and-parallel/checks.py), [bounded scaling](../05-sampling-and-parallel/frozen-scaling.json) | No large allocation, ordinal/asymmetric-binary metric, or rare-group/joint-distribution representativeness study |
| `utils.py` | Dtypes/encoding, converters, model detection/LightForest, lists/style, local/remote loaders, synthetic data/plots: [00](../../00-baseline.md), [1B](../../01b-all-relevant.md), [03](../../03-selector-contracts.md), [04](../../04-training-and-benchmarks.md), [06](../../06-utilities-and-workflows.md) | [Utility checks](../06-utilities-and-workflows/checks.py), [dataset ledger](../06-utilities-and-workflows/frozen-datasets.json), [training references](../04-training-and-benchmarks/checks.py) | No live OpenML download. Trusted bundled pickle only. Duplicate conversion-only validator and unused public helpers need an API decision, not automatic deletion |

Non-Python resources: unit 00 inspected wheel/sdist contents; unit 06 compared
both shipped data hashes and local schemas, and source-loaded housing separately.
No absent dataset package `__init__.py` is counted as a seventeenth source file.
Docs/CI/configuration ownership is units 00/06/07, not inferred from code coverage.

## All 17 notebook entry points

The [unit-00 inventory](../00-baseline/inventory.json) records imports and active
tests. The [unit-06 ledger](../06-utilities-and-workflows/frozen-workflow-ledger.json)
and its [newer counterpart](../06-utilities-and-workflows/current-workflow-ledger.json)
are the authority for exact executed cells/substitutions. **No notebook has a
complete clean-kernel passing run.** Source inventory alone is not execution.

| Notebook in `docs/notebooks/` | Ownership/evidence | Explicit remaining gap |
| --- | --- | --- |
| `basic_feature_selection.ipynb` | 03/06; bounded exact synthetic-data cells and plots | Optional fastshap cells and complete notebook; newer run stops on 05 C1 |
| `preprocessing.ipynb` | 02/06; bounded exact cells | Full pipeline and notebook; newer association failure; extra native-model probe rejects interval outputs |
| `arfs_classification.ipynb` | 1B/04/06; extracted import/fit cells and adjusted cancer smoke | Unchanged workflow fails; no remote Titanic or optional comparator run |
| `arfs_regression.ipynb` | 1B/04/06; native Boston excerpt/plot and missing benchmark import | Optional comparisons/full notebook |
| `association_and_feature_selection.ipynb` | 1A/03; algorithm references and source claims | Full notebook; repaired mixed association semantics |
| `mrmr_feature_selection.ipynb` | 1A; weighted statistics and MRMR checks | Full notebook; score comparability/calibration |
| `mrmr_fs_VS_arfs.ipynb` | 1A/1B/06; subset claim/source inventory | No validated comparison or theorem establishing nested selected sets |
| `lasso_feature_selection.ipynb` | 1C/02; family/exposure/intercept and transform checks | Full notebook; corrected deviance/design and post-selection interpretation |
| `arfs_non_normal_loss_and_sample_weight.ipynb` | 1B/1C/04; weighted method/source tracing | End-to-end weighted/categorical selector validation; does not test GBM offset persistence |
| `arfs_large_data_sampling.ipynb` | 05; overwrite source assertion and small allocation measurements | Full/large run and sampling quality; never allocated the 20,640-square output |
| `issue_collinearity.ipynb` | 1A/03; constant/weighted/graph counterexamples | Exact complete notebook not run |
| `issue_categoricals.ipynb` | 1B/02/03/06; category value/type references | Exact complete notebook and optional estimator behavior not run |
| `arfs_boruta_borutaShap_comparison.ipynb` | 00 inventory; 1B statistical screen and 04 comparison contracts | Optional third-party Boruta/BorutaShap comparison unexecuted; no relative-quality claim |
| `arfs_grootcv_custom_params.ipynb` | 07 source inspection; 1B/04 training helpers | CPU-count loop/regularization timing not run; its “physical cores” helper counts online/logical processors, not a physical-core measurement |
| `arfs_timeseries.ipynb` | 07 inspected all cells; [cell-10 counterexample](checks.py) | No OpenML/time-series model evaluation. Full-data selection followed by manual dropping is not a demonstrated fold-local selection evaluation |
| `arfs_on_GPU.ipynb` | 07 source inspection; device and fastshap settings recorded in source | GPU/CUDA builds/hardware, timing and backend equality untested; no device 1 assumed available |
| `arfs_shap_vs_fastshap.ipynb` | 07 source inspection; 100,000-row optional comparison | No optional backend installation or large benchmark; no speed/equality claim |

The remaining examples use previously owned algorithms. The newly inspected
time-series plotting error is report 07 F1. Optional backend execution,
documentation rendering, full estimator compliance, statistical calibration and
large-data performance stay in the final report's unresolved-check ledger and
the remediation exit criteria. They have not been silently marked passing or
removed from scope.
