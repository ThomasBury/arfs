# MRMR row sampling stays opt-in

MRMR now accepts `sample_size=None` and `random_state=None`. Full-data fitting
remains the default. A positive integer cap draws rows without replacement only
when the input exceeds that cap. Relevance and redundancy share the same draw;
`transform` and `fit_transform` still return every input row.

```python
from arfs.feature_selection import MinRedundancyMaxRelevance

selector = MinRedundancyMaxRelevance(
    10, task="classification", sample_size=10_000, random_state=42
)
X_selected = selector.fit_transform(X_train, y_train)
```

Classification draws are stratified by target; regression draws are uniform.
Weights remain positional and unscaled, and do not determine draw probabilities.
Samples with no positive weight mass or a lost positive-weight target class are
rejected. Rare predictor levels can still disappear. `sample_indices_` exposes
the original fit-input row positions in input order (`None` means all rows);
`n_samples_fit_` records their count before callback missing-value handling.
The [workflow documentation](../docs/workflows.rst) describes the same contract.

The implementation reuses installed scikit-learn
[`resample`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html),
without its newer weight-as-probability interface. No dependency, default metric,
relevance formula, feature-count setting, or redundancy-table layout changed.
Row sampling does not remove the feature-square storage cost or the caller's
full input allocation.

## Adult validation

Source: Becker and Kohavi (1996), [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult).
The official `adult.data` training file has SHA-256
`5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`.
After dropping incomplete rows, a fixed stratified split of its 30,162 rows
uses **24,129 for training and 6,033 for evaluation**, with split seed 20261016.
The official `adult.test` file is unused. Categorical predictors are preserved;
`fnlwgt` stays a predictor and no sample weights are used.

Both default Theil's U and the existing Cramér's V redundancy callback select
10 features. Caps of 5,000 and 10,000 and seeds 0–4 were fixed before inspecting
evaluation outcomes. All selection, relevance, and association calculations
use only training rows. Each selected feature set then trains the same
`HistGradientBoostingClassifier` on **all outer training rows**, with native
categorical support, 100 iterations, 15 maximum leaf nodes, L2 regularization 1,
no early stopping, and learner seed 0. Column order follows the original input,
so different MRMR selection orders do not perturb learner tie handling.

Log loss evaluates predicted probabilities; lower is better. All 14 predictors
give **0.284820**, compared with **0.341697** for full-training-row MRMR. Both
redundancy metrics happen to select the same full-row feature set on this split.
Full-data selection and its complete ranking exactly match the pre-change MRMR
class loaded from Git HEAD, so this result is not a changed-default regression.
The earlier [performance report](mrmr-performance.md) used all 30,162 complete
rows for selection; its selections and timings are a different experiment.

| Redundancy | Row cap | Overlap with full-row top ten | Median association error | Median log loss | Log-loss range |
| --- | ---: | ---: | ---: | ---: | --- |
| Theil's U | 5,000 | 7–9 | 0.010622 | 0.288821 | 0.288447–0.296610 |
| Theil's U | 10,000 | 9 | 0.005555 | 0.294842 | 0.294285–0.296610 |
| Cramér's V | 5,000 | 8–9 | 0.011330 | 0.296436 | 0.294267–0.297570 |
| Cramér's V | 10,000 | 9 | 0.005927 | 0.294842 | 0.294285–0.296610 |

For each draw, association error is the mean absolute difference from the
full-training association matrix, across its returned entries, before MRMR's
absolute-value/floor handling. The table reports the median of those five errors.
The worst individual matrix-entry discrepancy across all draws is **0.06949**.
Relevance-rank Spearman correlations range from **0.94286 to 0.99121**; high
univariate rank agreement does not guarantee identical MRMR selections.

All draws preserve the target classes and their proportions within one row of
the full-training fractions. Every 5,000-row draw loses **1–3 predictor levels**:
usually rare countries, with `Armed-Forces` occupation also missing in one draw.
Four 10,000-row draws retain every training predictor level; one loses
`Holand-Netherlands`. The same rows are drawn for both metrics at each cap/seed.

The sampled selections beat full-row MRMR on this learner/holdout, but all
selected subsets perform worse than using all 14 predictors. This is not evidence
that sampling improves MRMR generally. Frequency-weight relevance also changes
with the smaller retained mass, so the difference is not just sampling noise.
Do not tune the cap or seed against these evaluation outcomes. These five seeds
measure sensitivity on one split, not uncertainty across independent datasets.

MRMR fit seconds, measured after warm-up, are descriptive: the full-row fits
took 0.447 s (Theil) and 0.668 s (Cramér). Median sampled fits took 0.281/0.329 s
for Theil at 5,000/10,000 rows, and 0.469/0.492 s for Cramér. The full-row cases
have one timed fit each, and sampled cases have one fit per seed; this is not a
controlled five-repeat speed benchmark. No timing or predictive-quality pass/fail
threshold is introduced.

## Reproduction and verification

The existing [manual runner](../tests/mrmr_performance.py) now has a separate
`--sampling-validation` mode. Earlier full-row measurement artifacts remain
historical evidence. Local raw measurements in `evidence/mrmr-sampling/measurements.json`
record every draw, missing level, selected order, association error, class
proportion, log loss, timing, split/input hashes, source hashes, and settings.
The raw file stays ignored and is not part of the PR; the command below
reproduces the experiment.
Versions: Python 3.12.7, ARFS 4.0.1, NumPy 2.0.2, pandas 2.2.3, SciPy 1.13.1,
scikit-learn 1.6.1; further dependency versions are in the raw provenance.
There is one worker and one thread per native pool; tests and documentation
builds ran after the experiment.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
.venv/bin/python tests/mrmr_performance.py /tmp/arfs-mrmr-sampling-new \
    --adult-data /tmp/arfs-mrmr/adult-training.data --sampling-validation
/tmp/arfs-association/py312-locked/bin/python -m pytest \
    tests/test_mrmr_sampling.py tests/test_mrmr_ranking.py \
    tests/test_association_kernels.py tests/test_statistical_arithmetic.py \
    tests/test_observation_boundaries.py tests/test_category_identity.py \
    tests/test_parallel_contracts.py tests/test_documentation_workflows.py \
    -q --tb=short -p no:cacheprovider
```

The pre-existing Python 3.12 test environment supplies pytest; its numerical
dependency versions match the experiment. Tests check exact agreement with a
manual sampled fit, full-data/no-op caps, alignment with unrelated or duplicate
indexes, unchanged inputs, weights and mixed-type target identities, seeded
reproducibility and cloning, custom callbacks, full-row transforms, empty
selections, and invalid/lost-mass samples.

Verification: **534 tests passed**, with 224 warnings, in 88.46 s. The new
sampling tests and existing MRMR, association, and documentation workflow
contracts passed. A fresh strict Sphinx HTML build, Ruff 0.17.0 lint/format,
source/measurement reconciliation, local-link checks, and whitespace checks
passed. The historical wide-memory benchmark and the full installed-wheel
version matrix were not rerun.

Not validated: weighted Adult sampling, repeated outer splits, the official test
partition, other downstream learners, or production workloads. No default row cap
is recommended or chosen automatically.
