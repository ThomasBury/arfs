# Research proposal: distributional importance before estimator design

**Status:** Proposal only; experiments and implementation require separate
authorization. **Prepared:** 2026-10-22. **Constraint:** Python throughout,
including the research experiment.

## 1. Executive assessment

**There is a worthwhile research question, but no justification yet for another
public selector.** The recommendation is **Option C: reproduce and assess an
established method**, concentrating on Bénard et al.'s distributional importance.
Assess stability selection separately as a possible improvement to selection
reliability.

The opportunity is specific: **can ARFS identify features needed for
probabilistic prediction that its current mean-oriented workflows miss, at an
acceptable computational and maintenance cost?**

Three conclusions constrain the project:

- Distribution-sensitive importance already exists. Detecting variance effects
  would demonstrate capability, not originality.
- Stability can improve reproducibility while consistently preserving a biased
  selection.
- The earlier `PredictiveSubsetSelector` proposal is insufficiently
  differentiated from established elimination and projection methods. It should
  not proceed independently without comparative evidence.

The Python-only constraint applies throughout. No R execution, new forest
implementation, dependency changes or modifications to GrootCV are proposed in
this planning phase.

## 2. Literature comparison and statistical objectives

### Define relevance before measuring it

An operational definition of **all-relevant selection** is weak relevance:

\[
j\in S_{\mathrm{weak}}
\iff
\exists A\subseteq\{1,\ldots,p\}\setminus\{j\}:
Y\not\!\perp X_j\mid X_A.
\]

This can include informative proxies and interaction participants. It differs
from **conditional relevance**, which asks whether
\(Y\not\!\perp X_j\mid X_{-j}\).

For example, two identical informative predictors are individually redundant
conditional on the other. Both can nevertheless be weakly relevant. Removing
their entire group loses information.

**Predictive subset selection** instead seeks:

\[
\min_S |S|
\quad\text{subject to}\quad
R(f_S)-R(f_{\mathrm{full}})\leq\varepsilon.
\]

Its answer depends on the learner, loss, training procedure and acceptable
margin. Neither objective identifies causal drivers.

### Bénard et al.: strongest methodological starting point

Their population importance is based on

\[
D_j=\mathbb E\!\left[
\operatorname{MMD}^{2}
\left(P_{Y\mid X},P_{Y\mid X_{-j}}\right)
\right],
\]

normalized by the distributional information explained by all predictors. A
characteristic kernel makes the population discrepancy sensitive to conditional
distribution changes, including effects beyond the mean.

The estimator refits without each feature and subtracts a discrepancy between
independently randomized full forests to mitigate finite-sample bias. Their
consistency results require regularity conditions: independent observations,
continuous covariates with bounded positive density, smooth conditional
embeddings, and specified honest forest construction and asymptotics.
**Consistency does not supply a calibrated selection threshold or
false-discovery guarantee.** Their stated goal is compact distributional
prediction. [Bénard, Näf & Josse, 2024](https://proceedings.mlr.press/v238/benard24a/benard24a.pdf)

Two consequences require explicit testing:

- Under global independence, the population normalization denominator is zero.
  Near-null normalized values must not be interpreted as strong evidence.
- Exact duplicates and categorical inputs do not satisfy the continuous-density
  assumptions above. Practical support and theoretical coverage must remain
  separate.

DRFs represent conditional distributions through weights on training responses,
allowing means, quantiles and other functionals without selecting a parametric
response family. [Ćevid et al., 2022](https://www.jmlr.org/papers/v23/21-0585.html)

### Stability selection: useful, with a narrower guarantee than its name suggests

Classical stability selection bounds the **expected number of false
selections** under conditions including noise-selection exchangeability and
restrictions on the base procedure. That is not generally false-discovery-rate
control. [Meinshausen & Bühlmann, 2010](https://people.math.ethz.ch/~peterbu/Files/Manuscripts/stabilityselection.pdf)

Complementary-pairs stability selection, or CPSS, obtains more general bounds for
selecting variables with **low base-procedure selection probability**.
Translating these into guarantees about irrelevant variables requires additional
assumptions; stronger bounds also introduce shape restrictions. Half-sample
selection probabilities are not full-sample relevance probabilities.
[Shah & Samworth, 2013](https://www.statslab.cam.ac.uk/~rjs57/SSFinal.pdf)

Correlated substitutes illustrate the limitation: an informative group may be
selected reliably while its interchangeable members each have low frequency.
Group reporting helps interpretation; it does not automatically inherit
feature-level guarantees.

Existing Python work, including [Stabl](https://github.com/gregbellan/Stabl), also
makes a generic "stability plus artificial features" estimator difficult to
justify as an original contribution.

### Four distinct uncertainties

| Quantity | What varies | What it establishes |
| --- | --- | --- |
| Algorithmic randomness | Seeds, sampled features, tree construction, shadows | Sensitivity of a fitted procedure |
| Sampling uncertainty | Newly sampled observations | Variation across datasets from the population |
| Epistemic predictive uncertainty | Plausible fitted prediction functions | Model-dependent uncertainty about predictions |
| Feature-relevance uncertainty | An explicitly defined relevance estimand | Requires inference about that estimand |

Repeated SHAP or permutation scores can describe the first two sources when the
experimental design separates them. Their spread is not automatically a
confidence interval for relevance. Weak effects may remain indistinguishable
from noise; uncertainty should permit that conclusion.

Malinin et al.'s SGLB adds gradient noise and shrinkage, with asymptotic
posterior-sampling arguments. Virtual ensembles reuse correlated, truncated
stages of one boosting trajectory. They offer inexpensive predictive
uncertainty, but not independent dataset replications or posterior
feature-inclusion probabilities. Their dependence also limits ensemble
diversity. [Malinin et al., 2021](https://arxiv.org/pdf/2006.10562)

CatBoost exposes this mechanism, including CPU-only posterior-sampling
configuration. Its predictive uncertainty interface does not provide a
calibrated feature-selection rule.
[CatBoost documentation](https://github.com/catboost/catboost/blob/master/catboost/docs/en/concepts/python-reference_virtual_ensembles_predict.md)

### NGBoost: additional prediction targets, not Bayesian inclusion inference

NGBoost optimizes proper scores for a chosen conditional distribution family. It
already exposes separate importance for distribution parameters, such as Normal
location and scale. Sample-weight support exists, although categorical
preprocessing and compatibility require checking.
[NGBoost interpretation](https://stanfordmlgroup.github.io/ngboost/3-interpretation.html),
[implementation](https://github.com/stanfordmlgroup/ngboost/blob/master/ngboost/ngboost.py)

A Normal model can capture conditional variance, but cannot represent arbitrary
distributional shape. Under Gaussian negative log-likelihood, distributions
sharing their first two moments have the same optimal Gaussian approximation.
Changing the optimization geometry does not remove this restriction or produce
Bayesian inclusion probabilities.

Use **CRPS** as the common scalar distributional evaluation score. Negative
log-likelihood is appropriate when a predictive density is defined; an empirical
DRF distribution does not directly provide a continuous density. Calibration
must accompany scoring: broad intervals can achieve coverage without useful
sharpness. [Proper scoring rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

### Candidate comparison

Here, \(C\) denotes one base fit and \(B\) the number of complementary pairs.

| Direction | Scientific target and limitation | Practical contribution | Engineering, cost and compatibility |
| --- | --- | --- | --- |
| Existing GrootCV | Empirical shadow-relative model importance; no calibrated relevance guarantee | Established baseline | Existing categories/weights; repeated fits and SHAP |
| GrootCV stability diagnostics | Reproducibility of its scores and selections | Potentially useful reporting; little methodological novelty | Low engineering; repeated fits dominate; fold scores alone cannot supply sampling intervals |
| Stability selection / CPSS | Selection frequencies and conditional error bounds | Could reduce unstable noise selections; may lose weak signals and substitutes | Wrapper-level work; \(2BC\); guarantees need separate scrutiny for weighting/grouping |
| CatBoost virtual ensembles | Model-dependent predictive uncertainty | Cheap predictive diagnostic; feature-selection benefit unestablished | One training trajectory plus prediction; new relevance rule would need substantial validation |
| NGBoost distributional importance | Parameter reliance or score-dependent predictive contribution | Detect scale effects using established software | Additional dependency; numeric preprocessing; permutation costs predictions, drop-and-relearn costs roughly \(p+1\) fits |
| DRF/MMD importance | Conditional distribution information lost after relearning | Most direct extension beyond mean-oriented importance | Roughly \(p+1\) fits, plus an additional full fit for correction; kernel storage can be quadratic in observations |
| Projection predictive selection | Compact models approximating a trusted reference | Established answer to much of the earlier subset proposal | Reference-model quality and projection support govern results; mature implementation is primarily R |
| Model-X knockoffs | Conditional-relevance testing with FDR control under valid construction | Strongest discovery guarantee among these options | Requires suitable covariate modeling and exchangeability; arbitrary mixed data are difficult |

Projection predictive selection is already implemented in
[projpred](https://mc-stan.org/projpred/). Knockoff validity depends on the joint
predictor construction; estimating that distribution incorrectly can inflate
FDR. [Model-X assumptions](https://web.stanford.edu/group/candes/knockoffs/modeling.html),
[robustness results](https://arxiv.org/abs/1801.03896)

Rashomon approaches address another useful question: how importance differs
across well-performing models. Random-seed ensembles do not generally explore
that entire model class. This is useful context for correlated substitutes, not
an additional candidate for this experiment.
[Fisher, Rudin & Dominici, 2019](https://jmlr.org/papers/v20/18-760.html)

## 3. Repository assessment

Inspection of the current working tree confirms:

- [GrootCV](../src/arfs/feature_selection/allrelevant.py) normalizes each fold's
  training-row SHAP vector, takes feature medians, and compares them with the
  maximum shadow median divided by `cutoff`.
- Leshy already stores importance histories; BoostAGroota retains final-round
  iteration scores. These are reusable diagnostics, not independent sampling
  replications.
- Shared permutation importance uses weighted validation scoring, but its helper
  owns a fixed split and uses the estimator's default score. It is not currently
  a general proper-score importance interface.
- Existing validation, category handling, cloning, seed tests and transformation
  contracts should be reused. The SHAP reducer should not be repurposed to
  aggregate distribution parameters.
- The [selection screen](../tests/selection_screen.py) already supplies a useful
  protocol/evidence pattern, but needs distinct relevance labels and
  distributional metrics.
- [R33](R33-release-evidence.md) supplies warning evidence, not calibrated error
  rates.

ARFS supports Python 3.9–3.12. CatBoost is currently a documentation dependency;
NGBoost and a native DRF backend are absent from the declared dependencies.

A maintained R implementation already exposes MMD importance. Its Python wrapper
depends on R/rpy2, so it is excluded by the Python-only constraint.
[Current R reference](https://www.maths.bris.ac.uk/R/web/packages/drf/refman/drf.html),
[Python wrapper](https://github.com/lorismichel/drf/blob/master/python-package/setup.py)

The newer native Python/Rust `drforest` is an experimental possibility. It
requires Python 3.11+, lacks observation-weight and missing-value support, and
explicitly disclaims inference-valid forest asymptotics. It cannot currently be
treated as a compatible replacement or inherit Bénard's theorem.
[Backend scope](https://github.com/silaskoemen/drforest#statistical-scope),
[fit interface](https://github.com/silaskoemen/drforest/blob/main/drforest/forest.py)

## 4. Recommended direction

**Choose Option C: a Python-only assessment of Bénard-style distributional
importance.**

Reproduce the estimand and correction using an existing native backend; describe
this as an adaptation, not an exact reproduction of the authors' forest
implementation.

Investigate CPSS/GrootCV separately. Do not combine its frequencies with MMD
importance into a new score.

The contribution sought is practical: useful distributional feature assessment
with transparent dependence, uncertainty and cost boundaries. No claim of
methodological originality is warranted yet.

A relevant 2026 preprint reports that distributional splitting often fails to
outperform CART on scalar tabular problems. Treat this as preliminary evidence
requiring a **same-backend CART control**, not as a settled conclusion.
[Koemen, 2026](https://arxiv.org/abs/2607.23721)

## 5. Minimal experimental design

### First reviewable phase: feasibility, null outcomes and variance effects

Use only two serious candidates:

1. Bénard-style DRF/MMD importance.
2. CPSS around unchanged GrootCV.

Controls are unchanged GrootCV, the same DRF backend with CART splitting,
marginal prediction, and full/oracle-feature NGBoost predictions. NGBoost is a
predictive control, not another proposed selector.

Use 10 independently generated datasets per mechanism, three learner seeds per
dataset, 800 training rows, 400 development rows and 2,000 independent evaluation
rows, with 12 numeric predictors.

Start with:

- **Null:** \(Y=Z\), independent of every predictor.
- **Variance only:** \(Y=\exp(0.8X_1)Z\), with centered independent predictors
  and \(Z\sim N(0,1)\).

Freeze the protocol before results. Use 100-tree DRFs, honest splitting, fixed
kernel bandwidth learned from training responses, and exact kernel calculations
for importance evaluation. For GrootCV, use three folds, one repeat and a
100-round learner cap. CPSS uses ten complementary pairs and a fixed frequency
threshold of 0.9; five-pair results provide a predeclared sensitivity check.

Keep raw and normalized MMD values, the normalization denominator, and the
full-versus-full correction. Preserve negative corrected estimates. For this
screen, select at most one feature: the highest strictly positive raw corrected
score, with ties broken by input column order. Use the same rule for the CART
control. A run recovers the variance feature when it selects \(X_1\). With no
positive score, select the empty set. For the bounded Gaussian kernel, mark
normalized scores undefined when the denominator is at or below \(10^{-12}\).
The raw-score rule still applies. This screen makes no significance claim.

Fit each selector on training rows only. Freeze its selected columns, then fit
a fresh Normal NGBoost predictor on the combined training and development rows
using those columns. Use logarithmic loss, natural gradients, 100 boosting
rounds, a learning rate of 0.01, and depth-three regression trees with at least
10 observations per leaf. Use all rows and columns each round. An empty subset
uses the empirical marginal distribution of those same responses. Use the same
predictor and settings for the MMD, unchanged GrootCV, CPSS, CART, full and
oracle subsets. Pair methods on the same rows and seed, and score the untouched
evaluation rows with mean CRPS. Use learner seeds 11, 23 and 37 and record the
dataset and split seeds before fitting. No settings are tuned on evaluation rows.

### Conditional extension after reviewing the first phase

Use 20 fresh independent datasets per mechanism and add:

- Strong additive effects.
- Weak additive effects.
- Correlated proxies, plus an exact-duplicate stress case.
- Pure interactions.
- Shape effects with equal conditional mean and variance—for example, a
  predictor-dependent mixture of Normal and standardized Student-\(t_5\)
  distributions.

Record weak relevance, conditional relevance and informative groups separately.

For subset evaluation, use fixed ranking prefixes of sizes \(0,1,2,4,8,12\), a
common prediction backend, and development-only choice within a margin of 1% of
marginal-model CRPS. Refit before independent evaluation. This is an experimental
comparison rule, not a proposed production stopping rule.

### Measurements and replication

- **All-relevant assessment:** recall, false-discovery proportion, expected false
  count, feature/group retention and stability.
- **Conditional distribution assessment:** signal–null ranking separation, CRPS,
  interval coverage and width, and calibration conditional on the
  variance-driving predictor.
- **Subset assessment:** selected count, independent loss deterioration and
  margin violations.
- **Engineering:** total fits, elapsed time and peak memory.

Hold splits, shadows and complementary pairs fixed while varying learner seeds.
Across fresh datasets, estimate sampling variation separately using a balanced
variance-components analysis. Uncertainty summaries resample whole datasets;
three seeds do not turn ten datasets into thirty independent replications.

This screen cannot establish nominal FDR or confidence coverage. Any subsequent
guarantee requires a derivation and a dedicated calibration experiment, with
approximately 1,000 independent replications for a useful assessment near a 5%
error rate.

## 6. Conditional engineering plan

The first deliverable is a private experiment and evidence report. **No public
API changes.**

Reuse ARFS validation and evidence conventions. Keep three operations separate:
fitting a distribution predictor, computing importance, and assessing sampling
variability. Use existing backend predictions and weights; do not build a forest
engine or introduce a generalized uncertainty framework.

Before substantial runs, verify normalized forest weights, prediction
identities, kernel calculations through two independent formulations,
deterministic replay and separation of evaluation observations. Reject
unsupported weighted or missing-data inputs explicitly.

Estimate runtime from the initial feasibility cells. Stop for review if a
selection run exceeds three minutes or the projected first screen exceeds two
workstation hours. These are research budgets, not measured performance claims.

Only successful evidence would justify a separate distributional interface.
Public integration additionally requires a credible backend-maintenance path,
explicit Python-version support, categorical grouping and weight semantics. If
that requires rewriting the backend, abandon the integration rather than
expanding the project.

## 7. Decision gates

Freeze these screening criteria before execution:

- **Continue distributional research:** variance-feature recovery is at least
  80% across the 30 variance-only dataset/seed runs, and mean evaluation CRPS
  on those runs is at least 2% lower than for the paired unchanged GrootCV
  selections under the same predictor. Results must also survive the CART
  and NGBoost controls. Confirm any promising result using fresh replications.
- **Continue stability work:** false selections fall by at least 50% in the
  extended null/additive assessment, with no more than a ten-percentage-point
  loss of strong-signal recall. Report correlated-group losses separately.
- **Revise:** rankings improve but predictive utility, sampling reliability or
  computational cost remains inconclusive. Do not tune against the completed
  evaluation samples.
- **Abandon:** simpler controls match the benefit, gains require unstable
  normalization, weak-signal/group losses dominate, or Python integration costs
  exceed the demonstrated value.
- **Authorize a public estimator only later:** independently reproduced utility,
  an independently reviewed statistical contract, a defensible stopping rule and
  acceptable compatibility are all required.

Independent evidence means known generating mechanisms, untouched evaluation
observations, reproducible artifacts and statistical review—not agreement
between importance plots or agents.

No implementation, dependency installation or benchmark execution was performed
to prepare this proposal. The main remaining risk is that distributional
importance works scientifically while offering too little practical advantage
to justify maintaining it in ARFS.
