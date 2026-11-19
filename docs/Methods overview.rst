Methods overview
================

Leshy
-----

Leshy builds on `Boruta <https://doi.org/10.18637/jss.v036.i11>`_ and
`BorutaPy <https://github.com/scikit-learn-contrib/boruta_py>`_. It counts
how often real features beat a percentile of the shadow importances across
iterations. Choose native, permutation or SHAP importance for a compatible
tree estimator. Each choice measures a different aspect of the fitted model.

Leshy configures an owned estimator clone and leaves the caller's estimator
and constructor options unchanged. Its ``estimator_`` records the runtime
learner configuration; importance helpers train separate clones. Leshy and
BoostAGroota expose the fitted importance method in ``importance_``.

BoostAGroota
------------

BoostAGroota builds on `BoostARoota <https://github.com/chasedehan/BoostARoota>`_,
which removes low-scoring predictors over successive reduction rounds.

Use ``importance="shap"``, ``"pimp"`` or ``"native"``. Replace the removed
``importance="fastshap"`` mode with ``"shap"``. LightGBM uses native SHAP
contributions; other compatible estimators use TreeExplainer. Supply a tree
estimator and positive integer ``iters`` and ``max_rounds``. Current parameters
are checked during ``fit``, including changes through ``set_params``; invalid
round limits raise an error instead of being coerced. ``n_rounds_`` counts
completed rounds and ``n_features_in_`` records input width. Selection operations
require a successful fit, and a failed refit invalidates the previous selection.

In each model iteration, BoostAGroota uses a single normalization total across
real and shuffled features. For SHAP and native importances it sums all scores.
For permutation importance it sums their absolute values and preserves each
score's sign. A zero-total iteration contributes zeros.

In each reduction round it averages the normalized scores across iterations.
The threshold is the largest shadow mean divided by ``cutoff``. Real features
with positive means at or above that threshold are kept. For example, one
iteration with real scores ``[.3, .1]`` and shadow scores ``[.5, .1]`` has a
single total of ``1.0`` and a threshold of ``.5`` when ``cutoff=1``. These
empirical thresholds do not establish false-discovery control.

``sha_cutoff_df`` now contains only the final reduction round: one row per
iteration, with real candidate columns followed by their shadows. Features
eliminated in earlier rounds are absent. The boxplot shows those final-round
candidates against ``mean_shadow``, the threshold from the same normalized
scale, and colors boxes by selected feature names. Earlier versions mixed
rounds in this table; update consumers that expected all input columns.
``ranking_absolutes_`` orders every input feature by descending first-round
mean importance, preserving input order on ties. It is a model-dependent
ordering, separate from the selected/rejected ranks in ``ranking_``.

GrootCV
-------

GrootCV uses LightGBM's native SHAP contributions. Remove the former
``fastshap`` argument from existing calls. Boosting-round limits in
``lgbm_params`` follow LightGBM's parameter precedence instead of being
discarded. With no supplied limit, training uses a 10,000-round ceiling, with
20-round early stopping where supported. Set an explicit ``num_iterations``
for DART, which disables early stopping. A previously ignored limit can change
selected features, so refit and check downstream results.

- SHAP importance is the mean absolute contribution over each fold's training
  rows, summed over outputs for multi-output models, then normalized across
  real and shadow features; a zero-total fold contributes zeros. Validation
  rows drive early stopping where supported.
- Take each feature's median across all folds and repeats, then the maximum
  shadow median divided by ``cutoff``. Keep positive real medians at or above
  this threshold. Summary columns do not enter either calculation.
- For real scores ``[.57, .57, .16]`` and shadows ``[.35, .05, .70]`` and
  ``[.05, .35, .05]`` across three folds, the real median is ``.57`` and the
  threshold is ``.35`` at ``cutoff=1``, so the real feature is kept.
- Recognized multiclass classification objectives resolve to softmax, including
  binary classification with more than two target classes. SHAP aggregation
  follows the returned output axes rather than the objective's spelling.
- Default classification folds use ``RepeatedStratifiedKFold``; regression uses
  ``RepeatedKFold``. The resolved objective, including one supplied through
  ``lgbm_params``, determines the fold type. Default stratification requires at
  least two classes and at least ``n_folds`` observations per class. Reduce
  ``n_folds`` or supply custom ``folds`` when this is infeasible. Supplied
  splitters or iterables of index pairs take precedence over default folds and
  retain their own splitting policy. Iterators and generators are consumed by
  each fit; use a reusable list or splitter for refitting, or supply a new
  iterator. Empty or exhausted folds raise ``ValueError`` with this guidance.
- This empirical comparison does not establish calibrated false-discovery control.

Pass an integer ``random_state`` to replay selector shadows and GrootCV's
default repeated folds. Leshy also draws integer learner seeds. BoostAGroota
preserves the learner's seed: set that seed separately for reproducible
stochastic training. GrootCV's default seed is ``2652124``; supplied splitters
keep their own RNG policy and learner seeds stay in ``lgbm_params``. A
``RandomState`` instance advances across fits, while ``None`` uses NumPy's
global RNG. Imports no longer reset that global stream. The fixed internal
Leshy/BoostAGroota holdout is unchanged.

Use Leshy's ``verbose=0``, BoostAGroota/GrootCV's ``silent=True``, or
VariableImportance's ``verbose<=0`` to disable routine output and progress
bars. GradientBoosting's ``verbose_eval<=0`` also suppresses its split,
training, save and load messages. Warnings remain visible. Configure the
supplied Leshy/BoostAGroota learner's own logging separately.

MRMR
----

ARFS adapts the MRMR scheme with associations for continuous and categorical predictors.
Mixed-type relevance scores are ranking heuristics, not calibrated significance
tests. MRmr's selected set need not be a subset of an all-relevant selector's set.

``fit`` always returns the selector. Remove the former ``return_scores``
argument and read ``selected_features_``, ``relevance_``, ``redundancy_`` and
``ranking_`` after fitting. Default callbacks resolve from the current ``task``
and ``n_jobs`` on each fit, in ``relevance_func_`` and ``redundancy_func_``;
custom callbacks and the shared row-sampling contract are preserved.

MRMR's pair-score table, ``redundancy`` (without the trailing underscore), now
retains columns only for selected features used in later comparisons, in
selection order. The last selected feature has no column because no later step
uses it. With p available predictors and k selections, storage is O(p k), rather
than O(p²); input frames and temporary calculations still add to process memory.
Code inspecting the former full square must adapt. ``redundancy_`` and
``ranking_`` retain their meanings. The string reducers ``"mean"``/``"max"``
and NumPy ``mean``/``max`` use vectorized row reductions; other callables retain
the labeled-Series row path. Floors, perfect-association penalties, domain
restrictions and stable input-order ties are unchanged.

Association and relevance functions accept only ``handle_na="drop"``,
``"fill"`` or ``None``. The default ``"drop"`` deletes rows listwise across
the supplied frame and aligns responses and observation weights with those
rows. It does not use a separate row subset per feature. ``"fill"`` uses zero
for numeric missing values and a collision-free category for nominal missing
values, including strings and Booleans. Nominal columns with missing values
become categorical; existing categories (including unused levels) and their
order are retained. Caller data is not modified. ``None`` passes rows unchanged
to custom callbacks; built-in statistics still require complete inputs.
Relevance functions reject missing or nonfinite responses before applying the
predictor policy, and weights must remain finite, nonnegative and have positive
retained mass. Frequency-weight degrees of freedom are unchanged.

- Theil's U statistics for the categorical-categorical association (correlation).
- Variance ratio for continuous-categorical association.
- Pearson or Spearman correlation for continuous-continuous association.

``CollinearityThreshold`` takes callable association measures, with defaults
``weighted_theils_u``, ``weighted_corr`` (Spearman), and ``correlation_ratio``.
Use callables or ``functools.partial`` to change these measures; string names
are not supported. Its ``assoc_matrix_`` has a zero diagonal so self-association
does not remove unrelated features.

Lasso
-----

Performing a simple grid search with enforced lasso regularization.
The best model minimizes training BIC (``score="bic"``) or mean total validation
deviance across five folds (``score="deviance"``), and all non-zero coefficients are selected.
``EnetGLM.score`` returns negative response-scale deviance so sklearn searches
can maximize it. Weights enter as variance weights; the loss uses unit scale
and is not divided by fold size or weight mass.
Each fold fits an independent clone. Classifiers and binomial GLMs with discrete
targets use stratified folds; regressions, including two-valued numeric responses,
and fractional binomial responses use ordinary folds. Splits do not shuffle rows.
Fit records one design for prediction and transformation. Nonzero training
constants are preserved without adding another intercept; ``fit_intercept=False``
disables automatic addition and preserves supplied columns. A generated column
is named ``Intercept`` (with a numbered suffix to avoid input-name collisions).
It is still penalized and, if selected, included in transformed output.
``feature_names_in_``, ``n_features_in_`` and ``support_`` describe original
inputs; ``design_feature_names_`` and ``design_support_`` describe the selector's
coefficient design. Named inference inputs align to the fitted names; missing,
extra and duplicate columns are rejected. Arrays require fitted input width
and order. Family/link parameters are resolved at each fit.
Zero-weight rows are omitted after response/weight validation, before fitting,
scoring and forming GLM validation folds. Positive weights are not normalized.
``alpha`` is penalty strength; ``L1_wt`` is the L1 fraction of the selection
penalty. Surviving columns are refit without regularization, so returned
coefficients do not retain ridge shrinkage. Pure ridge (``L1_wt=0``) is unsupported.
Summary p-values do not account for using the same data to select variables.
The wrapper's AIC/BIC use the returned log-likelihood, retained row count and
selected design rank, counting mean parameters without dispersion. Backend
summary information criteria are unchanged and can differ from these corrected
attributes. A nonfinite candidate score aborts the search with its penalty value;
invalid candidates are not silently discarded. Variance-weighted fixed-scale
likelihoods are quasi-likelihoods; finite BIC does not establish calibrated
selection or post-selection significance.
The loss function can belong to the exponential family, as seen in the statsmodels GLM documentation.
Using the bic metric is faster since it is evaluated on the training data, making it unsuitable for the test data, whereas the deviance is cross-validated.

Combine ``TreeGrouper`` (categorical predictors) followed by ``TreeDiscretizer``
(numeric predictors) to introduce supervised univariate partitions. Both fit one
native LightGBM tree per selected feature using all positive-weight training rows;
the general GBM wrapper's validation split is not used. Validate outcomes and
weights before omitting zero-weight rows. Nominal levels use native categorical
splits; ordered categoricals use their declared order with contiguous groups.
Missing and unknown levels follow the fitted missing route, without an Other
level. Output category vocabularies are frozen at fit.

Fit these steps inside every training fold, forwarding observation weights to
each. Encode groups and numeric bins as factors (``num_bins_as_category=True``)
when they should have separate model effects. This does not implement group or
fused Lasso: selecting encoded columns does not jointly select the original
categorical variable. Supervised partitioning and selection invalidate ordinary
post-selection interpretations of reported p-values. Neither the partitions nor
a passing example establish predictive improvement.


See :doc:`arfs.feature_selection` for constructor signatures and fitted
attributes.
