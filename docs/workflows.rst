Documentation workflows
=======================

Runtime validation
------------------

The installed-wheel checks use Linux x86_64, with locked dependencies on CPython
3.9, 3.10, 3.11 and 3.12, exact direct runtime minima on 3.9, and a separately
pinned newer snapshot on 3.12. The requirement files under
``.github/requirements/`` identify the minimum and newer snapshots; ``uv.lock``
identifies the locked dependencies. The newer pins require Python 3.12 and
must not be copied to earlier interpreters. The metadata range
``>=3.9,<3.13`` does not certify every dependency combination or platform.

GPU/CUDA, Windows/macOS, other architectures, CatBoost/XGBoost, FastTreeSHAP
and live HoloViews/Panel plots need separate execution before claiming support.
Missing optional-backend errors are tested; that does not test
the backends themselves. Persistence checks use fresh processes in the same
environment, trusted artifacts and new prediction rows; they do not promise
cross-version compatibility or safe loading of untrusted pickle files.

Selection experiments
---------------------

The R33 release-evidence handoff records a fixed ten-seed screen of strong,
null, weak, correlated, mixed-type, weighted and interacting signals for
regression and binary classification. Selectors and category encoders learn
only from each outer training split. The independent evaluation rows compare
dummy, all-feature, univariate and known-signal baselines using mean squared
error or binary log loss. Weighted cases use the same covariate-based weights
for training and evaluation. No setting is chosen from evaluation outcomes.

These small synthetic experiments measure behavior for the recorded settings.
They do not calibrate false-discovery control, certify weak-signal recovery,
or establish superiority over other selectors, encodings or unweighted fits.
An empty selection uses the dummy predictor for evaluation. In cross-validation
or tuning, repeat preprocessing and selection inside every outer training fold.
GrootCV's internal folds do not replace an untouched outer evaluation set.

Opt-in MRMR row sampling
------------------------

MRMR uses all rows by default. To cap the rows used for both relevance and
redundancy estimation, set a positive ``sample_size`` and an integer seed:

.. code-block:: python

   from arfs.feature_selection import MinRedundancyMaxRelevance

   selector = MinRedundancyMaxRelevance(
       10, task="classification", sample_size=10_000, random_state=42
   )
   X_selected = selector.fit_transform(X_train, y_train)

Inputs at or below the cap use all rows. Larger inputs are sampled without
replacement, stratifying classification draws by target. Both callbacks see
the same subset; ``fit_transform`` still returns every input row. Inspect
``sample_indices_`` for the original row positions (``None`` means all rows)
and ``n_samples_fit_`` for the count before callback missing-value handling.

Weights are subset positionally without rescaling or changing draw
probabilities. Default frequency-weight relevance therefore uses the smaller
sample mass. A sample losing a positive-weight target class is rejected;
increase the cap or disable sampling. Rare predictor levels can still
disappear. Compare category coverage, association estimates, selected features
across seeds, and held-out utility before choosing a cap. Fit selection only
inside the training fold. Row sampling does not reduce the redundancy table's
feature-count cost. For p available predictors and k selected features, the table
retains k - 1 selected-feature columns (none for k = 0), requiring O(p k) values. Input
frames and temporary association calculations add to total process memory.

Build HTML locally
------------------

Use Python 3.12 and install the system ``pandoc`` executable on PATH. The Python
package named ``pandoc`` is not a substitute for that executable. Read the Docs
installs the system package and the project's ``docs`` dependency group using
``uv sync`` and ``uv.lock``; FastTreeSHAP is not part of either the core examples
or the documentation build.

From the checkout root:

.. code-block:: sh

   uv sync --locked --group docs --group test
   uv run --no-sync python -m pytest tests/test_documentation_workflows.py -q
   uv run --no-sync make -C docs html

The ``docs`` and ``test`` groups contain contributor tools and are not published
as package extras. Add documentation dependencies with
``uv add --group docs <package>``. After manually editing ``pyproject.toml``, run
``uv lock`` and commit both files together; ``uv lock --check`` verifies they
agree without changing the lockfile. Reserve ``[project.optional-dependencies]``
and ``uv add --optional <extra>`` for optional features installed by package
users with ``pip install 'arfs[<extra>]'``.

HTML is written to ``docs/_build/html``. On Windows, run ``make.bat html`` from
``docs`` after activating the same environment. The Windows launcher has the
same source path but has not been executed on Windows.

The build renders saved notebook outputs without executing cells. The core
tutorials use local fixtures. The association notebook ends with an optional
PPS reference and explains its dependency constraint.

The earlier execution ledgers are in the review records for
`documentation workflows <https://github.com/ThomasBury/arfs/blob/main/reviews/R32-documentation-workflows.md>`_
and `tree preprocessing <https://github.com/ThomasBury/arfs/blob/main/reviews/tree-preprocessing.md>`_.

Sampling limits
---------------

Exact Gower sampling through :func:`arfs.sampling.sample` stores each pair once
in a condensed float64 vector, requiring ``4 * N * (N - 1)`` bytes for N input
rows. That is about 10 GB at 50,000 rows and 40 GB at 100,000 rows (decimal units),
before clustering and other temporary memory. Requesting fewer output rows with ``n``
does not reduce distance storage. Exact Gower sampling is unsuitable when this
allocation exceeds your memory budget; reduce the input rows first.

The row sampling tutorial caps the Gower input at 600 rows, so its distance
vector alone is about 1.4 MB. Direct :func:`arfs.sampling.gower_matrix` calls
still return a dense matrix: ``8 * n_x * n_y`` bytes, or ``8 * N**2`` for self-comparison.

The sampler uses native numeric and categorical distance calculations and exact
complete linkage. Floating-point reduction order can change nearly tied
distances and therefore cluster memberships; recheck downstream results.

Cluster representatives are synthetic column means/modes; Gower sampling rejects
observation weights. Isolation sampling returns original rows, uses weights only
for its forest fit and keeps its draws and KS screen unweighted. Neither method
provides a representativeness or outlier-removal guarantee.
