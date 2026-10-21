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
Missing optional-backend errors and fallbacks are tested; that does not test
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

The build renders saved notebook outputs and never executes code cells. Some
saved outputs predate the current repairs; an HTML build is not evidence that
all notebooks execute. Execute examples separately in a fresh kernel, with
their data and optional dependencies recorded.

Example execution scope
-----------------------

The historical R32 validation handoff records exact cells, environments, output
notebooks and remaining gaps for the earlier tutorial revisions. Those checks
covered basic selection and preprocessing on the local synthetic regression fixture, classification on a declared local cancer
substitute for the remote Titanic download, and regression on bundled Boston
data. The classification encoder is also checked with categorical input.

CatBoost/XGBoost model comparisons require separate compatible installations.
FastTreeSHAP examples are opt-in; the basic notebook's two cells and the
classification/regression FastTreeSHAP model cells are commented out. Install a
compatible FastTreeSHAP build separately and uncomment an example to run it.
GPU, optional-backend speed comparisons and the remaining
notebooks require their own execution before their saved results can be treated
as current. No supported optional-backend version matrix is claimed here.

The time-series regression checks verify the exact reduced columns passed to
fit and predict. The example's manually reduced frame follows exploratory
full-series selection; its CV numbers are not an untouched performance estimate.
For evaluation, learn preprocessing and selection inside each training fold or
outer time window, then transform its held-out rows with the fitted objects.

Tree preprocessing tutorial revision
------------------------------------

The current ``preprocessing``, ``lasso_feature_selection`` and
``mrmr_feature_selection`` notebooks separate numeric binning from categorical
grouping and fit learned steps only on training rows. Their settings are fixed
before evaluating the same untouched rows. Fixture weights are illustrative
observation weights, not actuarial exposure. The separate Poisson illustration
generates counts from synthetic exposure times expected rate before forming rates.

The preprocessing notebook adds a 4,320-row factorial experiment with known
numeric thresholds, noncontiguous nominal groups and contiguous declared ordinal
groups. It also includes independent noise, constant/all-missing columns, missing
values and unknown-level probes. Its semantic assertions compare the fitted
partitions with the generating truth, including a category-renaming
counterexample. This is a correctness check on a specified process, not a
real-portfolio benchmark or evidence of universal predictive improvement.

.. list-table:: Current execution ledger
   :header-rows: 1
   :widths: 22 46 32

   * - Notebook
     - Automated source-cell checks
     - Full notebook and figure outputs
   * - ``preprocessing``
     - Crafted data, train/evaluation split, true partitions, renaming,
       native missing/unknown routes, stable categories and weighted MSE arms
     - Cleared; manual execution pending
   * - ``lasso_feature_selection``
     - Correct Poisson count generation and weighted factor-design comparison;
       training-only preprocessing, selection and finite GLM predictions
     - Cleared; manual execution pending
   * - ``mrmr_feature_selection``
     - Local classification comparison, identifier exclusion, weighted binary
       log loss and training-only preprocessing/selection
     - Cleared; manual execution pending

The source-cell checks are in ``tests/test_documentation_workflows.py`` and use
cell tags instead of fragile positions. They execute computational cells without
running a notebook kernel or figure cells. They do not replace the requested
manual top-to-bottom runs or visual inspection. Saved outputs were deliberately
cleared for this revision; no score or figure is claimed from an unexecuted
notebook. The original linear/Poisson plotting sections and the separate MRMR
regression example still need that manual run.

From the checkout root, after the documentation environment is installed:

.. code-block:: sh

   uv run --no-sync jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/notebooks/preprocessing.ipynb
   uv run --no-sync jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/notebooks/lasso_feature_selection.ipynb
   uv run --no-sync jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/notebooks/mrmr_feature_selection.ipynb
   uv run --no-sync python -m sphinx -W --keep-going -b html docs docs/_build/html

Restarting the kernel and running all cells in each notebook is equivalent. Use
the Python 3.12 locked environment and the separately pinned newer snapshot when
checking compatibility. Inspect the saved tables, truth-versus-fitted curves and
loss figures at reading size, and update this ledger after those runs. A strict
Sphinx build remains a separate check and keeps ``nbsphinx_execute="never"``.

Sampling limits
---------------

Exact Gower sampling through :func:`arfs.sampling.sample` stores the full dense
float64 distance matrix for N input rows, requiring ``8 * N**2`` bytes. That is
about 20 GB at 50,000 rows and 80 GB at 100,000 rows (decimal units), before
clustering and other temporary memory. Requesting fewer output rows with ``n``
does not reduce distance storage. Exact Gower sampling is unsuitable when this
allocation exceeds your memory budget; reduce the input rows first.

The large-data example retains a 10,000-row presample before Gower clustering,
so its distance output alone is about 800 MB. Reduce the presample for smaller
machines. The full allocation and large-data throughput are not part of the
bounded checks. :func:`arfs.sampling.gower_matrix` documents cross-matrix storage.

Cluster representatives are synthetic column means/modes; Gower sampling rejects
observation weights. Isolation sampling returns original rows, uses weights only
for its forest fit and keeps its draws and KS screen unweighted. Neither method
provides a representativeness or outlier-removal guarantee.
