Contributing and documentation workflows
========================================

Use Python 3.12, ``uv`` and the system ``pandoc`` executable on your PATH.
For example, on Debian or Ubuntu, install it with ``sudo apt install pandoc``.
Clone the repository and install the locked documentation and test groups:

.. code-block:: sh

   git clone https://github.com/ThomasBury/arfs.git
   cd arfs
   uv sync --python 3.12 --locked --group docs --group test
   uv run --no-sync python -m pytest tests/test_documentation_workflows.py -q

Run the notebooks locally
-------------------------

From the checkout root, register the project Python for this run and execute
the notebooks in fresh kernels. The command writes review copies to
``docs/_build/notebooks``; inspect their figures and tables before updating saved
outputs in ``docs/notebooks``.

.. code-block:: sh

   uv run --no-sync python -m ipykernel install --prefix docs/_build/kernels --name arfs-docs
   JUPYTER_PATH="$PWD/docs/_build/kernels/share/jupyter" \
     OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     uv run --no-sync python -m nbconvert --to notebook --execute \
     --ExecutePreprocessor.kernel_name=arfs-docs --ExecutePreprocessor.timeout=120 \
     --output-dir docs/_build/notebooks docs/notebooks/*.ipynb

The walkthroughs use local fixtures. The association notebook ends with the
optional PPS reference and explains its released dependency requirement.

Build and check HTML
--------------------

.. code-block:: sh

   uv run --no-sync python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` and check the tutorial links and rendered
outputs. Sphinx renders the saved outputs in ``docs/notebooks``. Notebook code
runs in the separate command above. Read the Docs uses Python 3.12, system
Pandoc and the locked ``docs`` group.

The ``docs`` and ``test`` groups are for contributors. If you change their
requirements, update ``uv.lock`` with ``uv lock`` and run ``uv lock --check``.
Use these review records for earlier execution ledgers and the runtime matrix:

* `Documentation workflows <https://github.com/ThomasBury/arfs/blob/main/reviews/R32-documentation-workflows.md>`_
* `Tree preprocessing <https://github.com/ThomasBury/arfs/blob/main/reviews/tree-preprocessing.md>`_
* `Release validation <https://github.com/ThomasBury/arfs/blob/main/reviews/R33-release-evidence.md>`_

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
