Documentation workflows
=======================

Build HTML locally
------------------

Use Python 3.12 and install the system ``pandoc`` executable on PATH. The Python
package named ``pandoc`` is not a substitute for that executable. Read the Docs
installs the system package and the project's ``docs`` extra; FastTreeSHAP is
not part of either the core examples or the documentation build.

From the checkout root:

.. code-block:: sh

   uv sync --locked --extra docs --extra test
   uv run --no-sync python -m pytest tests/test_documentation_workflows.py -q
   uv run --no-sync make -C docs html

HTML is written to ``docs/_build/html``. On Windows, run ``make.bat html`` from
``docs`` after activating the same environment. The Windows launcher has the
same source path but has not been executed on Windows.

The build renders saved notebook outputs and never executes code cells. Some
saved outputs predate the current repairs; an HTML build is not evidence that
all notebooks execute. Execute examples separately in a fresh kernel, with
their data and optional dependencies recorded.

Example execution scope
-----------------------

The R32 validation handoff records exact cells, environments, output notebooks
and remaining gaps. Core checks cover basic selection and preprocessing on the
local synthetic regression fixture, classification on a declared local cancer
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

Sampling limits
---------------

The large-data example retains a 10,000-row presample before Gower clustering.
Its dense float64 distance output alone is about 800 MB; temporary arrays and
clustering need additional memory. Reduce the presample for smaller machines.
The full allocation and large-data throughput are not part of the bounded checks.
Cluster representatives are synthetic column means/modes; Gower sampling rejects
observation weights. Isolation sampling returns original rows, uses weights only
for its forest fit and keeps its draws and KS screen unweighted. Neither method
provides a representativeness or outlier-removal guarantee.
