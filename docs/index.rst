ARFS feature selection
======================

ARFS provides feature selection and preprocessing for pandas data. Start with
the `offline GrootCV example <https://github.com/ThomasBury/arfs#first-fit-with-grootcv>`_
to fit a selector and reuse it on held-out rows. See
`Upgrading to 5.0 <https://github.com/ThomasBury/arfs/blob/main/CHANGELOG.md#upgrading-to-50>`_
for migration steps.

Choose a method
---------------

* **Basic filters** remove columns that fail rules for missingness, number of
  distinct values, cardinality or redundancy. Use them to screen a raw table.
* **All-relevant selectors** (GrootCV, Leshy and BoostAGroota) look for predictors
  that a tree model can use, including potentially redundant signals. They judge
  fitted importances against shuffled copies of predictors.
* **MRMR** ranks candidates by their relevance to the target and redundancy
  with features already selected. Use it when you need a feature budget.
* **Lasso** selects individual numeric or encoded columns in a generalized
  linear model (GLM) design and refits the surviving columns without a penalty.
  Use it when that design and model family match the intended analysis.

Selections depend on the training rows, predictor dtypes and encodings, and
any model, importance method or observation weights used. Fit learned steps
inside each training split and reuse them to transform held-out rows. Use
:doc:`arfs.feature_selection` for signatures and fitted attributes.

.. toctree::
   :maxdepth: 2
   :caption: User's guide

   Introduction
   Methods overview
   workflows
   modules


.. toctree::
   :maxdepth: 2
   :caption: Core tutorials

   notebooks/arfs_regression.ipynb
   notebooks/arfs_classification.ipynb
   notebooks/basic_feature_selection.ipynb
   notebooks/association_and_feature_selection.ipynb
   notebooks/preprocessing.ipynb
   notebooks/lasso_feature_selection.ipynb
   notebooks/mrmr_feature_selection.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Advanced tutorials

   notebooks/mrmr_fs_VS_arfs.ipynb
   notebooks/issue_categoricals.ipynb
   notebooks/issue_collinearity.ipynb
   notebooks/arfs_grootcv_custom_params.ipynb
   notebooks/arfs_timeseries.ipynb
   notebooks/arfs_large_data_sampling.ipynb
   notebooks/arfs_non_normal_loss_and_sample_weight.ipynb

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
