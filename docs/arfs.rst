arfs package
============

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   arfs.feature_selection

Submodules
----------

arfs.association module
-----------------------

.. automodule:: arfs.association
   :members:
   :undoc-members:
   :show-inheritance:

arfs.benchmark module
---------------------

.. automodule:: arfs.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

arfs.gbm module
---------------

.. automodule:: arfs.gbm
   :members:
   :undoc-members:
   :show-inheritance:

arfs.parallel module
--------------------

.. automodule:: arfs.parallel
   :members:
   :undoc-members:
   :show-inheritance:

arfs.preprocessing module
-------------------------

``TreeDiscretizer`` bins numeric predictors; ``TreeGrouper`` groups categorical
predictors. For mixed data, group first and then bin. Pass observation weights to
each supervised fit. Existing mixed discretizers must be refitted. ``tree_dic``
now stores native LightGBM Boosters: use ``tree_dic[column].predict(...)`` instead
of ``tree_dic[column].model.predict(...)``, supplying the fitted numeric/category
codes expected by that tree. Prefer the transformer's ``transform`` interface to
apply its complete fitted mapping. Category labels belong to ``TreeGrouper``.

.. automodule:: arfs.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

arfs.sampling module
--------------------

.. automodule:: arfs.sampling
   :members:
   :undoc-members:
   :show-inheritance:

arfs.utils module
-----------------

.. automodule:: arfs.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: arfs
   :members:
   :undoc-members:
   :show-inheritance:
