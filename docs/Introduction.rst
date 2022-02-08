Introduction
============

What is ARFS ?
--------------

ARFS is a package providing three methods to perform all-relevant feature selection.
All relevant feature selection means trying to find all features carrying information usable for prediction, 
rather than finding a possibly compact subset of features on which some particular model has a minimal error. 
This might include redundant predictors. 

All relevant feature selection is model agnostic in the sense that it doesn't optimize a scoring function for 
a specific model but rather tries to select all the predictors which are related to the response.

This package implements 3 different methods (Leshy is an evolution of Boruta, 
BoostAGroota is an evolution of BoostARoota and GrootCV is a new one). 
They are sklearn compatible. See hereunder for details about those methods. 
You can use any sklearn compatible estimator with Leshy and BoostAGroota but I recommend lightGBM. 
It's fast, accurate and has SHAP values builtin.

Installation
------------

.. code:: shell

   pip install arfs -U

Disclaimer
----------

The package is provided "as-is" and there is NO WARRANTY of any kind. 
Use it only if the content and output files make sense to you.

The Leshy class is heavily based on the BorutaPy port of Boruta.
Leshy might be merged with BorutaPy, if the authors would like to
