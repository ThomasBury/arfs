Introduction
============

All-relevant feature selection aims to find predictors that carry information
about the target, including redundant predictors. Leshy, BoostAGroota and
GrootCV compare model importances with those of shuffled predictors, called
shadow features. These empirical comparisons do not establish calibrated
false-discovery control.

Leshy and BoostAGroota use a compatible tree estimator and an importance
method it supports. GrootCV uses LightGBM and native SHAP contributions. The
data types, encodings, estimator, observation weights and rows used for fitting
or validation can all affect the result.

For supervised preprocessing, fit ``TreeGrouper`` on categorical predictors and
``TreeDiscretizer`` on numeric predictors inside each training split. Forward
observation weights to both steps when applicable. Reuse the fitted steps on
held-out rows. See :doc:`Methods overview` for the selection rules and :doc:`arfs`
for preprocessing interfaces.
