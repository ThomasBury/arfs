# ARFS feature selection

```sh
pip install arfs
```

The first fit uses ARFS 5.0 and pandas 2.x. ARFS 5.0's dtype check fails
with pandas 3.1.1. If you already have ARFS 5.0 from PyPI, run
`pip install "pandas<3"`. If PyPI offers an earlier ARFS release, install
the 5.0 wheel from GitHub with pandas 2.x:

```sh
pip install "pandas<3" https://github.com/ThomasBury/arfs/releases/download/5.0.0/arfs-5.0.0-py3-none-any.whl
```

## First fit with GrootCV

This offline example uses one strong predictor and two noise columns. Split the
rows before fitting so selection sees only the training data.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from arfs.feature_selection import GrootCV

rng = np.random.default_rng(42)
X = pd.DataFrame(rng.normal(size=(240, 3)), columns=["signal", "noise_a", "noise_b"])
y = 5 * X["signal"] + rng.normal(scale=0.1, size=len(X))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

selector = GrootCV(
    objective="rmse",
    n_folds=3,
    n_iter=1,
    n_jobs=1,
    random_state=42,
    lgbm_params={"num_iterations": 40},
).fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(selector.selected_features_.tolist())  # ['signal']
print(X_train_selected.shape, X_test_selected.shape)  # (180, 1) (60, 1)
```

GrootCV uses LightGBM to compare each predictor's importance with shuffled
copies of the predictors. The selected columns depend on the data, the model
and the training split. Use the held-out rows to evaluate a downstream model
after selection.

Start with the [method chooser](https://arfs.readthedocs.io/en/latest/) to pick
a selector for your task, or see the [local documentation workflow](./docs/workflows.rst)
to run the tutorials and build the docs.

## Tutorials

 - [Preprocessing](./docs/notebooks/preprocessing.ipynb)
 - [Basic FS (best before ARFS)](./docs/notebooks/basic_feature_selection.ipynb)
 - [Regression](./docs/notebooks/arfs_regression.ipynb)
 - [Classification](./docs/notebooks/arfs_classification.ipynb)
 - [LASSO and (G)LM feature selection](./docs/notebooks/lasso_feature_selection.ipynb)
 - [Passing custom params](./docs/notebooks/arfs_grootcv_custom_params.ipynb)
 - [Non-normal loss and sample weights](./docs/notebooks/arfs_non_normal_loss_and_sample_weight.ipynb)
 - [ARFS on GPU](./docs/notebooks/arfs_on_GPU.ipynb)
 - [GrootCV native SHAP](./docs/notebooks/arfs_shap_vs_fastshap.ipynb)
 - [Categoricals](./docs/notebooks/issue_categoricals.ipynb)
 - [Collinearity](./docs/notebooks/issue_collinearity.ipynb)
 - [Reducing run time for large data](./docs/notebooks/arfs_large_data_sampling.ipynb)
 - [Comparison to Boruta and BorutaShap](./docs/notebooks/arfs_boruta_borutaShap_comparison.ipynb)
 - [MRmr alternative](./docs/notebooks/mrmr_feature_selection.ipynb)
 - [MRmr vs ARFS](./docs/notebooks/mrmr_fs_VS_arfs.ipynb)

## Method origins

Leshy builds on [Boruta](https://doi.org/10.18637/jss.v036.i11) and
[BorutaPy](https://github.com/scikit-learn-contrib/boruta_py).
BoostAGroota builds on [BoostARoota](https://github.com/chasedehan/BoostARoota).
GrootCV compares LightGBM SHAP contributions across folds.
[Methods overview](./docs/Methods%20overview.rst) describes how ARFS implements
these methods, MRMR and Lasso.
