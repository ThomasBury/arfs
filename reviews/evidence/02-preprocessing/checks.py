"""Preprocessing review evidence. Strict xfails record defects, not release success.

Run unchanged against either unit 0 installed-wheel environment. No ARFS patches.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from patsy import PatsyError, dmatrix
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline

from arfs.preprocessing import (
    IntervalToMidpoint,
    OrdinalEncoderPandas,
    PatsyTransformer,
    TreeDiscretizer,
    cat_var,
    dtype_column_selector,
    find_interval_midpoint,
    transform_interval_to_midpoint,
)

PANDAS3 = int(pd.__version__.split('.')[0]) >= 3
PARAMS = dict(num_threads=1, min_data_in_leaf=5, learning_rate=1, verbosity=-1, seed=8)


def numeric_data():
    X = pd.DataFrame({'x': np.arange(240, dtype=float)}, index=np.arange(240) + 1000)
    return X, np.repeat([0., 2., -1., 3., 0., 2.], 40)


def categorical_data():
    return pd.DataFrame({'cat': np.tile(['a', 'b', 'c', 'd'], 60)}), np.tile([0., 1., 2., 3.], 60)


def discretizer(**kwargs):
    return TreeDiscretizer(n_bins=6, boost_params=PARAMS.copy(), **kwargs)


def interval_data():
    return pd.DataFrame({'interval': pd.arrays.IntervalArray.from_tuples([(0., 2.), (2., 4.)]),
                         'num': [1, 2]}, index=[10, 20])


def without_patsy_error(call):
    # Python 3.12.7 traceback suggestions crash on Patsy's evaluation namespace.
    error = None
    result = None
    try:
        result = call()
    except PatsyError as exc:
        error = str(exc)
    assert error is None, error
    return result


def test_ordinal_codes_unknown_missing_index_and_passthrough():
    X = pd.DataFrame({'cat': ['b', 'a', np.nan], 'num': [1., 2., 3.]}, index=[10, 20, 30])
    before = X.copy(deep=True)
    encoder = OrdinalEncoderPandas()
    actual = encoder.fit_transform(X)
    assert_allclose(actual.cat, [1, 0, np.nan], equal_nan=True)
    assert_array_equal(actual.index, X.index)
    assert_array_equal(actual.num, X.num)
    assert_frame_equal(X, before)
    assert np.isnan(encoder.transform(pd.DataFrame({'cat': ['new'], 'num': [4.]})).cat.iloc[0])


def test_ordinal_no_categories_is_passthrough():
    X, _ = numeric_data()
    assert_frame_equal(OrdinalEncoderPandas().fit_transform(X), X)


def test_dtype_selector_combines_filters():
    X = pd.DataFrame({'cat_a': ['a'], 'cat_b': ['b'], 'cat_num': [1.], 'other': ['c']})
    assert dtype_column_selector('^cat', dtype_include=['object'], exclude_cols=['cat_b'])(X) == ['cat_a']


def test_ordinal_clone_and_inverse_values():
    X, _ = categorical_data()
    encoder = clone(OrdinalEncoderPandas())
    encoded = encoder.fit_transform(X)
    assert_frame_equal(encoder.inverse_transform(encoded.copy()), X)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F8: metadata omits passthrough columns')
def test_ordinal_feature_names_describe_full_output():
    X = pd.DataFrame({'cat': ['a', 'b'], 'num': [1., 2.]})
    encoder = OrdinalEncoderPandas().fit(X)
    assert_array_equal(encoder.get_feature_names_out(), encoder.transform(X).columns)


@pytest.mark.xfail(strict=True, raises=ValueError, reason='F8: pandas output uses incomplete feature names')
def test_ordinal_pandas_output_mode():
    X = pd.DataFrame({'cat': ['a', 'b'], 'num': [1., 2.]})
    actual = OrdinalEncoderPandas().set_output(transform='pandas').fit_transform(X)
    assert actual.shape == X.shape


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F13: inverse transformation changes caller data')
def test_ordinal_inverse_preserves_input():
    X, _ = categorical_data()
    encoder = OrdinalEncoderPandas().fit(X)
    encoded = encoder.transform(X)
    before = encoded.copy(deep=True)
    encoder.inverse_transform(encoded)
    assert_frame_equal(encoded, before)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F14: output_dtype update does not update parent dtype')
def test_ordinal_set_params_dtype():
    X, _ = categorical_data()
    actual = OrdinalEncoderPandas().set_params(output_dtype=np.float32).fit_transform(X)
    assert actual.cat.dtype == np.dtype('float32')


def test_tree_weighted_fit_matches_native_lightgbm():
    X, y = numeric_data()
    w = np.linspace(.2, 3, len(X))
    before = X.copy(deep=True)
    actual = discretizer(raw=True).fit(X, y, w)
    train, _ = next(ShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X))
    params = dict(PARAMS, objective='rmse', metric='rmse', num_leaves=6)
    native = lgb.train(params, lgb.Dataset(X.iloc[train], label=y[train], weight=w[train]), num_boost_round=1)
    assert_allclose(actual.transform(X).x, native.predict(X), atol=1e-12)
    assert_frame_equal(X, before)
    assert_array_equal(actual.transform(X).index, X.index)
    unweighted = discretizer(raw=True).fit(X, y)
    assert not np.allclose(actual.transform(X).x, unweighted.transform(X).x)


def test_tree_categorical_missing_unknown_and_input_preservation():
    X, y = categorical_data()
    X.loc[0, 'cat'] = np.nan
    before = X.copy(deep=True)
    model = discretizer().fit(X, y)
    query = pd.DataFrame({'cat': ['a', np.nan, 'new']}, index=[10, 20, 30])
    result = model.transform(query)
    assert result.cat.notna().all()
    assert_array_equal(result.index, query.index)
    assert_frame_equal(X, before)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F1: response grouping merges separate numerical leaves')
@pytest.mark.parametrize('task', ['regression', 'classification'])
def test_tree_bins_preserve_fitted_leaf_partition(task):
    X, y = numeric_data()
    if task == 'classification':
        y = np.tile(np.r_[np.zeros(36), np.ones(4), np.zeros(28), np.ones(12),
                         np.zeros(12), np.ones(28)], 2)
    model = discretizer(task=task).fit(X, y)
    leaves = model.tree_dic['x'].model.predict(X, pred_leaf=True).ravel()
    bins = model.transform(X).x.astype(str).to_numpy()
    assert len(np.unique(leaves)) == 6
    # Compare partitions without assuming a particular naming/order of bins.
    assert_array_equal(leaves[:, None] == leaves[None, :], bins[:, None] == bins[None, :])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F1: observed maxima replace actual numerical split thresholds')
def test_tree_bins_preserve_leaf_between_observed_values():
    X, y = numeric_data()
    model = discretizer().fit(X, y)
    query = pd.DataFrame({'x': [120., 120.25]})
    leaves = model.tree_dic['x'].model.predict(query, pred_leaf=True).ravel()
    assert leaves[0] == leaves[1]
    result = model.transform(query)
    assert result.x.iloc[0] == result.x.iloc[1]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F2: temporary x_g column overwrites real predictor')
def test_tree_generated_column_name_cannot_change_another_feature_fit():
    X, y = numeric_data()
    X['x_g'] = np.tile(np.arange(40, dtype=float), 6)
    together = discretizer().fit(X, y)
    alone = discretizer().fit(X[['x_g']], y)
    assert_allclose(together.tree_dic['x_g'].model.predict(X[['x_g']]),
                    alone.tree_dic['x_g'].model.predict(X[['x_g']]), atol=1e-12)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F4: n_bins set_params does not rebuild training parameters')
def test_tree_updated_bin_limit_is_respected():
    X, y = numeric_data()
    model = discretizer().set_params(n_bins=2).fit(X, y)
    assert model.tree_dic['x'].model.dump_model()['tree_info'][0]['num_leaves'] <= 2


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F4: task set_params leaves old objective')
def test_tree_updated_task_is_respected():
    X, _ = numeric_data()
    model = discretizer().set_params(task='classification').fit(X, np.tile([0, 1], 120))
    assert model.tree_dic['x'].model.params['objective'] == 'binary'


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F4: n_estimators overrides one-tree constraint')
def test_tree_single_tree_contract_survives_boost_params():
    X, y = numeric_data()
    model = TreeDiscretizer(n_bins=6, boost_params=dict(PARAMS, n_estimators=3)).fit(X, y)
    assert model.tree_dic['x'].model.num_trees() == 1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F4: constructor mutates caller parameter dictionary')
def test_tree_constructor_preserves_parameter_dictionary():
    params = PARAMS.copy()
    TreeDiscretizer(boost_params=params)
    assert params == PARAMS


@pytest.mark.xfail(strict=True, raises=KeyError, reason='F6: fit replaces all selector with previous column names')
def test_tree_refit_rediscovers_columns():
    X, y = numeric_data()
    model = discretizer().fit(X, y)
    new = X.rename(columns={'x': 'z'})
    model.fit(new, y)
    assert_frame_equal(model.transform(new), discretizer().fit_transform(new, y))


@pytest.mark.xfail(strict=True, raises=IndexError, reason='F7: array conversion only happens during fit')
def test_tree_array_fit_transform():
    X, y = numeric_data()
    result = discretizer().fit_transform(X.to_numpy(), y)
    assert result.shape == X.shape


@pytest.mark.xfail(strict=True, raises=ValueError, reason='F5: raw mode skips fitted categorical encoder')
def test_tree_raw_categorical_transform():
    X, y = categorical_data()
    model = discretizer(raw=True).fit(X, y)
    assert np.isfinite(model.transform(X).cat).all()


@pytest.mark.xfail(strict=True, raises=ValueError, reason='F9: fixed missing category collides with real category')
def test_tree_existing_missing_sentinel_category():
    X, y = categorical_data()
    X = X.replace('a', 'missing_added')
    assert discretizer().fit_transform(X, y).cat.notna().all()


@pytest.mark.xfail(strict=True, raises=ValueError, reason='F9: all-missing numeric feature has no imputation candidate')
def test_tree_all_missing_numeric_feature():
    X, y = numeric_data()
    X['x'] = np.nan
    assert discretizer().fit_transform(X, y).shape == X.shape


def test_interval_midpoint_reference_includes_one_sided_infinite_and_missing():
    values = pd.Series(pd.arrays.IntervalArray.from_tuples(
        [(-np.inf, 2.), (2., 6.), (6., np.inf), (np.nan, np.nan)]))
    assert_allclose(find_interval_midpoint(values), [2., 4., 6., np.nan], equal_nan=True)


@pytest.mark.xfail(PANDAS3, strict=True, raises=TypeError, reason='F3: dtype-changing loc assignment fails on pandas 3')
@pytest.mark.parametrize('api', ['estimator', 'function'])
def test_interval_public_midpoints(api):
    X = interval_data()
    before = X.copy(deep=True)
    actual = IntervalToMidpoint().fit_transform(X) if api == 'estimator' else transform_interval_to_midpoint(X)
    assert_allclose(actual.interval, [1., 3.])
    assert_array_equal(actual.index, X.index)
    assert_frame_equal(X, before)


@pytest.mark.xfail(PANDAS3, strict=True, raises=TypeError, reason='F3: notebook tree-to-midpoint pipeline fails on pandas 3')
def test_tree_midpoint_pipeline():
    X, y = numeric_data()
    result = Pipeline([('tree', discretizer()), ('midpoint', IntervalToMidpoint())]).fit_transform(X, y)
    assert result.shape == X.shape
    assert np.isfinite(result.x).all()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F12: scalar column name is treated as individual characters')
@pytest.mark.parametrize('api', ['estimator', 'function'])
def test_interval_single_string_column(api):
    X = interval_data()
    actual = IntervalToMidpoint(cols='interval').fit_transform(X) if api == 'estimator' else transform_interval_to_midpoint(X, cols='interval')
    assert pd.api.types.is_float_dtype(actual.interval)


@pytest.mark.xfail(strict=True, raises=ValueError, reason='F6: fitted cols parameter is a pandas Index; second fit fails')
def test_interval_refit():
    X = interval_data()
    model = IntervalToMidpoint().fit(X)
    model.fit(X)
    assert model.columns_to_transform_ == ['interval']


def test_patsy_formula_matches_native_and_retains_fitted_center():
    X = pd.DataFrame({'x': [1., 2., 3.], 'cat': ['a', 'b', 'a']}, index=[10, 20, 30])
    model = PatsyTransformer('center(x) + C(cat)')
    actual = model.fit_transform(X)
    reference = dmatrix('center(x) + C(cat)', X, return_type='dataframe')
    assert_frame_equal(actual, reference)
    query = pd.DataFrame({'x': [10.], 'cat': ['b']}, index=[40])
    result = model.transform(query)
    assert result['center(x)'].iloc[0] == 8.
    assert_array_equal(result.index, query.index)
    assert_array_equal(result.columns, actual.columns)


def test_patsy_raise_policy_works_during_fit():
    with pytest.raises(PatsyError, match='missing'):
        PatsyTransformer('x', NA_action='raise').fit(pd.DataFrame({'x': [1., np.nan]}))


@pytest.mark.xfail(strict=True, reason='F10: transform omits NA_action and silently drops rows')
@pytest.mark.parametrize('return_type', ['dataframe', 'ndarray'])
def test_patsy_raise_policy_also_applies_during_transform(return_type):
    model = PatsyTransformer('x', NA_action='raise', return_type=return_type).fit(pd.DataFrame({'x': [1., 2.]}))
    with pytest.raises(PatsyError, match='missing'):
        model.transform(pd.DataFrame({'x': [3., np.nan]}, index=[10, 20]))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F11: add_intercept is never applied')
def test_patsy_add_intercept_false():
    result = PatsyTransformer('x', add_intercept=False).fit_transform(pd.DataFrame({'x': [1., 2.]}))
    assert list(result.columns) == ['x']


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F7: named dictionary columns are renamed to pred_N')
def test_patsy_named_dictionary():
    result = without_patsy_error(lambda: PatsyTransformer('x').fit_transform({'x': [1., 2.]}))
    assert_allclose(result.x, [1., 2.])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F7: array conversion only occurs in fit')
def test_patsy_array_transform():
    X = np.arange(6, dtype=float).reshape(3, 2)
    model = PatsyTransformer().fit(X)
    assert without_patsy_error(lambda: model.transform(X)).shape[0] == len(X)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='F6: inferred formula becomes a fixed constructor parameter')
def test_patsy_refit_rediscovers_automatic_formula():
    model = PatsyTransformer().fit(pd.DataFrame({'x': [1., 2.]}))
    result = without_patsy_error(lambda: model.fit_transform(pd.DataFrame({'z': [3., 4.]})))
    assert 'z' in result.columns


@pytest.mark.xfail(PANDAS3, strict=True, raises=TypeError, reason='F15: cat_var dtype-changing assignment fails on pandas 3')
def test_cat_var_public_roundtrip_without_missing():
    X = pd.DataFrame({'cat': ['b', 'a', 'b']})
    encoded, _, inverse, _ = cat_var(X, return_cat=False)
    assert_array_equal(encoded.cat.map(inverse['cat']), X.cat)


@pytest.mark.xfail(strict=True, raises=(AssertionError, TypeError), reason='F15: missing category aliases first level; pandas 3 fails earlier')
def test_cat_var_missing_is_distinct_from_observed_category():
    result, _, _, _ = cat_var(pd.DataFrame({'cat': ['a', None, 'b']}), return_cat=False)
    assert result.cat.iloc[0] != result.cat.iloc[1]
