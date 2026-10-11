"""Review 06 counterexamples. Controlled plot state is not a selection benchmark."""
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest
import scipy
from lightgbm import LGBMRegressor

from arfs import utils as u
from arfs.association import plot_association_matrix, plot_association_matrix_int
from arfs.benchmark import highlight_tick
from arfs.feature_selection import Leshy, BoostAGroota, GrootCV, VariableImportance
from arfs.feature_selection.allrelevant import BLUE


def defect(fid):
    return pytest.mark.xfail(strict=True, reason=fid)


@pytest.fixture(autouse=True)
def restore_state():
    state = np.random.get_state()
    with mpl.rc_context():
        yield
    np.random.set_state(state)
    plt.close('all')


def save(fig, name):
    if os.environ.get('REVIEW_PLOTS'):
        folder = Path(os.environ['REVIEW_PLOTS'])
        folder.mkdir(exist_ok=True)
        fig.savefig(folder / f'{name}.png', dpi=100, bbox_inches='tight')


def importance(threshold=0.9):
    selector = VariableImportance(threshold=threshold)
    selector.feature_importances_summary_ = pd.DataFrame({
        'feature': ['signal', 'second', 'noise'], 'importance': [6., 3., 1.],
        'normalized_importance': [.6, .3, .1], 'cumulative_importance': [.6, .9, 1.]})
    return selector


def test_utility_basics_and_explicit_side_effects():
    frame = pd.DataFrame({'num': [1., 2.], 'cat': ['a', 'b'],
                          'bool': [True, False], 'time': pd.date_range('2020', periods=2),
                          'interval': pd.arrays.IntervalArray.from_breaks([0., 1., 2.])})
    assert u.create_dtype_dict(frame) == {'num': 'num', 'cat': 'cat', 'bool': 'cat',
                                         'time': 'time', 'interval': 'num_interval'}
    with pytest.raises(TypeError): u.create_dtype_dict(frame.to_numpy())
    with pytest.raises(ValueError): u.create_dtype_dict(frame, 'invalid')
    out, names, positions = u.get_pandas_cat_codes(frame)
    assert out is frame  # Documented helper mutation; selector misuse is unit 1B F5.
    assert names == ['cat', 'bool', 'time'] and positions == [1, 2, 3]
    assert u.is_list_of_str(['a']) and u.is_list_of_bool([True])
    assert u.is_list_of_int([1]) and u.is_list_of_int([True])
    assert not u.is_list_of_str(('a',)) and u.is_list_of_str([])
    assert u.concat_or_group('x', pd.DataFrame({'x': ['a', 'a', 'b']})) == 'a / b'
    u.set_my_plt_style(height=4, width=6, linewidth=3)
    assert list(mpl.rcParams['figure.figsize']) == [6, 4]
    u.reset_plot()
    assert mpl.rcParams['lines.linewidth'] == mpl.rcParamsDefault['lines.linewidth']


def test_weight_converter_is_not_numeric_validation():
    for weights in [np.array([-1., 2.]), np.array([np.nan, 1.]),
                    np.array([np.inf, 1.]), np.zeros(2), np.ones((2, 1))]:
        assert u.validate_sample_weight(weights) is weights
    np.testing.assert_equal(u.validate_sample_weight(pd.Series([1., 2.])), [1., 2.])
    assert u.validate_sample_weight(None) is None
    with pytest.raises(ValueError): u.validate_sample_weight([1., 2.])


@defect('F1: stringification merges distinct categories')
@pytest.mark.parametrize('values', [[1, '1'], [np.nan, 'nan'], [None, 'None']])
def test_encoding_preserves_distinct_values(values):
    frame = pd.DataFrame({'category': pd.Series(values, dtype=object)})
    encoded, _, _ = u.get_pandas_cat_codes(frame)
    assert encoded.category.nunique() == 2


@defect('F2: scatter uses subplot column number as feature label')
@pytest.mark.parametrize('columns,ncols', [(['a','b','c','d'], 2), ([0,1,2,3], 2), (['a'], 1), (['a','b'], 2)])
def test_target_plot_uses_each_feature(columns, ncols):
    frame = pd.DataFrame(np.arange(12 * len(columns)).reshape(12, -1), columns=columns)
    fig = u.plot_y_vs_X(frame, np.arange(12), ncols=ncols)
    save(fig, 'target-integer-columns' if isinstance(columns[0], int) else 'target')
    for ax, column in zip(fig.axes, columns):
        np.testing.assert_array_equal(ax.collections[0].get_offsets()[:,0], frame[column])
        assert ax.get_title() == str(column)


@defect('F3: Leshy colors sorted boxes by counts instead of membership')
def test_leshy_plot_colors_match_support():
    selector = Leshy(LGBMRegressor())
    selector.feature_names_in_ = np.array(['steady_selected', 'spiky_rejected'])
    selector.imp_real_hist = np.array([[1., 0.], [1., 10.], [1., 0.]])
    selector.support_ = np.array([True, False])
    selector.support_weak_ = np.array([False, False])
    selector.sha_max = .5
    fig = selector.plot_importance(n_feat_per_inch=.5)
    save(fig, 'leshy-membership')
    colors = dict(zip([t.get_text() for t in fig.axes[0].get_yticklabels()],
                      [p.get_facecolor() for p in fig.axes[0].patches]))
    assert colors['steady_selected'] == to_rgba(BLUE)
    assert colors['spiky_rejected'] == to_rgba('gray')


@defect('F4: valid threshold 1 has no cumulative value strictly above it')
def test_importance_plot_at_threshold_one():
    fig = importance(threshold=1).plot_importance(log=False, figsize=(8, 4))
    fig.canvas.draw()


@defect('F5: plot_n changes height but never limits plotted data')
def test_importance_plot_respects_limit():
    fig = importance().plot_importance(plot_n=2, log=False, figsize=(8, 4))
    save(fig, 'importance-limit-two')
    assert len(fig.axes[0].collections[0].get_offsets()) == 2
    assert len(fig.axes[1].collections[0].get_offsets()) == 2


@defect('F6: default log scale on categorical rank hides the first feature')
def test_importance_cumulative_plot_includes_first_feature():
    fig = importance().plot_importance(figsize=(8,4))
    fig.canvas.draw()
    save(fig, 'importance-default-log')
    ax = fig.axes[1]
    first = ax.collections[0].get_offsets()[0]
    assert ax.get_xlim()[0] <= first[0] <= ax.get_xlim()[1]


def test_importance_plot_changes_global_style():
    mpl.rcParams['lines.linewidth'] = 7
    importance().plot_importance(log=False, figsize=(8,4))
    assert mpl.rcParams['lines.linewidth'] == mpl.rcParamsDefault['lines.linewidth']


@defect('F7: clustering cannot handle a one-feature matrix')
def test_one_feature_association_plot():
    fig, ax = plot_association_matrix(pd.DataFrame([[1.]], columns=['a'], index=['a']), show=False)
    fig.canvas.draw()


@pytest.mark.xfail(scipy.__version__ == "1.18.1", strict=True, reason="F11: hierarchy.distance is not a public SciPy API")
def test_association_sign_order_annotations_and_owner():
    frame = pd.DataFrame([[1., -.8, .2], [-.8, 1., .4], [.2, .4, 1.]],
                         index=['a','b','c'], columns=['a','b','c'])
    before = frame.copy()
    fig, ax = plt.subplots(figsize=(5,4))
    result, target_ax = plot_association_matrix(frame, ax=ax, show=False, annotate=True)
    assert result is fig and target_ax is ax
    labels = [t.get_text() for t in ax.get_xticklabels()]
    np.testing.assert_array_equal(ax.images[0].get_array(), frame.loc[labels,labels])
    assert ax.images[0].get_clim() == (-1, 1)
    assert '-0.8' in [t.get_text().replace('−', '-') for t in ax.texts]
    pd.testing.assert_frame_equal(before, frame)
    fig.sca(ax)  # highlight_tick intentionally uses the figure's current axes.
    highlight_tick('b', fig, axis='x')
    assert next(t for t in ax.get_xticklabels() if t.get_text()=='b').get_color() == 'red'
    with pytest.raises(ValueError): highlight_tick('b', fig, axis='bad')
    save(fig, 'association-signed')
    plt.close(fig)
    assert not plt.get_fignums()


@pytest.mark.xfail(scipy.__version__ == "1.18.1", strict=True, reason="F11: hierarchy.distance is not a public SciPy API")
def test_empty_association_rejected():
    with pytest.raises(ValueError): plot_association_matrix(pd.DataFrame(), show=False)


def test_optional_backend_missing():
    with pytest.raises(ImportError, match='Holoviews'):
        plot_association_matrix_int(pd.DataFrame(np.eye(2)))


def test_other_selector_plot_membership_and_empty_histories():
    selector = GrootCV()
    selector.sha_cutoff = .15
    selector.selected_features_ = np.array(['selected_low'])
    selector.cv_df = pd.DataFrame({'feature': ['selected_low','rejected_high','ShadowVar1','ShadowVar2'],
                                  'f0': [.1,.8,.2,.3], 'f1': [.2,.9,.3,.2], 'Med': [.15,.85,.25,.25]})
    fig = selector.plot_importance(n_feat_per_inch=.5)
    colors = dict(zip([t.get_text() for t in fig.axes[0].get_yticklabels()],
                      [p.get_facecolor() for p in fig.axes[0].patches]))
    assert colors['selected_low'] == to_rgba(BLUE)
    assert colors['rejected_high'] == to_rgba('gray')
    save(fig, 'grootcv-membership')
    selector = BoostAGroota(estimator=LGBMRegressor())
    selector.mean_shadow = .25
    selector.selected_features_ = np.array(['signal'])
    selector.sha_cutoff_df = pd.DataFrame({'noise':[.1,.2], 'signal':[.8,.9],
                                          'ShadowVar1':[.2,.3], 'ShadowVar2':[.3,.2]})
    fig = selector.plot_importance(n_feat_per_inch=.5)
    assert fig.axes[0].patches[1].get_facecolor() == to_rgba(BLUE)
    save(fig, 'boostagroota-ordinary')
    selector.sha_cutoff_df = selector.sha_cutoff_df.iloc[:0]
    with pytest.warns(UserWarning): assert selector.plot_importance() is None
    selector = Leshy(LGBMRegressor())
    selector.imp_real_hist = np.empty((0,1))
    selector.feature_names_in_ = ['x']
    with pytest.warns(UserWarning): assert selector.plot_importance() is None


@defect('F8: generators reset the global NumPy RNG')
@pytest.mark.parametrize('generator', [u._make_corr_dataset_regression, u._make_corr_dataset_classification])
def test_generator_does_not_reset_global_rng(generator):
    np.random.seed(19)
    generator(size=20)
    after19 = np.random.random(5)
    np.random.seed(83)
    generator(size=20)
    after83 = np.random.random(5)
    assert not np.array_equal(after19, after83)


def test_generator_repeatability_weights_and_signals():
    for generator in [u._make_corr_dataset_regression, u._make_corr_dataset_classification]:
        np.random.seed(11)
        x1,y1,w1 = generator(size=1000)
        np.random.seed(12)
        x2,y2,w2 = generator(size=1000)
        pd.testing.assert_frame_equal(x1,x2)
        np.testing.assert_array_equal(y1,y2)
        assert not np.array_equal(w1,w2)
        assert len(x1)==len(y1)==len(w1)==1000
        assert np.isfinite(w1).all() and (w1>0).all()
        assert x1.var10.nunique()==1 and x1.var12.isna().sum()==500
        assert np.corrcoef(x1.var0,y1)[0,1]>.7
        assert np.corrcoef(x1.var2,y1)[0,1]<-.4
        assert abs(np.corrcoef(x1.var5,y1)[0,1])<.12
        assert x1.shape[1] == (14 if generator is u._make_corr_dataset_regression else 13)


@defect('F9: regression high-cardinality column has size minus one for odd sizes')
def test_odd_size_regression():
    x,y,w = u._make_corr_dataset_regression(size=11)
    assert len(x)==len(y)==len(w)==11


@defect('F10: documented NumPy input has no values attribute')
def test_validate_pandas_input_accepts_array():
    x=np.arange(4).reshape(2,2)
    np.testing.assert_array_equal(u.validate_pandas_input(x),x)


def test_local_datasets_and_installed_housing_blocker():
    cancer = u.load_data('cancer')
    assert cancer.data.shape==(569,33) and cancer.target.shape==(569,)
    assert cancer.data.index.equals(cancer.target.index)
    assert set(cancer.target.unique())=={0,1}
    assert cancer.sample_weight is None
    boston = u.load_data('Boston')  # Trusted repository resource; hash checked by environment.py.
    assert len(boston.data)==len(boston.target)==506
    assert not boston.data.columns.duplicated().any()
    assert np.isfinite(boston.target).all()
    assert set(boston.categorical).issubset(boston.data.columns)
    with pytest.raises(FileNotFoundError, match='housing.rst'): u.load_data('housing')
    with pytest.raises(ValueError): u.load_data('not-a-dataset')
