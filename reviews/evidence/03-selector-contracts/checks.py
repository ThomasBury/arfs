"""Unit 03 review evidence, not repairs or a release gate.

Strict xfails assert desired contracts. --runxfail exposes the actual failures.
Controlled importance checks replace only the training boundary, explicitly;
real_* checks train the installed LightGBM and preserve the installed ARFS code.
"""

import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline

import arfs.feature_selection as fs
import arfs.feature_selection.variable_importance as vi
from arfs.feature_selection.base import BaseThresholdSelector
from arfs.feature_selection.unsupervised import _recursive_collinear_elimination
from arfs.preprocessing import OrdinalEncoderPandas

PANDAS3 = int(pd.__version__.split('.')[0]) >= 3
HAS_JINJA = importlib.util.find_spec('jinja2') is not None
PARAMS = dict(objective='rmse', metric='rmse', n_estimators=6, num_leaves=4,
              min_data_in_leaf=5, num_threads=1, verbosity=-1)
THRESHOLD_CLASSES = [fs.MissingValueThreshold, fs.UniqueValuesThreshold, fs.CardinalityThreshold]
ALL_CLASSES = THRESHOLD_CLASSES + [fs.CollinearityThreshold, fs.VariableImportance]


def defect(number):
    return pytest.mark.xfail(strict=True, reason=number)


def numeric_data():
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=['a', 'b', 'c'],
                     index=np.arange(120) + 1000)
    y = pd.Series(4 * X.a + 0.3 * X.b, index=X.index)
    return X, y


def importance_selector(**kwargs):
    options = dict(encode=False, n_iterations=1, lgb_kwargs=PARAMS.copy())
    options.update(kwargs)
    return fs.VariableImportance(**options)


def controlled_backend(monkeypatch, contributions):
    """Supply declared SHAP arrays, leaving aggregation/selection unmodified."""
    calls = []

    class TrainingDouble:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, sample_weight=None):
            matrix = np.asarray(contributions[len(calls) % len(contributions)], dtype=float)
            self.valid_features = X.iloc[:len(matrix)]
            self.model = SimpleNamespace(predict=lambda data, pred_contrib: matrix.copy())
            calls.append(dict(X=X.copy(), y=y, sample_weight=sample_weight, kwargs=self.kwargs))

    monkeypatch.setattr(vi, 'GradientBoosting', TrainingDouble)
    return calls


def test_missing_ratio_hand_reference_infinities_and_nullable():
    X = pd.DataFrame({'none': [1., 2., 3., 4.], 'quarter': [np.nan, 1., 2., 3.],
                      'half': [np.inf, -np.inf, 1., 2.], 'all': [np.nan] * 4,
                      'nullable': pd.Series([1, None, 2, 3], dtype='Int64')})
    selector = fs.MissingValueThreshold(0.5).fit(X)
    assert_allclose(selector.statistic_series_, [0., .25, .5, 1., .25])
    assert_array_equal(selector.get_support(), [True, True, False, False, True])
    assert_array_equal(fs.MissingValueThreshold(0).fit(X).get_support(), [False] * 5)
    assert_array_equal(fs.MissingValueThreshold(1).fit(X).get_support(), [True, True, True, False, True])


def test_unique_count_boundaries_exclude_missing_values():
    X = pd.DataFrame({'empty': [np.nan]*4, 'one': [1]*4, 'two': [1,2,1,2], 'three': [1,2,3,1]})
    selector = fs.UniqueValuesThreshold(2).fit(X)
    assert_array_equal(selector.statistic_series_, [0,1,2,3])
    assert_array_equal(selector.selected_features_, ['three'])
    assert_array_equal(fs.UniqueValuesThreshold(1).fit(X).selected_features_, ['two','three'])


def test_cardinality_hand_reference_and_equality():
    X = pd.DataFrame({'one': pd.Categorical(['a']*4), 'two': pd.Categorical(['a','b','a','b']),
                      'three': pd.Series(['a','b','c',None], dtype=object),
                      'boolean': [True,False,True,False], 'numeric': [1,2,3,4],
                      'empty': pd.Categorical([None]*4)})
    selector = fs.CardinalityThreshold(2).fit(X)
    assert_array_equal(selector.statistic_series_, [1,2,3,2,0,0])
    assert_array_equal(selector.selected_features_, ['one','numeric','empty'])


@pytest.mark.parametrize('cls', THRESHOLD_CLASSES)
def test_threshold_fit_transform_metadata_refit_reorder_and_input_preservation(cls):
    X = pd.DataFrame({'a': [1,2,3,4], 'b': [1,1,1,1], 'cat': pd.Categorical(['x','y','x','y'])}, index=[4,8,9,20])
    before = X.copy(deep=True)
    selector = clone(cls())
    out = selector.fit_transform(X)
    assert_frame_equal(out, cls().fit(X).transform(X))
    assert_array_equal(out.columns, selector.get_feature_names_out())
    assert_array_equal(np.flatnonzero(selector.support_), selector.get_support(indices=True))
    assert_frame_equal(selector.transform(X[X.columns[::-1]]), out)
    assert_frame_equal(X, before)
    missing_selected = X.drop(columns=selector.selected_features_[0])
    with pytest.raises(KeyError):
        selector.transform(missing_selected)
    renamed = X.rename(columns={c: f'new_{c}' for c in X.columns})
    selector.fit(renamed)
    assert_array_equal(selector.feature_names_in_, renamed.columns)
    assert_array_equal(selector.get_feature_names_out(), selector.transform(renamed).columns)


@pytest.mark.parametrize('cls', ALL_CLASSES)
def test_dataframe_only_input_is_enforced(cls):
    with pytest.raises(TypeError, match='dataframe'):
        cls().fit(np.ones((3,2)), np.arange(3))


def test_empty_selection_preserves_rows_and_index():
    X = pd.DataFrame({'a': [1,1,1]}, index=[3,5,8])
    selector = fs.UniqueValuesThreshold()
    out = selector.fit_transform(X)
    assert out.shape == (3,0)
    assert_array_equal(out.index, X.index)
    assert_array_equal(selector.get_feature_names_out(), [])


@defect('F4: StringDtype is not counted as categorical')
def test_cardinality_string_extension_dtype():
    X = pd.DataFrame({'text': pd.Series(['a','b','c'], dtype='string')})
    selector = fs.CardinalityThreshold(2).fit(X)
    assert selector.statistic_series_['text'] == 3
    assert selector.selected_features_.size == 0


def test_cardinality_inferred_strings():
    X = pd.DataFrame({'text': ['a','b','c']})
    assert fs.CardinalityThreshold(2).fit(X).selected_features_.size == 0


@pytest.mark.parametrize('cls', [fs.UniqueValuesThreshold, fs.CardinalityThreshold])
@defect('F5: duplicate names reintroduce or multiply columns')
def test_duplicate_names_either_rejected_or_positionally_selected(cls):
    X = pd.DataFrame([[1,1],[1,2],[1,3]], columns=['a','a'])
    selector = cls()
    try:
        out = selector.fit_transform(X)
    except ValueError:
        return  # Rejecting duplicate input is also safe.
    assert_frame_equal(out, X.iloc[:,selector.get_support()])


@defect('F6: base ignores statistic Series labels')
def test_custom_statistic_aligns_by_documented_feature_index():
    X = pd.DataFrame({'constant': [1,1,1], 'varying': [1,2,3]})
    selector = BaseThresholdSelector(1, lambda frame: frame.nunique().sort_values(ascending=False), True)
    assert_array_equal(selector.fit(X).selected_features_, ['varying'])


@pytest.mark.parametrize('cls,value', [(fs.MissingValueThreshold,-.1), (fs.MissingValueThreshold,1.1),
    (fs.MissingValueThreshold,np.nan), (fs.UniqueValuesThreshold,0), (fs.CardinalityThreshold,0),
    (fs.CollinearityThreshold,np.nan), (fs.CollinearityThreshold,1.1), (fs.VariableImportance,np.nan),
    (fs.VariableImportance,1.1)])
@defect('F7: invalid thresholds accepted at fit including set_params')
def test_invalid_thresholds_rejected_at_fit(cls, value, monkeypatch):
    X,y = numeric_data()
    controlled_backend(monkeypatch, [np.ones((2,4))])
    selector = cls().set_params(threshold=value)
    with pytest.raises(ValueError):
        selector.fit(X,y)


@pytest.mark.parametrize('iterations', [0,-1])
@defect('F7: nonpositive iteration count silently does no training')
def test_nonpositive_iterations_rejected(iterations):
    X,y = numeric_data()
    with pytest.raises(ValueError):
        importance_selector(n_iterations=iterations).fit(X,y)


@pytest.mark.parametrize('cls', ALL_CLASSES)
@defect('F12: transform lacks fitted-state guard')
def test_unfitted_transform_raises_not_fitted(cls):
    X,_ = numeric_data()
    with pytest.raises(NotFittedError):
        cls().transform(X)


@pytest.mark.parametrize('cls', ALL_CLASSES)
def test_unfitted_support_and_names_raise_not_fitted(cls):
    with pytest.raises(NotFittedError):
        cls().get_support()
    with pytest.raises(NotFittedError):
        cls().get_feature_names_out()


@pytest.mark.parametrize('cls', ALL_CLASSES)
@defect('F12: documented n_features_in_ is missing')
def test_fitted_feature_count(cls, monkeypatch):
    X,y = numeric_data()
    controlled_backend(monkeypatch, [np.ones((2,4))])
    assert cls().fit(X,y).n_features_in_ == X.shape[1]


def test_valid_parameter_updates_used_on_refit():
    X = pd.DataFrame({'a':[1,1,1], 'b':[1,2,1], 'c':[1,2,3]})
    selector = fs.UniqueValuesThreshold(1).fit(X)
    assert_array_equal(selector.selected_features_, ['b','c'])
    selector.set_params(threshold=2).fit(X)
    assert_array_equal(selector.selected_features_, ['c'])


def test_collinearity_negative_asymmetric_ties_disconnected_and_equality():
    # b is the unique most connected node; d is disconnected.
    matrix = pd.DataFrame([[0,-.9,.1,0],[-.9,0,.8,0],[.1,.2,0,0],[0,0,0,0]],
                          columns=list('abcd'), index=list('abcd'))
    before = matrix.copy(deep=True)
    assert _recursive_collinear_elimination(matrix,.7) == ['b']
    assert _recursive_collinear_elimination(matrix,.9) == []
    assert_frame_equal(matrix,before)
    tie = pd.DataFrame([[0,-1,0],[-1,0,0],[0,0,0]],index=list('abc'),columns=list('abc'))
    drop = _recursive_collinear_elimination(tie,.8)
    assert len(drop)==1 and drop[0] in ['a','b']


def test_real_collinearity_negative_duplicate_metadata_and_refit():
    X,y = numeric_data()
    X['negative'] = -X.a
    before = X.copy(deep=True)
    selector = fs.CollinearityThreshold(.8)
    out = selector.fit_transform(X)
    assert {'a','negative'}.intersection(out.columns) in [{'a'},{'negative'}]
    assert_array_equal(out.columns, selector.get_feature_names_out())
    assert_frame_equal(out, X.iloc[:,selector.get_support()])
    assert_frame_equal(selector.transform(X[X.columns[::-1]]),out)
    assert_frame_equal(X,before)
    selector.set_params(threshold=1).fit(X)
    assert_array_equal(selector.get_feature_names_out(),X.columns)


@pytest.mark.parametrize('remaining', [0,1])
@defect('F8: no association pairs crashes after earlier selection')
def test_collinearity_handles_fewer_than_two_features(remaining):
    X = pd.DataFrame({'constant':[1,1,1,1], 'varying':[1,2,3,4]})
    threshold = 1 if remaining else 4
    pipe = Pipeline([('unique',fs.UniqueValuesThreshold(threshold)),('collinear',fs.CollinearityThreshold())])
    assert pipe.fit_transform(X).shape == (4,remaining)


@defect('F9: tie results depend on process hash seed')
def test_collinearity_tie_is_reproducible_between_processes():
    code = """import json, pandas as pd
from arfs.feature_selection.unsupervised import _recursive_collinear_elimination
m=pd.DataFrame([[0,1,0],[1,0,0],[0,0,0]],index=['a','b','c'],columns=['a','b','c'])
print(json.dumps(_recursive_collinear_elimination(m,.8)))
"""
    outcomes={seed: json.loads(subprocess.check_output([sys.executable,'-c',code],text=True,
        env={**os.environ,'PYTHONHASHSEED':str(seed)},stderr=subprocess.DEVNULL).splitlines()[-1])
        for seed in [0,1,2,3]}
    assert len({tuple(value) for value in outcomes.values()}) == 1, outcomes


def test_controlled_signed_contributions_bias_exclusion_and_iteration_average(monkeypatch):
    X,y = numeric_data()
    controlled_backend(monkeypatch, [ [[1,-2,0,900],[-3,4,0,-900]],
                                     [[-2,0,6,1000],[2,-2,-2,1000]] ])
    result=vi._compute_varimp_lgb(X,y,encode=False,n_iterations=2,fastshap=False).set_index('feature').loc[X.columns]
    assert_allclose(result.importance,[2,2,2])
    assert_allclose(result.normalized_importance,[1/3]*3)
    assert_allclose(sorted(result.cumulative_importance),[1/3,2/3,1])


def test_controlled_multiclass_aggregation_reference(monkeypatch):
    X,y = numeric_data()
    y = pd.Series(np.tile([0,1,2],40))
    blocks=np.array([[[1,-2,3,100],[2,3,-4,200],[-3,4,5,300]],
                     [[-1,2,-3,400],[-2,-3,4,500],[3,-4,-5,600]]])
    controlled_backend(monkeypatch,[blocks.reshape(2,-1)])
    result=vi._compute_varimp_lgb(X,y,task='multiclass',n_iterations=1,fastshap=False).set_index('feature').loc[X.columns]
    assert_allclose(result.importance,[6,9,12])
    assert_allclose(result.normalized_importance,[2/9,3/9,4/9])


@pytest.mark.parametrize('threshold,expected',[(.5,['a']),(.9,['a','b']),(1.,['a','b','c'])])
@defect('F1: first feature reaching cumulative threshold is discarded')
def test_cumulative_threshold_includes_crossing_feature(monkeypatch, threshold, expected):
    X,y=numeric_data()
    controlled_backend(monkeypatch,[[[.6,.3,.1,100]]])
    selector=importance_selector(threshold=threshold).fit(X,y)
    assert_array_equal(selector.selected_features_,expected)


@defect('F2: all-zero importance normalizes to NaN and keeps everything')
def test_real_zero_importance_is_not_selected():
    X,y=numeric_data()
    selector=importance_selector().fit(X,y*0)
    assert_array_equal(selector.feature_importances_summary_.importance,[0,0,0])
    assert selector.selected_features_.size == 0


@pytest.mark.parametrize('value',[np.nan,np.inf])
@defect('F2: nonfinite importance silently retains invalid predictors')
def test_nonfinite_contributions_rejected(monkeypatch,value):
    X,y=numeric_data()
    controlled_backend(monkeypatch,[[[value,0,0,0]]])
    with pytest.raises(ValueError):
        importance_selector().fit(X,y)


@defect('F1: a real single-signal fit discards its only useful feature')
def test_real_dominant_feature_is_retained():
    X,y=numeric_data()
    X['b']=0.; X['c']=0.; y=4*X.a
    selector=importance_selector().fit(X,y)
    assert selector.feature_importances_summary_.iloc[0].feature == 'a'
    assert selector.feature_importances_summary_.iloc[0].normalized_importance == 1
    assert_array_equal(selector.selected_features_,['a'])


@defect('F3: every iteration trains identical data and model')
def test_real_iterations_actually_change_split_or_model(monkeypatch):
    X,y=numeric_data()
    original=vi.GradientBoosting
    models=[]

    class ObservedTraining(original):
        def fit(self,*args,**kwargs):
            super().fit(*args,**kwargs)
            models.append((tuple(self.valid_features.index),self.model.model_to_string()))

    monkeypatch.setattr(vi,'GradientBoosting',ObservedTraining)
    importance_selector(n_iterations=3).fit(X,y)
    assert len(models)==3
    assert len(set(models))>1, 'three identical validation splits and serialized fitted models'


def test_real_pipeline_weights_categories_and_native_contribution_reference(monkeypatch):
    X,y=numeric_data()
    X['cat']=pd.Categorical(np.tile(['a','b','c'],40))
    w=pd.Series(np.where(X.a>0,20.,1.),index=X.index)
    before=X.copy(deep=True)
    original=vi.GradientBoosting
    observed=[]

    class ObservedTraining(original):
        def fit(self,X,y,sample_weight=None):
            super().fit(X,y,sample_weight=sample_weight)
            observed.append((self,X.copy(),sample_weight.copy()))

    monkeypatch.setattr(vi,'GradientBoosting',ObservedTraining)
    pipe=Pipeline([('missing',fs.MissingValueThreshold()),('collinear',fs.CollinearityThreshold(1)),
                   ('encoder',OrdinalEncoderPandas()),('importance',importance_selector())])
    out=pipe.fit_transform(X,y,collinear__sample_weight=w,importance__sample_weight=w)
    model,encoded,forwarded=observed[0]
    assert_array_equal(forwarded,w)
    train,valid=next(ShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(encoded,y))
    assert_array_equal(model.valid_features.index,encoded.iloc[valid].index)
    # Independent native training uses the known partition and explicit weights.
    import lightgbm as lgb
    train_params={k:v for k,v in PARAMS.items() if k!='n_estimators'}
    reference=lgb.train(train_params,lgb.Dataset(encoded.iloc[train],label=y.iloc[train],weight=w.iloc[train]),
                        num_boost_round=PARAMS['n_estimators'],
                        valid_sets=[lgb.Dataset(encoded.iloc[valid],label=y.iloc[valid],weight=w.iloc[valid])],
                        callbacks=[lgb.early_stopping(10,verbose=False)])
    native=np.abs(reference.predict(encoded.iloc[valid],pred_contrib=True)[:,:-1]).mean(axis=0)
    summary=pipe[-1].feature_importances_summary_.set_index('feature').loc[encoded.columns]
    assert_allclose(summary.importance,native,rtol=1e-6,atol=1e-8)
    unweighted=original(params=PARAMS.copy(),return_valid_features=True)
    unweighted.fit(encoded,y)
    other=np.abs(unweighted.model.predict(unweighted.valid_features,pred_contrib=True)[:,:-1]).mean(axis=0)
    assert not np.allclose(native,other), 'fixture must detect ignored weights'
    assert_frame_equal(X,before)
    assert_array_equal(out.index,X.index)
    assert_array_equal(out.columns,pipe[-1].get_feature_names_out())
    assert_frame_equal(out,encoded[out.columns])


@pytest.mark.parametrize('task,objective', [('regression','rmse'),('classification','binary'),('multiclass','multiclass')])
def test_real_task_fit_with_consistent_objective_and_series(task,objective):
    X,y=numeric_data()
    if task=='classification': y=pd.Series((X.a>0).astype(int),index=X.index)
    if task=='multiclass': y=pd.Series(np.tile([0,1,2],40),index=X.index)
    params={**PARAMS,'objective':objective,'metric':{'rmse':'rmse','binary':'binary_logloss','multiclass':'multi_logloss'}[objective]}
    if task=='multiclass':params['num_class']=3
    selector=importance_selector(task=task,lgb_kwargs=params).fit(X,y)
    assert np.isfinite(selector.feature_importances_summary_.importance).all()
    assert_array_equal(selector.transform(X).columns,selector.get_feature_names_out())


@defect('F11: multiclass path calls fillna on ndarray labels')
def test_real_multiclass_accepts_array_target():
    X,y=numeric_data()
    params={**PARAMS,'objective':'multiclass','metric':'multi_logloss','num_class':3}
    importance_selector(task='multiclass',lgb_kwargs=params).fit(X,np.tile([0,1,2],40))


def test_importance_rejects_missing_target_and_unknown_task():
    X,y=numeric_data()
    with pytest.raises(ValueError,match='No training labels'):
        importance_selector().fit(X,None)
    with pytest.raises(ValueError,match='Task'):
        importance_selector(task='unknown').fit(X,y)


@defect('F13: optional fasttreeshap import error is constructed but not raised')
def test_missing_fasttreeshap_reports_import_error(monkeypatch):
    X,y=numeric_data()
    monkeypatch.setitem(sys.modules,'fasttreeshap',None)
    controlled_backend(monkeypatch,[[[1,2,3,0]]])
    with pytest.raises(ImportError,match='fasttreeshap'):
        importance_selector(fastshap=True).fit(X,y)


@pytest.mark.xfail(not HAS_JINJA or PANDAS3, strict=True,
                   reason='F14: undeclared Jinja2 dependency' if not HAS_JINJA else 'F10: Styler.applymap was removed')
def test_summary_real_pipeline_name_preserving_transformer_and_empty_selection():
    X=pd.DataFrame({'missing':[np.nan]*4,'constant':[1]*4,'varying':[1,2,3,4],
                    'cat':pd.Categorical(['a','b','a','b'])})
    pipe=Pipeline([('missing',fs.MissingValueThreshold(.5)),('unique',fs.UniqueValuesThreshold()),
                   ('encoder',OrdinalEncoderPandas()),('last',fs.UniqueValuesThreshold(10))])
    assert pipe.fit_transform(X).shape==(4,0)
    style=fs.make_fs_summary(pipe)
    expected=pd.DataFrame({'predictor':X.columns,'missing':[0.,1,1,1],
                           'unique':[np.nan,0,1,1],'encoder':[np.nan]*4,
                           'last':[np.nan,np.nan,0,0]})
    assert_frame_equal(style.data,expected,check_dtype=False)
    html=style.to_html()
    assert all(color in html for color in ['#ba0202','#0c8a30','#f57505'])


@defect('Inherited unit 0: BaseThresholdSelector declared but not imported')
def test_every_public_export_exists():
    missing=[name for name in fs.__all__ if not hasattr(fs,name)]
    assert not missing,missing


def test_real_pipeline_collinearity_receives_weights():
    X=pd.DataFrame({'a':[0.,1,2,3], 'b':[0.,2,1,3]})
    w=np.array([20.,1,1,20])
    pipe=Pipeline([('unique',fs.UniqueValuesThreshold()),('collinear',fs.CollinearityThreshold(.9))])
    unweighted=clone(pipe).fit_transform(X)
    weighted=pipe.fit_transform(X,collinear__sample_weight=w)
    assert unweighted.shape[1]==2
    assert weighted.shape[1]==1
    assert_frame_equal(pipe[-1].assoc_matrix_,fs.CollinearityThreshold(.9).fit(X,sample_weight=w).assoc_matrix_)


def test_threshold_statistics_are_currently_unweighted():
    X=pd.DataFrame({'a':[np.nan,1,2,3], 'b':[1,2,3,4]})
    for cls in THRESHOLD_CLASSES:
        weighted=cls().fit(X,sample_weight=np.array([100.,1,1,1]))
        unweighted=cls().fit(X)
        assert_allclose(weighted.statistic_series_,unweighted.statistic_series_)


@pytest.mark.parametrize('encode', [True,False])
def test_real_importance_categorical_input_refit_and_parameter_preservation(encode):
    X,y=numeric_data()
    X['cat']=pd.Categorical(np.tile(['a','b','c'],40))
    before=X.copy(deep=True)
    params=PARAMS.copy()
    selector=importance_selector(encode=encode,lgb_kwargs=params)
    selector.fit(X,y)
    out=selector.transform(X)
    assert_frame_equal(out,X.iloc[:,selector.get_support()])
    assert_array_equal(out.columns,selector.get_feature_names_out())
    assert_frame_equal(selector.transform(X[X.columns[::-1]]),out)
    renamed=X.rename(columns={c:f'new_{c}' for c in X.columns})
    selector.set_params(threshold=.7).fit(renamed,y)
    assert_array_equal(selector.feature_names_in_,renamed.columns)
    assert_array_equal(selector.transform(renamed).columns,selector.get_feature_names_out())
    assert_frame_equal(X,before)
    assert params==PARAMS


@defect('F7: invalid method bypasses constructor validation through set_params')
def test_collinearity_invalid_method_rejected_at_fit():
    X,_=numeric_data()
    with pytest.raises(ValueError):
        fs.CollinearityThreshold().set_params(method='typo').fit(X)
