"""Inspect trusted local resources and a network-free Titanic loader fixture."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from arfs import utils as u

records={}
for name in ['Boston','cancer']:
    np.random.seed(42)
    data=u.load_data(name)
    records[name]={'keys':list(data),'shape':list(data.data.shape),
                   'columns':list(data.data.columns),'dtypes':{str(k):str(v) for k,v in data.data.dtypes.items()},
                   'target_length':len(data.target),'target_dtype':str(data.target.dtype),
                   'categorical':data.get('categorical'),
                   'sample_weight':None if data.get('sample_weight') is None else {'length':len(data.sample_weight)},
                   'feature_names':list(data.feature_names) if 'feature_names' in data else None,
                   'description':data.get('DESCR'),
                   'missing_count':int(data.data.isna().sum().sum())}

# OpenML is replaced only with this small, fully specified local frame.
frame=pd.DataFrame({'pclass':[1,2,3,1], 'sex':['female','male','female','male'],
                    'embarked':['S','C','Q','S'], 'age':[20.,30.,40.,50.],
                    'parch':[0,0,1,1], 'sibsp':[0,1,0,1], 'fare':[10.,20.,30.,40.],
                    'name':['A, Mrs. One','B, Mr. Two','C, Miss. Three','D, Mr. Four']})
target=pd.Series(pd.Categorical(['1','0','1','0']))
with patch.object(u,'fetch_openml',return_value=(frame.copy(),target)) as fetch:
    result=u._get_titanic_data()
    assert result.data.shape==(4,10)
    assert len(result.target)==len(result.sample_weight)==4
    actual=result.data.is_alone.astype(int).tolist()
    expected=[1,0,0,0]
    records['Titanic_local_fixture']={'remote_download':False,'fetch_arguments':str(fetch.call_args),
                                    'shape':list(result.data.shape),'columns':list(result.data.columns),
                                    'categorical':result.categorical,'is_alone_actual':actual,
                                    'is_alone_expected':expected,'result':'F15: one-relative passengers are labeled alone' if actual!=expected else 'passed'}
    assert actual==[1,1,1,0]  # Pin the observed counterexample, not a repaired loader.
Path(sys.argv[1]).write_text(json.dumps(records,indent=2)+'\n')
