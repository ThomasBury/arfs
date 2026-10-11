import json
import numpy as np
import pandas as pd
from arfs.feature_selection.allrelevant import _create_shadow
from arfs.feature_selection import UniqueValuesThreshold, CardinalityThreshold
from arfs.utils import _make_corr_dataset_classification

result={'pandas':pd.__version__, 'numpy':np.__version__, 'shadow':{}}
np.random.seed(20261010)
x=pd.DataFrame({'a':[1.0,2.0,3.0], 'b':[4.0,5.0,6.0]})
original=x.copy(deep=True)
for mode in ('default','copy_on_write'):
    def run():
        try:
            augmented,names=_create_shadow(x)
            assert augmented.shape==(3,4)
            return {'result':'OK', 'shadow_names':names}
        except ValueError as error:
            return {'error':str(error)}
    if mode=='default': result['shadow'][mode]=run()
    else:
        with pd.option_context('mode.copy_on_write',True): result['shadow'][mode]=run()
    pd.testing.assert_frame_equal(x,original)
x,_,_=_make_corr_dataset_classification(size=10)
selector=UniqueValuesThreshold(threshold=2).fit(x)
result['unique']={'counts':x.nunique().to_dict(), 'rejected':selector.not_selected_features_.tolist(), 'rejected_at_threshold_1':UniqueValuesThreshold(threshold=1).fit(x).not_selected_features_.tolist()}
x,_,_=_make_corr_dataset_classification(size=100)
result['cardinality']={'columns':list(x.columns), 'nice_guys_present':'nice_guys' in x, 'rejected':CardinalityThreshold(threshold=5).fit(x).not_selected_features_.tolist()}
assert result['cardinality']['nice_guys_present'] is False
assert len(result['unique']['rejected'])>1
print(json.dumps(result,indent=2))
