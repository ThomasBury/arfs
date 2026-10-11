"""Bounded notebook excerpts and local resource/build audit; never full notebook runs."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
from urllib.parse import unquote

evidence = Path(__file__).resolve().parent
repo = evidence.parents[2]


def child(name, destination):
    if Path(destination).resolve().is_relative_to(repo):
        raise SystemExit('Rerun results must be outside the checkout to preserve saved evidence')
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    np.random.seed(20261011)
    ns = {'__name__': '__main__'}
    records = []
    if name.startswith('classification'):
        notebook='arfs_classification'
    elif name.startswith('regression'):
        notebook='arfs_regression'
    elif name=='basic': notebook='basic_feature_selection'
    else: notebook='preprocessing'
    cells=json.loads((repo/f'docs/notebooks/{notebook}.ipynb').read_text())['cells']
    def execute(cell, source, changes=None):
        entry={'cell':cell,'source':source,'changes':changes or [],'sha256':hashlib.sha256(source.encode()).hexdigest()}
        capture=io.StringIO()
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                exec(compile(source,f'{notebook}:cell-{cell}','exec'),ns)
            entry['status']='passed'
            if 'X' in ns: entry['input_shape']=list(ns['X'].shape)
            if 'X_trans' in ns: entry['transformed_shape']=list(ns['X_trans'].shape)
            if 'X_encoded' in ns: entry['encoded_shape']=list(ns['X_encoded'].shape)
            if 'feat_selector' in ns and hasattr(ns['feat_selector'],'selected_features_'):
                entry['selected_features']=list(ns['feat_selector'].selected_features_)
        except Exception as error:
            entry.update(status='failed',exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        entry['output_tail']=capture.getvalue().splitlines()[-15:]
        records.append(entry)
        Path(destination).write_text(json.dumps({'workflow':name,'notebook':notebook,'cell_numbering':'zero-based JSON index','records':records},indent=2)+'\n')
        return entry['status']=='passed'
    def cell(i, optional_imports=False):
        source=''.join(cells[i]['source'])
        changes=[]
        if source.startswith('%%time\n'):
            source=source.removeprefix('%%time\n'); changes.append('Removed timing-only %%time magic')
        if optional_imports:
            source='\n'.join(line for line in source.splitlines() if not line.startswith(('from xgboost import','from catboost import')))
            changes.append('Omitted optional XGBoost/CatBoost imports; all other statements retained')
        return execute(i,source,changes)
    if name.endswith('imports'):
        cell(1)
    elif name=='basic':
        for i in [2,6,11,13,17,21,25,27,32,33,39,42]:
            if not cell(i): break
    elif name=='preprocessing':
        for i in [2,5,10,12,14,18,20,22]:
            if not cell(i): break
        else:
            execute('integration-added', '''from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=5, n_jobs=1, verbose=-1)
model.fit(X_encoded, y, sample_weight=w)
prediction = model.predict(X_encoded)
assert len(prediction) == len(y) and np.isfinite(prediction).all()
''', ['Added five-tree weighted model smoke check on notebook output; training predictions only'])
    elif name=='classification-cell':
        cell(1,optional_imports=True)
        execute('local-data-substitution', '''from arfs.utils import load_data
cancer = load_data('cancer')
X, y = cancer.data, cancer.target
''', ['Local 569-row cancer data replaces remote Titanic; cell 7 not executed'])
        cell(9)
        cell(11)
    elif name=='classification-core':
        cell(1,optional_imports=True)
        execute('local-data-substitution', '''from arfs.utils import load_data
cancer = load_data('cancer')
X, y = cancer.data, cancer.target
X_encoded = X.copy()
''', ['Local cancer replaces remote Titanic; explicit numeric X_encoded repairs only missing setup for this smoke check'])
        cell(9)
        cell(11)
    elif name=='regression-native':
        cell(1,optional_imports=True)
        for i in [6,9,11]:
            if not cell(i): break
    elif name=='regression-benchmark':
        cell(1,optional_imports=True)
        for i in [6,9,34]:
            if not cell(i): break
    for i, number in enumerate(plt.get_fignums()):
        fig=plt.figure(number)
        fig.savefig(Path(destination).parent/f'{name}-figure-{i}.png',dpi=80,bbox_inches='tight')
    plt.close('all')


def parent(label, output_dir=None):
    assert label in {'frozen','current'}
    destination = (Path(output_dir).resolve() if output_dir is not None
                   else Path(tempfile.mkdtemp(prefix=f'arfs-review06-workflows-{label}-results-')))
    if destination.is_relative_to(repo):
        raise SystemExit('Rerun results must be outside the checkout to preserve saved evidence')
    destination.mkdir(parents=True, exist_ok=True)
    print(f'Rerun results: {destination}', flush=True)
    # Restrict this runner and its children only; no model hyperparameter edits.
    if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    folder=destination/f'{label}-workflows'
    folder.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONHASHSEED='0')
    ledger=[]
    with tempfile.TemporaryDirectory(prefix=f'arfs-review06-workflows-{label}-') as work:
        env['MPLCONFIGDIR']=str(Path(work)/'mpl')
        for name in ['classification-imports','regression-imports','basic','preprocessing','classification-cell','classification-core','regression-native','regression-benchmark']:
            cmd=[sys.executable,str(Path(__file__).resolve()),'child',name,str(folder/f'{name}.json')]
            try:
                result=subprocess.run(cmd,cwd=work,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
                entry={'workflow':name,'command':cmd,'cwd':work,'exit_code':result.returncode,'log':f'{name}.txt'}
                (folder/f'{name}.txt').write_text(result.stdout)
                if (folder/f'{name}.json').exists():
                    data=json.loads((folder/f'{name}.json').read_text())
                    entry['cells']=[{'cell':r['cell'],'status':r['status'],'exception':r.get('exception'),'message':r.get('message')} for r in data['records']]
            except subprocess.TimeoutExpired as error:
                entry={'workflow':name,'command':cmd,'cwd':work,'status':'timeout','timeout_seconds':120}
                output=error.stdout or b''
                (folder/f'{name}.txt').write_text(output.decode(errors='replace') if isinstance(output,bytes) else output)
            ledger.append(entry)
            (destination/f'{label}-workflow-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
            print(json.dumps(entry),flush=True)
        build=[]
        for title,cmd in [
            ('make-html',['make','-C',str(repo/'docs'),'html',f'SPHINXBUILD={sys.executable} -m sphinx',f'BUILDDIR={work}/docs-build']),
            ('configured-source',[sys.executable,'-m','sphinx','-b','html',str(repo/'docs'),str(Path(work)/'direct-build')])]:
            result=subprocess.run(cmd,cwd=work,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
            build.append({'attempt':title,'command':cmd,'exit_code':result.returncode,'output':result.stdout})
        (destination/f'{label}-build.json').write_text(json.dumps(build,indent=2)+'\n')
        source_env=env.copy(); source_env['PYTHONPATH']=str(repo/'src')
        source_code='''import json, arfs.utils as u
import numpy as np
frame=u._load_housing(as_frame=True)
bunch=u._load_housing(as_frame=False)
np.testing.assert_array_equal(frame[bunch.feature_names].to_numpy(),bunch.data)
np.testing.assert_array_equal(frame.target.to_numpy(),bunch.target)
assert frame.shape==(20640,9)
assert len(bunch.DESCR)>100
print(json.dumps({'module':u.__file__,'as_frame_shape':list(frame.shape),'bunch_shape':list(bunch.data.shape),'columns':list(frame.columns),'description_length':len(bunch.DESCR),'target_min':float(frame.target.min()),'target_max':float(frame.target.max())}))
'''
        result=subprocess.run([sys.executable,'-c',source_code],cwd=work,env=source_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        (destination/f'{label}-source-housing.txt').write_text(result.stdout)
        assert result.returncode==0,result.stdout
    text=(repo/'README.md').read_text()
    refs=re.findall(r'\]\((\.[^ )]+)\)',text)+re.findall(r'src="([^"h][^"]*)"',text)
    reference_ledger=[{'from':'README.md','target':t,'exists':(repo/unquote(t)).exists()} for t in refs]
    for file in [repo/'docs/index.rst',repo/'docs/modules.rst',repo/'docs/arfs.rst',repo/'docs/arfs.feature_selection.rst']:
        for line in file.read_text().splitlines():
            target=line.strip()
            if line.startswith('   ') and target and not target.startswith(':') and not ' ' in target and not target.startswith('arfs.'):
                path=file.parent / (target if target.endswith('.ipynb') else target+'.rst')
                reference_ledger.append({'from':str(file.relative_to(repo)),'target':target,'exists':path.exists()})
    for file in [repo/'docs/arfs.rst', repo/'docs/arfs.feature_selection.rst']:
        for module in re.findall(r'^\.\. automodule:: (\S+)', file.read_text(), re.M):
            path = repo/'src'/Path(*module.split('.'))
            reference_ledger.append({'from':str(file.relative_to(repo)), 'module':module,
                                     'exists':path.with_suffix('.py').exists() or (path/'__init__.py').exists()})
    reference_ledger.append({'from':'docs/Makefile and docs/make.bat','target':'docs/source/conf.py','exists':(repo/'docs/source/conf.py').exists()})
    (destination/'local-references.json').write_text(json.dumps(reference_ledger,indent=2)+'\n')


if __name__=='__main__':
    if sys.argv[1]=='child': child(sys.argv[2],sys.argv[3])
    elif len(sys.argv) in {2, 3}: parent(*sys.argv[1:])
    else: raise SystemExit('Usage: workflows.py frozen|current [output-dir]')
