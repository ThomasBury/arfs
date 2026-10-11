"""Run the installed-wheel review with bounded subprocesses outside the checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

evidence = Path(__file__).resolve().parent
repo = evidence.parents[2]
if len(sys.argv) not in {2, 3} or sys.argv[1] not in {'frozen', 'current'}:
    raise SystemExit('Usage: run.py frozen|current [output-dir]')
label = sys.argv[1]
destination = (Path(sys.argv[2]).resolve() if len(sys.argv) == 3
               else Path(tempfile.mkdtemp(prefix=f'arfs-review06-{label}-results-')))
if destination.is_relative_to(repo):
    raise SystemExit('Rerun results must be outside the checkout to preserve saved evidence')
destination.mkdir(parents=True, exist_ok=True)
print(f'Rerun results: {destination}', flush=True)
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONHASHSEED='0')
runs=[]
with tempfile.TemporaryDirectory(prefix=f'arfs-review06-{label}-') as work:
    env['MPLCONFIGDIR'] = str(Path(work)/'mpl')
    env['REVIEW_PLOTS'] = str(destination/f'{label}-plots')
    commands=[('environment',[str(evidence/'environment.py'),str(repo),str(destination/f'{label}-environment.json')],0),
              ('datasets',[str(evidence/'datasets.py'),str(destination/f'{label}-datasets.json')],0),
              ('checks',['-m','pytest',str(evidence/'checks.py'),'-q','-rx','--tb=short','-c','/dev/null',f'--rootdir={evidence}','-p','no:cacheprovider'],0),
              ('failures',['-m','pytest',str(evidence/'checks.py'),'-q','--runxfail','--tb=short','-c','/dev/null',f'--rootdir={evidence}','-p','no:cacheprovider'],1)]
    for name,args,expected in commands:
        cmd=[sys.executable,*args]
        result=subprocess.run(cmd,cwd=work,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (destination/f'{label}-{name}.txt').write_text(result.stdout)
        runs.append({'name':name,'command':cmd,'cwd':work,'exit_code':result.returncode,'expected_exit_code':expected,'tail':result.stdout.splitlines()[-5:]})
        (destination/f'{label}-runs.json').write_text(json.dumps(runs,indent=2)+'\n')
        print(json.dumps(runs[-1]),flush=True)
        if result.returncode != expected: raise SystemExit('Unexpected result; inspect evidence')
