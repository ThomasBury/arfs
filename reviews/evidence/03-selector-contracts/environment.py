"""Record interpreter, installed-wheel identity, versions, and exact source hashes."""

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2])
modules = {}
for name in ['arfs.feature_selection.base', 'arfs.feature_selection.unsupervised',
             'arfs.feature_selection.variable_importance', 'arfs.feature_selection.summary',
             'arfs.feature_selection', 'arfs.association', 'arfs.preprocessing',
             'arfs.gbm', 'arfs.utils', 'arfs.parallel']:
    installed = Path(importlib.import_module(name).__file__)
    source = repo / 'src' / Path(*name.split('.'))
    source = source / '__init__.py' if source.is_dir() else source.with_suffix('.py')
    assert installed.read_bytes() == source.read_bytes(), name
    assert not installed.is_relative_to(repo), installed
    modules[name] = {'installed_path': str(installed), 'matches_checkout': True,
                     'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
versions = dict(sorted((dist.metadata['Name'].lower(), dist.version)
                       for dist in importlib.metadata.distributions()))
destination.write_text(json.dumps({
    'review_date': '2026-10-11',
    'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
    'versions': versions, 'modules': modules,
    'workaround': 'Unit 0 setuptools==80.9.0 retained; historical environments unchanged; no on-disk ARFS patches; optional styling overlay appears in versions; scoped test doubles/spies are identified in checks.py',
}, indent=2) + '\n')
destination.with_suffix('.requirements.txt').write_text('\n'.join(
    f'{name}=={version}' for name, version in versions.items() if name != 'arfs') + '\n')
