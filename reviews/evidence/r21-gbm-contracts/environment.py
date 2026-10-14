"""Record an installed R21 environment and assert import provenance."""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

import arfs

assert Path(sys.prefix) in Path(arfs.__file__).resolve().parents
assert importlib.util.find_spec("setuptools") is None
modules = {}
for name in [
    "arfs.preprocessing",
    "arfs.feature_selection.base",
    "arfs.feature_selection.unsupervised",
    "arfs.feature_selection.variable_importance",
    "arfs.association",
    "arfs.parallel",
    "arfs.utils",
    "arfs.gbm",
    "arfs.benchmark",
    "arfs.feature_selection.allrelevant",
    "arfs.feature_selection.lasso",
    "arfs.feature_selection.mrmr",
]:
    path = Path(importlib.import_module(name).__file__).resolve()
    assert Path(sys.prefix) in path.parents
    modules[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

print(
    json.dumps(
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "arfs_version": arfs.__version__,
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "modules": modules,
            "setuptools_installed": False,
        },
        indent=2,
        sort_keys=True,
    )
)
