"""Record installed-wheel identity; invoke from a temporary working directory."""

import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from pathlib import Path

import joblib

repo = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2])
modules = {}
for name in ["arfs.utils", "arfs.association", "arfs.benchmark", "arfs.preprocessing", "arfs.feature_selection.allrelevant", "arfs.feature_selection.variable_importance", "arfs.feature_selection.unsupervised"]:
    installed = Path(importlib.import_module(name).__file__)
    source = repo / "src" / Path(*name.split(".")).with_suffix(".py")
    assert installed.read_bytes() == source.read_bytes(), name
    assert not installed.is_relative_to(repo), installed
    modules[name] = {"installed_path": str(installed), "matches_checkout": True,
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
versions = dict(sorted((dist.metadata["Name"].lower(), dist.version)
                       for dist in importlib.metadata.distributions()))
resources = {}
import arfs
installed_root = Path(arfs.__file__).parent
for relative in ["dataset/data/boston_bunch.joblib", "dataset/data/housing.zip"]:
    source = repo / "src/arfs" / relative
    installed = installed_root / relative
    assert source.read_bytes() == installed.read_bytes()
    resources[relative] = {"path": str(installed), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "matches_checkout": True}
notebooks = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo / "docs/notebooks").glob("*.ipynb")}
destination.write_text(json.dumps({
    "review_date": "2026-10-11",
    "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "versions": versions, "modules": modules,
    "notebook_sha256": notebooks, "resources": resources,
    "cpu_counts": {"multiprocessing": multiprocessing.cpu_count(), "joblib": joblib.cpu_count()},
    "thread_settings": {key: os.environ.get(key) for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]},
    "workaround": "Unit 0 setuptools==80.9.0 retained; no installs or ARFS patches in this unit",
}, indent=2) + "\n")
destination.with_suffix(".requirements.txt").write_text("\n".join(
    f"{name}=={version}" for name, version in versions.items() if name != "arfs") + "\n")
