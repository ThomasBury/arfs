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
for name in ["arfs.sampling", "arfs.parallel", "arfs.association", "arfs.utils"]:
    installed = Path(importlib.import_module(name).__file__)
    source = repo / "src" / Path(*name.split(".")).with_suffix(".py")
    assert installed.read_bytes() == source.read_bytes(), name
    assert not installed.is_relative_to(repo), installed
    modules[name] = {"installed_path": str(installed), "matches_checkout": True,
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
versions = dict(sorted((dist.metadata["Name"].lower(), dist.version)
                       for dist in importlib.metadata.distributions()))
notebook = repo / "docs/notebooks/arfs_large_data_sampling.ipynb"
destination.write_text(json.dumps({
    "review_date": "2026-10-11",
    "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "versions": versions, "modules": modules,
    "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
    "cpu_counts": {"multiprocessing": multiprocessing.cpu_count(), "joblib": joblib.cpu_count()},
    "thread_settings": {key: os.environ.get(key) for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]},
    "workaround": "Unit 0 setuptools==80.9.0 retained; no installs or ARFS patches in this unit",
}, indent=2) + "\n")
destination.with_suffix(".requirements.txt").write_text("\n".join(
    f"{name}=={version}" for name, version in versions.items() if name != "arfs") + "\n")
