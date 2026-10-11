"""Capture environment and source identity using the interpreter under review."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import arfs.feature_selection.lasso as lasso

repo = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2])
versions = dict(sorted((d.metadata["Name"].lower(), d.version)
                       for d in importlib.metadata.distributions()))
installed = Path(lasso.__file__)
source = repo / "src/arfs/feature_selection/lasso.py"
same_source = installed.read_bytes() == source.read_bytes()
assert same_source
destination.write_text(json.dumps({
    "review_date": "2026-10-11",
    "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "versions": versions, "installed_module": str(installed), "source_matches_checkout": same_source,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "workaround": "setuptools==80.9.0 retained from unit 0; no package installation or ARFS patch",
}, indent=2) + "\n")
destination.with_suffix(".requirements.txt").write_text("\n".join(
    f"{name}=={version}" for name, version in versions.items() if name != "arfs") + "\n")
