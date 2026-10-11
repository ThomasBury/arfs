"""Record installed-wheel identity and dependencies; run outside the checkout."""

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
for name in [
    "arfs.gbm", "arfs.benchmark", "arfs.utils", "arfs.preprocessing",
    "arfs.feature_selection.variable_importance", "arfs.feature_selection.allrelevant",
]:
    installed = Path(importlib.import_module(name).__file__)
    source = repo / "src" / Path(*name.split(".")).with_suffix(".py")
    assert installed.read_bytes() == source.read_bytes(), name
    assert not installed.is_relative_to(repo), installed
    modules[name] = {"installed_path": str(installed), "matches_checkout": True,
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
versions = dict(sorted((dist.metadata["Name"].lower(), dist.version)
                       for dist in importlib.metadata.distributions()))
destination.write_text(json.dumps({
    "review_date": "2026-10-11",
    "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "versions": versions, "modules": modules,
    "workaround": "Unit 0 setuptools==80.9.0 retained; no installs or ARFS patches in this unit",
}, indent=2) + "\n")
destination.with_suffix(".requirements.txt").write_text("\n".join(
    f"{name}=={version}" for name, version in versions.items() if name != "arfs") + "\n")
