"""Resolve a bounded compatibility matrix; never install or update the lockfile."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

repo = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
results = []
cells = [("offline-py39", "3.9", True, False)]
cells += [(f"runtime-py{v.replace('.', '')}", v, False, False) for v in ["3.9", "3.10", "3.11", "3.12"]]
cells += [("readthedocs-py310", "3.10", False, True)]
cells += [("docs-extra-py310", "3.10", False, "extra-only")]
with tempfile.TemporaryDirectory(prefix="arfs-final-resolution-") as run:
    for name, version, offline, docs in cells:
        output = Path(run) / f"{name}.txt"
        command = ["uv", "pip", "compile", str(repo / "pyproject.toml"), "--python-version", version,
                   "--no-build", "--output-file", str(output)]
        if offline:
            command.append("--offline")
        if docs:
            if docs is True:
                command += [str(repo / "docs/requirements.txt")]
            command += ["--extra", "docs"]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=run, env=os.environ | {
                "UV_CACHE_DIR": "/tmp/arfs-review-03f67d0/cache", "UV_PYTHON_DOWNLOADS": "never"
            }, capture_output=True, text=True, timeout=50)
            result = {"cell": name, "command": command, "cwd": run, "returncode": completed.returncode,
                      "stdout": completed.stdout, "stderr": completed.stderr,
                      "requirements": output.read_text() if output.exists() else None}
        except subprocess.TimeoutExpired as exc:
            result = {"cell": name, "command": command, "cwd": run, "status": "timeout",
                      "stdout": str(exc.stdout), "stderr": str(exc.stderr)}
        result["seconds"] = time.perf_counter() - started
        results.append(result)
        (evidence / "resolution.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, result.get("returncode", result.get("status")), flush=True)
assert len(results) == 7
