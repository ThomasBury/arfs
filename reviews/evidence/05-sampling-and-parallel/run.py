"""Run one installed-wheel environment outside the checkout, with time limits.

Usage: /path/to/environment/bin/python /absolute/path/to/run.py frozen|current
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

evidence = Path(__file__).resolve().parent
repo = evidence.parents[2]
label = sys.argv[1]
assert label in {"frozen", "current"}
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONHASHSEED="0")
runs = []
with tempfile.TemporaryDirectory(prefix=f"arfs-review05-{label}-") as work:
    env["MPLCONFIGDIR"] = str(Path(work) / "mpl")
    commands = [
        ("environment", [str(evidence / "environment.py"), str(repo), str(evidence / f"{label}-environment.json")], 0),
        ("checks", ["-m", "pytest", str(evidence / "checks.py"), "-q", "-rx", "--tb=short", "-c", "/dev/null",
                    f"--rootdir={evidence}", "-p", "no:cacheprovider"], 0),
        ("failures", ["-m", "pytest", str(evidence / "checks.py"), "-q", "--runxfail", "--tb=short", "-c", "/dev/null",
                      f"--rootdir={evidence}", "-p", "no:cacheprovider"], 1),
        ("scaling", [str(evidence / "scaling.py"), str(evidence / f"{label}-scaling.json")], 0),
    ]
    for name, args, expected_exit in commands:
        command = [sys.executable, *args]
        result = subprocess.run(command, cwd=work, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        if name in {"checks", "failures"}:
            (evidence / f"{label}-{name}.txt").write_text(result.stdout)
        runs.append({"name": name, "command": command, "cwd": work,
                     "exit_code": result.returncode, "expected_exit_code": expected_exit,
                     "tail": result.stdout.splitlines()[-5:]})
        print(json.dumps(runs[-1]), flush=True)
        (evidence / f"{label}-runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        if result.returncode != expected_exit:
            raise SystemExit(f"Unexpected {name} exit; inspect evidence before continuing")
