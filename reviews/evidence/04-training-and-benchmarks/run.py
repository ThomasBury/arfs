"""Run one saved wheel snapshot outside the checkout and record exact commands.

Usage: python run.py frozen /tmp/arfs-review-03f67d0/locked-env/bin/python
       python run.py current /tmp/arfs-review-03f67d0/wheel-env/bin/python
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

if len(sys.argv) not in {3, 4} or sys.argv[1] not in {"frozen", "current"}:
    raise SystemExit("Usage: run.py frozen|current interpreter [output-dir]")
label, interpreter = sys.argv[1:3]
evidence = Path(__file__).resolve().parent
repo = evidence.parents[2]
destination = (Path(sys.argv[3]).resolve() if len(sys.argv) == 4
               else Path(tempfile.mkdtemp(prefix=f"arfs-review04-{label}-results-")))
if destination.is_relative_to(repo):
    raise SystemExit("Rerun results must be outside the checkout to preserve saved evidence")
destination.mkdir(parents=True, exist_ok=True)
print(f"Rerun results: {destination}", flush=True)
environment = os.environ.copy()
settings = dict(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
environment.update(settings)
records = []
with tempfile.TemporaryDirectory(prefix=f"arfs-review04-{label}-", dir="/tmp") as work:
    environment["MPLCONFIGDIR"] = str(Path(work) / "mpl")
    identity = [interpreter, str(evidence / "environment.py"), str(repo),
                str(destination / f"{label}-environment.json")]
    subprocess.run(identity, cwd=work, env=environment, check=True, timeout=60)
    base = [interpreter, "-m", "pytest", str(evidence / "checks.py"), "-q", "-rx",
            "--tb=short", "-c", "/dev/null", f"--rootdir={evidence}", "-p", "no:cacheprovider"]
    for suffix, extra, expected in [("checks", [], 0), ("failures", ["--runxfail"], 1)]:
        command = base + extra
        start = time.monotonic()
        with (destination / f"{label}-{suffix}.txt").open("w") as output:
            output.write("Command: " + shlex.join(command) + "\n")
            output.flush()
            result = subprocess.run(command, cwd=work, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT, timeout=180)
        records.append(dict(command=command, cwd=work, exit_code=result.returncode,
                            elapsed_seconds=round(time.monotonic() - start, 3)))
        assert result.returncode == expected, records[-1]
    (destination / f"{label}-runs.json").write_text(json.dumps({
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "identity_command": identity,
        "environment": settings | {"MPLCONFIGDIR": environment["MPLCONFIGDIR"]},
        "evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in evidence.glob("*.py")},
        "runs": records,
    }, indent=2) + "\n")
print(json.dumps(records, indent=2))
