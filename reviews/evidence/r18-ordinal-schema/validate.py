import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path("/home/bsatom/Documents/arfs")
work = Path("/tmp/arfs-r18/final")
evidence = root / "reviews/evidence/r18-ordinal-schema"
cell = sys.argv[1]
venv = work / cell
python = venv / "bin/python"
run_dir = work / ("run-" + cell)
requirements = (
    work / "locked-requirements.txt"
    if cell.endswith("locked")
    else root / ".github/requirements/runtime-newer-py312.txt"
)
env = dict(
    os.environ,
    UV_CACHE_DIR="/tmp/arfs-r04/cache",
    MPLBACKEND="Agg",
    MPLCONFIGDIR=str(work / ("mpl-" + cell)),
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
    LOKY_MAX_CPU_COUNT="2",
    PYTHONHASHSEED="0",
    PYTHONDONTWRITEBYTECODE="1",
)
env.pop("PYTHONPATH", None)
commands = []


def check(name, argv, cwd=root):
    record = {"command": [str(x) for x in argv], "cwd": str(cwd)}
    commands.append(record)
    with (evidence / (cell + "-" + name + ".txt")).open("w") as log:
        result = subprocess.run(
            record["command"],
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["exit_code"] = result.returncode
    (evidence / (cell + "-commands.json")).write_text(
        json.dumps(commands, indent=2) + "\n"
    )
    assert result.returncode == 0, (name, result.returncode)
    print(cell, name, "passed", flush=True)


check("venv", ["uv", "venv", "--python", "/tmp/arfs-r17/" + cell + "/bin/python", venv])
check(
    "install",
    [
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "-r",
        requirements,
        *sorted((work / "dist").glob("*.whl")),
    ],
)
check("pip-check", ["uv", "pip", "check", "--python", python])
check("font-cache", [python, "-I", "-c", "import matplotlib.font_manager"])
check("environment", [python, "-I", work / "environment.py"])
metadata = json.loads((evidence / (cell + "-environment.txt")).read_text())
(evidence / (cell + "-environment.json")).write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)
(evidence / (cell + "-environment.txt")).unlink()
for name, info in metadata["modules"].items():
    source = root / "src" / (name.replace(".", "/") + ".py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == info["sha256"], name
run_dir.mkdir()
shutil.copytree(
    root / "tests",
    run_dir / "tests",
    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
)
for source in (root / "tests").glob("*.py"):
    assert source.read_bytes() == (run_dir / "tests" / source.name).read_bytes()
check(
    "tests",
    [
        python,
        "-I",
        "-m",
        "pytest",
        "tests",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
    ],
    run_dir,
)
if cell.startswith("py312"):
    check(
        "historical",
        [
            python,
            "-I",
            "-m",
            "pytest",
            root / "reviews/evidence/02-preprocessing/checks.py",
            "-k",
            "ordinal",
            "--runxfail",
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ],
        run_dir,
    )
