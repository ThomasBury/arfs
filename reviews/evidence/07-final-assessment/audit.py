"""Record source ownership, imports, installed metadata and bounded import costs.

Run with either saved review interpreter; writes only the supplied destination.
No dependency installation or source modification is performed.
"""

import ast
from collections import Counter
import hashlib
import importlib.metadata as metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


repo = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
if destination.is_relative_to(repo):
    raise SystemExit('Rerun results must be outside the checkout to preserve saved evidence')
owners = {
    "__init__.py": ["00"],
    "association.py": ["01a", "05", "06", "07"],
    "benchmark.py": ["04", "06"],
    "gbm.py": ["04"],
    "parallel.py": ["01a", "05"],
    "preprocessing.py": ["02", "06"],
    "sampling.py": ["05"],
    "utils.py": ["00", "01b", "03", "04", "06"],
    "feature_selection/__init__.py": ["00", "03"],
    "feature_selection/allrelevant.py": ["01b", "04", "06"],
    "feature_selection/base.py": ["03"],
    "feature_selection/lasso.py": ["01c"],
    "feature_selection/mrmr.py": ["01a"],
    "feature_selection/summary.py": ["03"],
    "feature_selection/unsupervised.py": ["03", "06"],
    "feature_selection/variable_importance.py": ["03", "06"],
}
source_files = sorted((repo / "src/arfs").rglob("*.py"))
assert {str(p.relative_to(repo / "src/arfs")) for p in source_files} == owners.keys()
installed_root = Path(metadata.distribution("arfs").locate_file("arfs"))
assert not installed_root.resolve().is_relative_to(repo)
modules = {}
for source in source_files:
    relative = str(source.relative_to(repo / "src/arfs"))
    installed = installed_root / relative
    assert source.read_bytes() == installed.read_bytes(), relative
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    imports = [
        {"line": node.lineno, "statement": ast.get_source_segment(source.read_text(), node),
         "top_level": node in tree.body}
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    modules[relative] = {
        "owners": owners[relative], "installed_path": str(installed),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "public_definitions": [{"name": node.name, "line": node.lineno} for node in definitions if not node.name.startswith("_")],
        "duplicate_definitions": [name for name, n in Counter(node.name for node in definitions).items() if n > 1],
        "imports": imports,
    }

distributions = {}
for dist in metadata.distributions():
    requires_python = dist.metadata.get("Requires-Python", "")
    distributions[dist.metadata["Name"].lower()] = {
        "version": dist.version, "requires_python": requires_python,
        "python_metadata_allows": {v: SpecifierSet(requires_python).contains(v) for v in ["3.9.0", "3.10.0", "3.11.0", "3.12.7", "3.13.0"]},
        "requires_dist": dist.requires or [],
    }
direct = [Requirement(r).name.lower() for r in metadata.requires("arfs") if not Requirement(r).marker]

history = subprocess.check_output([
    "git", "-C", str(repo), "log", "--no-merges", "-40", "--format=COMMIT %H %s", "--name-only", "--", "src/arfs"
], text=True)
commits = []
for chunk in history.split("COMMIT ")[1:]:
    heading, *files = chunk.strip().splitlines()
    sha, subject = heading.split(" ", 1)
    commits.append({"sha": sha, "subject": subject, "files": [f for f in files if f.startswith("src/arfs/")]})
cochanges = Counter()
for commit in commits:
    if "ruff formatting" not in commit["subject"]:
        cochanges.update(itertools.combinations(sorted(set(commit["files"])), 2))

probe = '''import importlib, json, resource, sys, time
started = time.perf_counter()
importlib.import_module(sys.argv[1])
print(json.dumps({"seconds": time.perf_counter()-started,
 "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 "loaded": {n: n in sys.modules for n in ["numpy","pandas","scipy","sklearn","matplotlib","lightgbm","shap","numba","statsmodels","patsy","pkg_resources"]}}))
'''
costs = {}
with tempfile.TemporaryDirectory(prefix="arfs-final-import-") as run:
    env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "MPLCONFIGDIR": run}
    env.update({name: "1" for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]})
    for name in ["arfs", "arfs.parallel", "arfs.association", "arfs.feature_selection.unsupervised"]:
        costs[name] = []
        for _ in range(3):
            result = subprocess.run([sys.executable, "-I", "-c", probe, name], cwd=run, env=env, text=True, capture_output=True, timeout=45)
            assert result.returncode == 0, result.stderr
            costs[name].append(json.loads(result.stdout))

result = {
    "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "workaround": "setuptools==80.9.0 remains installed from unit 00; no clean-install claim",
    "modules": modules, "direct_runtime_dependencies": direct, "distributions": distributions,
    "history": commits, "cochanges_excluding_ruff_formatting": [{"files": list(pair), "count": n} for pair, n in cochanges.most_common(12)],
    "import_costs": costs,
    "import_measurement_limits": "Three fresh interpreters, warm filesystem cache, one CPU thread per native library; whole-process peak RSS, not marginal allocation or a general benchmark.",
}
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"modules": len(modules), "distributions": len(distributions), "history_commits": len(commits), "output": str(destination)}))
