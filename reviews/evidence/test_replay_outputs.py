"""Replay runners must keep logs and manifests outside the historical checkout."""

import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

CASES = [
    ("04-training-and-benchmarks/run.py", ["frozen", sys.executable]),
    ("05-sampling-and-parallel/run.py", ["frozen"]),
    ("06-utilities-and-workflows/run.py", ["frozen"]),
    ("06-utilities-and-workflows/workflows.py", ["frozen"]),
    ("07-final-assessment/resolve.py", []),
]


@pytest.fixture(params=CASES, ids=[case[0] for case in CASES])
def runner(request, tmp_path):
    relative, args = request.param
    repo = tmp_path / "checkout"
    path = repo / "reviews/evidence" / relative
    path.parent.mkdir(parents=True)
    path.write_bytes((Path(__file__).parent / relative).read_bytes())
    (repo / "README.md").write_text("")
    (repo / "docs").mkdir()
    for name in ("index.rst", "modules.rst", "arfs.rst", "arfs.feature_selection.rst"):
        (repo / "docs" / name).write_text("")
    return path, args, repo


@pytest.mark.parametrize("default_output", [False, True])
def test_rerun_preserves_saved_evidence(runner, tmp_path, monkeypatch, default_output):
    path, args, repo = runner
    saved = path.parent / "frozen-checks.txt"
    saved.write_text("historical evidence")
    output = tmp_path / "results"
    mkdtemp = tempfile.mkdtemp

    def temporary(suffix=None, prefix=None, dir=None):
        return mkdtemp(suffix=suffix, prefix=prefix, dir=tmp_path)

    def run(command, **kwargs):
        for part in command:
            if str(part).endswith(".json"):
                assert not Path(part).resolve().is_relative_to(repo)
        if "REVIEW_PLOTS" in kwargs.get("env", {}):
            assert not Path(kwargs["env"]["REVIEW_PLOTS"]).is_relative_to(repo)
        stream = kwargs.get("stdout")
        if hasattr(stream, "write"):
            stream.write("rerun output")
        return subprocess.CompletedProcess(
            command, 1 if "--runxfail" in command else 0, stdout="rerun output", stderr=""
        )

    argv = [str(path), *args] + ([] if default_output else [str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(tempfile, "mkdtemp", temporary)
    monkeypatch.setattr(subprocess, "run", run)
    if hasattr(os, "sched_setaffinity"):
        monkeypatch.setattr(os, "sched_setaffinity", lambda *args: None)
    runpy.run_path(str(path), run_name="__main__")
    assert saved.read_text() == "historical evidence"
    assert not list(repo.rglob("*.json"))
    assert any(file.suffix == ".json" for file in tmp_path.rglob("*") if not file.is_relative_to(repo))


@pytest.mark.parametrize("inside_evidence", [False, True])
def test_reject_checkout_output(runner, monkeypatch, inside_evidence):
    path, args, repo = runner
    output = path.parent if inside_evidence else repo / "results"
    monkeypatch.setattr(sys, "argv", [str(path), *args, str(output)])
    with pytest.raises(SystemExit, match="outside the checkout"):
        runpy.run_path(str(path), run_name="__main__")
    if not inside_evidence:
        assert not output.exists()


@pytest.mark.parametrize("script", ["audit.py", "workflows.py"])
def test_direct_audit_or_child_rejects_checkout_output(tmp_path, monkeypatch, script):
    folder = "07-final-assessment" if script == "audit.py" else "06-utilities-and-workflows"
    repo = tmp_path / "checkout"
    path = repo / "reviews/evidence" / folder / script
    path.parent.mkdir(parents=True)
    path.write_bytes((Path(__file__).parent / folder / script).read_bytes())
    output = path.parent / "saved.json"
    args = [str(repo), str(output)] if script == "audit.py" else ["child", "basic", str(output)]
    monkeypatch.setattr(sys, "argv", [str(path), *args])
    with pytest.raises(SystemExit, match="outside the checkout"):
        runpy.run_path(str(path), run_name="__main__")
    assert not output.exists()
