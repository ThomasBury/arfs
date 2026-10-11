"""Reproduction output must not overwrite the historical checkout."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path):
    evidence = tmp_path / "checkout/reviews/evidence/05-sampling-and-parallel"
    evidence.mkdir(parents=True)
    path = evidence / "run.py"
    path.write_bytes(Path(__file__).with_name("run.py").read_bytes())
    return path


def test_rerun_writes_only_to_requested_output(runner, tmp_path, monkeypatch):
    saved = runner.parent / "frozen-checks.txt"
    saved.write_text("historical evidence")
    output = tmp_path / "results"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 1 if "--runxfail" in command else 0, stdout="rerun output"
        )

    monkeypatch.setattr(sys, "argv", [str(runner), "frozen", str(output)])
    monkeypatch.setattr(subprocess, "run", run)
    runpy.run_path(str(runner), run_name="__main__")
    assert saved.read_text() == "historical evidence"
    assert (output / "frozen-checks.txt").read_text() == "rerun output"
    assert (output / "frozen-failures.txt").read_text() == "rerun output"
    assert (output / "frozen-runs.json").is_file()
    assert commands[0][-1] == str(output / "frozen-environment.json")
    assert commands[-1][-1] == str(output / "frozen-scaling.json")


@pytest.mark.parametrize("inside_evidence", [False, True])
def test_reject_output_inside_checkout(runner, monkeypatch, inside_evidence):
    output = runner.parent if inside_evidence else runner.parent.parents[2] / "results"
    monkeypatch.setattr(sys, "argv", [str(runner), "frozen", str(output)])
    with pytest.raises(SystemExit, match="outside the checkout"):
        runpy.run_path(str(runner), run_name="__main__")
    if not inside_evidence:
        assert not output.exists()
