"""Reject new Ruff diagnostics while the recorded review debt is repaired."""

import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / ".github" / "ruff-baseline.json"


def regressions(diagnostics, baseline, root):
    # shortcut: counts can hide a replacement issue in the same file/rule;
    # remove each allowance when repaired, then retire the baseline at R33.
    counts = Counter(
        (Path(item["filename"]).relative_to(root).as_posix(), item["code"])
        for item in diagnostics
    )
    return {
        (path, code): count - baseline.get(path, {}).get(code, 0)
        for (path, code), count in counts.items()
        if count > baseline.get(path, {}).get(code, 0)
    }


def main():
    result = subprocess.run(
        ["ruff", "check", "--output-format", "json", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise SystemExit(result.stderr or f"Ruff failed: {result.returncode}")
    diagnostics = json.loads(result.stdout)
    extra = regressions(diagnostics, json.loads(BASELINE.read_text()), ROOT)
    for (path, code), count in sorted(extra.items()):
        print(f"{path}: {count} new {code} diagnostic(s)")
    print(f"Ruff: {len(diagnostics)} total diagnostics; {sum(extra.values())} new")
    raise SystemExit(bool(extra))


if __name__ == "__main__":
    main()
