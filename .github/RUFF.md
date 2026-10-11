# Ruff during remediation

CI uses Ruff 0.17.0 and checks the whole repository. The checkpoint has 432
existing diagnostics, recorded by file and rule in `ruff-baseline.json`.
`scripts/check_ruff_baseline.py` fails if a new file or rule has a diagnostic,
or a file's count for a rule increases. Ruff execution errors also fail CI.
The baseline is an exception for existing debt; it does not mean the code is
lint-clean. No source, notebook, or saved review evidence was rewritten for it.

Run the same gate locally from the repository root:

```sh
uvx ruff==0.17.0 --version
uvx --from ruff==0.17.0 python .github/scripts/check_ruff_baseline.py
python -m unittest discover -s .github/scripts -p 'test_*.py'
```

When a repair removes diagnostics, lower the corresponding baseline counts;
do not add allowances to pass CI. Review the actual Ruff output on changed files
as well: counts can miss a new issue replacing an old one in the same file and
rule. At R33, remove the baseline gate and require a clean `ruff check .`.
Changing Ruff's version requires reviewing any changed diagnostics explicitly.
