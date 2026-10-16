# Ruff

CI requires a clean `ruff check .` using Ruff 0.17.0. R33 retired the temporary
diagnostic-count baseline and its scripts. Historical review evidence remains
unchanged and ignored by the ordinary repository check.

Run the same gate locally from the repository root:

```sh
uv tool run --from ruff==0.17.0 ruff check .
```

The project keeps runtime-compatible type annotations on Python 3.9 using
`lint.pyupgrade.keep-runtime-typing`. Native `list`/`dict`/`tuple` annotations are
supported there; `Union`/`Optional` remain evaluable through `typing.get_type_hints`.
The setting avoids replacing them with Python-3.10-only union syntax.

Local `noqa` comments preserve real compiled-backend import probes and deliberate
third-party exception wrappers, including plot failures that must not discard a
fitted model. They name the reason at the relevant statement. The documentation
tests retain their existing exception for executing repository-owned cells.
No count-based debt allowances remain. Review changes to Ruff's version or
these exceptions explicitly.
