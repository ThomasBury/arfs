# Pre-remediation checkpoint assessment

Checked on 2026-10-11. The completed review can be preserved as a known baseline,
but its artifacts are not yet saved in Git. No repair phase has started.

## Verified state

| Surface | Result |
| --- | --- |
| Current checkout and remote main | Both `03f67d0a54b69fac5ddbb83e306c8e8e72e2d3a2`; remote confirmed with `git ls-remote` |
| Tracked working tree and index | No changes; both unstaged and staged diffs empty |
| Git object integrity | `git fsck --full --no-reflogs --no-dangling` succeeds |
| Existing release tag | Lightweight tag `3.0.0` points to `ccbed4940df995e5ef10293e8932a220d39c93e2`; local and remote agree |
| Difference from release | Current main is three commits after the release tag; the review target is current main, not the published release |
| GitHub release | [3.0.0](https://github.com/ThomasBury/arfs/releases/tag/3.0.0), published 2025-04-02, neither draft nor prerelease; no separately uploaded GitHub assets |
| PyPI release | [3.0.0](https://pypi.org/project/arfs/3.0.0/) is the latest reported version; wheel and sdist were uploaded 2025-04-02 and are not yanked |
| Local release files | Both files under ignored `dist/` exactly match PyPI's SHA-256 digests; all 16 wheel source modules match tag `3.0.0` |
| Completed review | All queued reviews complete; `reviews/` remains entirely untracked. Before this assessment it contained 201 files, 2,397,009 bytes, excluding ignored bytecode |
| Main branch protection | One approving PR review required; force pushes/deletion disabled; administrators exempt; no required status checks |
| CI evidence | Ruff workflow exists, but its runs endpoint returned no runs. Recent successful jobs on this commit are Dependabot jobs; they do not establish passing package tests |

Published release hashes:

```text
arfs-3.0.0-py3-none-any.whl
5a748fb81a291ee30b1d6ee3ce078cd3d0d331d0acfa6f9f1ef00d5b366f341d

arfs-3.0.0.tar.gz
58a70a063d2ce1596d4f0ce2c45cbdcad0e69767507cc880de5fbc23ec5b78c0
```

The published wheel matches only five current-main source files byte for byte.
The remaining eleven were changed after the release, including formatting.
This comparison does not establish semantic differences in every changed file.
The earlier review's disposable wheel was built from current main and has its
own hashes; do not substitute the older `dist/` release files when reproducing
those checks.

## Checkpoint steps before R01

1. Save the completed reports, evidence, plans, remediation handoff, and this
   assessment in a review-only commit, for example
   `docs(review): record completed audit and remediation baseline`.
   Use a branch and PR to respect main's review protection. Check the staged
   paths so ignored environments, caches, builds, and unrelated files stay out.
2. Once that commit is on the intended base branch, create an annotated tag such
   as `checkpoint/pre-remediation-2026-10-11`. Its message should state that the
   review is complete, the underlying code is based on `03f67d0`, and known
   defects remain. Preserve the existing `3.0.0` tag unchanged.
3. Push the checkpoint commit/tag through the authorized Git workflow. Confirm
   the remote tag resolves to the intended commit and can be checked out in a
   fresh directory. A local tag alone does not provide an off-machine backup.
4. Start `fix/r01-installed-package` from that checkpoint. Implement only R01,
   run its prescribed fresh-wheel checks, and stop for review before another
   repair phase. See [REMEDIATION.md](REMEDIATION.md).
5. For rollback or comparison, create a fresh branch/worktree from the checkpoint
   tag. For an already merged repair, revert its commit through the normal PR
   workflow rather than rewriting shared main history.

A package release is not required to establish this checkpoint. The existing
3.0.0 release is already published. A future repair release needs a new version,
matching metadata/lock/changelog, passing installed-package regression checks,
and the support evidence described in [the final assessment](07-final-assessment.md).
Current code still has the demonstrated release blockers.

## Scope and limits

This assessment queried Git/GitHub/PyPI and read local release archives. It did
not rerun the known-failing runtime suites because the reviewed source is
unchanged. It does not certify hosted documentation, package correctness, or
full dependency/platform support. Branch-protection and registry status are
point-in-time observations.

At the end of the assessment, only this file had been added. No commit,
checkpoint tag, branch, PR, push, release, branch-protection change, or repair
had been created by that check.

## Authorized checkpoint execution

The user subsequently authorized saving the checkpoint. The review artifacts
are being committed on `docs/review-baseline-2026-10-11` for a PR into main.
The annotated backup tag is `checkpoint/pre-remediation-2026-10-11`.
It preserves the review branch's commit independently of PR approval; main's
one-review requirement remains in force. The tag can be checked out immediately
even while the PR awaits approval.

The next local branch is `fix/r01-installed-package`, based on that tag. Creating
the branch does not start implementation. Existing runtime failures remain
documented, and raw diagnostic logs retain their original whitespace.
