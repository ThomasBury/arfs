# 05 — Sampling, distances, and parallel execution

**Status:** Reviewed; findings open. **Outcome:** [Completed report](05-sampling-and-parallel.md) and [executable evidence](evidence/05-sampling-and-parallel/checks.py). Fourteen scoped findings and two additional association/relevance compatibility blockers. Seventy-nine checks in each saved installed-wheel environment, with strict and unmasked logs, independent distance references, bounded retry probes, and small allocation measurements. No library fixes applied.

**Purpose:** Check that distance calculations and sampling retain the intended
observations and weighting, and that parallel execution preserves the serial
result with understandable resource costs.

**Read first**

- [Queue and shared review procedure](README.md).
- [Sampling and Gower distance functions](../src/arfs/sampling.py).
- [Parallel helpers](../src/arfs/parallel.py) and their association/relevance
  callers in [association.py](../src/arfs/association.py).
- Relevant validation/type helpers in [utils.py](../src/arfs/utils.py).
- [Large-data sampling notebook](../docs/notebooks/arfs_large_data_sampling.ipynb).

**Review questions and checks**

1. Compute small mixed numeric/categorical Gower distances independently. Cover
   constant numeric columns, missing values, feature weights, declared category
   masks, separate reference/query frames, and unsupported sparse input. Check
   shape, row/column meaning, symmetry where applicable, and zero diagonals.
2. Check `smallest_indices` and `gower_topn` with ties, self-neighbors, nonfinite
   distances, `n=0`, and `n` beyond the available candidates. Validate returned
   positions against the actual input rows and confirm documented ordering.
3. Exercise each sampling method with mixed data, duplicated/non-default
   indexes, tiny datasets, and requests at or beyond the dataset size. Determine
   whether outputs are original rows or synthesized/grouped representatives;
   verify the stated sample-weight meaning and conservation rule where claimed.
4. Review isolation-forest sampling and split helpers for termination, empty
   results, deterministic behavior, mutation, and what their statistical checks
   can actually establish. A nonsignificant comparison is not proof that a
   sample represents the full joint distribution.
5. Compare serial and parallel helper results on the same small inputs, including
   weighted calls, empty work lists, worker counts, ordering, and propagated
   errors. Reuse unit 1A references instead of re-auditing the formulas.
6. Inspect time/memory complexity and measure only a bounded small scaling case
   if needed to support a finding. Identify pairwise matrix allocation and
   oversubscription risks against the notebook's advertised large-data use.
   Avoid an unrestricted quadratic allocation on the review machine.

**Known findings to carry forward**

- [Unit 1A](01a-statistical-foundations.md) traced parallel callers as part of
  statistical calculations; distinguish arithmetic issues from execution issues.
- [Unit 0's coverage map](00-baseline.md) records no original sampling tests.
  That is historical context, not proof that a function is incorrect.

**Deliverables and completion**

- Report: `reviews/05-sampling-and-parallel.md`.
- Evidence: `reviews/evidence/05-sampling-and-parallel/`.
- State the actual sampling contract, numerical counterexamples, serial/parallel
  results, and resource limits. Follow the queue's completion/update procedure.
- Next queued item: [06 — utilities and workflows](06-utilities-and-workflows.todo.md).

**Boundary:** No distributed framework, performance rewrite, or broad sampling
study. Document statistical uncertainty separately from concrete implementation
failures.
