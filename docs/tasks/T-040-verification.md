# T-040 intake refactor verification

Implemented 2026-09-10 in the shared worktree. Existing unrelated edits were preserved.

The reusable coordinator now holds immutable dependencies. A new session owns each
invocation's identities, events, and commits. Resource inventory and parsing return typed
results independently of persistence. The public intake entry imports without the CLI,
design, analysis, preparation, presentation, runtime, model, or graph modules.

The final focused run passed **186 tests** across intake, CLI, runtime composition,
and design entry. Two warnings came from existing statistical-library fixtures: a NumPy
timedelta deprecation and a PyFixest divide-by-zero warning. No live-provider run was used.

```sh
.venv/bin/python -m pytest tests/intake tests/cli tests/runtime/test_composition.py tests/design/test_entry.py -q
.venv/bin/python -m ruff check src/causal/intake src/causal/runtime/composition.py tests/intake
.venv/bin/python -m mypy src/causal/intake src/causal/runtime/composition.py
```

Ruff passes. Strict mypy passes all 14 checked production source files. Diff whitespace
checks pass. Boundary review found no new regression after the source-factory failure
and refused-archive extraction paths were corrected.

New regression coverage verifies:

- Duplicate archive members refuse admission without extracting bytes.
- Unsafe members and excluded siblings retain resource rows; corrupt archives refuse.
- Corrupt individual members become failed resources while good tables can finish.
- CSV, TSV, and Parquet profiles with NaN/infinity serialize canonically, retain explicit
  non-finite counts, and compute correct finite summaries.
- Separate coordinator invocations preserve their own question, artifact, and event identities.
- Exact replay makes no provider call or client construction, and source-factory failure
  commits a sanitized refusal that can itself be replayed.
- Public intake execution and valid handoff work with only intake dependencies.

## Complexity and remaining limits

Using the existing checker, intake is **1,200 / 1,200** significant Python lines,
compared with 1,192 in the captured starting worktree. Its largest module is 155 lines
and its largest function is 49, within the existing 350/75 limits. Three new modules
separate application lifecycle, resource processing, and workflow orchestration.

The [whole-worktree report](../../evals/reports/T-040-intake-budget.json) remains
`blocked_complexity_budget`: other packages, test/declarative totals, production/grand
totals, module count, and existing largest-module/function limits remain exceeded.
No ceilings were raised. The report compares HEAD with the entire shared worktree,
so its overall changes also include work that preceded this task.

The [intake notes](../../src/causal/intake/README.md) describe the remaining eager CLI
runtime construction, dataset-scoped mutable catalogue lookups, and partial-capture /
persistence recovery limitations. This is verification of the intake refactor, not a
claim that the repository has passed its release or whole-system complexity gates.
