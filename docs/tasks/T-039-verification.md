# T-039 verification

The live post-analysis implementation is consolidated in
[`src/causal/post_analysis`](../../src/causal/post_analysis/). The previous
presentation implementation, runtime coordinator, claim judge, curator, mandatory
figure builders and visualization catalogue have been removed from execution.
Scientific support calculations remain in analysis. Shared storage, events,
checkpointing and model transport remain shared services.

## Behavioral coverage

Verification exercises the actual LangGraph author/tool/reviewer loop and runtime
handoff. It covers exact approved design/role/DAG bindings for RCT, AIPW, DiD and
RDD; missing, extra and failed diagnostic records; failure before a numerical
response; exact attempted-plan receipts; compatible sensitivity comparisons with
per-row sources; immutable numeric values, uncertainty and units through rendering;
image-aware review and revision; stale review and tampered export rejection;
idempotent replay; interrupted checkpoints; recovery with already-spent budgets;
and tracing failures before authoring and after an otherwise successful review.

Chart, DAG, wide-table and report previews were rendered and inspected. The final
renderer cleanup was replayed against saved fixtures and preserved exact bytes,
paths and hashes. The composed runtime test reaches numerical completion, report
review and verified delivery. Provider requests are scripted in these tests;
no live model call or live LangSmith transmission was made during this refactor.

The shared tracing tests verify complete application prompts, schemas, responses,
provider-exposed reasoning summaries and actual preview image content; nested
parent spans; credential redaction; and fail-closed trace delivery. This is not a
claim that unavailable internal model thoughts can be captured.

Ruff passes across source, tests and tools. Strict mypy passes all 129 production
source files. A wheel builds successfully; all new post-analysis source and prompt
bytes match the workspace, and the removed execution paths are absent. The full
suite passes **1,811 tests, with 3 skips and 71 warnings**, in 169.45 seconds.
Warnings are retained in the run output, including third-party statistical fixture
disclosures. After the final narrow handoff error-reporting change, all **40
entry/graph tests** pass. Missing required references now identify the exact upstream
source and path before any model call. Final full-source Ruff/mypy and wheel-byte
verification were rerun successfully after that change.

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests tools conftest.py
.venv/bin/mypy src/causal
.venv/bin/python tools/budget_check.py --task-id T-039 --base HEAD --worktree --json
uv build --wheel --out-dir tmp/post_analysis_wheel
```

## Complexity accounting

| Dimension | Actual | Limit |
| --- | ---: | ---: |
| Post-analysis (retained presentation allocation) | 2,229 | 1,500 |
| Production total | 20,658 | 16,500 |
| Grand total | 52,580 | 35,000 |
| Production modules | 129 | 86 |

Post-analysis has 18 production modules. Its largest module/function are 270/75
significant lines, within the 350/75 local limits. The stage allocation is exceeded;
this is recorded as a remaining gate, not hidden through scope reassignment or a
ceiling increase. Shared/design/analysis and other full-worktree dimensions also
remain over their recorded limits. The old presentation directory is still counted
in the historical baseline, so its deletion is not misclassified as shared code.

## Explicit limits

- The active runtime reads the numerical integration artifacts. The standalone
  public analysis API now emits numerical v2 plans/evidence; it is not a second
  post-analysis entry. Historical public v1 plans need fresh compilation/approval.
- The producer does not supply diagnostics for each sensitivity branch or complete
  quantity/unit descriptions for every support series. The report must disclose
  those gaps; charts cannot invent missing units or scientific interpretations.
- Code checks citations and preserves source values. Whether authored prose is
  scientifically entailed remains an LLM-review responsibility, not a proved
  property of a valid citation.
- Historical bundles are readable. Historical failed presentation runs and the
  old claim/template gold rubric require explicit migration before new execution
  or comparable evaluation.
- Complexity ceilings remain unchanged. The machine-readable
  [budget report](../../evals/reports/T-039-post-analysis-budget.json) includes the
  full existing dirty worktree, so its HEAD delta is not solely this task's work.
  Functional verification is separate from the unresolved complexity and
  product-release gates.

Changes remain in the shared worktree. No broad checkpoint commit was created
over the user's pre-existing design, analysis and intake changes.
