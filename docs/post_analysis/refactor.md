# Refactor ownership

| Responsibility | Single destination |
| --- | --- |
| Input validation and frozen joins | post_analysis/input.py, input_sources.py and input_tables.py |
| Contract, actions, report and issues | post_analysis/contracts.py |
| Autonomous workflow | post_analysis/graph.py and agent.py |
| Scientific/display review | post_analysis/review.py and resources/review.v1.txt |
| Visual encoding and rendering | post_analysis/visualization/ |
| Report pages/export | post_analysis/presentation/ |
| Entry, recovery and CLI dispatch | post_analysis/entry.py and runtime.py |
| Shared tracing/model transport | shared/tracing.py and gateway.py |

Removed the old presentation coordinator, harness, catalogue, compiler, renderer,
curator and prompts; the numerical claim judge and its prompt; the old runtime
presentation implementation; and the obsolete visualization catalogue. Useful
safe-rendering behavior was consolidated into the new visualization package.
The analysis coordinator now closes a numerical bundle immediately after collecting
numerical evidence. It does not invoke a claim LLM or mandatory figure builders.

Runtime composition and evaluation recovery call post_analysis.runtime. CLI export
delegates to post_analysis.presentation.delivery. Old bundle bytes remain readable
through a small type-dispatched delivery reader; no old execution pipeline remains.
Stage-specific tests live under post_analysis/tests; shared services retain shared tests.

## Explicit compatibility boundaries

- `presentation.runs`, the CLI `presentation` command and shared Stage.PRESENTATION
  remain operational names. Renaming database history is not required for ownership.
- Historical artifact registry entries remain so committed records can be read.
- Legacy numerical schemas/metadata still serialize figure/capacity fields where
  removing them would invalidate saved approvals or change hash-derived numerical
  seeds. They do not direct the new reporting agent.
- Public analysis now emits v2 numerical plans/evidence with raw supporting_data.
  Historical v1 plot-bearing records remain readable, but execution requires a
  new compilation and approval. Explicit numerical seeds are preserved.
- New design outputs contain no required plots and select no report templates.
  Their cardinality summaries remain descriptive. Saved old approvals remain
  intact; post-analysis ignores their historical visual prescriptions.
- The live evaluator in `tools/model_quality.py` still loads the approved
  `human-gold.v1` and `development-expectations.v1` rubrics. Those require the removed
  `ClaimJudgment`, `FigurePlan`, template IDs and claim/figure task boundaries.
  When current `PostAnalysis*` artifacts or author/review calls are present, scoring
  stops with classification `rubric_compatibility` and issue
  `post_analysis_rubric_migration_required`. The evaluation gate cannot pass; this
  is an incompatible rubric, not a scientific or rendering failure. Actual terminal
  outcomes and recorded call metrics remain available in the report. Historical-only outputs
  retain their original scoring. No gold labels, approval hashes or scores are
  rewritten; current reports need a separately reviewed and approved rubric migration.

The dead numerical figure builders, visual point helpers, claim ceiling computation
and duplicated capacity recheck were deleted. Scientific supporting computations
such as RDD bins/densities and DiD event estimates remain upstream. Their legacy
field names do not authorize a display choice.
