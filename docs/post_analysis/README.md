# Post-analysis

The implementation lives in `src/causal/post_analysis/`. It owns interpretation,
visual choices, report composition, visual inspection and delivery. One LangGraph
controls the workflow; shared services own persistence, tracing and model transport.

1. [Contract](contract.md): exact inputs, immutable facts and controlled failure paths.
2. [Graph](graph.md): nodes, tools, budgets, review and recovery.
3. [Coverage](coverage.md): what the live analysis producer supplies and remaining gaps.
4. [Refactor](refactor.md): ownership, deleted paths and historical compatibility.
5. [AGENTS.md](../../src/causal/post_analysis/AGENTS.md): instructions for future changes.
6. [Verification](../tasks/T-039-verification.md): test evidence and remaining gates.

The three reporting outcomes are `complete`, `blocked`, `incomplete`. An explicit
failed diagnostic is evidence; a missing expected diagnostic is an upstream defect.
Visualization decisions are made here after reading the evidence.
