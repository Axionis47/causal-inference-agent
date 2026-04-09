# Bounded analysis contract

## Scope and status

New runs use `bounded-agents-1`. Historical runs retain their original contracts and remain readable.
This replacement covers analysis, its applied context inputs, and saved result discussion.
The [verification guide](verification.md) describes numerical checks and acceptance procedures.
Implementation does not establish numerical or scientific correctness by itself.

## Applied context boundary

The applied context pack is the agreed study input when analysis starts.
The service binds the exact dataset version, checksum, context revision, context hash, brief, question, and prior exposure.
Launch checks those identities and the submitted readiness fingerprint transactionally.
Recovery verifies integrity without reopening context preparation.

Agents can retrieve existing keys from the frozen pack.
They cannot add sources, edit context, fetch study evidence externally, or request renewed context review.
Necessary conditional assumptions remain explicit. Optional unknowns do not block progress.
A genuinely unsupported or unidentified request produces a final outcome under the same frozen inputs.

## Four agents and accepted handoffs

| ID | Agent | Tools and responsibility | Accepted handoff |
| --- | --- | --- | --- |
| `study` | Study understanding | Bind the target, factors, typed values, assignment processes, and comparisons through small submissions | Source-linked study specification, inventory, independent review, unresolved issues |
| `design` | Analysis design | Assess relevant roles, structure, identification, and source relationships; freeze the causal account before diagnostics; specify the method | Frozen snapshot, assumption ledger, diagnostic evidence, independent review |
| `computation` | Computation | Inspect canonical input headers, submit code, obtain independent review, wait for exact approval, validate saved execution | Execution identity, validated facts, artifacts, program and snapshot hashes |
| `report` | Results explanation | Read saved numerical evidence, draft a grounded explanation, obtain independent review | Saved report, numerical references, limitations, and frozen assumptions |

Each agent has a private model/tool/observation loop in LangGraph.
Agents can revise rejected proposals and request bounded evidence-reading subtasks.
Accepted upstream findings travel by producer identity and content hash, without upstream conversations.
The backend assembles typed submissions and checks scientific invariants before publication.
An accepted handoff is immutable. Its envelope includes schema version, frozen input identities, dependency references, evidence, assumptions, and unresolved issues.
Reviews and evidence readers have separate sessions charged to their parent agent.

All columns remain inventoried. The backend retains unused columns as unresolved and withheld.
There is no mandatory model call for every column.
A treatment used in another effect retains its already accepted variable binding.

## Source authority and qualifications

Supplied facts, interpretations, conditional assumptions, and computed evidence have separate authority.
A supplied claim selects one compatible reviewed source record. The backend restores its exact content.
Multiple supporting records retain separate references and excerpts.
A definition does not establish a causal role, and an assignment rule does not establish an observed treatment effect.

Relationship selections reference exact frozen records.
The backend preserves original endpoints, conditions, qualifications, and timing alongside the agent's applicability explanation.
Assignment rules retain participants without inventing directed graph edges.
Unknown optional properties remain unknown.
Only necessary, supported causal obligations can admit explicit working assumptions before causal freeze.
Neither diagnostics nor reporting can introduce new identifying premises afterward.

Native claim submissions can omit empty prerequisite lists.
An omitted non-assumption obligation defaults to an empty string.
These defaults confer no authority. A working assumption must still name its exact required obligation.

## Bounds

| Resource | Ceiling |
| --- | --- |
| Study understanding | 10 model calls |
| Analysis design | 16 model calls |
| Computation | 14 model calls |
| Results explanation | 8 model calls |
| Concurrent model calls per run | 3 |
| Handoff submissions per agent | 3 |
| Evidence-reading subtask | 2 calls, charged to its parent |
| Independent review | 3 calls, charged to its parent |
| Diagnostic programs | 4, including failures |
| Program revisions, including human corrections and execution repairs | 2, each separately reviewed and approved |
| Retrieved evidence per request | 32,000 characters |
| Complete native request | 192,000 characters, including schemas and provider signatures |
| Model output | 8,192 tokens; 16,384 for computation |

These are ceilings, not required work counts.
Every call is charged before dispatch. A failed or uncertain provider call retains its charge.
Subtask exhaustion leaves any remaining parent allowance available.
No retry, renamed task, or restarted worker creates credits.

## Persistence and recovery

`AnalysisTask` owns each session's immutable packet, provisional records, submission count, and accepted handoff.
`AnalysisTaskAttempt` owns the reserved request, native response, usage, and tool receipts.
PostgreSQL constraints protect completed responses, settled tools, accepted handoffs, and spent allowances.
The additive migration preserves historical records.

The worker holds a run advisory lock. Procrastinate delivers jobs; LangGraph checkpoints retain workflow position.
Native provider messages, function-call identities, opaque signatures, and function responses are replayed in order.
Completed tools return their saved receipts rather than executing again.
A lost provider response becomes uncertain and remains charged.
Accepted upstream agents do not rerun.

Provider interruption permits retry while allowance remains.
Malformed arguments and rejected scientific submissions return targeted feedback to the responsible agent.
Exhausted allowances and unsupported conclusions remain visible final outcomes.
Pending saved tool work can settle even when no new model call is available.

## Freeze, approval, and computation

Diagnostics use a physically restricted input projection after causal freeze.
Target outcomes remain withheld until exact program approval.
A program's hash binds the source, frozen snapshot, runtime, verification, and independent review.
Approval names that exact hash and snapshot. A stale or different program cannot execute.

Execution uses a deterministic identity derived from the approved program hash.
Saved runner receipts support recovery without refitting.
Validated results must satisfy the frozen output contract.
A reviewer can request a correction to the exact proposed program. The rejected version and feedback remain saved; only the computation agent resumes, within its original allowance. Each replacement needs independent review and exact approval.
Programming repairs receive the error without failed target values and cannot change the science.
A scientific failure cannot authorize a replacement estimator.

Static checks cover Python names, control flow, recognized numerical fact writers, and certain explicit probability arguments.
The probability check rejects a resolvable SciPy interval coverage that contradicts one explicitly frozen confidence percentage.
SciPy's interval method takes confidence coverage, rather than significance probability. See the [SciPy interval API](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.rv_continuous.interval.html).
Dynamic expressions, other formulas, and scientific meaning still require independent review and numerical verification.
No static check proves complete statistical correctness.

Computation produces `facts.json`, a result chart, and a numerical table when the frozen result supports them.
The report uses saved numerical handles and carries the existing assumption ledger.
Saved chat can discuss authorized reports, programs, evidence, and downloadable artifacts without recomputing results.

## Interface

The primary Analysis journey is question, progress, approval, then results.
Agent transcripts, budgets, native tool receipts, scientific validation, and historical checkpoints remain under optional inspection.
The event stream reports stage starts, accepted handoffs, approval requests, execution status, and failures.
The agent detail endpoint is owner-scoped. Frontend contracts are generated from OpenAPI.

## Implementation sources

- [Runtime and orchestration](../backend/analysis/agents.py), [native loop](../backend/analysis/agent_runtime.py), [durable store](../backend/analysis/agent_store.py)
- [Study tools](../backend/analysis/agent_study.py), [design tools](../backend/analysis/agent_design.py), [computation tools](../backend/analysis/agent_computation.py), [reporting tools](../backend/analysis/agent_report.py)
- [Source admission](../backend/analysis/authority.py), [numerical API checks](../backend/analysis/numerical_api.py), [public service](../backend/analysis/service.py)
- [LangGraph subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs), [Vertex function calling](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tools/function-calling)
