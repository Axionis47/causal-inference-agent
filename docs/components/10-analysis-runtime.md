# 10. Shared analysis runtime

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md) · [Evidence and discrepancies](evidence.md)

## Purpose and boundary

This is the shared code that helps all three analysis workflows run their steps in order, check model answers, handle failures and save results. In this handbook, **execution machinery**, **harness** and **shared runtime** refer to that supporting code.

For example, adjustment may need the model to choose among several allowed estimators. The adjustment component supplies the choices and the rules for judging them. The shared code gives the model its instructions and permitted tools, records what it returns, runs the supplied checks, and asks again or stops if those checks fail. After acceptance, the workflow continues to its next step. The family's engine later performs the statistical calculation.

There are three responsibilities to keep separate:

| Responsibility | Concrete question it answers | Owner |
|---|---|---|
| Analysis reasoning | What can this study support, and which analysis specification is justified? | The specialist family's reasoning code and model judgments |
| Workflow support | How do we run those steps, check their answers, record attempts and handle a stop? | This shared component, using checks supplied by the family |
| Statistical execution | How do we calculate the effect and uncertainty from the agreed specification? | The specialist family's execution code and engine |

The shared code operates throughout the workflow. The narrower term **statistical execution** means the calculation after the analysis specification has been fixed. Other terms used below are explained in the [plain-language guide](terms.md).

## Internal parts

| Part | Current implementation | Responsibility |
|---|---|---|
| Workflow declarations | [rungs](../../causal_agent/lane/rungs.py), [graph](../../causal_agent/lane/graph.py), [state](../../causal_agent/lane/state.py), [ladder](../../causal_agent/lane/ladder.py) | Load graph shape, create nodes/transitions, common state/records |
| Intake/admission | [intake](../../causal_agent/lane/intake.py), [endpoints](../../causal_agent/lane/endpoints.py) | Apply declared scope and validate admitted context |
| Episodes/tools | [episode](../../causal_agent/lane/episode.py), [tools](../../causal_agent/lane/tools.py) | Bounded calls, tool facts, typed answers, gate attempts |
| Evidence | [ledger](../../causal_agent/lane/ledger.py) | Entries, provenance and accepted revision history |
| Knowledge and stops | [knowledge](../../causal_agent/lane/knowledge.py), [space](../../causal_agent/lane/space.py), [nodes](../../causal_agent/lane/nodes.py), [terminal](../../causal_agent/lane/terminal.py) | Declared capabilities, support, common stop/status machinery |
| Output | [records](../../causal_agent/lane/records.py), [figures](../../causal_agent/lane/figures.py), [prompts](../../causal_agent/lane/prompts.py), [words](../../causal_agent/lane/words.py) | Common result envelope, figure gates and shared language |

## Inputs and outputs

The graph builder takes the declared step order (`LadderSpec`), the functions that perform those steps, and the definition of the information passed between them. It connects those parts into an executable workflow.

A **model episode** is one bounded task for the model. The calling component supplies instructions, permitted evidence and tools, a required answer format (schema), and checks (gates). The episode returns either an accepted answer or a failure, together with an `EpisodeLog` recording its attempts.

At the end, result assembly collects the workflow's saved information into the result dictionary the desk reads.

## Contracts

| Boundary | Enforcement | Failure |
|---|---|---|
| Rung declaration → graph | `graph.build` implements exits, workers, stops and tail; family ladder tests compare graph/declarations | Invalid/mismatched structure fails checks |
| Model → accepted record | Schema validation, call/try budgets, address gate and optional evidence/completion gates | Refusals retained; exhaustion yields no accepted record |
| Evidence → ledger | `ledger.apply` checks resolvable sources and causal-evidence authority | Rejected changes do not silently promote proposals |
| Reasoning → execution | [FrozenDesign](../../causal_agent/common/contracts/frozen.py) seal and family input checks | Changed nested specification fails integrity checks |
| Lane → result | `records.result`, `records.artifacts`, `terminal.status_of` | Typed deficiency/feasibility plus explicit status for non-success |

The result envelope is currently assembled as dictionaries with shared keys, not one exhaustive strongly typed cross-family result schema. `RunRecord` wraps that result for the desk.

## Decision authority

Code enforces admissible shapes, budgets, provenance and allowed transitions. Models judge within scoped evidence and offered choices. Family declarations supply the causal policy. The shared runtime never interviews a person.

## Dependencies

Lives below exploration and family packages in the import layers. Shared data tools are in `lane`; sandboxed code tools are supplied from `explore` by family callers. Family graphs use this builder, while normal desk execution invokes them in a child process.

## State and storage

Episode logs, addressed facts, ledger history and candidate transitions are retained in lane state and assembled records. Analysis files go through `trace.write`. Standalone tests can compile with in-memory checkpoints; lane execution inside the pipeline has no user interrupt/checkpoint conversation of its own.

## Worked walkthrough

1. A specialist reaches a step that needs a model judgment, such as choosing an allowed estimator.
2. It prepares the information that judgment may use and the tools the model may call.
3. The shared episode code sends the task, runs requested tools within the budget, and records their results.
4. The model returns an answer in the required format, including evidence references.
5. A check finds a reference that does not exist. The episode records the rejection and gives the model the reason for another attempt.
6. If the answer passes, the workflow continues. If the allowed attempts are exhausted, it records a stop.
7. The family assembles its result with the accepted decisions, attempts and reasons, so the desk can explain what happened.

## Verification

**Code-verified:** graph builder, episode and shared record interfaces. **Test-supported:** [episodes](../../causal_agent/lane/tests/test_episode.py), [ledger](../../causal_agent/lane/tests/test_ledger.py), [tools](../../causal_agent/lane/tests/test_tools.py), [rungs](../../causal_agent/lane/tests/test_rungs.py), [family ladders](../../causal_agent/families/tests/test_ladders.py) and [trace policy](../../causal_agent/common/tests/test_trace_policy.py). Exact supported trajectories remain in [generated ladders](../../docs/ladders), not duplicated here.
