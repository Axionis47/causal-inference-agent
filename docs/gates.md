# Judgements and code gates

Every model call goes through `common.llm.structured`, `common.llm.decide`, or `lane.episode.run_episode`. The trace retains its prompt, returned record, model summary, tool calls, each gate attempt and its errors. Files written under a design/run directory go through `trace.write`.

| Call | Evidence | Output | Code authority |
|---|---|---|---|
| Desk reader | Initial material or human words, catalogue, unresolved fields | Sourced draft or explicit human update | `memory.ops.apply` controls status and beliefs |
| Frame | Question, column notes and profile | Target, scope, relevant columns | Names and addresses must resolve |
| Family selection | Surviving structural families and requirements | Chosen family with reasons | Only offered families can be selected |
| Context readiness | Endpoint catalogue and source/status fields | Endpoint admissions | Pure code, no model-generated eligibility rules |
| Advisory design brief/judge | Prepared context and its admitted requirements | Explanation and caveats | Cannot grant missing admission |
| Lane episode | A scoped view and permitted read-only facts | Typed proposal or assumption | Source, citation, transition and bounded retry gates |
| Identification/support | Structural graph or assignment account and permitted diagnostics | Certificates, support, candidate decisions | Registered family predicates and engines |
| Freeze | Pre-effect candidate decisions, target, rows, weights, estimator and inference | Sealed designs and fixed lead | Integrity checked before every fit |
| Interpretation/chat | Design-qualified estimates, checks, ledgers and artifacts | Cited explanation | Every citation resolves and stated numbers match |

The model gets three gate attempts. Every refusal has a trace gate with its reason. A declared pre-freeze revisit is bounded to one; its episode and evidence addresses are unique. Exceeding essential structural/specification bounds produces a visible limitation. A lane never invents a new endpoint or changes the target after an effect is seen.

A diagnostic cannot prove a causal assumption, erase a supplied hidden cause, or turn a draft into a confirmed account. Unknowns and contradictions are admitted only under explicit endpoint policies. Hard diagnostic failures stay hard under human reassertion.

Unexpected essential missing context returns a typed deficiency. The desk takes the human clarification through memory gates, prepares a new snapshot and waits for another run authorization. It does not silently resume an already authorized lane on changed context.

See [architecture](architecture.md) and [generated ladders](ladders/) for responsibilities and exact declarations.
