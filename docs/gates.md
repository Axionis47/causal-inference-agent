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

When adjustment has several eligible estimators, its selection episode may run three arbitrary Python diagnostic scripts in the isolated sandbox. The supplied table contains treatment and identified adjustment variables without outcome values. The returned estimator and design identifiers remain constrained to the offered options; citations must resolve, and each pick must cite a successful code result with nonempty, finite numerical facts. Failed scripts remain visible and cannot support a choice. Missing numerical evidence is reported after each script; exhausting the code budget without usable evidence stops before a final model answer. Generated diagnostic code does not change the endpoint, target, graph, adjustment set or admitted population. The numerical evidence implementation is shared in `explore/diagnostics.py`.

DiD’s existing `probe` rung and discontinuity’s existing `line` rung can optionally use the same isolated coding tool, each with three calls. Sufficient existing evidence permits a no-code judgment. Once code is attempted, a successful finite numerical probe must be cited; failed scripts cannot be ignored. The permitted CSV column names and usable probe addresses are shown explicitly.

DiD preserves all panel metadata rows and exposes outcomes and confirmed safe covariates only before the earliest adoption anywhere. Later never-treated and not-yet-treated outcomes are withheld too. `ProbeReading.diagnostic_cites` records numerical findings without requiring a new causal-ledger operation. Discontinuity preserves admitted rows independently of outcome missingness and exposes only score, eligibility, optional receipt/cluster and approved predetermined covariates. Outcomes and known outcome proxies are excluded. Source-role and reserved-name checks prevent metadata aliases from bypassing these restrictions. Standard estimator, inference, bandwidth and window policies retain their existing authority.

Unexpected essential missing context returns a typed deficiency. The attempt stops visibly. If the person supplies a correction, the shared context gate commits a new revision. Preparing it and authorizing a new run are explicit actions. It does not silently resume an already authorized lane on changed context.

See [architecture](architecture.md) and [generated ladders](ladders/) for responsibilities and exact declarations.
