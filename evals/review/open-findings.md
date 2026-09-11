# Session issue register

This is the side record of issues observed during design, implementation, and live-data evaluation. It is historical: an item remains listed even if a repair is underway or has since landed. A validation wall correctly rejecting an unsafe output is recorded separately from the product successfully completing the journey.

## Overall result observed

- The first eight-case live evaluation completed **0 of 8 journeys successfully**. Some safety walls worked correctly, but safe rejection is not a successful product outcome.
- Later four-case runs still failed to complete all four release journeys. Provider truncation, semantic-output failures, graph failures, and downstream compiler incompatibilities remained visible.
- Deterministic and scripted tests demonstrated infrastructure behavior, but they did not by themselves prove that the live LLM could complete one real dataset from each analysis class.

## Cross-cutting design and agency issues

### SI-001: The LLM did not participate in a statistical investigation loop

The method worker initially chose a method before deterministic diagnostics ran. Diagnostics were executed afterward, so the LLM never saw their results and could not revise its design. This made the system model-assisted rather than genuinely agentic at the method boundary.

Required repair: provisional proposal, bounded diagnostic request, deterministic execution, results returned to the model, typed assessment, revised/final proposal, and deterministic compilation. The model must have no arbitrary code or argument surface and a hard tool-call limit.

### SI-002: Some workers received the wrong amount or shape of context

The system sometimes held useful information but presented only names, references, or an incomplete parent artifact. Earlier examples included evidence text reduced to a filename, table-profile statistics not shown to semantic workers, and role workers not receiving the semantic cards whose timing and meaning they needed.

Required repair: every task gets a purpose-built, bounded briefing containing the exact authoritative facts, evidence text, registered vocabulary, parent decisions, user answers, remaining budgets, and relevant diagnostic results needed for that decision.

### SI-003: Canonical reference ergonomics caused repeated safe failures

Workers paraphrased or mechanically altered valid references instead of copying their canonical IDs. Observed forms included case-only evidence aliases, uppercase README references, table-qualified column names, and the DiD role worker paraphrasing a concept ID. The reference wall correctly rejected these and, in the DiD instance, supplied the exact allowed vocabulary.

Required repair: present canonical IDs next to their human-readable meaning, make field expectations explicit, deterministically normalize only unambiguous mechanical aliases, and retain exact rejection for ambiguous or invented references.

### SI-004: Mechanical output fields consumed semantic correction budgets

The model repeatedly emitted treatment timing as `pre_treatment` when the product contract required `concurrent`. Other shape failures involved misplaced or missing `timing` and `units` slots. These were mechanical serialization failures rather than useful causal disagreements.

Required repair: compiler-own derivable mechanical values, simplify response schemas, and reserve model corrections for genuine semantic judgment.

### SI-005: The causal-synthesis task was overloaded

A single synthesis boundary was responsible for both causal-context construction and role-ledger binding. This mixed graph reasoning with executable column-role commitments and made corrections less targeted.

Required repair: separate `causal_context` and `role_ledger` tasks with different prompts, schemas, evidence needs, and validation messages.

### SI-006: Graph and role consistency failures were common

Observed failures included `graph_cycle`, `confounder_edges_missing`, `disputed_edge_without_alternative`, missing required graph relationships, and incorrect treatment/running-variable role overlap. The walls generally rejected these safely, but the prompts and context did not make the legal graph commitments easy enough to produce.

Required repair: show the closed concept/edge vocabulary and role-specific graph obligations directly in each prompt, then validate required and forbidden relationships deterministically.

### SI-007: Missing-requirement contracts were hard for the model to use

Observed failures included `invalid_context_requirement`, invented requirement IDs, a `needs_context` response without required missing-requirement rows, and unnecessary context stops. At times the model treated `requirement_id` as a new instance handle instead of selecting a registered policy ID.

Required repair: expose only task-relevant registered requirement IDs, show the expected `(requirement_id, scope_id)` form, and compile questions deterministically. If the fact is not knowable from offered evidence, the system should ask; otherwise it should not.

### SI-008: Candidate and role references needed a stronger early wall

The intent/semantic path could emit candidate columns or role bindings that did not exist exactly in the selected CSV. Some failures then surfaced later as `unresolved_column` rather than at the original decision boundary.

Required repair: reject nonexistent `candidate_columns` and unresolvable role columns immediately, while allowing only unique mechanical alias normalization.

### SI-009: Model-authored prose weakened deterministic safety

Final claims, assumptions, risks, sensitivities, captions, and qualifications contained too much free-form model-authored language. That made it harder to guarantee claim ceilings, preserve every qualification, and prevent invented numbers or unsupported statements.

Required repair: the model selects registered finding, qualification, assumption, risk, sensitivity, alternative-explanation, and cannot-conclude IDs. Deterministic compilers own final wording and all numeric rendering.

### SI-010: The figure curator had a nonfunctional tool surface

The curator advertised a model-facing layout tool that was not actually part of a useful executable interaction. This created the appearance of agency without a functioning tool loop.

Required repair: remove the tool, hydrate bounded layout facts into context, allow only registered evidence/template/panel choices, and keep the curator tool budget at zero.

### SI-011: Correction and escalation needed an explicit bounded contract

Some correction messages were useful and named the failing path and allowed vocabulary. Others carried too little detail, repeated the same failure, or exhausted the correction budget without distinguishing a model-repairable error from a compiler/provider failure. During diagnostic-loop work, a fail-open branch was also identified where an unbindable fact set or method could look like an empty successful diagnostic result; valid investigation turns and repair attempts were initially counted together.

Required repair: stable error code, exact JSON path, rejected value, allowed set, responsible actor, permitted repair action, independent correction count, and explicit escalation after exhaustion. An unexecuted requested diagnostic must never count as successful investigation.

### SI-012: Observability did not initially prove the agentic behavior

Transport-level tracing did not clearly show the bounded sequence of model decision, diagnostic request, deterministic result, revision, and escalation. Raw transport tracing also risked retaining more prompt material than necessary.

Required repair: sanitized gateway spans and explicit `agent.diagnostic_requested`, `diagnostic.completed`, `agent.design_revised`, and `agent.escalated` events. Store hashes, IDs, counts, model profile, correction attempt, and evaluation identifiers, never raw rows, credentials, hidden reasoning, or gold labels.

## Release-journey failures

### Lalonde / NSW randomized experiment

- Provider returned `model_output_truncated` in multiple runs.
- Evidence references were mechanically altered and rejected as `unresolved_evidence`.
- Treatment timing was emitted incorrectly and rejected as `treatment_timing_invalid`.
- A disputed causal edge was emitted without the required alternative explanation.
- Earlier designs struggled to express row-derived unit identity when one row genuinely represented one randomized unit.
- In one later run the case stopped before all eight model-decision boundaries, causing the evaluator to report many downstream missing-artifact symptoms.

### Groupon AIPW

- Provider returned `model_output_truncated` in one run.
- Treatment timing failed the concurrent-treatment contract.
- Confounder candidates were bound without the graph edges required to support their role.
- Invalid or unnecessary missing-context requirements were emitted.
- A stopping state outside the task’s allowed states caused `schema_invalid`.
- Shape validation failed on a timing slot.
- The journey did not reliably reach compiled AIPW estimation and presentation.

### Minimum-wage difference-in-differences

- The role worker paraphrased a concept ID; the reference wall safely rejected it and returned the exact allowed vocabulary.
- The period column was returned as a table-qualified alias and rejected as `unresolved_column` before unique normalization was introduced.
- Causal graphs contained cycles and exhausted correction attempts.
- Treatment timing and, in one run, running-variable timing were invalid.
- Required comparator, treatment role, time role, timing, adoption fact, and causal relationships were not consistently selected.
- The design/preparation/estimation handoff failed with `estimator_schema_mismatch`.
- Required parallel-trends qualification and downstream figures could not be evaluated when the upstream design failed.

### Senate sharp regression discontinuity

- Required causal relationships were missing.
- Treatment and running-variable roles were incorrectly bound to the same column in one proposal.
- Treatment timing was invalid.
- Confounder candidates lacked required graph edges.
- A `needs_context` response omitted the required missing-requirement contract.
- Preparation failed with `unknown_rule_target`.
- Cutoff, sharp-assignment, role, claim, and figure expectations became unscorable after the upstream failure.

## Stress-journey failures

### Resume-audit randomized experiment

- Unit identity did not match the accepted design.
- Evidence aliases and treatment timing failed validation.
- Semantic shape validation failed.
- Downstream routing failed with `draft_shape_invalid`.

### NHEFS AIPW

- README evidence was returned with a case-changed reference and rejected.
- Confounder roles lacked required graph edges.
- Treatment timing and missing-requirement output were invalid.
- Corrections exhausted at the role-ledger boundary.

### Card and Krueger DiD

- Semantic shape and treatment timing failed.
- Invalid missing requirements were emitted.
- Causal graph cycles remained after correction.
- The journey terminated `needs_context` instead of reaching its declared analysis outcome.

### Head Start RDD

- A table-qualified outcome column was rejected as unresolved.
- Comparator and required graph relationships were incorrect or missing.
- Preparation failed with `unknown_rule_target`.

## Downstream compiler and execution issues

### SI-013: Design-to-estimator schema mismatch

The minimum-wage DiD design produced an estimator schema that the next stage did not accept. This is a compiler-contract incompatibility, not something the LLM should repair by guessing a different payload.

### SI-014: Registered preparation rules did not cover compiled targets

RDD journeys reached `unknown_rule_target` during stabilization. The design compiler could name a rule target that the preparation operation registry could not execute.

### SI-015: Coordinator draft shape mismatch

The resume-audit journey reached `draft_shape_invalid` at the stage coordinator, showing that independently valid upstream artifacts did not match the downstream routing contract.

### SI-016: Data manipulation boundaries were not yet explicit enough

The system needed a closed operation vocabulary for selection, casting, derivation, reshaping, eligibility filtering, missing-value handling, exact deduplication, categorical encoding, and lineage. High-impact operations such as aggregation, overlap trimming, weight clipping, donut RDD exclusions, and ambiguous deduplication must require explicit contracts and impact previews rather than arbitrary dataframe code.

## Evaluation and infrastructure issues

### SI-017: Live modes initially lacked required observability credentials

LangSmith credentials were missing during part of the session. Live calibration and gate modes correctly failed closed, but this blocked proof on real datasets until the environment was configured.

### SI-018: Provider truncation was treated as a terminal journey failure

Large structured responses from Gemini 2.5 Flash were truncated. The system surfaced `model_output_truncated`, but prompt/schema size and response-token pressure need reduction so normal journeys do not depend on exceptional output capacity.

### SI-019: Evaluator failures could cascade into noisy secondary failures

When a journey failed before producing `DesignFactSet`, `CompiledDesign`, claims, or figures, scoring sometimes reported every absent downstream gold expectation. This obscured the primary failing boundary.

Required repair: preserve hard failure status but distinguish the root failure from downstream `not_reached` checks.

### SI-020: Development reruns were expensive and slow

Eight-case stress runs took roughly an hour or more and repeated unaffected boundaries. Recorded token use ranged widely, with full runs costing a few dollars rather than cents. Development needs affected-boundary reruns while release gating must still run every frozen case exactly once.

### SI-021: Scripted success was initially mistaken for product success

Repository tests and deterministic fixtures could pass while live LLM journeys failed. Acceptance must require terminal success on actual datasets, not merely schema tests or successful infrastructure startup.

## Implementation-quality issues

### SI-022: Code-size reporting was initially misleading

A broad line-count report included material outside the intended production scope and produced an alarming total around 132,476 lines. The governed budget checker’s scoped significant-line count is the relevant measure. Reports should always separate production, tests/tools, declarative configuration, documentation, generated files, and dependencies.

### SI-023: Complexity ceilings left almost no integration margin

The repository reached or approached the production-line, design-line, test-line, module-count, module-size, and function-size ceilings while the diagnostic loop was being connected. The first controller implementation exceeded the harness module and total budgets.

Required repair: delete obsolete synthesis/tool/compatibility paths, consolidate repeated controller behavior, and keep one execution path rather than increasing the approved ceilings or stacking another patch layer.

### SI-024: Evaluation policy initially conflicted with bounded tools

The evaluator originally required every model call to have an empty tool allowlist. That was correct for the old no-tool architecture but inconsistent with the intended method worker’s single bounded statistical-diagnostic capability.

Required repair: allow exactly one generic diagnostic capability for `method_design`, require zero tools for all other model boundaries, and verify that requested diagnostic IDs remain compiler-bound.

## Safety behavior that worked

- Invented, paraphrased, ambiguous, or case-altered references were rejected instead of silently accepted.
- Graph cycles, missing confounder edges, invalid timing, role overlap, unknown rule targets, and schema incompatibilities produced stable failure codes.
- Missing observability caused live evaluation to fail closed.
- The correction response could supply the exact allowed reference vocabulary.
- The deterministic system did not permit arbitrary model-authored dataframe code.

These are necessary safety properties, but none should be counted as journey success until the agent repairs the issue or escalates to the correct human/system owner and the case reaches its declared terminal outcome.

## Repair order

1. Make the bounded diagnostic investigation loop real and observable.
2. Fix prompt/context ergonomics for canonical references, timing, graph obligations, and missing-requirement contracts.
3. Repair DiD estimator-schema and RDD preparation-rule compatibility.
4. Reduce prompt/schema size to prevent provider truncation.
5. Make evaluation report root failures separately from downstream `not_reached` checks.
6. Run one real dataset per analysis class, then the fresh four-case release gate.
