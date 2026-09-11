# Post-analysis boundary — minimal contract

**Status: implemented live boundary; producer gaps and compatibility are listed in [coverage](coverage.md).**
This is the current boundary specification. It replaces the larger draft that
defined a separate handoff envelope, capability catalog and operational protocol.

## 1. One entry through the common harness

The stage opens from one **handoff ID**, resolved by the common harness. Reuse
the existing artifact envelopes, exact references, handoff manifest, storage,
model-task envelopes and operational lifecycle. The current entry delivers a fixed
HTML/SVG/PNG report with accessible source tables; audience, language and export-format
preferences are not yet public inputs.

The harness supplies exact references; the post-analysis reader derives a typed,
read-only input view. It references original artifacts and does not create a second
editable scientific record. The generic handoff gate checks transport; the reader
owns the scientific joins described below.

| Input section | Required contents | Authority |
| --- | --- | --- |
| Source reference | Existing handoff identity/version and exact artifact references belonging to this analysis | Common harness |
| Approved context | Scientific question/targets, population/time, column meanings and assigned roles, causal structure and sourced assumptions/limitations | Design |
| Analysis attempts | Exact approved request plus actual analysis, diagnostic and sensitivity evidence; or exact attempted request plus an explicit failure receipt when no response exists | Analysis, packaged by harness |
| Supporting data | Scientific tables/quantities referenced by the evidence, with units, dimensions, uncertainty meaning and scope; explicitly empty when none was produced | Analysis |

Do not reconstruct the request from results. Do not join an old result to the
latest design. Do not request upstream charts, diagram layouts or report order.

## 2. Minimum scientific meaning

| Item | What must be unambiguous |
| --- | --- |
| Question/target | Estimand, treatment/comparator, outcome, contrast direction, population and time scope |
| Columns and roles | Stable column/variable references, actual field names, meaning, coding/units, assigned role and target scope; any prepared-field mapping is traceable |
| Actual use | Which fields each fit or diagnostic used; assigned role alone does not prove use in the effect fit |
| Causal structure | Exact approved graph reference, nodes linked to variables or explicitly defined concepts, directed edges and sourced assumptions; layout is downstream |
| Result quantities | Quantity meaning, value type, units/scale, uncertainty type/method, population/denominator and computation scope |
| Diagnostics/sensitivities | Actual status, criterion/threshold where used, interpretation limits, exact scope; a sensitivity identifies its parent and change |

A causal graph is required by the current live handoff. A future producer that
does not create a graph needs an explicitly versioned absence contract; no such
fallback is currently accepted. A promised graph with a missing object is an input defect. Post-analysis never
creates new causal edges to complete it; it may choose how to display supplied
relationships. A layout is not evidence of a causal relationship.

Scientific assumptions retain their sources and uncertainty. Approval is not
proof. An unknown identifying assumption can qualify a report; missing essential
outcome coding or contrast identity prevents reliable interpretation.

New diagnostic names do not require new post-analysis graph branches or catalog
entries. They require a supported payload schema and sufficient authoritative
scientific descriptors. The LLM reasons about those descriptors and findings.
Missing meaning is a source issue; lack of receiver support is a receiver issue.

## 3. One expected-versus-received check

The common harness resolves and checks transport/lineage. Post-analysis then
performs a small deterministic scientific input check before any authoring LLM
call. Reuse upstream schemas and existing validation machinery where possible.

| Check | Expected source | Reject when |
| --- | --- | --- |
| Identity and revision | Handoff and approved source references | Wrong analysis, wrong revision, unresolved required artifact or inconsistent binding |
| Context/request compatibility | Approved question, roles, coding and target | The requested computation addresses a different target or contradicts the context |
| Column/DAG mapping | Context and preparation references | Used fields or graph concepts cannot be resolved, or declared roles disagree |
| Request/result binding | Exact approved request and execution references | Evidence comes from another request, dataset or attempt |
| Computation coverage | The actual compiled request's inventory | Missing expected record, duplicate ID or unauthorized extra computation |
| Quantity and scope validity | Source result schemas and scientific descriptors | Malformed values/uncertainty, missing essential meaning or incorrect scope |

**Read the inventory from the actual request; do not author a second inventory.**
The agent may receive a derived index, but its authority remains the source plan.
Current analysis includes all catalog diagnostics in its compiled plan and only
selected sensitivity branches. Those differences must be preserved by the reader.

Computation reconciliation applies when evidence was returned. If no response
exists, validate the exact attempted request and authoritative failure receipt,
without attesting that the request was approved or executed successfully. Disclose
that execution-level failure; do not require or invent per-computation outcomes.

An expected computation with explicit `failed`, `blocked`, `unavailable`,
`inapplicable` or `not_selected` status is accounted for. It is never relabeled a
pass. A computation outside the declared inventory needs no invented placeholder.
Primary diagnostics do not apply to a sensitivity unless scoped evidence says so.

Large bodies can be fetched as needed, with the same checks before use. A defect
discovered later takes the same input-rejection path; it cannot enter a prose
repair loop. Optional display data may be absent without invalidating the science.

## 4. Rejection names the responsible owner

The input check returns **accepted** or **rejected with structured issues**.
The boundary InputIssue carries `code`, `source` (a shared artifact reference),
`path`, `expected`, `received`, `owner` and `required_action`. This narrow boundary
record preserves the shared artifact identity without replacing common validation.

| Example | Owner and action |
| --- | --- |
| Planned diagnostic `d17` has no outcome record | Analysis: supply the actual terminal record or a traceable execution-failure explanation |
| Approved outcome refers to a column that the request does not use | Design/analysis integration: reconcile the conflicting source bindings |
| Required causal graph reference cannot be resolved | Design or handoff assembler: provide the exact promised artifact |
| Supporting table lacks the units needed to interpret it | Producing analysis component: provide authoritative quantity metadata |
| Valid producer schema is outside the receiver's agreed support | Post-analysis/integration: add support or reconcile compatibility; do not call it a failed analysis |
| Object storage is temporarily unreachable | Common runtime: operational recovery; do not claim the scientific source is missing or corrupt |

For instance: “Analysis input rejected: expected diagnostic `d17` for execution
`e4`; received no matching record. Owner: analysis. Required action: provide the
record and a new handoff.” No reporting-agent repair is attempted.

Post-analysis returns the issue through the common harness. The owner corrects
or clarifies the source and the harness supplies a new handoff. A traceable
clarification may preserve existing numerical results if upstream verifies that
the executed question, roles, coding, target, configuration and data are unchanged.
A substantive change needs upstream reassessment and, where necessary, reanalysis.
Post-analysis neither chooses nor performs that repair. Original artifacts remain
unchanged, and a clarification is an explicit additional source reference.

## 5. Downstream responsibility and output

After acceptance, one LLM agent reasons about the causal claim, diagnostic and
sensitivity findings, visual choices and narrative through LangGraph tools. A
separate bounded LLM review assesses claims and actual final previews alongside
code checks. No upstream defect may be waived by either model.

Persist a versioned report workspace using shared artifacts. Within it, keep
separately identifiable interpretations, display derivations, visual specs/renders,
report content and reviewed export. These need revision/dependency references,
not seven new public APIs or a second artifact storage system.

The following remain enforced:

- Source IDs, approved roles/DAG, requests, values and statuses cannot be edited.
- Every scientific claim/value resolves to source evidence or a recorded permitted
  display derivation. Tools do not fit models, invent tests or calculate new uncertainty.
- For returned evidence, every expected computation has an interpretation or
  explicit disclosure; a compact appendix is sufficient. An attempt with no
  response has an execution-level failure disclosure instead.
- A changed caption, visual or export invalidates the affected review. Release
  binds the exact inspected files and cannot use a stale passing review.

The stage-specific outcomes are **complete**, **blocked** or **incomplete**.
`complete` includes the exact report reference; `blocked` includes input issues
and their owners; `incomplete` means reporting could not finish. Running status,
attempts and recovery remain part of the common harness lifecycle. The LLM cannot
set these outcomes or replenish budgets.

See the [LangGraph plan](graph.md), [current coverage audit](coverage.md),
and [folder/refactor plan](refactor.md).

## Implemented artifact surface

The entry is `run_post_analysis(PostAnalysisDeps, analysis_id, stage_run_id,
revision, outcome: ArtifactRef, handoff: HandoffManifestV1)`. The shared envelope
owns identities, hashes, parents and producer/run metadata; post-analysis never
accepts a model-supplied analysis ID or scientific result.

| Artifact | Contents and allowed writer |
| --- | --- |
| PostAnalysisContext | Code pins the exact source chain and expected evidence |
| PostAnalysisVisual | Code validates LLM display choices and freezes source-bound files |
| PostAnalysisDraft | LLM sections/statements/citations; code validates coverage and references |
| PostAnalysisExport | Code-generated final HTML/SVG/PNG files and their hashes |
| PostAnalysisReview | Independent read-only LLM verdict plus code binding to exact dependencies |
| PostAnalysisBundle | Code-only release of the reviewed dependency set for one operational run |

There are three input paths. A complete NumericalBundle opens the normal authoring
loop. An authoritative execution failure with exactly one attempted-plan parent
opens a failure-disclosure report without inventing diagnostic outcomes. A missing
or inconsistent source rejects the handoff with InputIssue(code, owner, source,
path, expected, received, required_action). Temporary storage failures terminate as
incomplete and preserve exact references for recovery.
