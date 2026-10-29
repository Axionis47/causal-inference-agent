# The system's terms, in plain language

[Component map](README.md)

Use this page alongside the component chapters. Each definition explains what the term means in this codebase and gives a concrete example. The examples describe behavior; they are not new analysis results.

## Running the system

| Term | What it means here | Example |
|---|---|---|
| Component | A part of the system with a clear job and responsibility. It may span several files. | Dataset intake receives and registers a CSV. |
| Boundary | The point where one component hands responsibility to another. | Intake asks the profiler to describe the uploaded table. |
| Orchestration | Coordinating which operation happens next and passing it the required information. | The desk receives an exploration request, calls the exploration agent, then displays its result. |
| Workflow | The connected steps that carry out a task, including branches and stops. | Prepare context, reason about the study, fix the analysis specification, calculate, then explain. |
| Graph, node, transition | A graph describes the workflow. A node is one step; a transition says which step can follow. | The desk can go from a completed run to discussion, or return to preparation when essential context is missing. |
| Family and lane | A family is a category of causal analysis. A lane is the implemented workflow that performs it. Some declared families do not yet have an implemented analysis lane. | Difference-in-differences is one of the three implemented families. |
| Rung and ladder | A rung is a declared analysis step. The ladder describes the sequence and permitted routes through those steps. | Adjustment has a `pick_estimator` rung followed by `freeze_design`. |
| Runtime, harness, execution machinery | Supporting code that makes the workflow run: connecting steps, handling model/tool calls, checking answers, recording attempts and producing results. The exact responsibilities depend on whether we mean session runtime or analysis runtime. | Shared analysis code retries a rejected model answer within its allowed budget. |
| Engine and statistical execution | The engine implements statistical operations, often using a library. Statistical execution applies those operations to the fixed analysis specification. | The DiD engine fits the specified model to its saved panel data. |
| Model episode | One task given to the model, with a limited set of tools and attempts, ending in a checked answer or an explicit failure. | Inspect permitted diagnostics, then choose one of the offered estimators. |
| Budget | A limit on calls, attempts or time for a task. These are separate limits. | An estimator-selection step can allow three diagnostic scripts. |
| Worker, process, subprocess | A worker performs a task. A process is a running program; a subprocess is another program started by it. | The API runs desk steps in worker threads; the desk starts a separate process for a lane. |
| Sandbox | Restrictions around generated code's execution. The strength depends on the configured backend and mode. | Strict diagnostic mode limits file reads to permitted inputs and refuses backends that cannot enforce that restriction. |

For a full example of execution machinery, read [shared analysis runtime](10-analysis-runtime.md). For conversation workers and checkpoints, read [API and sessions](05-api-sessions.md).

## Passing information and checking it

| Term | What it means here | Example |
|---|---|---|
| Contract | The agreement between components about what is supplied, returned, allowed and rejected. It includes meaning and rules, as well as fields. | A handoff carries a dataset identity; the lane must reject it if the dataset bytes have changed. |
| Schema and typed record | A schema defines an answer or data object's required structure and value types. A typed record is an object following that structure. | `IngestedDataset` declares the saved dataset and version IDs, filename and original context. |
| Gate and validation | Code checks whether an input or proposed answer satisfies specified rules. A gate can reject an answer and provide the reason for another attempt. | A model cites a nonexistent fact; the citation gate rejects its answer. |
| Invariant | A rule that must continue to hold throughout the relevant operation. | Execution must use the specification fixed before effect fitting. |
| Admission | Checking supplied context against the requirements of a supported analysis. It is not proof that every causal assumption is true. | An analysis needing a cutoff cannot be admitted with that required cutoff missing. |
| Authorization | The person's instruction to run on the prepared context. | The person reviews the ready context and says to run. |
| Handoff | The saved input package passed from the desk to the analysis workflow. | `handoff.json` contains the question, scope, dataset reference and sourced context. |
| Snapshot | A saved copy of information at a particular point in time. | The memory snapshot records what was known when an analysis was authorized. |
| Freeze and frozen design | Fixing the analysis specification before calculating the target effect. The record captures the choices execution must follow. | Rows, target, estimator and inference policy are fixed before fitting. |
| Integrity seal, fingerprint, digest | A value computed from content so code can detect that the content differs from what was recorded. It does not prove the study's assumptions. | A dataset SHA256 detects changed file bytes; a design seal detects a changed specification. |
| Projection | A selected or reformatted view of existing information. The meaning depends on context. | The API builds a display view from internal state; exploration receives a table with the known outcome removed before freeze. |
| Adapter | Code that translates between the application's own records and a library or service's interface. | An engine adapter turns a frozen design into library arguments and converts the output into estimates. |
| Result envelope | The common outer structure used to return a result, including status and relevant records. | A lane returns designs, estimates, attempts and any context deficiency to the desk. |

A schema-valid answer can still fail a gate. For example, an answer may contain a correctly formatted citation string that names no real evidence. The [handoff chapter](08-readiness-handoff.md) shows how these checks connect.

## Remembering and explaining what happened

| Term | What it means here | Example |
|---|---|---|
| State | Information an operation currently carries while running. Some state is saved in checkpoints. | The current question, pending prompt and recorded run references in `DeskState`. |
| Durable memory | Saved knowledge about the dataset, including sources and confirmation status. It outlives an individual conversation. | The confirmed meaning of a column is available to a new analysis. |
| Checkpoint | Saved workflow progress and state used to resume the conversation. | The server recovers which step or user reply the desk was waiting for. |
| Ledger | The analysis's record of evidence, assumptions and accepted revisions. It is separate from dataset memory. | A lane records a proposed relationship and why a later judgment contested it. |
| Journal | The conversation's record of actions, what they read and what they left behind. | A journal step records that a new table version was selected. |
| Transcript | The messages shown in the conversation. | The user's question and the assistant's reply. |
| Provenance | Where a fact, decision or product came from and how it was produced. | A field retains the person's sentence; an artifact retains the script and source versions. |
| Address and citation | An address names a particular fact or record. A citation refers to it so a statement can be checked. | `artifact:<id>.<fact>` names a number produced by an exploration. |
| Artifact | A saved product. In the specific `Artifact` type, it is an exploration result with facts and optional picture/table files. | A requested distribution plot and the numbers supporting its caption. |
| Persistence and lifecycle | Persistence means saving information beyond the current operation. Lifecycle covers its creation, use, replacement or deletion. | Dataset memory survives a new conversation and is removed when its dataset is deleted. |
| Cache | A saved or in-process copy used to avoid repeating work. It has different authority and cleanup rules from durable knowledge. | A content-keyed profile cache avoids profiling the same file bytes repeatedly. |
| Trace | A detailed execution record of calls, tools, answers, rejections and linked files. | A trace shows which diagnostic script failed before an accepted estimator choice. |

Read [memory](07-memory.md) for knowledge rules and [persistence](09-persistence.md) for the actual storage locations. A transcript, journal, checkpoint and ledger can concern the same analysis while recording different things.

## Choosing and interpreting an analysis

| Term | What it means here | Example |
|---|---|---|
| Analysis endpoint | One declared supported kind of causal question/design, with its requirements, target and permitted methods. | Discontinuity distinguishes a cutoff eligibility effect from a cutoff receipt effect. |
| API endpoint | A URL and HTTP operation through which the frontend calls the server. This is a different use of the word endpoint. | `POST /api/datasets` ingests a CSV and optional original context. |
| Target or estimand | The causal quantity the question asks for, including whose effect, which contrast and how it is averaged. | The average effect among treated units differs from the average across everyone. |
| Estimator | The calculation method used to estimate that target from the data. | Regression, weighting and matching can be offered as alternative methods under particular supported designs. |
| Identification | The reasoning and assumptions that connect available observations to the requested causal quantity. | An adjustment design needs an account of which variables address confounding. |
| Support | Whether the data contain the observations or comparisons required by the proposed calculation. | A threshold analysis needs usable observations on the relevant sides of the cutoff. |
| Inference | How uncertainty about an estimate is calculated and reported. | An interval can be unavailable when the supported inference method lacks enough independent assignment clusters. |
| Diagnostic | A calculation that checks a relevant property of the data or design. | A score histogram reveals heaping but does not prove a threshold rule is causally valid. |
| Sensitivity analysis | Examining how conclusions vary under specified alternative assumptions or settings. | Report a range of results under the declared sensitivity procedure. |
| Candidate and lead design | A candidate is a possible specification under consideration. The lead is selected before fitting as the main design for the requested target. | A successful companion estimate does not automatically replace a failed lead. |

When reading a component, translate its terms into four questions: **What does it do? What does it receive? What does it return? Who checks it?** Code names should make those answers traceable, rather than substitute for an explanation.
