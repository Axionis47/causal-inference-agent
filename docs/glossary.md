# Technical term register

Use these meanings throughout the current manual.
Literal schema values can differ from display labels.

| Term | Meaning |
| --- | --- |
| Analysis brief | The applied question and references needed to start analysis preparation. |
| Analysis run | The durable record for one analysis attempt with exact input bindings. |
| API | Application programming interface. Here, the FastAPI HTTP boundary. |
| Application | The complete Causal Workspace product. |
| Approval | The owner's authorization of an exact scientific snapshot and program revision. |
| Artifact | A saved product such as code, a receipt, a table, or a figure. |
| Assignment process | The process that determines treatment assignment, including unit, timing, and history. |
| Assumption | A proposition used conditionally because supplied evidence does not establish it. |
| Assumption ledger | The recorded candidate, relied-upon, and rejected assumptions with their dependencies. |
| Attempt | One reserved unit of work, including a failed or interrupted provider call. |
| Authority | The permitted use of a claim as a source statement, deduction, or conditional premise. |
| Backend | The Python API, domain operations, and background workflows. |
| Budget | A persisted limit on model calls, retrievals, computations, or repairs. |
| Causal account | The graph, identification claims, assumptions, and unresolved threats for the requested effects. |
| Causal graph | The accepted variables and directed causal relations for an analysis. |
| Checkpoint | A scientific decision with prerequisites, gates, and a persisted receipt. |
| Checkpointer | The LangGraph component that stores workflow position. It does not own scientific authority. |
| Claim | A scoped proposition submitted for authority admission. |
| Column | One physical field in a saved CSV dataset version. |
| Computation | Execution of numerical or data-processing code in the sandbox runner. |
| Context pack | The version-owned collection of context revisions. |
| Context revision | One immutable snapshot of keyed statements and an optional analysis brief. |
| Contrast | The exact comparison that defines a requested effect. |
| Conversation | The saved chat attached to a dataset version. |
| Dataset | The stable application identity for a collection of source data. |
| Dataset version | Immutable source bytes, checksum, headers, and associated metadata. |
| Deduction | A claim admitted by a supported rule with compatible recorded premises. |
| Diagnostic | A declared computation that assesses feasibility before target-effect execution. |
| Effect | A causal quantity requested for a specified contrast, population, and time scope. |
| Estimand | The mathematical quantity that the analysis intends to estimate. |
| Estimator | The numerical procedure used to estimate the estimand. |
| Event | A persisted lifecycle observation with a sequence number. |
| Execution | One approved program's computation against its approved inputs and runtime. |
| Fingerprint | A hash used to compare exact structured inputs or records. |
| Freeze | The operation that makes a specified scientific or input record immutable. |
| Frontend | The React application that runs in the browser. |
| Gate | Checks that determine whether a submission can be accepted. |
| Harness | The code that controls task inputs, capabilities, budgets, validation, and publication. |
| Identification | The argument that connects an estimand to observable data under stated assumptions. |
| Inference | A procedure for uncertainty statements, such as confidence intervals. |
| LangGraph | The workflow library used for chat and analysis execution position. |
| LangSmith | The external service used for correlated diagnostic traces. |
| Model proposal | Model output before the harness determines its permitted authority. |
| Owner | The authenticated account through which access to a record is checked. |
| Packet | The bounded input assembled for a specific model task. |
| Prerequisite | An accepted decision or required input that a task depends on. |
| Program revision | One immutable version of generated analysis code and its review records. |
| Projection | A derived selection or masking of data for a declared purpose. |
| Procrastinate | The PostgreSQL-backed background job delivery library. |
| Provenance | The record of where information came from and which inputs produced it. |
| Readiness | The backend's structural assessment of an applied analysis brief. |
| Receipt | A durable record of inputs, outputs, checks, identities, and consumed work. |
| Recovery | Continuation that reuses saved work without resetting identities or allowances. |
| Role | A variable's relation to a particular effect and treatment time. |
| Sandbox runner | The separate service that executes generated Python with restricted capabilities. |
| Scientific snapshot | The frozen causal account and executable scientific specification. |
| Scientific variable | A domain concept that can bind one or more source columns. |
| Semantic task | A task that interprets permitted context and accepted evidence without computing data values. |
| Source material | An immutable uploaded text file or note used as context evidence. |
| SSE | Server-sent events. The HTTP mechanism for analysis lifecycle updates. |
| Submission ID | A client-generated identity used to recognize a repeated request. |
| Target | The outcome or effect being studied, with its population and timing. |
| Treatment | A scientific intervention factor. It is separate from its values and assignment process. |
| Treatment value | One explicitly typed level or state of a treatment. |
| Turn | One saved user message and its processing record within a conversation. |
| Widening | Bounded retrieval or dependent work needed to resolve a specific task obligation. |
| Worker | The separate process that consumes jobs and runs application workflows. |
| Workflow graph | The task and dependency view. It is separate from the causal graph. |

## Important distinctions

**Reviewed context** records a review action.
It does not prove that a statement is true.

**Accepted checkpoint** records a passed gate and persisted decision receipt.
It does not prove that all scientific judgments are correct.

**Ready with assumptions** permits preparation from the applied brief.
It does not activate every candidate assumption.

**Succeeded** is an operation status.
Read the result kind and execution receipt before you infer that an effect was estimated.
