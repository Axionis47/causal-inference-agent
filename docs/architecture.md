# Context preparation and analysis lanes

For the functional breakdown, boundary contracts and source ownership, start with the [component handbook](components/README.md). This page is the architectural overview; the handbook links each responsibility to its implementation and evidence.

One dataset and one causal question enter the desk. The desk prepares and admits a fixed context snapshot with the person. One chosen family reasons over that snapshot and executes its frozen designs. The conversation continues over the resulting artifacts, and corrections require another admitted snapshot and another authorized run.

```text
Project
├── Intake and profile
│   ├── CSV upload and deterministic column facts
│   └── Initial study material, read as sourced drafts
├── Desk
│   ├── Question, target, restrictions and family selection
│   ├── Human interview and confirmation of relevant drafts
│   ├── Endpoint readiness matrix
│   ├── Context snapshot and human run authorization
│   └── Discussion before and after analysis
├── Dataset memory
│   ├── Field values, status, source statements and reasons
│   ├── Data checks and contradictions
│   ├── Explicit human updates to causal beliefs
│   └── Owned artifact repository interface and dataset versions
├── Endpoint catalogue
│   ├── endpoints.yaml: configurations and proof obligations
│   ├── space.yaml: dimensions and numerical support limits
│   └── rungs.yaml: decisions, transitions and bounded revisits
├── Analysis lanes
│   ├── Adjustment: observed adjustment, IV, frontdoor, restricted mediation and modifiers
│   ├── Difference in differences: paired, projection, imputation, cohort cells and clean projections
│   └── Discontinuity: sharp, fuzzy or eligibility targets under continuity or local randomisation
│       Each lane owns:
│       ├── reason: evidence ledger and candidates
│       ├── engine: library adapters and deterministic checks
│       ├── execute: frozen specifications and attempts
│       └── artifacts: results, interpretations and figures
├── Exploration agent
│   ├── Restricted scripts in a separate sandbox
│   ├── Addressed facts and requested artifacts
│   └── New table versions on the person's word
├── API
│   └── Sessions, typed projections, artifacts and generated OpenAPI
└── Frontend
    └── Conversation, context admission, designs, history and result inspectors
```

## Ownership

| Component | Owns | Must not decide |
|---|---|---|
| Intake/profile | File parsing, deterministic facts, initial material | Causal validity |
| Desk | Necessary context, relevant confirmations, target restrictions, snapshot admission and authorization | Effect-driven specification selection |
| Memory | Durable sourced knowledge and its gated writes | Confirmation from a model proposal |
| Lane reasoning | Structural/assignment assumptions, certificates, support and frozen design | Human interview or post-effect redesign |
| Engine/execution | Exact frozen estimators, inference and declared fallback | Unrecorded target, clustering, horizon or population changes |
| Artifacts/chat | Cited explanations, separate falsifications and sensitivities | Invented evidence or silent lead promotion |
| Frontend | Showing the API's records | Readiness, identification or method selection |

## Context sufficiency

Good context is sufficient for the requested endpoint obligations. There is no universal context score. The authoritative requirements identify missing, failed, satisfied and permitted-unknown obligations. Statistical diagnostics cannot establish every causal assumption. Optional configurations activate only when their relevant study account or question is supplied. A discriminating missing structural fact returns a desk question. A discontinuity question also fixes whether the requested target is at the cutoff or inside a score window, and whether it concerns receipt or eligibility. The lane cannot infer those choices from a method preference. A difference-in-differences question similarly admits its averaging policy, so treated-cell averages and native projection targets remain distinct.

The existing family-by-claim matrix remains useful for routing. It also contains endpoint admissions within each surviving family. Every relevant activated endpoint must be admitted, explicitly excluded, or unavailable with a reason. The desk gathers evidence from initial material, profiles, memory and conversation, confirms relevant drafts together, and asks unresolved requirements. A model draft cannot make context ready.

The handoff contains its content identity, family contract fingerprint, dataset SHA256 and dataset/memory versions, facts with source/status/sentence/reason, scope and restrictions, and endpoint admissions. The person's run word authorizes this snapshot. The lane re-evaluates the admission and verifies its identity at entry. An essential unexpected omission becomes a `ContextDeficiency`; the desk clears readiness and requires a new authorization after correction.

## Lane reasoning and freeze

The ledger is separate from dataset memory. Each accepted revision retains the previous and new entry, stage and reason. Evidence distinguishes study accounts, data facts, measurement timing, associations, assumptions and diagnostics. Causal support requires a sourced study account; justified measurement timing can support the absence of an edge. A diagnostic alone cannot certify a causal edge or erase a supplied hidden cause. Data checks can confirm noncausal facts. Model proposals remain proposals or conditional assumptions. Each episode visit has a unique address, and every gate attempt survives.

Structural specifications are sealed before target-effect inspection. DiD diagnostics use only the declared pre-treatment outcome support. Adjustment outcome-involving graph diagnostics annotate results after freeze. RD fixes its selector policy before deterministic bandwidth computation. A frozen design carries endpoint/context identity, ledger revision, identification certificate, target, support, weights, estimator, inference and parameters. Its integrity seal covers nested fields and is checked by execution.

The bounds are three gate attempts, one declared pre-freeze revisit and at most eight executable specifications. Adjustment admits at most sixteen combinations from four uncertain structural choices. An essential bound overflow is an explicit limitation. Execution never returns to effect-informed redesign. Only an equivalent-target fallback declared before freeze can be attempted.

## Results and comparison

A result contains `designs`, `lead_design_id`, `lead_selection`, candidate decisions, the ledger with history, all episode visits, checks, estimates, falsifications, sensitivities, comparisons and execution attempts. Addresses include design identity. The recorded lead ranking uses target alignment, assumption and support evidence, catalogue rank and stable identity before fitting. If the requested target is unidentified, successful companion estimates remain a partial result without a replacement lead. If the lead fails, that failure stays visible.

Numerical comparison requires matching targets, populations, contrasts, scales and aggregation/support definitions. Different designs can be explained side by side without claiming equivalence. Confidence sets retain disconnected regions, unresolved search tails, empty results and computational failure. A receipt-effect failure does not erase a separately valid eligibility effect.

## Layers, tracing and storage

Import order is common → profile → memory → viz → lane → explore → families → desk → server. The registry is the only entry to family implementations. Import contracts forbid reasoning from effect-fitting modules and execution from reopening reasoning.

Model calls use `common/llm.py` or `lane/episode.py`; named graph nodes, gate reasons, tool calls and written files appear in the trace. A lane subprocess receives its parent's trace and thread. Analysis writers use `trace.write`.

Dataset knowledge lives in one immutable PostgreSQL revision chain, starting with extraction. Review and chat update that same context. Dataset and column browsing project its selected revision; original extraction output is audit provenance only. Reviews, selected chat revisions, messages, commands, questions and receipts are also stored in PostgreSQL. Graph checkpoints in SQLite contain execution references only. The desk derives an ephemeral Memory from the selected revision and never writes a filesystem knowledge copy. Owned analysis snapshots live under `data/conversations/<id>/analyses/<operation>/`; lane runtime files remain under `.artifacts/`. Original uploads and registered artifact files are private S3 objects. Fixture memory remains a CLI evaluation input, separate from chat authority.

Question entry and context review precede chat creation. One semantic interpretation path handles discussion, explicit edits, references and tool requests. Fast-track applies endpoint obligations but never grants run authorization. A compact sealed reference resolves only the authorized snapshot. Essential deficiencies stop visibly without an automatic interview or analysis loop. Generated scripts remain Docker-only.

See the generated [endpoint ladders](ladders/) for exact requirements, trajectories, estimator capabilities and stopping states.


Owned chat entry and shared exploration/analysis publication are implemented in `server/conversations`. Origin and format are distinct; registered evidence is loaded on demand for grounded discussion. Compose retains the persistent A2A agent and restricted script worker. See the [conversation runtime](components/05-api-sessions.md) and [storage lifecycle](components/09-persistence.md).
