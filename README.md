# Causal desk

One CSV and one causal question enter a conversation. The desk assembles and admits the required context with the person. A human run instruction authorizes that exact snapshot. One family then reasons over it, freezes its valid designs, runs their engines and returns artifacts for discussion.

```text
Intake and deterministic profile
    → Desk: target, sourced context, endpoint readiness and authorization
    → Lane: evidence ledger, bounded decisions, certificates and frozen designs
    → Execution: fixed estimator, target, support, weights and inference
    → Artifacts: every design, attempt, comparison, falsification and sensitivity
    → Conversation: explanation, correction or hypothetical context
```

The complete [ownership tree and context policy](docs/architecture.md) keeps intake, frontend, desk, durable memory, exploration, reasoning and execution separate. There is no universal context score: context is sufficient when the relevant endpoint obligations are admitted or explicitly excluded under their policy. Causal validity remains conditional on the stated assumptions.

## Supported analysis

- Adjustment: twelve executable endpoints for empty/backdoor/general adjustment, Wald/linear IV, singleton linear frontdoor, additive natural mediation, frozen baseline modifier groups and graph-implied zero.
- Difference in differences: six structural endpoints using FEOLS, did2s, saturated event-study and LP-DiD, with explicit pooled, horizon and cohort targets.
- Discontinuity: sharp, fuzzy and eligibility effects under polynomial continuity or fixed-window local randomisation.

See the generated [endpoint catalogues](docs/lanes.md) and [acceptance report](docs/acceptance/context-lane-policy.md) for exact trajectories, real Gemini evidence, stopping behavior and unsupported configurations. Graph motifs illustrate reusable structures rather than enumerating every DAG.

## Information and execution boundaries

Memory retains values, statuses, source sentences, reasons and versions. Models draft; human statements or applicable checks confirm. A lane keeps a separate revision ledger. Every model judgement and number cites evidence, with bounded code gates. The lead is selected before inspecting target-effect estimates and remains visible if it fails.

Unrestricted exploration always withholds the outcome before a matching design freezes. Profiles own deterministic outcome summaries. DiD diagnostics use pre-treatment outcomes; adjustment graph diagnostics annotate designs after freeze; RD locks its selector before deterministic bandwidth computation. Essential omissions return typed deficiencies to the desk. Invalid population restrictions stop instead of changing the target.

Corrections and suppositions prepare a new snapshot and require another authorization. A hypothetical copy never overwrites durable memory. Superseded layouts, analyses and checkpoints are deleted rather than migrated.

## Run and verify

Copy `.env.example` to `.env`. Gemini runs on Vertex AI using gcloud application-default credentials. LangSmith traces show the workflow, including lane subprocesses under the desk.

```bash
make dev-api
make dev-web
```

The API listens on port 8000; Vite serves the frontend on 5173. Upload a CSV, give the question and study account, confirm the relevant context, then authorize the ready snapshot.

A stored study fixture exercises a lane directly:

```bash
uv run python -m causal_agent.evals.lane adjustment --handoff causal_agent/families/adjustment/evals/handoffs/backdoor_protocol.json
make trace
```

```bash
make check
make build-web
```

Checks cover imports, types, numerical engines, policy invariants, API/frontend serialization and generated schema/ladders. Real Gemini runs and trace inspection validate judgement behavior. `make` lists every command. [docs/README.md](docs/README.md) indexes the detailed documentation.
