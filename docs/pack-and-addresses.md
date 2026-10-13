# The authorized context and its evidence addresses

The desk builds one `Handoff` for the chosen family from the question, target, dataset memory and deterministic profile. Its `context_id` hashes the complete admitted snapshot, including the dataset SHA256, dataset/memory versions, source statements, evidence statuses, endpoint admissions and accepted uncertainty. The run instruction authorizes the preview exactly as shown. The lane reads that handoff and its declared CSV only.

Initial documents and model readings are drafts. Relevant human confirmations and applicable data checks produce confirmed fields. An unknown or contradiction can enter only under its endpoint's explicit policy. The lane keeps its own assumption/evidence ledger; it cannot silently rewrite the person's account or remove a hidden cause.

Population restrictions use an executable grammar: column comparisons or membership, joined by `and`. Time windows use explicit bounds. Unsupported, missing or unparseable restrictions stop execution. No supported endpoint silently substitutes the whole dataset for a requested population. Outcome, treatment and post-treatment population selection are unsupported in this finite catalogue.

## Address grammar

| Address | Meaning |
|---|---|
| `col:<key>.<field>` | Sourced column meaning, timing or relationship |
| `col:<key>.profile.<facet>` | Deterministic profile fact |
| `claim:<kind>.<field>` | Sourced study condition or accepted human belief |
| `claim:story.text`, `user:turn:<n>` | Original study account and human statements |
| `context.id`, `context.admission.*`, `context.fields.*` | Authorized identity and recorded admission |
| `matrix:<family>.endpoint.<id>` | Endpoint readiness and obligations |
| `rel:<id>`, `ledger:revision.<n>` | Lane evidence entries and preserved revisions |
| `probe:<episode>.<visit>.<n>` | Unique read-only episode observation |
| `design:<id>.*` | One frozen specification, target and certificate |
| `check:<design>.<contrast>.<name>` | Diagnostic for its design and contrast |
| `estimate:<design>.<contrast>.*` | Point, interval, inference, target or confidence set |
| `falsification:<design>.<contrast>.<name>` | A distinct falsification record |
| `sensitivity:<design>.<contrast>.<method>` | A distinct sensitivity record |
| `figure:<id>`, `artifact:<id>.*` | Figures, exploration products and their facts |
| `version:<id>`, `step:<n>` | Dataset versions and conversation provenance |

A numeric comparison requires matching target, population, contrast, scale, support and aggregation weights. Otherwise results are explained side by side. Disconnected confidence regions and unresolved search tails remain explicit rather than becoming one fabricated interval. The selected lead is recorded before target-effect inspection and remains selected if execution fails.

Every model judgement cites only addresses in its displayed view or its own episode facts. An unresolved cite triggers a visible gate refusal and bounded retry. Post-run chat material projects all designs, source context, revisions, attempts, checks, falsifications, sensitivities and artifacts. Numeric statements must match their cited material within one percent. Figure `draws_on` addresses are checked before publication.

Stored eval handoffs under `families/<name>/evals/handoffs/` use this same contract. They are explicit study fixtures, not compatibility records. `desk/handoff.py::forced` constructs them from fixture memory; the ordinary API requires the conversation and authorization.
