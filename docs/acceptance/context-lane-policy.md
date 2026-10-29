# Real-data validation

Analysis acceptance uses real-source datasets, including live Kaggle datasets supplied through the ordinary intake path. The original example datasets and source context under `data/` remain available for regression checks.

Generated evaluation tables, dataset generators, dependent handoffs and their saved acceptance reports have been removed. Their former endpoint-coverage claims are no longer presented as current validation evidence.

## Acceptance requirements

Each new evaluation records the dataset source, the supplied study account, the question and requested target. The desk must prepare and admit the context before the person authorizes the run. The resulting trace must show the endpoint reached, its assumptions and empirical checks, the alternatives rejected, the frozen specification and the execution outcome. Artifacts must explain that recorded result faithfully.

A real dataset does not establish every causal assumption. Passing code tests is also separate from real-model trajectory acceptance. Small deterministic unit inputs remain useful for checking numerical behavior and workflow invariants.

The endpoint declarations and generated [ladders](../lanes.md) describe implemented configurations. Complete endpoint coverage on live datasets remains to be established. No new Kaggle evaluation is claimed by this cleanup.

## Retained empirical evidence

- [Card and Krueger paired analysis](readbacks/did-paired.md): the recorded real-source run reached the paired two-wave endpoint and reported the limitation from two assignment clusters.
- [Cigarette-sales analysis](readbacks/did-cigar.md): the recorded real-source run stopped at its pre-treatment design checks.

These records cover those behaviors only. They do not establish validation of every supported endpoint or the full API conversation on live data.

[Code-assisted estimator selection](code-assisted-estimator-selection.md) records a separate, conditional selector-level run on the retained students CSV, including its diagnostic and interpretation limits.

[Restricted lane diagnostic coding](lane-diagnostic-code.md) records the shared sandbox and evidence gate in DiD and discontinuity, targeted real-code runs and ordinary complete analyses.

## Running a case

Use the existing real-source example to run a lane directly:

```bash
uv run python -m causal_agent.evals.lane adjustment --handoff causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json
make trace
```

For a new live dataset, upload its CSV, supply its source material and study account, complete context preparation, authorize the admitted snapshot and inspect the resulting trace. Add a sourced evaluation case after the dataset and expected design behavior have been reviewed.
