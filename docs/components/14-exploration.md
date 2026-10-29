# 14. Exploration and transformation agent

[Component map](README.md) · [Source ownership](source-index.md) · [Full contract and runtime](../exploration.md)

The desk reaches the whole-dataset exploration agent through A2A. The supplied question, complete CSV version, context and explicit artifact references are its full input scope. `inputs.py` verifies exact bytes; `inspection.py` exposes bounded previews, plot pixels and code execution; `agent.py` reuses the shared episode. `gates.py` validates the explanation and artifacts before the client publishes them.

Generated scripts run only in Docker through `sandbox.py`. Family diagnostics reuse this runner while their owners continue to restrict the diagnostic data. Transformations have a separate `TransformRequest` and retain row identities, outcome withholding and new-version validation. Neither exploration nor diagnostics update column context or causal assumptions.

Successful exploration returns methods, findings, limitations and products. Failures remain visible; missing information returns explicit requests to the chatbot. Eight executions, sixteen inspections, three final validation attempts and sixty seconds per script are the defaults.

`server/conversations` owns conversation identity and the PostgreSQL/S3 publication transaction. `memory/artifacts.py` is the owner-bound repository interface used by desk material and exploration; local output folders are only scratch/cache. Compose runs a persistent A2A agent and a separate, network-disabled worker with an enforced per-script filesystem and process boundary. There is no Docker socket mount or host execution fallback.

The owning tests are under `causal_agent/explore/tests`; shared diagnostic regressions remain in each family's tests. See [container deployment](../production/containerization.md) for setup and the distinction between page/API liveness and working generated-code execution.
